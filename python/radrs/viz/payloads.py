"""Payload dataclasses and the ``prepare_*`` adapters that build them.

Payloads are shaped for transport, not for storage. Two rules run through all
five of them:

*Structure over coordinates.* A radar volume is a regular grid, so a gate's
azimuth, elevation, range and time are all recoverable from its row and column.
The payloads ship one value per row and let the widget do that arithmetic
instead of repeating six coordinates per point.

*Values are quantized.* Every value buffer is ``uint16`` over the payload's
``[vmin, vmax]``, with :data:`QUANT_NAN` for missing data. On the reflectivity
scale that is a 0.0016 dB step, far below what the instrument resolves, and it
halves the largest buffer in every payload.

Together those hold a full 360x1832 PPI sweep, which used to need about 19 MB
of float32 point buffers, to roughly 1.3 MB. What is left over is spent by
:data:`DEFAULT_BYTE_BUDGET`, which every ``prepare_*`` function honours by
choosing its own reduction rather than making the caller guess one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Final

import numpy as np
import xarray as xr

from . import assets
from .geometry import _polar_to_cartesian, beam_geometry
from .scales import (
    FALLBACK_COLORMAP,
    MOMENT_NAMES,
    ColorScale,
    _resolve_color_scale,
)

#: Ceiling on a payload's widget state, in bytes. Sits just under marimo's 5 MB
#: default output limit with room for the surrounding notebook output, so a
#: payload built with the default budget displays without the caller tuning
#: anything. Pass ``byte_budget=None`` to any ``prepare_*`` function to opt out.
DEFAULT_BYTE_BUDGET: Final[int] = 3_500_000

#: Highest finite quantization code. Codes 0..65534 map linearly onto
#: ``[vmin, vmax]``, so the step is ``(vmax - vmin) / 65534``.
QUANT_MAX: Final[int] = 65_534

#: Reserved code for NaN and missing data.
QUANT_NAN: Final[int] = 65_535

#: Seed for the payload downsampler. Fixed, so the same inputs always yield the
#: same picture; see :func:`_sample_indices` for why it is not a plain stride.
SAMPLE_SEED: Final[int] = 0x5241_4452

#: Bytes each sampled point costs in a :class:`VolumePayload`: a slot into the
#: per-return arrays, a gate index, and a quantized value.
_VOLUME_POINT_BYTES: Final[int] = 4 + 2 + 2

#: Bytes each referenced return costs in a :class:`VolumePayload`: its dataset
#: row index plus the four geometry fields the widget needs to place a gate.
_VOLUME_RETURN_BYTES: Final[int] = 4 + 4 * 4

#: Bytes per shipped row of a :class:`PolarPayload`.
_POLAR_ROW_BYTES: Final[int] = 4 + 4 + 8 + 4 + 4 + 4

#: Bytes per shipped row of a :class:`WaterfallPayload`.
_WATERFALL_ROW_BYTES: Final[int] = 4 + 4 + 8 + 2 + 4 + 4

#: Bytes per shipped row of a :class:`FoldedWaterfallPayload`.
_FOLDED_ROW_BYTES: Final[int] = 2 + 8 + 2 + 4 + 4 + 8 + 4 + 4 + 2

#: Bytes per quantized grid cell.
_CELL_BYTES: Final[int] = 2

#: Slack left for the JSON ``meta`` dict when solving a budget. Generous: the
#: grid payloads put a per-sweep list in there.
_META_RESERVE: Final[int] = 8_192


def quantize(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Map ``values`` onto ``uint16`` codes over ``[vmin, vmax]``.

    Codes run 0..:data:`QUANT_MAX` across the closed interval, so the step is
    ``(vmax - vmin) / QUANT_MAX``. Values outside the interval **clamp** to its
    ends rather than wrapping or dropping out, which matches how the widgets
    already coloured them. NaN, and anything else non-finite, becomes
    :data:`QUANT_NAN`.
    """

    array = np.asarray(values, dtype=np.float32).reshape(-1)
    span = float(vmax) - float(vmin)
    if not np.isfinite(span) or span <= 0.0:
        span = 1.0

    codes = np.full(array.shape, QUANT_NAN, dtype=np.uint16)
    finite = np.isfinite(array)
    if finite.any():
        scaled = (array[finite].astype(np.float64) - float(vmin)) / span * QUANT_MAX
        codes[finite] = np.clip(np.rint(scaled), 0.0, QUANT_MAX).astype(np.uint16)
    return codes


def dequantize(codes: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Invert :func:`quantize`, mapping :data:`QUANT_NAN` back to NaN."""

    array = np.asarray(codes, dtype=np.uint16)
    span = float(vmax) - float(vmin)
    if not np.isfinite(span) or span <= 0.0:
        span = 1.0
    values = float(vmin) + array.astype(np.float64) / QUANT_MAX * span
    return np.where(array == QUANT_NAN, np.nan, values).astype(np.float32)


def widget_state_nbytes(state: dict[str, object]) -> int:
    """Size of one widget state on the wire: binary traits plus JSON ``meta``."""

    total = 0
    for value in state.values():
        if isinstance(value, (bytes, bytearray)):
            total += len(value)
        elif isinstance(value, memoryview):
            total += value.nbytes
        else:
            total += len(json.dumps(value, default=str).encode("utf-8"))
    return total


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-int(numerator) // max(1, int(denominator)))


def _fit_grid_strides(
    n_rows: int,
    n_cols: int,
    row_bytes: int,
    cell_bytes: int,
    budget: int | None,
    *,
    min_rows: int = 1,
    min_cols: int = 1,
    row_stride: int = 1,
    col_stride: int = 1,
) -> tuple[int, int]:
    """Grow ``(row_stride, col_stride)`` until the shipped grid fits ``budget``.

    Decimation is shared between the two axes: whichever is currently longer
    gives ground first, so a 5000x1832 sweep shrinks squarely rather than
    collapsing one axis to nothing. ``min_rows``/``min_cols`` are floors below
    which an axis stops being cut even if the budget is still exceeded; a caller
    that hits them gets the smallest payload this shape can make.
    """

    if budget is None or budget <= 0 or n_rows <= 0 or n_cols <= 0:
        return max(1, row_stride), max(1, col_stride)

    def size(rows_stride: int, cols_stride: int) -> int:
        rows = _ceil_div(n_rows, rows_stride)
        cols = _ceil_div(n_cols, cols_stride)
        return rows * row_bytes + rows * cols * cell_bytes

    floor_rows = max(1, min(int(min_rows), n_rows))
    floor_cols = max(1, min(int(min_cols), n_cols))
    row_stride = max(1, int(row_stride))
    col_stride = max(1, int(col_stride))

    # Multiplicative growth, so even a 1000x overshoot converges in a few dozen
    # passes; the cap is only there to make non-termination impossible.
    for _ in range(512):
        if size(row_stride, col_stride) <= budget:
            break
        rows = _ceil_div(n_rows, row_stride)
        cols = _ceil_div(n_cols, col_stride)
        can_row = rows > floor_rows
        can_col = cols > floor_cols
        if not can_row and not can_col:
            break
        if can_col and (not can_row or cols >= rows):
            col_stride += max(1, col_stride // 2)
        else:
            row_stride += max(1, row_stride // 2)
    return row_stride, col_stride


def _fit_grid_size(grid_size: int, budget: int | None) -> int:
    """Largest square CAPPI/cross-section grid that fits ``budget``."""

    grid_size = max(1, int(grid_size))
    if budget is None or budget <= 0:
        return grid_size
    available = max(1, int(budget) - _META_RESERVE)
    limit = int(np.sqrt(available / _CELL_BYTES))
    return max(1, min(grid_size, limit))


def _volume_point_budget(budget: int | None, n_returns_total: int) -> int | None:
    """Point cap for a :class:`VolumePayload` under ``budget``.

    Two regimes, because the per-return arrays only carry the returns a sampled
    point actually lands on. Sample more points than there are returns and every
    return is referenced, so the return arrays cost a fixed amount; sample fewer
    and each point can drag in at most one new return.
    """

    if budget is None or budget <= 0:
        return None
    available = max(1, int(budget) - _META_RESERVE)
    all_returns = (available - n_returns_total * _VOLUME_RETURN_BYTES) // _VOLUME_POINT_BYTES
    if all_returns >= n_returns_total:
        return max(1, int(all_returns))
    return max(1, int(available // (_VOLUME_POINT_BYTES + _VOLUME_RETURN_BYTES)))


class _PayloadBase:
    """Shared reporting for the payload dataclasses."""

    def to_widget_state(self) -> dict[str, object]:  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def nbytes(self) -> int:
        """Size of this payload's widget state on the wire, in bytes.

        This is the number a byte budget is measured against, so it counts the
        quantized buffers actually shipped rather than the float arrays the
        dataclass holds.
        """

        return widget_state_nbytes(self.to_widget_state())


@dataclass(frozen=True)
class SweepInfo:
    """Human-friendly sweep metadata for selector controls."""

    index: int
    sweep_number: int
    elevation_deg: float
    num_returns: int
    label: str


@dataclass(frozen=True)
class PolarPayload(_PayloadBase):
    """One sweep as per-return geometry plus a dense ``(return, gate)`` grid.

    Per-point coordinates are not shipped. Every gate in row ``i``, column ``c``
    sits at azimuth ``azimuth_deg[i]`` and range
    ``base_range_m[i] + c * gate_stride * range_step_m[i]``, which is exactly
    what Python used to precompute and repeat for each of the sweep's several
    hundred thousand gates.
    """

    values: np.ndarray  # float32 (n_returns, n_gates), NaN = no data
    azimuth_deg: np.ndarray  # float32 (n_returns,)
    elevation_deg: np.ndarray  # float32 (n_returns,)
    return_time_ms: np.ndarray  # float64 (n_returns,)
    base_range_m: np.ndarray  # float32 (n_returns,)
    range_step_m: np.ndarray  # float32 (n_returns,)
    return_index: np.ndarray  # uint32 (n_returns,), dataset row of each row
    moment: str
    sweep_number: int
    sweep_time_ms: int
    max_range_m: float
    vmin: float
    vmax: float
    return_stride: int = 1
    gate_stride: int = 1
    n_returns_orig: int = 0
    n_gates_orig: int = 0
    colormap: str = FALLBACK_COLORMAP
    units: str = ""

    @property
    def n_returns(self) -> int:
        return int(self.values.shape[0]) if self.values.ndim == 2 else 0

    @property
    def n_gates(self) -> int:
        return int(self.values.shape[1]) if self.values.ndim == 2 else 0

    @property
    def point_count(self) -> int:
        """Gates with data in the shipped grid."""

        return int(np.count_nonzero(np.isfinite(self.values)))

    @property
    def cell_count(self) -> int:
        """Cells in the shipped grid, data or not."""

        return int(self.values.size)

    @property
    def gate_index(self) -> np.ndarray:
        """Original gate index of each shipped column."""

        return (np.arange(self.n_gates, dtype=np.uint32) * self.gate_stride).astype(
            np.uint16
        )

    @property
    def range_m(self) -> np.ndarray:
        """Slant range of every cell, shape ``(n_returns, n_gates)``."""

        if self.n_returns == 0 or self.n_gates == 0:
            return np.empty((0, 0), dtype=np.float32)
        gates = np.arange(self.n_gates, dtype=np.float32) * float(self.gate_stride)
        return (
            self.base_range_m[:, None] + gates[None, :] * self.range_step_m[:, None]
        ).astype(np.float32)

    def to_widget_state(self) -> dict[str, object]:
        """Serialize payload to binary traits for widget transport."""

        return {
            "value_bytes": quantize(self.values, self.vmin, self.vmax).tobytes(),
            "azimuth_bytes": np.ascontiguousarray(
                self.azimuth_deg, dtype=np.float32
            ).tobytes(),
            "elevation_bytes": np.ascontiguousarray(
                self.elevation_deg, dtype=np.float32
            ).tobytes(),
            "base_range_bytes": np.ascontiguousarray(
                self.base_range_m, dtype=np.float32
            ).tobytes(),
            "range_step_bytes": np.ascontiguousarray(
                self.range_step_m, dtype=np.float32
            ).tobytes(),
            "return_index_bytes": np.ascontiguousarray(
                self.return_index, dtype=np.uint32
            ).tobytes(),
            "return_time_ms_bytes": np.ascontiguousarray(
                self.return_time_ms, dtype=np.float64
            ).tobytes(),
            "meta": {
                "n_returns": self.n_returns,
                "n_gates": self.n_gates,
                "point_count": self.point_count,
                "return_stride": int(self.return_stride),
                "gate_stride": int(self.gate_stride),
                "n_returns_orig": int(self.n_returns_orig),
                "n_gates_orig": int(self.n_gates_orig),
                "moment": self.moment,
                "sweep_number": int(self.sweep_number),
                "sweep_time_ms": int(self.sweep_time_ms),
                "max_range_m": float(self.max_range_m),
                "vmin": float(self.vmin),
                "vmax": float(self.vmax),
                "colormap": self.colormap,
                "units": self.units,
            },
        }

    def to_html(self, width: int = 760, height: int = 760) -> str:
        """Render as a self-contained HTML document."""
        return assets.payload_to_html(
            self.to_widget_state(),
            assets.assemble_esm("polar.js"),
            assets.widget_css(),
            width,
            height,
        )


@dataclass(frozen=True)
class VolumePayload(_PayloadBase):
    """Sampled gates, or ray endpoints, as slots into per-return geometry.

    deck.gl wants an explicit positions buffer, so one is built, in JS, from
    this. Each point is a ``return_slot`` into the per-return arrays plus a
    ``gate_index``; the widget runs the same 4/3 effective-earth mapping
    :func:`radrs.viz.geometry.beam_geometry` does and lands on the same
    east/north/up metres. The Cartesian triple and the slant range are still
    available here as derived properties, they are simply not transported.
    """

    values: np.ndarray  # float32 (n_points,)
    return_slot: np.ndarray  # uint32 (n_points,), index into the arrays below
    gate_index: np.ndarray  # uint16 (n_points,)
    return_index: np.ndarray  # uint32 (n_used,), dataset row of each slot
    azimuth_deg: np.ndarray  # float32 (n_used,)
    elevation_deg: np.ndarray  # float32 (n_used,)
    base_range_m: np.ndarray  # float32 (n_used,)
    range_step_m: np.ndarray  # float32 (n_used,)
    moment: str
    render_mode: str
    max_abs_m: float
    vmin: float
    vmax: float
    colormap: str = FALLBACK_COLORMAP
    units: str = ""

    @property
    def point_count(self) -> int:
        return int(self.values.size)

    @property
    def return_count(self) -> int:
        """Distinct returns the sampled points land on."""

        return int(self.return_index.size)

    @property
    def point_azimuth_deg(self) -> np.ndarray:
        return self.azimuth_deg[self.return_slot]

    @property
    def point_elevation_deg(self) -> np.ndarray:
        return self.elevation_deg[self.return_slot]

    @property
    def point_return_index(self) -> np.ndarray:
        return self.return_index[self.return_slot]

    @property
    def range_m(self) -> np.ndarray:
        """Slant range of every point, shape ``(n_points,)``."""

        if self.point_count == 0:
            return np.empty(0, dtype=np.float32)
        slots = self.return_slot
        return (
            self.base_range_m[slots]
            + self.gate_index.astype(np.float32) * self.range_step_m[slots]
        ).astype(np.float32)

    def _cartesian(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.point_count == 0:
            empty = np.empty(0, dtype=np.float32)
            return empty, empty, empty
        x, y, z = _polar_to_cartesian(
            self.point_azimuth_deg, self.point_elevation_deg, self.range_m
        )
        return (
            x.astype(np.float32),
            y.astype(np.float32),
            z.astype(np.float32),
        )

    @property
    def x_m(self) -> np.ndarray:
        return self._cartesian()[0]

    @property
    def y_m(self) -> np.ndarray:
        return self._cartesian()[1]

    @property
    def z_m(self) -> np.ndarray:
        return self._cartesian()[2]

    def to_widget_state(self) -> dict[str, object]:
        return {
            "value_bytes": quantize(self.values, self.vmin, self.vmax).tobytes(),
            "return_slot_bytes": np.ascontiguousarray(
                self.return_slot, dtype=np.uint32
            ).tobytes(),
            "gate_index_bytes": np.ascontiguousarray(
                self.gate_index, dtype=np.uint16
            ).tobytes(),
            "return_index_bytes": np.ascontiguousarray(
                self.return_index, dtype=np.uint32
            ).tobytes(),
            "azimuth_bytes": np.ascontiguousarray(
                self.azimuth_deg, dtype=np.float32
            ).tobytes(),
            "elevation_bytes": np.ascontiguousarray(
                self.elevation_deg, dtype=np.float32
            ).tobytes(),
            "base_range_bytes": np.ascontiguousarray(
                self.base_range_m, dtype=np.float32
            ).tobytes(),
            "range_step_bytes": np.ascontiguousarray(
                self.range_step_m, dtype=np.float32
            ).tobytes(),
            "meta": {
                "point_count": self.point_count,
                "return_count": self.return_count,
                "moment": self.moment,
                "render_mode": self.render_mode,
                "max_abs_m": float(self.max_abs_m),
                "vmin": float(self.vmin),
                "vmax": float(self.vmax),
                "colormap": self.colormap,
                "units": self.units,
            },
        }

    def to_html(
        self,
        width: int = 760,
        height: int = 760,
        yaw_deg: float = 35.0,
        pitch_deg: float = 30.0,
    ) -> str:
        """Render as a self-contained HTML document."""
        return assets.payload_to_html(
            self.to_widget_state(),
            assets.assemble_esm("volume.js"),
            assets.widget_css(),
            width,
            height,
            extra_state={"yaw_deg": yaw_deg, "pitch_deg": pitch_deg},
        )


@dataclass(frozen=True)
class GridPayload(_PayloadBase):
    """2D gridded payload for CAPPI or vertical cross-section views."""

    grid: np.ndarray  # float32 (n_rows, n_cols), NaN = no data
    grid_mode: str  # "cappi" or "xsec"
    moment: str
    vmin: float
    vmax: float
    x_min: float  # world-coordinate extents (meters)
    x_max: float
    y_min: float
    y_max: float
    x_label: str  # "East (km)" or "Ground range (km)"
    y_label: str  # "North (km)" or "Altitude (km)"
    # Mode-specific metadata
    cappi_altitude_m: float | None = None
    cappi_tolerance_m: float | None = None
    xsec_azimuth_deg: float | None = None
    xsec_azimuth_tolerance_deg: float | None = None
    # Sweep context for overlay diagrams
    sweep_elevations_deg: np.ndarray | None = None  # float32, one per sweep
    sweep_num_returns: np.ndarray | None = None  # int32, one per sweep
    # Curved beam height each sweep reaches at the far edge of the plot, so the
    # elevation inset does not have to redo the geometry with flat trigonometry.
    sweep_max_altitude_m: np.ndarray | None = None  # float32, one per sweep
    colormap: str = FALLBACK_COLORMAP
    units: str = ""

    @property
    def n_rows(self) -> int:
        return int(self.grid.shape[0])

    @property
    def n_cols(self) -> int:
        return int(self.grid.shape[1])

    def to_widget_state(self) -> dict[str, object]:
        """Serialize payload to binary traits for widget transport."""

        meta: dict[str, object] = {
            "grid_mode": self.grid_mode,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "moment": self.moment,
            "vmin": float(self.vmin),
            "vmax": float(self.vmax),
            "x_min": float(self.x_min),
            "x_max": float(self.x_max),
            "y_min": float(self.y_min),
            "y_max": float(self.y_max),
            "x_label": self.x_label,
            "y_label": self.y_label,
            "colormap": self.colormap,
            "units": self.units,
        }
        if self.cappi_altitude_m is not None:
            meta["cappi_altitude_m"] = float(self.cappi_altitude_m)
        if self.cappi_tolerance_m is not None:
            meta["cappi_tolerance_m"] = float(self.cappi_tolerance_m)
        if self.xsec_azimuth_deg is not None:
            meta["xsec_azimuth_deg"] = float(self.xsec_azimuth_deg)
        if self.xsec_azimuth_tolerance_deg is not None:
            meta["xsec_azimuth_tolerance_deg"] = float(self.xsec_azimuth_tolerance_deg)
        if self.sweep_elevations_deg is not None:
            meta["sweep_elevations_deg"] = [
                float(e) for e in self.sweep_elevations_deg
            ]
        if self.sweep_num_returns is not None:
            meta["sweep_num_returns"] = [int(n) for n in self.sweep_num_returns]
        if self.sweep_max_altitude_m is not None:
            meta["sweep_max_altitude_m"] = [
                float(a) for a in self.sweep_max_altitude_m
            ]

        return {
            "grid_bytes": quantize(self.grid, self.vmin, self.vmax).tobytes(),
            "meta": meta,
        }

    def to_html(self, width: int = 760, height: int = 760) -> str:
        """Render as a self-contained HTML document."""
        return assets.payload_to_html(
            self.to_widget_state(),
            assets.assemble_esm("grid.js"),
            assets.widget_css(),
            width,
            height,
        )


@dataclass(frozen=True)
class WaterfallPayload(_PayloadBase):
    """2D heatmap of the raw (return_time, range) moment matrix."""

    grid: np.ndarray  # float32 (n_returns_ds, n_range_ds), NaN = no data
    azimuth_deg: np.ndarray  # float32 (n_returns_ds,)
    elevation_deg: np.ndarray  # float32 (n_returns_ds,)
    return_time_ms: np.ndarray  # float64 (n_returns_ds,) — ms since epoch
    sweep_number: np.ndarray  # uint16 (n_returns_ds,)
    base_range_m: np.ndarray  # float32 (n_returns_ds,), first gate of the row
    range_step_m: np.ndarray  # float32 (n_returns_ds,)
    sweep_boundaries: np.ndarray  # int32 — indices where sweep changes
    moment: str
    vmin: float
    vmax: float
    n_returns_orig: int
    n_range_orig: int
    # Median range axis, for tick placement only. Ranges reported on hover come
    # from the per-row pair above, which is exact even when rows disagree.
    axis_start_m: float
    axis_step_m: float
    axis_end_m: float
    return_stride: int = 1
    range_stride: int = 1
    colormap: str = FALLBACK_COLORMAP
    units: str = ""

    @property
    def n_returns(self) -> int:
        return int(self.grid.shape[0]) if self.grid.ndim == 2 else 0

    @property
    def n_range(self) -> int:
        return int(self.grid.shape[1]) if self.grid.ndim == 2 else 0

    def to_widget_state(self) -> dict[str, object]:
        """Serialize payload to binary traits for widget transport."""

        return {
            "grid_bytes": quantize(self.grid, self.vmin, self.vmax).tobytes(),
            "azimuth_bytes": np.ascontiguousarray(
                self.azimuth_deg, dtype=np.float32
            ).tobytes(),
            "elevation_bytes": np.ascontiguousarray(
                self.elevation_deg, dtype=np.float32
            ).tobytes(),
            "return_time_ms_bytes": np.ascontiguousarray(
                self.return_time_ms, dtype=np.float64
            ).tobytes(),
            "sweep_number_bytes": np.ascontiguousarray(
                self.sweep_number, dtype=np.uint16
            ).tobytes(),
            "base_range_bytes": np.ascontiguousarray(
                self.base_range_m, dtype=np.float32
            ).tobytes(),
            "range_step_bytes": np.ascontiguousarray(
                self.range_step_m, dtype=np.float32
            ).tobytes(),
            "sweep_boundary_bytes": np.ascontiguousarray(
                self.sweep_boundaries, dtype=np.int32
            ).tobytes(),
            "meta": {
                "n_returns": self.n_returns,
                "n_range": self.n_range,
                "moment": self.moment,
                "vmin": float(self.vmin),
                "vmax": float(self.vmax),
                "n_returns_orig": self.n_returns_orig,
                "n_range_orig": self.n_range_orig,
                "return_stride": int(self.return_stride),
                "range_stride": int(self.range_stride),
                "axis_start_m": float(self.axis_start_m),
                "axis_step_m": float(self.axis_step_m),
                "axis_end_m": float(self.axis_end_m),
                "colormap": self.colormap,
                "units": self.units,
            },
        }

    def to_html(self, width: int = 760, height: int = 760) -> str:
        """Render as a self-contained HTML document."""
        return assets.payload_to_html(
            self.to_widget_state(),
            assets.assemble_esm("waterfall.js"),
            assets.widget_css(),
            width,
            height,
        )


@dataclass(frozen=True)
class FoldedWaterfallPayload(_PayloadBase):
    """Row-per-return view of the folded returns matrix.

    Unlike :class:`WaterfallPayload`, rows are *not* strided: a window of
    consecutive returns is shown verbatim so every fold is visible. Returns
    chunked from one radial share a ``return_time``, and ``group_starts``
    marks where each such set of folds begins.
    """

    grid: np.ndarray  # float32 (n_rows, n_range), NaN = no data
    vcp_index: np.ndarray  # uint16 (n_rows,) — volume ordinal within the batch
    vcp_time_ms: np.ndarray  # float64 (n_rows,)
    sweep_number: np.ndarray  # uint16 (n_rows,)
    azimuth_deg: np.ndarray  # float32 (n_rows,)
    elevation_deg: np.ndarray  # float32 (n_rows,)
    return_time_ms: np.ndarray  # float64 (n_rows,)
    base_range_m: np.ndarray  # float32 (n_rows,) — first gate of this fold
    range_step_m: np.ndarray  # float32 (n_rows,)
    fold_index: np.ndarray  # uint16 (n_rows,) — position within its fold set
    group_starts: np.ndarray  # int32 — rows beginning a new return_time
    moment: str
    vmin: float
    vmax: float
    row_offset: int
    row_total: int
    range_stride: int
    n_range_orig: int
    colormap: str = FALLBACK_COLORMAP
    units: str = ""

    @property
    def n_rows(self) -> int:
        return int(self.grid.shape[0]) if self.grid.ndim == 2 else 0

    @property
    def n_range(self) -> int:
        return int(self.grid.shape[1]) if self.grid.ndim == 2 else 0

    @property
    def n_groups(self) -> int:
        """Number of fold sets in the window."""
        return int(self.group_starts.size) + (1 if self.n_rows else 0)

    def to_widget_state(self) -> dict[str, object]:
        """Serialize payload to binary traits for widget transport."""

        return {
            "grid_bytes": quantize(self.grid, self.vmin, self.vmax).tobytes(),
            "vcp_index_bytes": np.ascontiguousarray(
                self.vcp_index, dtype=np.uint16
            ).tobytes(),
            "vcp_time_ms_bytes": np.ascontiguousarray(
                self.vcp_time_ms, dtype=np.float64
            ).tobytes(),
            "sweep_number_bytes": np.ascontiguousarray(
                self.sweep_number, dtype=np.uint16
            ).tobytes(),
            "azimuth_bytes": np.ascontiguousarray(
                self.azimuth_deg, dtype=np.float32
            ).tobytes(),
            "elevation_bytes": np.ascontiguousarray(
                self.elevation_deg, dtype=np.float32
            ).tobytes(),
            "return_time_ms_bytes": np.ascontiguousarray(
                self.return_time_ms, dtype=np.float64
            ).tobytes(),
            "base_range_bytes": np.ascontiguousarray(
                self.base_range_m, dtype=np.float32
            ).tobytes(),
            "range_step_bytes": np.ascontiguousarray(
                self.range_step_m, dtype=np.float32
            ).tobytes(),
            "fold_index_bytes": np.ascontiguousarray(
                self.fold_index, dtype=np.uint16
            ).tobytes(),
            "group_start_bytes": np.ascontiguousarray(
                self.group_starts, dtype=np.int32
            ).tobytes(),
            "meta": {
                "n_rows": self.n_rows,
                "n_range": self.n_range,
                "n_groups": self.n_groups,
                "moment": self.moment,
                "vmin": float(self.vmin),
                "vmax": float(self.vmax),
                "row_offset": self.row_offset,
                "row_total": self.row_total,
                "range_stride": self.range_stride,
                "n_range_orig": self.n_range_orig,
                "colormap": self.colormap,
                "units": self.units,
            },
        }

    def to_html(self, width: int = 900, height: int = 900) -> str:
        """Render as a self-contained HTML document."""
        return assets.payload_to_html(
            self.to_widget_state(),
            assets.assemble_esm("folded_waterfall.js"),
            assets.widget_css(),
            width,
            height,
        )


def get_returns_and_sweeps(dt: xr.DataTree) -> tuple[xr.Dataset, xr.Dataset]:
    """Extract returns/sweeps datasets from a raystack DataTree."""

    if "returns" not in dt.children:
        raise KeyError("DataTree missing 'returns' node")
    if "sweeps" not in dt.children:
        raise KeyError("DataTree missing 'sweeps' node")
    returns = dt["returns"].dataset
    sweeps = dt["sweeps"].dataset
    return returns, sweeps


def available_moments(returns: xr.Dataset, include_qc: bool = True) -> list[str]:
    """List renderable moment variables from returns dataset."""

    moments = [name for name in MOMENT_NAMES if name in returns.data_vars]
    if include_qc:
        moments.extend(sorted(name for name in returns.data_vars if name.startswith("qc.")))
    return moments


def _sweep_counts(sweeps: xr.Dataset) -> np.ndarray:
    if "num_returns" in sweeps:
        return np.asarray(sweeps["num_returns"].values, dtype=np.int64)
    if "n_radials" in sweeps:
        return np.asarray(sweeps["n_radials"].values, dtype=np.int64)
    raise KeyError("sweeps dataset missing 'num_returns' and 'n_radials'")


def sweep_offsets(sweeps: xr.Dataset) -> np.ndarray:
    """Return cumulative return offsets per sweep, shape (n_sweeps + 1,)."""

    counts = _sweep_counts(sweeps)
    offsets = np.zeros(counts.size + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts, dtype=np.int64)
    return offsets


def sweep_infos(sweeps: xr.Dataset) -> list[SweepInfo]:
    """Build sweep selector metadata with stable labels."""

    offsets = sweep_offsets(sweeps)
    counts = np.diff(offsets)

    if "sweep_number" in sweeps:
        numbers = np.asarray(sweeps["sweep_number"].values, dtype=np.int64)
    else:
        numbers = np.arange(counts.size, dtype=np.int64)
    if "elevation_angle" in sweeps:
        elevations = np.asarray(sweeps["elevation_angle"].values, dtype=np.float32)
    elif "sweep_fixed_angle" in sweeps:
        elevations = np.asarray(sweeps["sweep_fixed_angle"].values, dtype=np.float32)
    else:
        elevations = np.full(counts.size, np.nan, dtype=np.float32)

    infos: list[SweepInfo] = []
    for idx, (number, elev, count) in enumerate(zip(numbers, elevations, counts)):
        label = f"{idx:02d} | swp {int(number)} | elev {float(elev):.2f} deg | {int(count)} returns"
        infos.append(
            SweepInfo(
                index=idx,
                sweep_number=int(number),
                elevation_deg=float(elev),
                num_returns=int(count),
                label=label,
            )
        )
    return infos


def _sample_indices(n_points: int, max_points: int) -> np.ndarray:
    """Choose ``max_points`` of ``n_points``, sorted, without aliasing.

    A constant stride is the obvious choice and the wrong one. The flat arrays
    these indices address are return-major, so striding by ``s`` over a matrix
    ``n_range`` gates wide only ever lands on gates that are multiples of
    ``gcd(s, n_range)``. With a power-of-two fold size, which is the normal
    case, that collapses a PPI to a handful of concentric rings.

    A fixed-seed permutation has no such structure and is just as reproducible:
    the same inputs give the same picture on every run and every machine, since
    NumPy pins ``default_rng``'s stream.
    """

    n_points = int(n_points)
    max_points = int(max_points)
    if max_points >= n_points:
        return np.arange(n_points, dtype=np.int64)
    keep = np.random.default_rng(SAMPLE_SEED).permutation(n_points)[:max_points]
    keep.sort()
    return keep.astype(np.int64, copy=False)


def _empty_polar_payload(
    moment: str,
    sweep_number: int,
    sweep_time_ms: int,
    colormap: str,
    units: str,
    n_returns_orig: int = 0,
    n_gates_orig: int = 0,
) -> PolarPayload:
    return PolarPayload(
        values=np.empty((0, 0), dtype=np.float32),
        azimuth_deg=np.empty(0, dtype=np.float32),
        elevation_deg=np.empty(0, dtype=np.float32),
        return_time_ms=np.empty(0, dtype=np.float64),
        base_range_m=np.empty(0, dtype=np.float32),
        range_step_m=np.empty(0, dtype=np.float32),
        return_index=np.empty(0, dtype=np.uint32),
        moment=moment,
        sweep_number=sweep_number,
        sweep_time_ms=sweep_time_ms,
        max_range_m=0.0,
        vmin=0.0,
        vmax=1.0,
        n_returns_orig=n_returns_orig,
        n_gates_orig=n_gates_orig,
        colormap=colormap,
        units=units,
    )


def prepare_polar_payload(
    returns: xr.Dataset,
    sweeps: xr.Dataset,
    sweep_index: int,
    moment: str,
    max_points: int | None = None,
    color_scale: ColorScale | None = None,
    *,
    byte_budget: int | None = DEFAULT_BYTE_BUDGET,
) -> PolarPayload:
    """Create a single-sweep polar payload.

    Parameters
    ----------
    max_points
        Explicit ceiling on shipped grid cells. ``None`` (default) leaves the
        sizing to ``byte_budget``; a value here only ever tightens it further.
    color_scale
        Overrides the :data:`COLOR_SCALES` default for the moment.
    byte_budget
        Ceiling on :attr:`PolarPayload.nbytes`. Returns and range gates are
        decimated together until the payload fits. ``None`` ships the sweep
        whole, which for a 360x1832 sweep is about 1.3 MB.
    """

    if moment not in returns.data_vars:
        raise KeyError(f"moment '{moment}' not found in returns dataset")

    offsets = sweep_offsets(sweeps)
    n_sweeps = offsets.size - 1
    if sweep_index < 0 or sweep_index >= n_sweeps:
        raise IndexError(f"sweep_index {sweep_index} out of range [0, {n_sweeps - 1}]")

    start = int(offsets[sweep_index])
    end = int(offsets[sweep_index + 1])
    sweep_slice = slice(start, end)

    moment_matrix = np.asarray(returns[moment].values[sweep_slice], dtype=np.float32)
    if moment_matrix.ndim != 2:
        raise ValueError(f"moment '{moment}' must be 2D on (return_time, range)")

    n_returns_orig, n_gates_orig = moment_matrix.shape

    if "sweep_number" in sweeps:
        sweep_number = int(np.asarray(sweeps["sweep_number"].values, dtype=np.int64)[sweep_index])
    else:
        sweep_number = int(sweep_index)
    if "sweep_time" in sweeps.coords:
        sweep_time = np.asarray(sweeps["sweep_time"].values, dtype="datetime64[ms]")
        sweep_time_ms = int(np.asarray(sweep_time, dtype=np.int64)[sweep_index])
    else:
        return_time_full = np.asarray(returns["return_time"].values, dtype="datetime64[ms]")
        sweep_time_ms = (
            int(np.asarray(return_time_full, dtype=np.int64)[start]) if start < return_time_full.size else 0
        )

    if n_returns_orig == 0 or n_gates_orig == 0 or not np.any(np.isfinite(moment_matrix)):
        _, _, empty_colormap, empty_units = _resolve_color_scale(
            moment, moment_matrix[:0], color_scale, returns, sweeps
        )
        return _empty_polar_payload(
            moment,
            sweep_number,
            sweep_time_ms,
            empty_colormap,
            empty_units,
            n_returns_orig,
            n_gates_orig,
        )

    # An explicit cell cap first, then the byte budget on top of whatever that
    # left. Both only ever coarsen, so applying them in series is safe.
    return_stride = gate_stride = 1
    if max_points is not None and max_points > 0:
        return_stride, gate_stride = _fit_grid_strides(
            n_returns_orig, n_gates_orig, 0, 1, int(max_points), min_rows=2, min_cols=2
        )
    return_stride, gate_stride = _fit_grid_strides(
        n_returns_orig,
        n_gates_orig,
        _POLAR_ROW_BYTES,
        _CELL_BYTES,
        None if byte_budget is None else max(1, int(byte_budget) - _META_RESERVE),
        min_rows=2,
        min_cols=2,
        row_stride=return_stride,
        col_stride=gate_stride,
    )

    grid = moment_matrix[::return_stride, ::gate_stride]
    rows = slice(start, end, return_stride)

    azimuth = np.asarray(returns["azimuth"].values[rows], dtype=np.float32)
    elevation = np.asarray(returns["elevation"].values[rows], dtype=np.float32)
    base_range = np.asarray(returns["base_range"].values[rows], dtype=np.float32)
    range_step = np.asarray(returns["range_step"].values[rows], dtype=np.float32)
    return_time = np.asarray(returns["return_time"].values[rows], dtype="datetime64[ms]")
    return_time_ms = np.asarray(return_time, dtype=np.int64).astype(np.float64, copy=False)
    return_index = np.arange(start, end, return_stride, dtype=np.uint32)

    vmin, vmax, colormap, units = _resolve_color_scale(
        moment, grid, color_scale, returns, sweeps
    )

    last_gate = float((grid.shape[1] - 1) * gate_stride)
    max_range_m = float(np.nanmax(base_range + last_gate * range_step))

    return PolarPayload(
        values=np.ascontiguousarray(grid, dtype=np.float32),
        azimuth_deg=np.ascontiguousarray(azimuth, dtype=np.float32),
        elevation_deg=np.ascontiguousarray(elevation, dtype=np.float32),
        return_time_ms=np.ascontiguousarray(return_time_ms, dtype=np.float64),
        base_range_m=np.ascontiguousarray(base_range, dtype=np.float32),
        range_step_m=np.ascontiguousarray(range_step, dtype=np.float32),
        return_index=return_index,
        moment=moment,
        sweep_number=sweep_number,
        sweep_time_ms=sweep_time_ms,
        max_range_m=max_range_m,
        vmin=vmin,
        vmax=vmax,
        return_stride=return_stride,
        gate_stride=gate_stride,
        n_returns_orig=int(n_returns_orig),
        n_gates_orig=int(n_gates_orig),
        colormap=colormap,
        units=units,
    )


def _empty_volume_payload(
    moment: str, render_mode: str, colormap: str, units: str
) -> VolumePayload:
    return VolumePayload(
        values=np.empty(0, dtype=np.float32),
        return_slot=np.empty(0, dtype=np.uint32),
        gate_index=np.empty(0, dtype=np.uint16),
        return_index=np.empty(0, dtype=np.uint32),
        azimuth_deg=np.empty(0, dtype=np.float32),
        elevation_deg=np.empty(0, dtype=np.float32),
        base_range_m=np.empty(0, dtype=np.float32),
        range_step_m=np.empty(0, dtype=np.float32),
        moment=moment,
        render_mode=render_mode,
        max_abs_m=0.0,
        vmin=0.0,
        vmax=1.0,
        colormap=colormap,
        units=units,
    )


def _build_volume_payload(
    returns: xr.Dataset,
    moment: str,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    values: np.ndarray,
    render_mode: str,
    color_scale: ColorScale | None,
) -> VolumePayload:
    """Assemble a :class:`VolumePayload` from per-point ``(row, gate)`` pairs.

    Only the returns the points actually land on are shipped, remapped to dense
    slots. A sampled cloud usually touches most rows once, so the arrays stay
    small while the per-point cost drops to a slot, a gate and a value.
    """

    used_rows, slots = np.unique(row_idx, return_inverse=True)
    used_rows = used_rows.astype(np.int64, copy=False)

    azimuth = np.asarray(returns["azimuth"].values, dtype=np.float32)[used_rows]
    elevation = np.asarray(returns["elevation"].values, dtype=np.float32)[used_rows]
    base_range = np.asarray(returns["base_range"].values, dtype=np.float32)[used_rows]
    range_step = np.asarray(returns["range_step"].values, dtype=np.float32)[used_rows]

    vmin, vmax, colormap, units = _resolve_color_scale(
        moment, values, color_scale, returns
    )

    # The extent the widget zooms to still comes from real coordinates, so they
    # are computed here once and then thrown away rather than transported.
    range_m = base_range[slots] + col_idx.astype(np.float32) * range_step[slots]
    x, y, z = _polar_to_cartesian(azimuth[slots], elevation[slots], range_m)
    max_abs_m = float(
        max(
            np.nanmax(np.abs(x)) if x.size else 0.0,
            np.nanmax(np.abs(y)) if y.size else 0.0,
            np.nanmax(np.abs(z)) if z.size else 0.0,
        )
    )

    return VolumePayload(
        values=np.ascontiguousarray(values, dtype=np.float32),
        return_slot=np.ascontiguousarray(slots, dtype=np.uint32),
        gate_index=np.ascontiguousarray(col_idx, dtype=np.uint16),
        return_index=np.ascontiguousarray(used_rows, dtype=np.uint32),
        azimuth_deg=np.ascontiguousarray(azimuth, dtype=np.float32),
        elevation_deg=np.ascontiguousarray(elevation, dtype=np.float32),
        base_range_m=np.ascontiguousarray(base_range, dtype=np.float32),
        range_step_m=np.ascontiguousarray(range_step, dtype=np.float32),
        moment=moment,
        render_mode=render_mode,
        max_abs_m=max_abs_m,
        vmin=vmin,
        vmax=vmax,
        colormap=colormap,
        units=units,
    )


def _effective_point_cap(
    max_points: int | None, byte_budget: int | None, n_returns_total: int
) -> int | None:
    """Tightest of the caller's cap and the budget's own."""

    budget_cap = _volume_point_budget(byte_budget, n_returns_total)
    caps = [c for c in (max_points, budget_cap) if c is not None and c > 0]
    return min(caps) if caps else None


def prepare_volume_payload(
    returns: xr.Dataset,
    moment: str,
    max_points: int | None = None,
    color_scale: ColorScale | None = None,
    *,
    byte_budget: int | None = DEFAULT_BYTE_BUDGET,
) -> VolumePayload:
    """Create a volume-wide gate cloud from all finite gates.

    Parameters
    ----------
    max_points
        Explicit ceiling on sampled gates. ``None`` (default) leaves the sizing
        to ``byte_budget``; a value here only ever tightens it further.
    color_scale
        Overrides the :data:`COLOR_SCALES` default for the moment.
    byte_budget
        Ceiling on :attr:`VolumePayload.nbytes`, spent on sample count.
    """

    if moment not in returns.data_vars:
        raise KeyError(f"moment '{moment}' not found in returns dataset")

    moment_matrix = np.asarray(returns[moment].values, dtype=np.float32)
    if moment_matrix.ndim != 2:
        raise ValueError(f"moment '{moment}' must be 2D on (return_time, range)")

    n_returns, n_range = moment_matrix.shape
    finite = np.isfinite(moment_matrix)
    n_finite = int(finite.sum())

    if n_returns == 0 or n_range == 0 or n_finite == 0:
        _, _, empty_colormap, empty_units = _resolve_color_scale(
            moment, moment_matrix[:0], color_scale, returns
        )
        return _empty_volume_payload(moment, "points", empty_colormap, empty_units)

    row_idx, col_idx = np.nonzero(finite)
    values_sel = moment_matrix[row_idx, col_idx]

    cap = _effective_point_cap(max_points, byte_budget, n_returns)
    if cap is not None and n_finite > cap:
        keep = _sample_indices(n_finite, cap)
        row_idx = row_idx[keep]
        col_idx = col_idx[keep]
        values_sel = values_sel[keep]

    return _build_volume_payload(
        returns, moment, row_idx, col_idx, values_sel, "points", color_scale
    )


def prepare_ray_payload(
    returns: xr.Dataset,
    moment: str,
    max_points: int | None = None,
    color_scale: ColorScale | None = None,
    *,
    byte_budget: int | None = DEFAULT_BYTE_BUDGET,
) -> VolumePayload:
    """Create a ray-centric payload with one full-length ray per return.

    Each ray is coloured by its strongest finite gate in this moment.

    Parameters
    ----------
    max_points
        Explicit ceiling on sampled rays. ``None`` (default) leaves the sizing
        to ``byte_budget``.
    color_scale
        Overrides the :data:`COLOR_SCALES` default for the moment.
    byte_budget
        Ceiling on :attr:`VolumePayload.nbytes`, spent on sample count.
    """

    if moment not in returns.data_vars:
        raise KeyError(f"moment '{moment}' not found in returns dataset")

    moment_matrix = np.asarray(returns[moment].values, dtype=np.float32)
    if moment_matrix.ndim != 2:
        raise ValueError(f"moment '{moment}' must be 2D on (return_time, range)")

    n_returns, n_range = moment_matrix.shape
    if n_returns == 0 or n_range == 0:
        _, _, empty_colormap, empty_units = _resolve_color_scale(
            moment, moment_matrix[:0], color_scale, returns
        )
        return _empty_volume_payload(moment, "rays", empty_colormap, empty_units)

    row_idx = np.arange(n_returns, dtype=np.int64)
    col_idx = np.full(n_returns, n_range - 1, dtype=np.int64)

    # Colour each full ray by its strongest finite gate in this moment.
    finite = np.isfinite(moment_matrix)
    has_finite = np.any(finite, axis=1)
    safe_matrix = np.where(finite, moment_matrix, -np.inf)
    ray_max = np.max(safe_matrix, axis=1)
    values_sel = np.where(has_finite, ray_max, np.nan).astype(np.float32, copy=False)

    cap = _effective_point_cap(max_points, byte_budget, n_returns)
    if cap is not None and n_returns > cap:
        keep = _sample_indices(n_returns, cap)
        row_idx = row_idx[keep]
        col_idx = col_idx[keep]
        values_sel = values_sel[keep]

    return _build_volume_payload(
        returns, moment, row_idx, col_idx, values_sel, "rays", color_scale
    )


def _extract_all_gates(
    returns: xr.Dataset, moment: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract all finite gates and return (azimuth, elevation, range, values, finite_mask) arrays.

    Returns flat arrays for azimuth_deg, elevation_deg, range_m, values
    over all finite gates.
    """
    moment_matrix = np.asarray(returns[moment].values, dtype=np.float32)
    if moment_matrix.ndim != 2:
        raise ValueError(f"moment '{moment}' must be 2D on (return_time, range)")

    n_returns, n_range = moment_matrix.shape
    flat_values = moment_matrix.reshape(-1)
    finite = np.isfinite(flat_values)

    if n_returns == 0 or n_range == 0 or not np.any(finite):
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, empty, empty, finite

    azimuth = np.asarray(returns["azimuth"].values, dtype=np.float32)
    elevation = np.asarray(returns["elevation"].values, dtype=np.float32)
    base_range = np.asarray(returns["base_range"].values, dtype=np.float32)
    range_step = np.asarray(returns["range_step"].values, dtype=np.float32)

    gate_indices = np.tile(np.arange(n_range, dtype=np.float32), n_returns)
    azimuth_full = np.repeat(azimuth, n_range)
    elevation_full = np.repeat(elevation, n_range)
    base_range_full = np.repeat(base_range, n_range)
    range_step_full = np.repeat(range_step, n_range)

    range_full = base_range_full + gate_indices * range_step_full

    return (
        azimuth_full[finite],
        elevation_full[finite],
        range_full[finite],
        flat_values[finite],
        finite,
    )


def _sweep_metadata(
    sweeps: xr.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract per-sweep elevations and return counts."""
    infos = sweep_infos(sweeps)
    elevations = np.array([s.elevation_deg for s in infos], dtype=np.float32)
    counts = np.array([s.num_returns for s in infos], dtype=np.int32)
    return elevations, counts


def _sweep_max_altitudes(
    elevations_deg: np.ndarray, range_m: float
) -> np.ndarray:
    """Height each sweep's beam reaches at ``range_m``, in meters.

    The widget's elevation inset draws every sweep as a straight line and marks
    which ones reach the CAPPI altitude. Computing that reach here, with the
    same 4/3 model that placed the gates, keeps the diagram honest: a flat-earth
    ``range * sin(elevation)`` understates a low sweep by kilometres at long
    range and would grey out sweeps that do carry data.
    """

    _, heights = beam_geometry(
        float(abs(range_m)), np.asarray(elevations_deg, dtype=np.float64)
    )
    return np.ascontiguousarray(heights, dtype=np.float32)


def prepare_cappi_payload(
    returns: xr.Dataset,
    sweeps: xr.Dataset,
    moment: str,
    altitude_m: float,
    tolerance_m: float | None = None,
    grid_size: int = 500,
    max_range_m: float | None = None,
    color_scale: ColorScale | None = None,
    *,
    byte_budget: int | None = DEFAULT_BYTE_BUDGET,
) -> GridPayload:
    """Create a CAPPI (constant altitude) horizontal slice through the volume.

    ``color_scale`` overrides the :data:`COLOR_SCALES` default for the moment.
    ``byte_budget`` caps :attr:`GridPayload.nbytes`; a quantized 500x500 grid is
    half a megabyte, so the cap only bites on unusually fine grids.
    """

    if moment not in returns.data_vars:
        raise KeyError(f"moment '{moment}' not found in returns dataset")

    grid_size = _fit_grid_size(grid_size, byte_budget)

    az, el, rng, vals, finite = _extract_all_gates(returns, moment)

    if az.size == 0:
        grid = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
        sweep_el, sweep_nr = _sweep_metadata(sweeps)
        _, _, empty_colormap, empty_units = _resolve_color_scale(
            moment, az, color_scale, returns, sweeps
        )
        return GridPayload(
            grid=grid,
            grid_mode="cappi",
            moment=moment,
            vmin=0.0,
            vmax=1.0,
            x_min=-1.0,
            x_max=1.0,
            y_min=-1.0,
            y_max=1.0,
            x_label="East (km)",
            y_label="North (km)",
            cappi_altitude_m=altitude_m,
            cappi_tolerance_m=tolerance_m or 500.0,
            sweep_elevations_deg=sweep_el,
            sweep_num_returns=sweep_nr,
            sweep_max_altitude_m=_sweep_max_altitudes(sweep_el, 1.0),
            colormap=empty_colormap,
            units=empty_units,
        )

    x, y, z = _polar_to_cartesian(az, el, rng)

    # Auto-compute tolerance if not provided
    if tolerance_m is None:
        z_finite = z[np.isfinite(z)]
        if z_finite.size > 1:
            unique_els = np.unique(np.round(el, decimals=1))
            n_els = max(len(unique_els), 1)
            z_range = float(np.ptp(z_finite))
            tolerance_m = max(z_range / (2.0 * n_els), 100.0)
        else:
            tolerance_m = 500.0

    # Select gates in altitude band
    mask = np.abs(z - altitude_m) < tolerance_m
    x_sel = x[mask]
    y_sel = y[mask]
    vals_sel = vals[mask]

    # Determine grid extents
    if max_range_m is not None:
        extent = float(max_range_m)
    elif x_sel.size > 0:
        extent = float(
            max(np.nanmax(np.abs(x_sel)), np.nanmax(np.abs(y_sel)), 1.0)
        )
    else:
        extent = float(max(np.nanmax(np.abs(x)), np.nanmax(np.abs(y)), 1.0))

    x_min, x_max = -extent, extent
    y_min, y_max = -extent, extent

    # Bin into regular grid
    grid = np.full((grid_size, grid_size), np.nan, dtype=np.float64)
    count = np.zeros((grid_size, grid_size), dtype=np.int32)

    if x_sel.size > 0:
        col = ((x_sel - x_min) / (x_max - x_min) * grid_size).astype(np.int64)
        row = ((y_max - y_sel) / (y_max - y_min) * grid_size).astype(np.int64)
        valid = (col >= 0) & (col < grid_size) & (row >= 0) & (row < grid_size)
        col = col[valid]
        row = row[valid]
        v = vals_sel[valid].astype(np.float64)

        # Use np.add.at for accumulation
        accum = np.zeros((grid_size, grid_size), dtype=np.float64)
        np.add.at(accum, (row, col), v)
        np.add.at(count, (row, col), 1)

        has_data = count > 0
        grid[has_data] = accum[has_data] / count[has_data]

    grid = grid.astype(np.float32)
    vmin, vmax, colormap, units = _resolve_color_scale(
        moment, grid, color_scale, returns, sweeps
    )

    sweep_el, sweep_nr = _sweep_metadata(sweeps)

    return GridPayload(
        grid=grid,
        grid_mode="cappi",
        moment=moment,
        vmin=vmin,
        vmax=vmax,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        x_label="East (km)",
        y_label="North (km)",
        cappi_altitude_m=altitude_m,
        cappi_tolerance_m=tolerance_m,
        sweep_elevations_deg=sweep_el,
        sweep_num_returns=sweep_nr,
        sweep_max_altitude_m=_sweep_max_altitudes(sweep_el, x_max),
        colormap=colormap,
        units=units,
    )


def prepare_xsec_payload(
    returns: xr.Dataset,
    sweeps: xr.Dataset,
    moment: str,
    azimuth_deg: float,
    azimuth_tolerance_deg: float = 2.0,
    grid_size: int = 500,
    color_scale: ColorScale | None = None,
    *,
    byte_budget: int | None = DEFAULT_BYTE_BUDGET,
) -> GridPayload:
    """Create a vertical cross-section along a target azimuth (and its opposite).

    ``color_scale`` overrides the :data:`COLOR_SCALES` default for the moment.
    ``byte_budget`` caps :attr:`GridPayload.nbytes` by capping ``grid_size``.
    """

    if moment not in returns.data_vars:
        raise KeyError(f"moment '{moment}' not found in returns dataset")

    grid_size = _fit_grid_size(grid_size, byte_budget)

    az, el, rng, vals, finite = _extract_all_gates(returns, moment)

    sweep_el, sweep_nr = _sweep_metadata(sweeps)

    if az.size == 0:
        grid = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
        _, _, empty_colormap, empty_units = _resolve_color_scale(
            moment, az, color_scale, returns, sweeps
        )
        return GridPayload(
            grid=grid,
            grid_mode="xsec",
            moment=moment,
            vmin=0.0,
            vmax=1.0,
            x_min=-1.0,
            x_max=1.0,
            y_min=0.0,
            y_max=1.0,
            x_label="Ground range (km)",
            y_label="Altitude (km)",
            xsec_azimuth_deg=azimuth_deg,
            xsec_azimuth_tolerance_deg=azimuth_tolerance_deg,
            sweep_elevations_deg=sweep_el,
            sweep_num_returns=sweep_nr,
            sweep_max_altitude_m=_sweep_max_altitudes(sweep_el, 1.0),
            colormap=empty_colormap,
            units=empty_units,
        )

    x, y, z = _polar_to_cartesian(az, el, rng)

    # Compute angular difference with wraparound for forward direction
    target = azimuth_deg % 360.0
    opposite = (azimuth_deg + 180.0) % 360.0
    tol = azimuth_tolerance_deg

    diff_fwd = np.abs(((az - target + 180.0) % 360.0) - 180.0)
    diff_bwd = np.abs(((az - opposite + 180.0) % 360.0) - 180.0)

    mask_fwd = diff_fwd < tol
    mask_bwd = diff_bwd < tol

    # Signed ground range: positive = forward, negative = backward
    ground_range = np.sqrt(x**2 + y**2)
    signed_range = np.where(mask_fwd, ground_range, np.where(mask_bwd, -ground_range, 0.0))

    mask = mask_fwd | mask_bwd
    sr_sel = signed_range[mask]
    z_sel = z[mask]
    vals_sel = vals[mask]

    # Determine grid extents
    if sr_sel.size > 0:
        r_max = float(np.nanmax(np.abs(sr_sel)))
        z_max = float(np.nanmax(z_sel)) if z_sel.size > 0 else 1.0
        z_min_val = float(np.nanmin(z_sel)) if z_sel.size > 0 else 0.0
    else:
        r_max = float(np.nanmax(ground_range)) if ground_range.size > 0 else 1.0
        z_max = float(np.nanmax(z)) if z.size > 0 else 1.0
        z_min_val = 0.0

    x_min = -r_max
    x_max = r_max
    y_min = min(z_min_val, 0.0)
    y_max = max(z_max, 1.0)

    # Bin into regular grid (rows=altitude, cols=ground_range)
    grid = np.full((grid_size, grid_size), np.nan, dtype=np.float64)
    count = np.zeros((grid_size, grid_size), dtype=np.int32)

    if sr_sel.size > 0:
        col = ((sr_sel - x_min) / (x_max - x_min) * grid_size).astype(np.int64)
        row = ((y_max - z_sel) / (y_max - y_min) * grid_size).astype(np.int64)
        valid = (col >= 0) & (col < grid_size) & (row >= 0) & (row < grid_size)
        col = col[valid]
        row = row[valid]
        v = vals_sel[valid].astype(np.float64)

        accum = np.zeros((grid_size, grid_size), dtype=np.float64)
        np.add.at(accum, (row, col), v)
        np.add.at(count, (row, col), 1)

        has_data = count > 0
        grid[has_data] = accum[has_data] / count[has_data]

    grid = grid.astype(np.float32)
    vmin, vmax, colormap, units = _resolve_color_scale(
        moment, grid, color_scale, returns, sweeps
    )

    return GridPayload(
        grid=grid,
        grid_mode="xsec",
        moment=moment,
        vmin=vmin,
        vmax=vmax,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        x_label="Ground range (km)",
        y_label="Altitude (km)",
        xsec_azimuth_deg=azimuth_deg,
        xsec_azimuth_tolerance_deg=azimuth_tolerance_deg,
        sweep_elevations_deg=sweep_el,
        sweep_num_returns=sweep_nr,
        sweep_max_altitude_m=_sweep_max_altitudes(sweep_el, x_max),
        colormap=colormap,
        units=units,
    )


def prepare_waterfall_payload(
    returns: xr.Dataset,
    moment: str,
    max_returns: int = 2048,
    max_range: int = 1024,
    fold_size: int | None = None,
    color_scale: ColorScale | None = None,
    *,
    byte_budget: int | None = DEFAULT_BYTE_BUDGET,
) -> WaterfallPayload:
    """Create a waterfall heatmap payload from the raw (return_time, range) matrix.

    Parameters
    ----------
    max_returns, max_range
        Explicit ceilings on shipped rows and range columns. ``byte_budget``
        decimates further on top of these if the result is still too large.
    fold_size : int | None
        If given, crop the range dimension to at most this many gates before
        downsampling.  ``None`` (default) keeps all gates.
    color_scale : ColorScale | None
        Overrides the :data:`COLOR_SCALES` default for the moment.
    byte_budget
        Ceiling on :attr:`WaterfallPayload.nbytes`. Rows and columns are
        decimated together, so the picture shrinks squarely.
    """

    if moment not in returns.data_vars:
        raise KeyError(f"moment '{moment}' not found in returns dataset")

    moment_matrix = np.asarray(returns[moment].values, dtype=np.float32)
    if moment_matrix.ndim != 2:
        raise ValueError(f"moment '{moment}' must be 2D on (return_time, range)")

    n_returns, n_range_full = moment_matrix.shape

    # Crop range dimension to fold_size if requested
    n_range = n_range_full
    if fold_size is not None and fold_size > 0 and fold_size < n_range_full:
        moment_matrix = moment_matrix[:, :fold_size]
        n_range = fold_size

    if n_returns == 0 or n_range == 0:
        _, _, empty_colormap, empty_units = _resolve_color_scale(
            moment, moment_matrix[:0], color_scale, returns
        )
        return WaterfallPayload(
            grid=np.empty((0, 0), dtype=np.float32),
            azimuth_deg=np.empty(0, dtype=np.float32),
            elevation_deg=np.empty(0, dtype=np.float32),
            return_time_ms=np.empty(0, dtype=np.float64),
            sweep_number=np.empty(0, dtype=np.uint16),
            base_range_m=np.empty(0, dtype=np.float32),
            range_step_m=np.empty(0, dtype=np.float32),
            sweep_boundaries=np.empty(0, dtype=np.int32),
            moment=moment,
            vmin=0.0,
            vmax=1.0,
            n_returns_orig=n_returns,
            n_range_orig=n_range_full,
            axis_start_m=0.0,
            axis_step_m=0.0,
            axis_end_m=0.0,
            colormap=empty_colormap,
            units=empty_units,
        )

    # Explicit ceilings first, then the byte budget coarsens whatever is left.
    return_stride = max(1, int(np.ceil(n_returns / max(1, max_returns))))
    range_stride = max(1, int(np.ceil(n_range / max(1, max_range))))
    return_stride, range_stride = _fit_grid_strides(
        n_returns,
        n_range,
        _WATERFALL_ROW_BYTES,
        _CELL_BYTES,
        None if byte_budget is None else max(1, int(byte_budget) - _META_RESERVE),
        min_rows=64,
        min_cols=64,
        row_stride=return_stride,
        col_stride=range_stride,
    )

    grid = moment_matrix[::return_stride, ::range_stride]

    # Per-return metadata
    azimuth = np.asarray(returns["azimuth"].values, dtype=np.float32)[::return_stride]
    elevation = np.asarray(returns["elevation"].values, dtype=np.float32)[::return_stride]
    return_time_raw = returns["return_time"].values
    return_time_ms = (
        return_time_raw.astype("datetime64[ms]").astype(np.float64)[::return_stride]
    )
    sweep_num = np.asarray(returns["sweep_number"].values, dtype=np.uint16)[::return_stride]
    base_range_all = np.asarray(returns["base_range"].values, dtype=np.float32)
    range_step_all = np.asarray(returns["range_step"].values, dtype=np.float32)
    base_range = base_range_all[::return_stride]
    range_step = range_step_all[::return_stride]

    # Sweep boundaries: indices where sweep_number changes
    changes = np.where(np.diff(sweep_num) != 0)[0] + 1
    sweep_boundaries = changes.astype(np.int32)

    # Median range axis, for tick placement.
    axis_start_m = float(np.median(base_range_all))
    axis_step_m = float(np.median(range_step_all))
    axis_end_m = axis_start_m + (n_range - 1) * axis_step_m

    vmin, vmax, colormap, units = _resolve_color_scale(
        moment, grid, color_scale, returns
    )

    return WaterfallPayload(
        grid=np.ascontiguousarray(grid, dtype=np.float32),
        azimuth_deg=np.ascontiguousarray(azimuth, dtype=np.float32),
        elevation_deg=np.ascontiguousarray(elevation, dtype=np.float32),
        return_time_ms=np.ascontiguousarray(return_time_ms, dtype=np.float64),
        sweep_number=np.ascontiguousarray(sweep_num, dtype=np.uint16),
        base_range_m=np.ascontiguousarray(base_range, dtype=np.float32),
        range_step_m=np.ascontiguousarray(range_step, dtype=np.float32),
        sweep_boundaries=sweep_boundaries,
        moment=moment,
        vmin=vmin,
        vmax=vmax,
        n_returns_orig=n_returns,
        n_range_orig=n_range_full,
        axis_start_m=axis_start_m,
        axis_step_m=axis_step_m,
        axis_end_m=axis_end_m,
        return_stride=return_stride,
        range_stride=range_stride,
        colormap=colormap,
        units=units,
    )


def prepare_folded_waterfall_payload(
    returns: xr.Dataset,
    moment: str,
    *,
    row_offset: int = 0,
    row_count: int | None = None,
    max_range: int | None = None,
    color_scale: ColorScale | None = None,
    byte_budget: int | None = DEFAULT_BYTE_BUDGET,
) -> FoldedWaterfallPayload:
    """Create a row-per-return payload over a window of the returns matrix.

    Rows are taken verbatim (no striding), so every fold in the window is drawn.
    Folds chunked from one radial share a ``return_time``; the boundaries between
    those sets are returned in ``group_starts`` for the renderer to divide on.

    Parameters
    ----------
    row_offset
        Where the window starts in the ``return_time`` dimension, clamped to the
        dataset.
    row_count
        How many rows the window holds. ``None`` (default) takes as many as
        ``byte_budget`` allows, which is what makes this view scale with fold
        size instead of the caller doing that arithmetic.
    max_range
        Cap on drawn range columns; ``None`` (default) keeps every gate. Folds
        are usually narrow enough to draw whole.
    color_scale
        Overrides the :data:`COLOR_SCALES` default for the moment.
    byte_budget
        Ceiling on :attr:`FoldedWaterfallPayload.nbytes`, spent on the row
        window: a wide fold buys fewer rows, a narrow one buys more.
    """

    if moment not in returns.data_vars:
        raise KeyError(f"moment '{moment}' not found in returns dataset")

    moment_matrix = np.asarray(returns[moment].values, dtype=np.float32)
    if moment_matrix.ndim != 2:
        raise ValueError(f"moment '{moment}' must be 2D on (return_time, range)")

    row_total, n_range_orig = moment_matrix.shape
    start = max(0, min(int(row_offset), row_total))

    range_stride = 1
    if max_range is not None and max_range > 0 and n_range_orig > max_range:
        range_stride = int(np.ceil(n_range_orig / max_range))

    def _empty() -> FoldedWaterfallPayload:
        _, _, empty_colormap, empty_units = _resolve_color_scale(
            moment, moment_matrix[:0], color_scale, returns
        )
        return FoldedWaterfallPayload(
            grid=np.empty((0, 0), dtype=np.float32),
            vcp_index=np.empty(0, dtype=np.uint16),
            vcp_time_ms=np.empty(0, dtype=np.float64),
            sweep_number=np.empty(0, dtype=np.uint16),
            azimuth_deg=np.empty(0, dtype=np.float32),
            elevation_deg=np.empty(0, dtype=np.float32),
            return_time_ms=np.empty(0, dtype=np.float64),
            base_range_m=np.empty(0, dtype=np.float32),
            range_step_m=np.empty(0, dtype=np.float32),
            fold_index=np.empty(0, dtype=np.uint16),
            group_starts=np.empty(0, dtype=np.int32),
            moment=moment,
            vmin=0.0,
            vmax=1.0,
            row_offset=start,
            row_total=row_total,
            range_stride=1,
            n_range_orig=n_range_orig,
            colormap=empty_colormap,
            units=empty_units,
        )

    if n_range_orig == 0 or row_total == 0:
        return _empty()

    n_range = _ceil_div(n_range_orig, range_stride)

    # The budget buys rows: one row costs its metadata plus its quantized fold.
    # If a single row does not fit, decimate the fold until one does.
    if byte_budget is not None and byte_budget > 0:
        available = max(1, int(byte_budget) - _META_RESERVE)
        while n_range > 1 and _FOLDED_ROW_BYTES + n_range * _CELL_BYTES > available:
            range_stride += max(1, range_stride // 2)
            n_range = _ceil_div(n_range_orig, range_stride)
        budget_rows = max(1, available // (_FOLDED_ROW_BYTES + n_range * _CELL_BYTES))
    else:
        budget_rows = row_total

    window = budget_rows if row_count is None else max(0, int(row_count))
    window = min(window, budget_rows)
    stop = min(row_total, start + window)

    if stop <= start:
        return _empty()

    grid = moment_matrix[start:stop, ::range_stride]

    rows = slice(start, stop)
    return_time_raw = returns["return_time"].values[rows]
    return_time_ms = return_time_raw.astype("datetime64[ms]").astype(np.float64)

    # Volume ordinal is resolved against the whole batch so it does not shift with the window.
    vcp_time_all = returns["vcp_time"].values
    vcp_uniq = np.unique(vcp_time_all)
    vcp_index = np.searchsorted(vcp_uniq, vcp_time_all[rows]).astype(np.uint16)
    vcp_time_ms = vcp_time_all[rows].astype("datetime64[ms]").astype(np.float64)

    # A fold set is one radial's chunks, which share a return_time.
    group_starts = (np.where(np.diff(return_time_ms) != 0)[0] + 1).astype(np.int32)

    # Position within the fold set: rows since the last group start.
    n_rows = stop - start
    starts_incl = np.concatenate([[0], group_starts]).astype(np.int64)
    group_of_row = np.searchsorted(starts_incl, np.arange(n_rows), side="right") - 1
    fold_index = (np.arange(n_rows) - starts_incl[group_of_row]).astype(np.uint16)

    vmin, vmax, colormap, units = _resolve_color_scale(
        moment, grid, color_scale, returns
    )

    return FoldedWaterfallPayload(
        grid=np.ascontiguousarray(grid, dtype=np.float32),
        vcp_index=np.ascontiguousarray(vcp_index, dtype=np.uint16),
        vcp_time_ms=np.ascontiguousarray(vcp_time_ms, dtype=np.float64),
        sweep_number=np.ascontiguousarray(
            np.asarray(returns["sweep_number"].values[rows]), dtype=np.uint16
        ),
        azimuth_deg=np.ascontiguousarray(
            np.asarray(returns["azimuth"].values[rows]), dtype=np.float32
        ),
        elevation_deg=np.ascontiguousarray(
            np.asarray(returns["elevation"].values[rows]), dtype=np.float32
        ),
        return_time_ms=np.ascontiguousarray(return_time_ms, dtype=np.float64),
        base_range_m=np.ascontiguousarray(
            np.asarray(returns["base_range"].values[rows]), dtype=np.float32
        ),
        range_step_m=np.ascontiguousarray(
            np.asarray(returns["range_step"].values[rows]), dtype=np.float32
        ),
        fold_index=fold_index,
        group_starts=group_starts,
        moment=moment,
        vmin=vmin,
        vmax=vmax,
        row_offset=start,
        row_total=row_total,
        range_stride=range_stride,
        n_range_orig=n_range_orig,
        colormap=colormap,
        units=units,
    )


Payload = (
    PolarPayload | VolumePayload | GridPayload | WaterfallPayload | FoldedWaterfallPayload
)
