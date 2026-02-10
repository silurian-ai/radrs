"""Visualization helpers for interactive raystack visualization.

This module provides:

- sweep and moment selectors over raystack DataTree nodes
- a vectorized adapter from sweep data to polar point buffers
- an optional anywidget polar renderer for marimo notebooks
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Final

import numpy as np
import xarray as xr

MOMENT_NAMES: Final[tuple[str, ...]] = (
    "DBZH",
    "VRADH",
    "WRADH",
    "ZDR",
    "PHIDP",
    "RHOHV",
    "CCORH",
)


@dataclass(frozen=True)
class SweepInfo:
    """Human-friendly sweep metadata for selector controls."""

    index: int
    sweep_number: int
    elevation_deg: float
    num_returns: int
    label: str


@dataclass(frozen=True)
class PolarPayload:
    """Flat point buffers for a single sweep+moment polar view."""

    azimuth_deg: np.ndarray
    range_m: np.ndarray
    values: np.ndarray
    elevation_deg: np.ndarray
    return_time_ms: np.ndarray
    gate_index: np.ndarray
    return_index: np.ndarray
    moment: str
    sweep_number: int
    sweep_time_ms: int
    max_range_m: float
    vmin: float
    vmax: float

    @property
    def point_count(self) -> int:
        return int(self.values.size)

    def to_widget_state(self) -> dict[str, object]:
        """Serialize payload to binary traits for widget transport."""

        return {
            "azimuth_bytes": np.ascontiguousarray(self.azimuth_deg, dtype=np.float32).tobytes(),
            "range_bytes": np.ascontiguousarray(self.range_m, dtype=np.float32).tobytes(),
            "value_bytes": np.ascontiguousarray(self.values, dtype=np.float32).tobytes(),
            "elevation_bytes": np.ascontiguousarray(
                self.elevation_deg, dtype=np.float32
            ).tobytes(),
            "gate_index_bytes": np.ascontiguousarray(
                self.gate_index, dtype=np.uint16
            ).tobytes(),
            "return_index_bytes": np.ascontiguousarray(
                self.return_index, dtype=np.uint32
            ).tobytes(),
            "return_time_ms_bytes": np.ascontiguousarray(
                self.return_time_ms, dtype=np.float64
            ).tobytes(),
            "meta": {
                "point_count": self.point_count,
                "moment": self.moment,
                "sweep_number": int(self.sweep_number),
                "sweep_time_ms": int(self.sweep_time_ms),
                "max_range_m": float(self.max_range_m),
                "vmin": float(self.vmin),
                "vmax": float(self.vmax),
            },
        }

    def to_html(self, width: int = 760, height: int = 760) -> str:
        """Render as a self-contained HTML document."""
        return _payload_to_html(self.to_widget_state(), _WIDGET_ESM, _WIDGET_CSS, width, height)


@dataclass(frozen=True)
class VolumePayload:
    """Flat point buffers across all rays in a volume."""

    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    values: np.ndarray
    gate_index: np.ndarray
    return_index: np.ndarray
    moment: str
    render_mode: str
    max_abs_m: float
    vmin: float
    vmax: float

    @property
    def point_count(self) -> int:
        return int(self.values.size)

    def to_widget_state(self) -> dict[str, object]:
        return {
            "x_bytes": np.ascontiguousarray(self.x_m, dtype=np.float32).tobytes(),
            "y_bytes": np.ascontiguousarray(self.y_m, dtype=np.float32).tobytes(),
            "z_bytes": np.ascontiguousarray(self.z_m, dtype=np.float32).tobytes(),
            "value_bytes": np.ascontiguousarray(self.values, dtype=np.float32).tobytes(),
            "gate_index_bytes": np.ascontiguousarray(
                self.gate_index, dtype=np.uint16
            ).tobytes(),
            "return_index_bytes": np.ascontiguousarray(
                self.return_index, dtype=np.uint32
            ).tobytes(),
            "meta": {
                "point_count": self.point_count,
                "moment": self.moment,
                "render_mode": self.render_mode,
                "max_abs_m": float(self.max_abs_m),
                "vmin": float(self.vmin),
                "vmax": float(self.vmax),
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
        return _payload_to_html(
            self.to_widget_state(),
            _VOLUME_WIDGET_ESM,
            _WIDGET_CSS,
            width,
            height,
            extra_state={"yaw_deg": yaw_deg, "pitch_deg": pitch_deg},
        )


@dataclass(frozen=True)
class GridPayload:
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

        return {
            "grid_bytes": np.ascontiguousarray(
                self.grid.ravel(), dtype=np.float32
            ).tobytes(),
            "meta": meta,
        }

    def to_html(self, width: int = 760, height: int = 760) -> str:
        """Render as a self-contained HTML document."""
        return _payload_to_html(self.to_widget_state(), _GRID_WIDGET_ESM, _WIDGET_CSS, width, height)


@dataclass(frozen=True)
class WaterfallPayload:
    """2D heatmap of the raw (return_time, range) moment matrix."""

    grid: np.ndarray  # float32 (n_returns_ds, n_range_ds), NaN = no data
    azimuth_deg: np.ndarray  # float32 (n_returns_ds,)
    elevation_deg: np.ndarray  # float32 (n_returns_ds,)
    return_time_ms: np.ndarray  # float64 (n_returns_ds,) — ms since epoch
    sweep_number: np.ndarray  # uint16 (n_returns_ds,)
    sweep_boundaries: np.ndarray  # int32 — indices where sweep changes
    moment: str
    vmin: float
    vmax: float
    n_returns_orig: int
    n_range_orig: int
    range_start_m: float
    range_step_m: float
    range_end_m: float

    @property
    def n_returns(self) -> int:
        return int(self.grid.shape[0]) if self.grid.ndim == 2 else 0

    @property
    def n_range(self) -> int:
        return int(self.grid.shape[1]) if self.grid.ndim == 2 else 0

    def to_widget_state(self) -> dict[str, object]:
        """Serialize payload to binary traits for widget transport."""

        return {
            "grid_bytes": np.ascontiguousarray(
                self.grid.ravel(), dtype=np.float32
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
            "sweep_number_bytes": np.ascontiguousarray(
                self.sweep_number, dtype=np.uint16
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
                "range_start_m": float(self.range_start_m),
                "range_step_m": float(self.range_step_m),
                "range_end_m": float(self.range_end_m),
            },
        }

    def to_html(self, width: int = 760, height: int = 760) -> str:
        """Render as a self-contained HTML document."""
        return _payload_to_html(self.to_widget_state(), _WATERFALL_WIDGET_ESM, _WIDGET_CSS, width, height)


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
    stride = int(np.ceil(n_points / max_points))
    return np.arange(0, n_points, stride, dtype=np.int64)


def _value_bounds(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 1.0
    p05 = float(np.nanpercentile(values, 5.0))
    p95 = float(np.nanpercentile(values, 95.0))
    if not np.isfinite(p05) or not np.isfinite(p95) or p05 == p95:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            return 0.0, 1.0
        return vmin, vmax
    return p05, p95


def _polar_to_cartesian(
    azimuth_deg: np.ndarray,
    elevation_deg: np.ndarray,
    range_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    azimuth_rad = np.deg2rad(azimuth_deg.astype(np.float64))
    elevation_rad = np.deg2rad(elevation_deg.astype(np.float64))
    horizontal = range_m.astype(np.float64) * np.cos(elevation_rad)

    # x=east, y=north, z=up
    x = horizontal * np.sin(azimuth_rad)
    y = horizontal * np.cos(azimuth_rad)
    z = range_m.astype(np.float64) * np.sin(elevation_rad)
    return x, y, z


def prepare_polar_payload(
    returns: xr.Dataset,
    sweeps: xr.Dataset,
    sweep_index: int,
    moment: str,
    max_points: int | None = None,
) -> PolarPayload:
    """Create a single-sweep polar payload with finite gates only."""

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

    n_returns, n_range = moment_matrix.shape
    flat_values = moment_matrix.reshape(-1)
    finite = np.isfinite(flat_values)

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

    if n_returns == 0 or n_range == 0 or not np.any(finite):
        return PolarPayload(
            azimuth_deg=np.empty(0, dtype=np.float32),
            range_m=np.empty(0, dtype=np.float32),
            values=np.empty(0, dtype=np.float32),
            elevation_deg=np.empty(0, dtype=np.float32),
            return_time_ms=np.empty(0, dtype=np.float64),
            gate_index=np.empty(0, dtype=np.uint16),
            return_index=np.empty(0, dtype=np.uint32),
            moment=moment,
            sweep_number=sweep_number,
            sweep_time_ms=sweep_time_ms,
            max_range_m=0.0,
            vmin=0.0,
            vmax=1.0,
        )

    azimuth = np.asarray(returns["azimuth"].values[sweep_slice], dtype=np.float32)
    elevation = np.asarray(returns["elevation"].values[sweep_slice], dtype=np.float32)
    base_range = np.asarray(returns["base_range"].values[sweep_slice], dtype=np.float32)
    range_step = np.asarray(returns["range_step"].values[sweep_slice], dtype=np.float32)
    return_time = np.asarray(returns["return_time"].values[sweep_slice], dtype="datetime64[ms]")
    return_time_ms = np.asarray(return_time, dtype=np.int64).astype(np.float64, copy=False)

    gate_index_full = np.tile(np.arange(n_range, dtype=np.uint16), n_returns)
    return_index_full = np.repeat(np.arange(start, end, dtype=np.uint32), n_range)
    azimuth_full = np.repeat(azimuth, n_range)
    elevation_full = np.repeat(elevation, n_range)
    base_range_full = np.repeat(base_range, n_range)
    range_step_full = np.repeat(range_step, n_range)
    return_time_full = np.repeat(return_time_ms, n_range)

    azimuth_sel = azimuth_full[finite]
    elevation_sel = elevation_full[finite]
    gate_index_sel = gate_index_full[finite]
    return_index_sel = return_index_full[finite]
    return_time_sel = return_time_full[finite]
    values_sel = flat_values[finite]
    range_sel = base_range_full[finite] + gate_index_sel.astype(np.float32) * range_step_full[finite]

    n_points = int(values_sel.size)
    if max_points is not None and max_points > 0 and n_points > max_points:
        keep = _sample_indices(n_points, max_points)
        azimuth_sel = azimuth_sel[keep]
        elevation_sel = elevation_sel[keep]
        gate_index_sel = gate_index_sel[keep]
        return_index_sel = return_index_sel[keep]
        return_time_sel = return_time_sel[keep]
        values_sel = values_sel[keep]
        range_sel = range_sel[keep]

    vmin, vmax = _value_bounds(values_sel)
    max_range_m = float(np.nanmax(range_sel)) if range_sel.size else 0.0

    return PolarPayload(
        azimuth_deg=np.ascontiguousarray(azimuth_sel, dtype=np.float32),
        range_m=np.ascontiguousarray(range_sel, dtype=np.float32),
        values=np.ascontiguousarray(values_sel, dtype=np.float32),
        elevation_deg=np.ascontiguousarray(elevation_sel, dtype=np.float32),
        return_time_ms=np.ascontiguousarray(return_time_sel, dtype=np.float64),
        gate_index=np.ascontiguousarray(gate_index_sel, dtype=np.uint16),
        return_index=np.ascontiguousarray(return_index_sel, dtype=np.uint32),
        moment=moment,
        sweep_number=sweep_number,
        sweep_time_ms=sweep_time_ms,
        max_range_m=max_range_m,
        vmin=vmin,
        vmax=vmax,
    )


def prepare_volume_payload(
    returns: xr.Dataset,
    moment: str,
    max_points: int | None = None,
) -> VolumePayload:
    """Create a volume-wide Cartesian payload from all finite gates."""

    if moment not in returns.data_vars:
        raise KeyError(f"moment '{moment}' not found in returns dataset")

    moment_matrix = np.asarray(returns[moment].values, dtype=np.float32)
    if moment_matrix.ndim != 2:
        raise ValueError(f"moment '{moment}' must be 2D on (return_time, range)")

    n_returns, n_range = moment_matrix.shape
    finite = np.isfinite(moment_matrix)
    n_finite = int(finite.sum())

    if n_returns == 0 or n_range == 0 or n_finite == 0:
        return VolumePayload(
            x_m=np.empty(0, dtype=np.float32),
            y_m=np.empty(0, dtype=np.float32),
            z_m=np.empty(0, dtype=np.float32),
            values=np.empty(0, dtype=np.float32),
            gate_index=np.empty(0, dtype=np.uint16),
            return_index=np.empty(0, dtype=np.uint32),
            moment=moment,
            render_mode="points",
            max_abs_m=0.0,
            vmin=0.0,
            vmax=1.0,
        )

    row_idx, col_idx = np.nonzero(finite)
    values_sel = moment_matrix[row_idx, col_idx]

    if max_points is not None and max_points > 0 and n_finite > max_points:
        keep = _sample_indices(n_finite, max_points)
        row_idx = row_idx[keep]
        col_idx = col_idx[keep]
        values_sel = values_sel[keep]

    azimuth = np.asarray(returns["azimuth"].values, dtype=np.float32)
    elevation = np.asarray(returns["elevation"].values, dtype=np.float32)
    base_range = np.asarray(returns["base_range"].values, dtype=np.float32)
    range_step = np.asarray(returns["range_step"].values, dtype=np.float32)

    azimuth_sel = azimuth[row_idx]
    elevation_sel = elevation[row_idx]
    range_sel = base_range[row_idx] + col_idx.astype(np.float32) * range_step[row_idx]
    gate_index_sel = col_idx.astype(np.uint16)
    return_index_sel = row_idx.astype(np.uint32)

    x_sel, y_sel, z_sel = _polar_to_cartesian(azimuth_sel, elevation_sel, range_sel)

    vmin, vmax = _value_bounds(values_sel)
    max_abs_m = float(
        max(
            np.nanmax(np.abs(x_sel)) if x_sel.size else 0.0,
            np.nanmax(np.abs(y_sel)) if y_sel.size else 0.0,
            np.nanmax(np.abs(z_sel)) if z_sel.size else 0.0,
        )
    )

    return VolumePayload(
        x_m=np.ascontiguousarray(x_sel, dtype=np.float32),
        y_m=np.ascontiguousarray(y_sel, dtype=np.float32),
        z_m=np.ascontiguousarray(z_sel, dtype=np.float32),
        values=np.ascontiguousarray(values_sel, dtype=np.float32),
        gate_index=np.ascontiguousarray(gate_index_sel, dtype=np.uint16),
        return_index=np.ascontiguousarray(return_index_sel, dtype=np.uint32),
        moment=moment,
        render_mode="points",
        max_abs_m=max_abs_m,
        vmin=vmin,
        vmax=vmax,
    )


def prepare_ray_payload(
    returns: xr.Dataset,
    moment: str,
    max_points: int | None = None,
) -> VolumePayload:
    """Create a ray-centric payload with one full-length ray per return."""

    if moment not in returns.data_vars:
        raise KeyError(f"moment '{moment}' not found in returns dataset")

    moment_matrix = np.asarray(returns[moment].values, dtype=np.float32)
    if moment_matrix.ndim != 2:
        raise ValueError(f"moment '{moment}' must be 2D on (return_time, range)")

    n_returns, n_range = moment_matrix.shape
    if n_returns == 0 or n_range == 0:
        return VolumePayload(
            x_m=np.empty(0, dtype=np.float32),
            y_m=np.empty(0, dtype=np.float32),
            z_m=np.empty(0, dtype=np.float32),
            values=np.empty(0, dtype=np.float32),
            gate_index=np.empty(0, dtype=np.uint16),
            return_index=np.empty(0, dtype=np.uint32),
            moment=moment,
            render_mode="rays",
            max_abs_m=0.0,
            vmin=0.0,
            vmax=1.0,
        )

    azimuth = np.asarray(returns["azimuth"].values, dtype=np.float32)
    elevation = np.asarray(returns["elevation"].values, dtype=np.float32)
    base_range = np.asarray(returns["base_range"].values, dtype=np.float32)
    range_step = np.asarray(returns["range_step"].values, dtype=np.float32)
    return_rows = np.arange(n_returns, dtype=np.uint32)
    last_gate = np.full(n_returns, n_range - 1, dtype=np.uint16)

    endpoint_range = base_range + last_gate.astype(np.float32) * range_step
    x_sel, y_sel, z_sel = _polar_to_cartesian(azimuth, elevation, endpoint_range)
    gate_index_sel = last_gate
    return_index_sel = return_rows

    # Color each full ray by its strongest finite gate in this moment.
    finite = np.isfinite(moment_matrix)
    has_finite = np.any(finite, axis=1)
    safe_matrix = np.where(finite, moment_matrix, -np.inf)
    ray_max = np.max(safe_matrix, axis=1)
    ray_values = np.where(has_finite, ray_max, np.nan).astype(np.float32, copy=False)
    values_sel = ray_values

    n_points = int(values_sel.size)
    if max_points is not None and max_points > 0 and n_points > max_points:
        keep = _sample_indices(n_points, max_points)
        x_sel = x_sel[keep]
        y_sel = y_sel[keep]
        z_sel = z_sel[keep]
        values_sel = values_sel[keep]
        gate_index_sel = gate_index_sel[keep]
        return_index_sel = return_index_sel[keep]

    finite_values = values_sel[np.isfinite(values_sel)]
    vmin, vmax = _value_bounds(finite_values if finite_values.size else values_sel)
    max_abs_m = float(
        max(
            np.nanmax(np.abs(x_sel)) if x_sel.size else 0.0,
            np.nanmax(np.abs(y_sel)) if y_sel.size else 0.0,
            np.nanmax(np.abs(z_sel)) if z_sel.size else 0.0,
        )
    )

    return VolumePayload(
        x_m=np.ascontiguousarray(x_sel, dtype=np.float32),
        y_m=np.ascontiguousarray(y_sel, dtype=np.float32),
        z_m=np.ascontiguousarray(z_sel, dtype=np.float32),
        values=np.ascontiguousarray(values_sel, dtype=np.float32),
        gate_index=np.ascontiguousarray(gate_index_sel, dtype=np.uint16),
        return_index=np.ascontiguousarray(return_index_sel, dtype=np.uint32),
        moment=moment,
        render_mode="rays",
        max_abs_m=max_abs_m,
        vmin=vmin,
        vmax=vmax,
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


def prepare_cappi_payload(
    returns: xr.Dataset,
    sweeps: xr.Dataset,
    moment: str,
    altitude_m: float,
    tolerance_m: float | None = None,
    grid_size: int = 500,
    max_range_m: float | None = None,
) -> GridPayload:
    """Create a CAPPI (constant altitude) horizontal slice through the volume."""

    if moment not in returns.data_vars:
        raise KeyError(f"moment '{moment}' not found in returns dataset")

    az, el, rng, vals, finite = _extract_all_gates(returns, moment)

    if az.size == 0:
        grid = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
        sweep_el, sweep_nr = _sweep_metadata(sweeps)
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
    finite_grid = grid[np.isfinite(grid)]
    vmin, vmax = _value_bounds(finite_grid) if finite_grid.size > 0 else (0.0, 1.0)

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
    )


def prepare_xsec_payload(
    returns: xr.Dataset,
    sweeps: xr.Dataset,
    moment: str,
    azimuth_deg: float,
    azimuth_tolerance_deg: float = 2.0,
    grid_size: int = 500,
) -> GridPayload:
    """Create a vertical cross-section along a target azimuth (and its opposite)."""

    if moment not in returns.data_vars:
        raise KeyError(f"moment '{moment}' not found in returns dataset")

    az, el, rng, vals, finite = _extract_all_gates(returns, moment)

    sweep_el, sweep_nr = _sweep_metadata(sweeps)

    if az.size == 0:
        grid = np.full((grid_size, grid_size), np.nan, dtype=np.float32)
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
    finite_grid = grid[np.isfinite(grid)]
    vmin, vmax = _value_bounds(finite_grid) if finite_grid.size > 0 else (0.0, 1.0)

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
    )


def prepare_waterfall_payload(
    returns: xr.Dataset,
    moment: str,
    max_returns: int = 2048,
    max_range: int = 1024,
    fold_size: int | None = None,
) -> WaterfallPayload:
    """Create a waterfall heatmap payload from the raw (return_time, range) matrix.

    Parameters
    ----------
    fold_size : int | None
        If given, crop the range dimension to at most this many gates before
        downsampling.  ``None`` (default) keeps all gates.
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
        return WaterfallPayload(
            grid=np.empty((0, 0), dtype=np.float32),
            azimuth_deg=np.empty(0, dtype=np.float32),
            elevation_deg=np.empty(0, dtype=np.float32),
            return_time_ms=np.empty(0, dtype=np.float64),
            sweep_number=np.empty(0, dtype=np.uint16),
            sweep_boundaries=np.empty(0, dtype=np.int32),
            moment=moment,
            vmin=0.0,
            vmax=1.0,
            n_returns_orig=n_returns,
            n_range_orig=n_range_full,
            range_start_m=0.0,
            range_step_m=0.0,
            range_end_m=0.0,
        )

    # Compute strides for downsampling
    return_stride = max(1, int(np.ceil(n_returns / max_returns)))
    range_stride = max(1, int(np.ceil(n_range / max_range)))

    grid = moment_matrix[::return_stride, ::range_stride]

    # Per-return metadata
    azimuth = np.asarray(returns["azimuth"].values, dtype=np.float32)[::return_stride]
    elevation = np.asarray(returns["elevation"].values, dtype=np.float32)[::return_stride]
    return_time_raw = returns["return_time"].values
    return_time_ms = (
        return_time_raw.astype("datetime64[ms]").astype(np.float64)[::return_stride]
    )
    sweep_num = np.asarray(returns["sweep_number"].values, dtype=np.uint16)[::return_stride]

    # Sweep boundaries: indices where sweep_number changes
    changes = np.where(np.diff(sweep_num) != 0)[0] + 1
    sweep_boundaries = changes.astype(np.int32)

    # Range metadata
    base_range = np.asarray(returns["base_range"].values, dtype=np.float32)
    range_step_arr = np.asarray(returns["range_step"].values, dtype=np.float32)
    range_start_m = float(np.median(base_range))
    range_step_m = float(np.median(range_step_arr))
    range_end_m = range_start_m + (n_range - 1) * range_step_m

    vmin, vmax = _value_bounds(grid[np.isfinite(grid)]) if np.any(np.isfinite(grid)) else (0.0, 1.0)

    return WaterfallPayload(
        grid=np.ascontiguousarray(grid, dtype=np.float32),
        azimuth_deg=np.ascontiguousarray(azimuth, dtype=np.float32),
        elevation_deg=np.ascontiguousarray(elevation, dtype=np.float32),
        return_time_ms=np.ascontiguousarray(return_time_ms, dtype=np.float64),
        sweep_number=np.ascontiguousarray(sweep_num, dtype=np.uint16),
        sweep_boundaries=sweep_boundaries,
        moment=moment,
        vmin=vmin,
        vmax=vmax,
        n_returns_orig=n_returns,
        n_range_orig=n_range_full,
        range_start_m=range_start_m,
        range_step_m=range_step_m,
        range_end_m=range_end_m,
    )


_WIDGET_ESM: Final[str] = r"""
function toArrayBuffer(raw) {
  if (!raw) {
    return new ArrayBuffer(0);
  }
  if (raw instanceof ArrayBuffer) {
    return raw;
  }
  if (ArrayBuffer.isView(raw)) {
    return raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
  }
  return new Uint8Array(raw).buffer;
}

function decodeArray(raw, ctor) {
  const buffer = toArrayBuffer(raw);
  if (buffer.byteLength === 0) {
    return new ctor(0);
  }
  const bytesPerElement = ctor.BYTES_PER_ELEMENT;
  const trimmed = buffer.byteLength - (buffer.byteLength % bytesPerElement);
  return new ctor(buffer.slice(0, trimmed));
}

function clamp01(x) {
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function colorMap(t) {
  const x = clamp01(t);
  const stops = [
    [68, 1, 84],
    [59, 82, 139],
    [33, 145, 140],
    [94, 201, 98],
    [253, 231, 37],
  ];
  const scaled = x * (stops.length - 1);
  const lo = Math.floor(scaled);
  const hi = Math.min(stops.length - 1, lo + 1);
  const local = scaled - lo;
  return [
    Math.round(lerp(stops[lo][0], stops[hi][0], local)),
    Math.round(lerp(stops[lo][1], stops[hi][1], local)),
    Math.round(lerp(stops[lo][2], stops[hi][2], local)),
  ];
}

function drawGuides(ctx, width, height, maxR) {
  const cx = width / 2;
  const cy = height / 2;
  const radius = Math.max(8, Math.min(width, height) / 2 - 24);

  ctx.save();
  ctx.strokeStyle = "rgba(110, 118, 129, 0.5)";
  ctx.lineWidth = 1;
  for (let ring = 1; ring <= 4; ring += 1) {
    const rr = radius * (ring / 4.0);
    ctx.beginPath();
    ctx.arc(cx, cy, rr, 0, Math.PI * 2);
    ctx.stroke();
  }

  ctx.beginPath();
  ctx.moveTo(cx - radius, cy);
  ctx.lineTo(cx + radius, cy);
  ctx.moveTo(cx, cy - radius);
  ctx.lineTo(cx, cy + radius);
  ctx.stroke();

  ctx.fillStyle = "rgba(17, 24, 39, 0.8)";
  ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, monospace";
  ctx.fillText("N", cx + 4, cy - radius - 6);
  ctx.fillText("E", cx + radius + 6, cy + 4);
  ctx.fillText("S", cx + 4, cy + radius + 14);
  ctx.fillText("W", cx - radius - 14, cy + 4);

  if (Number.isFinite(maxR) && maxR > 0) {
    const km = (maxR / 1000.0).toFixed(1);
    ctx.fillText(`max range: ${km} km`, 12, 18);
  }
  ctx.restore();
}

function nearestPick(pick, width, height, x, y) {
  const ix = Math.round(x);
  const iy = Math.round(y);
  for (let radius = 0; radius <= 3; radius += 1) {
    for (let dy = -radius; dy <= radius; dy += 1) {
      const py = iy + dy;
      if (py < 0 || py >= height) continue;
      for (let dx = -radius; dx <= radius; dx += 1) {
        const px = ix + dx;
        if (px < 0 || px >= width) continue;
        const idx = pick[py * width + px];
        if (idx >= 0) {
          return idx;
        }
      }
    }
  }
  return -1;
}

export default {
  render({ model, el }) {
    const root = document.createElement("div");
    root.className = "radrs-viz-root";
    const canvas = document.createElement("canvas");
    canvas.className = "radrs-viz-canvas";
    const tooltip = document.createElement("div");
    tooltip.className = "radrs-viz-tooltip";
    tooltip.style.display = "none";

    root.appendChild(canvas);
    root.appendChild(tooltip);
    el.appendChild(root);

    let pick = new Int32Array(0);
    let azimuth = new Float32Array(0);
    let range = new Float32Array(0);
    let values = new Float32Array(0);
    let elevation = new Float32Array(0);
    let gateIndex = new Uint16Array(0);
    let returnIndex = new Uint32Array(0);
    let returnTime = new Float64Array(0);
    let lastHover = -1;

    function redraw() {
      const width = Number(model.get("width")) || 760;
      const height = Number(model.get("height")) || 760;
      canvas.width = width;
      canvas.height = height;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, width, height);

      azimuth = decodeArray(model.get("azimuth_bytes"), Float32Array);
      range = decodeArray(model.get("range_bytes"), Float32Array);
      values = decodeArray(model.get("value_bytes"), Float32Array);
      elevation = decodeArray(model.get("elevation_bytes"), Float32Array);
      gateIndex = decodeArray(model.get("gate_index_bytes"), Uint16Array);
      returnIndex = decodeArray(model.get("return_index_bytes"), Uint32Array);
      returnTime = decodeArray(model.get("return_time_ms_bytes"), Float64Array);

      const meta = model.get("meta") || {};
      const n = Math.min(
        azimuth.length,
        range.length,
        values.length,
        elevation.length,
        gateIndex.length,
        returnIndex.length,
        returnTime.length,
      );
      if (n === 0) {
        drawGuides(ctx, width, height, Number(meta.max_range_m) || 0);
        return;
      }

      const vmin = Number(meta.vmin);
      const vmax = Number(meta.vmax);
      const denom = vmax > vmin ? (vmax - vmin) : 1.0;
      const maxRange = Math.max(1.0, Number(meta.max_range_m) || 1.0);

      const cx = width / 2;
      const cy = height / 2;
      const radiusPx = Math.max(8, Math.min(width, height) / 2 - 24);

      const image = ctx.createImageData(width, height);
      const pixels = image.data;
      pick = new Int32Array(width * height);
      pick.fill(-1);

      for (let i = 0; i < n; i += 1) {
        const vv = values[i];
        if (!Number.isFinite(vv)) {
          continue;
        }
        const theta = ((90.0 - azimuth[i]) * Math.PI) / 180.0;
        const rr = (range[i] / maxRange) * radiusPx;
        const x = Math.round(cx + rr * Math.cos(theta));
        const y = Math.round(cy - rr * Math.sin(theta));
        if (x < 0 || x >= width || y < 0 || y >= height) {
          continue;
        }

        const normalized = (vv - vmin) / denom;
        const [r, g, b] = colorMap(normalized);
        const pxOffset = (y * width + x) * 4;
        pixels[pxOffset] = r;
        pixels[pxOffset + 1] = g;
        pixels[pxOffset + 2] = b;
        pixels[pxOffset + 3] = 255;
        pick[y * width + x] = i;
      }

      ctx.putImageData(image, 0, 0);
      drawGuides(ctx, width, height, maxRange);
    }

    function hideTooltip() {
      tooltip.style.display = "none";
      if (lastHover !== -1) {
        lastHover = -1;
        model.set("hover", {});
        model.save_changes();
      }
    }

    function onMove(event) {
      if (pick.length === 0) {
        hideTooltip();
        return;
      }

      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const width = canvas.width;
      const height = canvas.height;
      const idx = nearestPick(pick, width, height, x, y);
      if (idx < 0) {
        hideTooltip();
        return;
      }

      const timeMs = returnTime[idx];
      const iso = Number.isFinite(timeMs) ? new Date(timeMs).toISOString() : "n/a";
      tooltip.style.left = `${Math.max(8, Math.round(x + 10))}px`;
      tooltip.style.top = `${Math.max(8, Math.round(y + 10))}px`;
      tooltip.style.display = "block";
      tooltip.textContent =
        `value=${values[idx].toFixed(2)} az=${azimuth[idx].toFixed(2)}deg ` +
        `r=${(range[idx] / 1000.0).toFixed(2)}km gate=${gateIndex[idx]} ` +
        `ret=${returnIndex[idx]} el=${elevation[idx].toFixed(2)}deg t=${iso}`;

      if (idx !== lastHover) {
        lastHover = idx;
        model.set("hover", {
          index: idx,
          value: Number(values[idx]),
          azimuth_deg: Number(azimuth[idx]),
          range_m: Number(range[idx]),
          elevation_deg: Number(elevation[idx]),
          gate_index: Number(gateIndex[idx]),
          return_index: Number(returnIndex[idx]),
          return_time_ms: Number(timeMs),
          return_time_iso: iso,
        });
        model.save_changes();
      }
    }

    function onLeave() {
      hideTooltip();
    }

    const watched = [
      "width",
      "height",
      "meta",
      "azimuth_bytes",
      "range_bytes",
      "value_bytes",
      "elevation_bytes",
      "gate_index_bytes",
      "return_index_bytes",
      "return_time_ms_bytes",
    ];
    for (const key of watched) {
      model.on(`change:${key}`, redraw);
    }

    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("mouseleave", onLeave);

    redraw();

    return () => {
      for (const key of watched) {
        model.off(`change:${key}`, redraw);
      }
      canvas.removeEventListener("mousemove", onMove);
      canvas.removeEventListener("mouseleave", onLeave);
    };
  },
};
"""

_VOLUME_WIDGET_ESM: Final[str] = r"""
import { COORDINATE_SYSTEM, Deck, LineLayer, OrbitView, PointCloudLayer } from "https://esm.sh/deck.gl@9.2.2?bundle";

function toArrayBuffer(raw) {
  if (!raw) return new ArrayBuffer(0);
  if (raw instanceof ArrayBuffer) return raw;
  if (ArrayBuffer.isView(raw)) {
    return raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
  }
  return new Uint8Array(raw).buffer;
}

function decodeArray(raw, ctor) {
  const buffer = toArrayBuffer(raw);
  if (buffer.byteLength === 0) return new ctor(0);
  const bytesPerElement = ctor.BYTES_PER_ELEMENT;
  const trimmed = buffer.byteLength - (buffer.byteLength % bytesPerElement);
  return new ctor(buffer.slice(0, trimmed));
}

function clamp01(x) {
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function clamp(v, lo, hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

function colorMap(t) {
  const x = clamp01(t);
  const stops = [
    [68, 1, 84],
    [59, 82, 139],
    [33, 145, 140],
    [94, 201, 98],
    [253, 231, 37],
  ];
  const scaled = x * (stops.length - 1);
  const lo = Math.floor(scaled);
  const hi = Math.min(stops.length - 1, lo + 1);
  const local = scaled - lo;
  return [
    Math.round(lerp(stops[lo][0], stops[hi][0], local)),
    Math.round(lerp(stops[lo][1], stops[hi][1], local)),
    Math.round(lerp(stops[lo][2], stops[hi][2], local)),
  ];
}

function computeHover(xVals, yVals, zVals, values, gateIndex, returnIndex, idx) {
  const xx = xVals[idx];
  const yy = yVals[idx];
  const zz = zVals[idx];
  const rr = Math.sqrt(xx * xx + yy * yy + zz * zz);
  const az = ((Math.atan2(xx, yy) * 180.0) / Math.PI + 360.0) % 360.0;
  const el = (Math.atan2(zz, Math.sqrt(xx * xx + yy * yy)) * 180.0) / Math.PI;
  return {
    index: idx,
    value: Number(values[idx]),
    x_m: Number(xx),
    y_m: Number(yy),
    z_m: Number(zz),
    azimuth_deg: Number(az),
    elevation_deg: Number(el),
    range_m: Number(rr),
    gate_index: Number(gateIndex[idx]),
    return_index: Number(returnIndex[idx]),
  };
}

function formatTooltip(hover) {
  return (
    `value=${hover.value.toFixed(2)} xyz=(${hover.x_m.toFixed(0)},${hover.y_m.toFixed(0)},${hover.z_m.toFixed(0)})m ` +
    `az=${hover.azimuth_deg.toFixed(2)}deg el=${hover.elevation_deg.toFixed(2)}deg ` +
    `r=${(hover.range_m / 1000.0).toFixed(2)}km ret=${hover.return_index} gate=${hover.gate_index}`
  );
}

function fitZoomForExtent(maxAbsMeters, width, height) {
  if (!Number.isFinite(maxAbsMeters) || maxAbsMeters <= 0) {
    return -7.0;
  }
  const pixelRadius = Math.max(1.0, 0.45 * Math.min(width, height));
  const pixelsPerMeter = pixelRadius / maxAbsMeters;
  return Math.log2(pixelsPerMeter);
}

function pointSizeForCount(n) {
  if (n > 240000) return 0.9;
  if (n > 160000) return 1.1;
  if (n > 100000) return 1.3;
  return 1.6;
}

function lineWidthForCount(n) {
  if (n > 40000) return 0.8;
  if (n > 20000) return 1.0;
  return 1.2;
}

export default {
  render({ model, el }) {
    const root = document.createElement("div");
    root.className = "radrs-viz-root";
    const deckHost = document.createElement("div");
    deckHost.className = "radrs-viz-deck-host";
    const overlay = document.createElement("div");
    overlay.className = "radrs-viz-overlay";
    const tooltip = document.createElement("div");
    tooltip.className = "radrs-viz-tooltip";
    tooltip.style.display = "none";

    root.appendChild(deckHost);
    root.appendChild(overlay);
    root.appendChild(tooltip);
    el.appendChild(root);

    /** @type {Deck | null} */
    let deck = null;
    let xVals = new Float32Array(0);
    let yVals = new Float32Array(0);
    let zVals = new Float32Array(0);
    let values = new Float32Array(0);
    let gateIndex = new Uint16Array(0);
    let returnIndex = new Uint32Array(0);
    let positions = new Float32Array(0);
    let colors = new Uint8Array(0);
    let n = 0;
    let lastHover = -1;
    let fitZoom = -7.0;
    let viewState = null;

    function readWidth() {
      return Number(model.get("width")) || 760;
    }

    function readHeight() {
      return Number(model.get("height")) || 760;
    }

    function readMeta() {
      const meta = model.get("meta");
      return meta && typeof meta === "object" ? meta : {};
    }

    function readAngle(name, fallback) {
      const value = Number(model.get(name));
      return Number.isFinite(value) ? value : fallback;
    }

    function hideTooltip() {
      tooltip.style.display = "none";
      if (lastHover !== -1) {
        lastHover = -1;
        model.set("hover", {});
        model.save_changes();
      }
    }

    function loadAndColorize() {
      xVals = decodeArray(model.get("x_bytes"), Float32Array);
      yVals = decodeArray(model.get("y_bytes"), Float32Array);
      zVals = decodeArray(model.get("z_bytes"), Float32Array);
      values = decodeArray(model.get("value_bytes"), Float32Array);
      gateIndex = decodeArray(model.get("gate_index_bytes"), Uint16Array);
      returnIndex = decodeArray(model.get("return_index_bytes"), Uint32Array);

      n = Math.min(
        xVals.length,
        yVals.length,
        zVals.length,
        values.length,
        gateIndex.length,
        returnIndex.length,
      );

      positions = new Float32Array(n * 3);
      colors = new Uint8Array(n * 3);
      const meta = readMeta();
      const vmin = Number(meta.vmin);
      const vmax = Number(meta.vmax);
      const denom = Number.isFinite(vmin) && Number.isFinite(vmax) && vmax > vmin ? vmax - vmin : 1.0;
      for (let i = 0; i < n; i += 1) {
        const posOffset = 3 * i;
        positions[posOffset] = xVals[i];
        positions[posOffset + 1] = yVals[i];
        positions[posOffset + 2] = zVals[i];
        const [r, g, b] = colorMap((values[i] - vmin) / denom);
        colors[posOffset] = r;
        colors[posOffset + 1] = g;
        colors[posOffset + 2] = b;
      }

      const width = readWidth();
      const height = readHeight();
      fitZoom = fitZoomForExtent(Number(meta.max_abs_m), width, height);
      if (viewState === null) {
        viewState = {
          target: [0, 0, 0],
          rotationOrbit: readAngle("yaw_deg", 35),
          rotationX: readAngle("pitch_deg", 30),
          zoom: fitZoom,
          minZoom: fitZoom - 5.0,
          maxZoom: fitZoom + 8.0,
          minRotationX: -89,
          maxRotationX: 89,
        };
      } else {
        const zoomOffset = viewState.zoom - fitZoom;
        viewState = {
          ...viewState,
          target: [0, 0, 0],
          zoom: clamp(fitZoom + zoomOffset, fitZoom - 5.0, fitZoom + 8.0),
          minZoom: fitZoom - 5.0,
          maxZoom: fitZoom + 8.0,
          minRotationX: -89,
          maxRotationX: 89,
        };
      }
    }

    function buildLayer() {
      const meta = readMeta();
      const renderMode = String(meta.render_mode || "points");
      if (renderMode === "rays") {
        return new LineLayer({
          id: "radrs-volume-rays",
          data: {
            length: n,
            attributes: {
              getTargetPosition: { value: positions, size: 3 },
              getColor: { value: colors, size: 3 },
            },
          },
          getSourcePosition: [0, 0, 0],
          coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
          pickable: true,
          widthUnits: "pixels",
          getWidth: lineWidthForCount(n),
          opacity: 0.95,
        });
      }
      return new PointCloudLayer({
        id: "radrs-volume-points",
        data: {
          length: n,
          attributes: {
            getPosition: { value: positions, size: 3 },
            getColor: { value: colors, size: 3 },
          },
        },
        coordinateSystem: COORDINATE_SYSTEM.CARTESIAN,
        pickable: true,
        pointSize: pointSizeForCount(n),
        opacity: 0.95,
      });
    }

    function updateOverlay() {
      const meta = readMeta();
      if (n === 0) {
        overlay.textContent = "Volume 3D (deck.gl) | no finite points";
        return;
      }
      const renderMode = String(meta.render_mode || "points");
      const itemLabel = renderMode === "rays" ? "rays" : "gates";
      overlay.textContent = (
        `Volume 3D (deck.gl) | ${itemLabel}=${n.toLocaleString()} ` +
        `moment=${String(meta.moment || "")} ` +
        `range=${(Number(meta.max_abs_m) / 1000.0).toFixed(2)}km`
      );
    }

    function ensureDeck() {
      if (deck !== null) {
        return;
      }
      deck = new Deck({
        parent: deckHost,
        width: readWidth(),
        height: readHeight(),
        views: [new OrbitView({ id: "orbit" })],
        controller: true,
        viewState,
        layers: [buildLayer()],
        parameters: {
          clearColor: [246, 248, 250, 255],
        },
        getCursor: ({ isDragging }) => (isDragging ? "grabbing" : "grab"),
        onViewStateChange: ({ viewState: nextViewState }) => {
          if (!nextViewState) return;
          viewState = {
            ...nextViewState,
            minRotationX: -89,
            maxRotationX: 89,
          };
          deck?.setProps({ viewState });
        },
        onHover: (info) => {
          if (!info || typeof info.index !== "number" || info.index < 0 || info.index >= n) {
            hideTooltip();
            return;
          }
          const idx = info.index;
          const hover = computeHover(xVals, yVals, zVals, values, gateIndex, returnIndex, idx);
          const px = Number.isFinite(info.x) ? info.x : 8;
          const py = Number.isFinite(info.y) ? info.y : 8;
          tooltip.style.left = `${Math.max(8, Math.round(px + 10))}px`;
          tooltip.style.top = `${Math.max(8, Math.round(py + 10))}px`;
          tooltip.style.display = "block";
          tooltip.textContent = formatTooltip(hover);
          if (idx !== lastHover) {
            lastHover = idx;
            model.set("hover", hover);
            model.save_changes();
          }
        },
      });
    }

    function renderDeck() {
      const width = readWidth();
      const height = readHeight();
      root.style.width = `${width}px`;
      root.style.height = `${height}px`;
      ensureDeck();
      deck?.setProps({
        width,
        height,
        viewState,
        layers: [buildLayer()],
      });
      updateOverlay();
    }

    function refreshFromModel() {
      loadAndColorize();
      renderDeck();
      hideTooltip();
    }

    function onSizeChange() {
      const width = readWidth();
      const height = readHeight();
      const nextFit = fitZoomForExtent(Number(readMeta().max_abs_m), width, height);
      if (viewState !== null) {
        const zoomOffset = viewState.zoom - fitZoom;
        fitZoom = nextFit;
        viewState = {
          ...viewState,
          zoom: clamp(nextFit + zoomOffset, nextFit - 5.0, nextFit + 8.0),
          minZoom: nextFit - 5.0,
          maxZoom: nextFit + 8.0,
        };
      }
      renderDeck();
    }

    function onAngleChange() {
      if (viewState === null) {
        return;
      }
      viewState = {
        ...viewState,
        rotationOrbit: readAngle("yaw_deg", viewState.rotationOrbit),
        rotationX: readAngle("pitch_deg", viewState.rotationX),
      };
      renderDeck();
    }

    const redrawWatched = ["meta"];
    const dataWatched = [
      "x_bytes",
      "y_bytes",
      "z_bytes",
      "value_bytes",
      "gate_index_bytes",
      "return_index_bytes",
    ];
    for (const key of redrawWatched) {
      model.on(`change:${key}`, refreshFromModel);
    }
    for (const key of dataWatched) {
      model.on(`change:${key}`, refreshFromModel);
    }
    model.on("change:width", onSizeChange);
    model.on("change:height", onSizeChange);
    model.on("change:yaw_deg", onAngleChange);
    model.on("change:pitch_deg", onAngleChange);

    refreshFromModel();
    // deck.gl may skip the first paint if the container isn't visible yet
    // (e.g. marimo defers widget display).  Re-render once it enters the
    // viewport so the WebGL canvas gets the correct dimensions.
    let visObs = new IntersectionObserver((entries) => {
      if (entries[0]?.isIntersecting && deck) {
        visObs.disconnect();
        visObs = null;
        deck.setProps({ width: readWidth(), height: readHeight(), layers: [buildLayer()] });
      }
    }, { threshold: 0.01 });
    visObs.observe(root);

    return () => {
      if (visObs) { visObs.disconnect(); visObs = null; }
      for (const key of redrawWatched) {
        model.off(`change:${key}`, refreshFromModel);
      }
      for (const key of dataWatched) {
        model.off(`change:${key}`, refreshFromModel);
      }
      model.off("change:width", onSizeChange);
      model.off("change:height", onSizeChange);
      model.off("change:yaw_deg", onAngleChange);
      model.off("change:pitch_deg", onAngleChange);
      if (deck !== null) {
        deck.finalize();
        deck = null;
      }
    };
  },
};
"""

_GRID_WIDGET_ESM: Final[str] = r"""
function toArrayBuffer(raw) {
  if (!raw) return new ArrayBuffer(0);
  if (raw instanceof ArrayBuffer) return raw;
  if (ArrayBuffer.isView(raw)) {
    return raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
  }
  return new Uint8Array(raw).buffer;
}

function decodeArray(raw, ctor) {
  const buffer = toArrayBuffer(raw);
  if (buffer.byteLength === 0) return new ctor(0);
  const bytesPerElement = ctor.BYTES_PER_ELEMENT;
  const trimmed = buffer.byteLength - (buffer.byteLength % bytesPerElement);
  return new ctor(buffer.slice(0, trimmed));
}

function clamp01(x) {
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function colorMap(t) {
  const x = clamp01(t);
  const stops = [
    [68, 1, 84],
    [59, 82, 139],
    [33, 145, 140],
    [94, 201, 98],
    [253, 231, 37],
  ];
  const scaled = x * (stops.length - 1);
  const lo = Math.floor(scaled);
  const hi = Math.min(stops.length - 1, lo + 1);
  const local = scaled - lo;
  return [
    Math.round(lerp(stops[lo][0], stops[hi][0], local)),
    Math.round(lerp(stops[lo][1], stops[hi][1], local)),
    Math.round(lerp(stops[lo][2], stops[hi][2], local)),
  ];
}

/** Compute nice round tick positions within [lo, hi] in world units. */
function niceTicks(lo, hi, maxTicks) {
  const range = hi - lo;
  if (range <= 0) return [];
  const rough = range / maxTicks;
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  let step = mag;
  if (rough / mag >= 5) step = mag * 5;
  else if (rough / mag >= 2) step = mag * 2;
  const ticks = [];
  const start = Math.ceil(lo / step) * step;
  for (let v = start; v <= hi; v += step) {
    ticks.push(v);
  }
  return ticks;
}

function drawGridImage(ctx, grid, nRows, nCols, vmin, vmax, marginLeft, marginTop, plotW, plotH) {
  if (nRows === 0 || nCols === 0) return;
  const denom = vmax > vmin ? (vmax - vmin) : 1.0;
  const image = ctx.createImageData(plotW, plotH);
  const pixels = image.data;

  for (let py = 0; py < plotH; py++) {
    const row = Math.floor((py / plotH) * nRows);
    for (let px = 0; px < plotW; px++) {
      const col = Math.floor((px / plotW) * nCols);
      const val = grid[row * nCols + col];
      const off = (py * plotW + px) * 4;
      if (!Number.isFinite(val)) {
        pixels[off + 3] = 0;
        continue;
      }
      const [r, g, b] = colorMap((val - vmin) / denom);
      pixels[off] = r;
      pixels[off + 1] = g;
      pixels[off + 2] = b;
      pixels[off + 3] = 255;
    }
  }
  ctx.putImageData(image, marginLeft, marginTop);
}

function drawAxes(ctx, meta, marginLeft, marginTop, plotW, plotH, canvasW, canvasH) {
  const xMin = Number(meta.x_min);
  const xMax = Number(meta.x_max);
  const yMin = Number(meta.y_min);
  const yMax = Number(meta.y_max);
  const xLabel = String(meta.x_label || "");
  const yLabel = String(meta.y_label || "");

  ctx.save();
  ctx.strokeStyle = "rgba(110, 118, 129, 0.5)";
  ctx.fillStyle = "rgba(17, 24, 39, 0.85)";
  ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, Monaco, monospace";
  ctx.lineWidth = 1;

  // X-axis ticks (bottom)
  const xTicks = niceTicks(xMin / 1000, xMax / 1000, 8);
  const marginBottom = canvasH - marginTop - plotH;
  for (const km of xTicks) {
    const px = marginLeft + ((km * 1000 - xMin) / (xMax - xMin)) * plotW;
    ctx.beginPath();
    ctx.moveTo(px, marginTop + plotH);
    ctx.lineTo(px, marginTop + plotH + 4);
    ctx.stroke();
    ctx.fillText(km.toFixed(0), px - 8, marginTop + plotH + 14);
  }
  ctx.fillText(xLabel, marginLeft + plotW / 2 - 30, canvasH - 4);

  // Y-axis ticks (left)
  const yTicks = niceTicks(yMin / 1000, yMax / 1000, 8);
  for (const km of yTicks) {
    const py = marginTop + (1 - (km * 1000 - yMin) / (yMax - yMin)) * plotH;
    ctx.beginPath();
    ctx.moveTo(marginLeft - 4, py);
    ctx.lineTo(marginLeft, py);
    ctx.stroke();
    ctx.fillText(km.toFixed(0), 4, py + 4);
  }
  ctx.save();
  ctx.translate(10, marginTop + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, -30, 0);
  ctx.restore();

  // Border around plot area
  ctx.strokeStyle = "rgba(110, 118, 129, 0.7)";
  ctx.strokeRect(marginLeft, marginTop, plotW, plotH);

  ctx.restore();
}

function drawCappiGuides(ctx, meta, marginLeft, marginTop, plotW, plotH) {
  const xMin = Number(meta.x_min);
  const xMax = Number(meta.x_max);
  const yMin = Number(meta.y_min);
  const yMax = Number(meta.y_max);

  ctx.save();

  // Range rings centered on radar (x=0, y=0)
  const cx = marginLeft + ((0 - xMin) / (xMax - xMin)) * plotW;
  const cy = marginTop + ((yMax - 0) / (yMax - yMin)) * plotH;
  const maxExtent = Math.max(Math.abs(xMin), xMax, Math.abs(yMin), yMax);
  const ringStep = maxExtent > 200000 ? 100000 : maxExtent > 50000 ? 50000 : 10000;

  ctx.strokeStyle = "rgba(110, 118, 129, 0.25)";
  ctx.lineWidth = 0.5;
  for (let r = ringStep; r < maxExtent * 1.5; r += ringStep) {
    const rpx = (r / (xMax - xMin)) * plotW;
    ctx.beginPath();
    ctx.arc(cx, cy, rpx, 0, Math.PI * 2);
    ctx.stroke();
  }

  // Crosshairs through radar
  ctx.strokeStyle = "rgba(110, 118, 129, 0.35)";
  ctx.lineWidth = 0.5;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(marginLeft, cy);
  ctx.lineTo(marginLeft + plotW, cy);
  ctx.moveTo(cx, marginTop);
  ctx.lineTo(cx, marginTop + plotH);
  ctx.stroke();
  ctx.setLineDash([]);

  // N/E/S/W labels
  ctx.fillStyle = "rgba(17, 24, 39, 0.7)";
  ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, Monaco, monospace";
  ctx.fillText("N", cx + 2, marginTop + 10);
  ctx.fillText("S", cx + 2, marginTop + plotH - 4);
  ctx.fillText("E", marginLeft + plotW - 10, cy - 4);
  ctx.fillText("W", marginLeft + 2, cy - 4);

  ctx.restore();
}

function drawXsecGuides(ctx, meta, marginLeft, marginTop, plotW, plotH) {
  const xMin = Number(meta.x_min);
  const xMax = Number(meta.x_max);
  const yMin = Number(meta.y_min);
  const yMax = Number(meta.y_max);

  ctx.save();
  ctx.strokeStyle = "rgba(110, 118, 129, 0.3)";
  ctx.lineWidth = 0.5;
  ctx.setLineDash([4, 4]);

  // Horizontal lines at round km altitudes
  const altTicks = niceTicks(yMin / 1000, yMax / 1000, 6);
  for (const km of altTicks) {
    const py = marginTop + (1 - (km * 1000 - yMin) / (yMax - yMin)) * plotH;
    ctx.beginPath();
    ctx.moveTo(marginLeft, py);
    ctx.lineTo(marginLeft + plotW, py);
    ctx.stroke();
  }

  // Vertical lines at round km ranges
  const rngTicks = niceTicks(xMin / 1000, xMax / 1000, 8);
  for (const km of rngTicks) {
    const px = marginLeft + ((km * 1000 - xMin) / (xMax - xMin)) * plotW;
    ctx.beginPath();
    ctx.moveTo(px, marginTop);
    ctx.lineTo(px, marginTop + plotH);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // Emphasized vertical line at range=0 (the radar)
  const radarX = marginLeft + ((0 - xMin) / (xMax - xMin)) * plotW;
  ctx.strokeStyle = "rgba(220, 38, 38, 0.5)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(radarX, marginTop);
  ctx.lineTo(radarX, marginTop + plotH);
  ctx.stroke();

  ctx.restore();
}

function drawElevationDiagram(ctx, meta, canvasW, canvasH) {
  const sweepElevations = meta.sweep_elevations_deg;
  if (!sweepElevations || sweepElevations.length === 0) return;

  const dw = 160;
  const dh = 100;
  const dx = canvasW - dw - 12;
  const dy = canvasH - dh - 12;

  ctx.save();
  // Background panel
  ctx.fillStyle = "rgba(17, 24, 39, 0.80)";
  ctx.beginPath();
  ctx.roundRect(dx, dy, dw, dh, 4);
  ctx.fill();

  const mode = String(meta.grid_mode);
  const ox = dx + 10;
  const oy = dy + dh - 10;
  const maxLen = dw - 20;
  const maxAlt = dh - 20;

  // Determine active sweeps
  const cappiAlt = Number(meta.cappi_altitude_m) || 0;
  const cappiTol = Number(meta.cappi_tolerance_m) || 0;
  const xMaxM = Number(meta.x_max) || 1;

  for (let i = 0; i < sweepElevations.length; i++) {
    const el = sweepElevations[i];
    const rad = (el * Math.PI) / 180;
    let active = false;

    if (mode === "cappi") {
      const maxZ = xMaxM * Math.sin(rad);
      active = el > 0 && (cappiAlt - cappiTol) < maxZ && (cappiAlt + cappiTol) > 0;
    } else {
      active = true;
    }

    ctx.strokeStyle = active ? "rgba(34, 197, 94, 0.8)" : "rgba(110, 118, 129, 0.5)";
    ctx.lineWidth = active ? 1.5 : 0.8;
    ctx.beginPath();
    ctx.moveTo(ox, oy);
    const endX = ox + maxLen * Math.cos(rad);
    const endY = oy - maxLen * Math.sin(rad);
    ctx.lineTo(endX, endY);
    ctx.stroke();
  }

  // CAPPI altitude line
  if (mode === "cappi") {
    const maxElRad = Math.max(...sweepElevations) * Math.PI / 180;
    const altScale = maxAlt / (xMaxM * Math.sin(Math.max(maxElRad, 0.01)));
    const altPx = Math.min(cappiAlt * altScale, maxAlt);
    ctx.strokeStyle = "rgba(253, 231, 37, 0.7)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(ox, oy - altPx);
    ctx.lineTo(ox + maxLen, oy - altPx);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Label
  ctx.fillStyle = "rgba(249, 250, 251, 0.8)";
  ctx.font = "9px ui-monospace, SFMono-Regular, Menlo, Monaco, monospace";
  ctx.fillText("Elevation beams", dx + 6, dy + 12);

  ctx.restore();
}

function drawAzimuthIndicator(ctx, meta, canvasW) {
  const mode = String(meta.grid_mode);
  const dw = 80;
  const dh = 80;
  const dx = canvasW - dw - 12;
  const dy = 12;

  ctx.save();
  ctx.fillStyle = "rgba(17, 24, 39, 0.80)";
  ctx.beginPath();
  ctx.roundRect(dx, dy, dw, dh, 4);
  ctx.fill();

  const cx = dx + dw / 2;
  const cy = dy + dh / 2 + 4;
  const r = 28;

  ctx.strokeStyle = "rgba(110, 118, 129, 0.5)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.stroke();

  // N marker
  ctx.fillStyle = "rgba(249, 250, 251, 0.8)";
  ctx.font = "9px ui-monospace, SFMono-Regular, Menlo, Monaco, monospace";
  ctx.fillText("N", cx - 3, dy + 10);

  if (mode === "cappi") {
    // Full circle fill for 360-degree coverage
    ctx.fillStyle = "rgba(34, 197, 94, 0.2)";
    ctx.beginPath();
    ctx.arc(cx, cy, r - 2, 0, Math.PI * 2);
    ctx.fill();
  } else if (mode === "xsec") {
    // Line through center at target azimuth
    const az = Number(meta.xsec_azimuth_deg) || 0;
    const rad = ((90 - az) * Math.PI) / 180;
    ctx.strokeStyle = "rgba(253, 231, 37, 0.8)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx - r * Math.cos(rad), cy + r * Math.sin(rad));
    ctx.lineTo(cx + r * Math.cos(rad), cy - r * Math.sin(rad));
    ctx.stroke();
    ctx.fillStyle = "rgba(253, 231, 37, 0.9)";
    ctx.fillText(`${az.toFixed(0)}°`, dx + 4, dy + dh - 6);
  }

  ctx.restore();
}

function drawSweepInfo(ctx, meta, marginLeft) {
  const sweepElevations = meta.sweep_elevations_deg;
  const sweepCounts = meta.sweep_num_returns;
  if (!sweepElevations || sweepElevations.length === 0) return;

  const mode = String(meta.grid_mode);
  const cappiAlt = Number(meta.cappi_altitude_m) || 0;
  const cappiTol = Number(meta.cappi_tolerance_m) || 0;
  const xMaxM = Number(meta.x_max) || 1;

  ctx.save();
  ctx.font = "9px ui-monospace, SFMono-Regular, Menlo, Monaco, monospace";

  let y = 40;
  const x = marginLeft + 4;

  // Build sweep labels, max 2 lines
  const parts = [];
  for (let i = 0; i < sweepElevations.length; i++) {
    const el = sweepElevations[i];
    const cnt = sweepCounts ? sweepCounts[i] : "?";
    let active = false;
    if (mode === "cappi") {
      const rad = (el * Math.PI) / 180;
      const maxZ = xMaxM * Math.sin(rad);
      active = el > 0 && (cappiAlt - cappiTol) < maxZ && (cappiAlt + cappiTol) > 0;
    } else {
      active = true;
    }
    parts.push({ text: `${el.toFixed(1)}°(${cnt})`, active });
  }

  // Render up to 2 lines, 10 items per line
  const perLine = 10;
  for (let line = 0; line < 2 && line * perLine < parts.length; line++) {
    let lineX = x;
    const start = line * perLine;
    const end = Math.min(start + perLine, parts.length);
    for (let i = start; i < end; i++) {
      const p = parts[i];
      ctx.fillStyle = p.active ? "rgba(34, 197, 94, 0.9)" : "rgba(110, 118, 129, 0.7)";
      ctx.fillText(p.text, lineX, y);
      lineX += ctx.measureText(p.text).width + 6;
    }
    if (end < parts.length && line === 1) {
      ctx.fillStyle = "rgba(110, 118, 129, 0.7)";
      ctx.fillText("...", lineX, y);
    }
    y += 12;
  }

  ctx.restore();
}

function drawModeOverlay(ctx, meta, marginLeft) {
  const mode = String(meta.grid_mode);
  const moment = String(meta.moment || "");

  ctx.save();
  ctx.fillStyle = "rgba(17, 24, 39, 0.75)";
  ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, Monaco, monospace";

  let label;
  if (mode === "cappi") {
    const altKm = ((Number(meta.cappi_altitude_m) || 0) / 1000).toFixed(1);
    const tolKm = ((Number(meta.cappi_tolerance_m) || 0) / 1000).toFixed(1);
    label = `CAPPI alt=${altKm}km | moment=${moment} | tol=${tolKm}km`;
  } else {
    const az = (Number(meta.xsec_azimuth_deg) || 0).toFixed(1);
    label = `Cross-section az=${az}° | moment=${moment}`;
  }

  // Background for overlay text
  const tw = ctx.measureText(label).width;
  ctx.fillStyle = "rgba(17, 24, 39, 0.75)";
  ctx.fillRect(marginLeft, 4, tw + 12, 20);
  ctx.fillStyle = "#f9fafb";
  ctx.fillText(label, marginLeft + 6, 18);

  ctx.restore();
}

export default {
  render({ model, el }) {
    const root = document.createElement("div");
    root.className = "radrs-viz-root";
    const canvas = document.createElement("canvas");
    canvas.className = "radrs-viz-canvas";
    const tooltip = document.createElement("div");
    tooltip.className = "radrs-viz-tooltip";
    tooltip.style.display = "none";

    root.appendChild(canvas);
    root.appendChild(tooltip);
    el.appendChild(root);

    let grid = new Float32Array(0);
    let lastHover = "";

    const marginLeft = 50;
    const marginTop = 20;
    const marginRight = 20;
    const marginBottom = 40;

    function redraw() {
      const width = Number(model.get("width")) || 760;
      const height = Number(model.get("height")) || 760;
      canvas.width = width;
      canvas.height = height;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, width, height);

      const meta = model.get("meta") || {};
      grid = decodeArray(model.get("grid_bytes"), Float32Array);

      const nRows = Number(meta.n_rows) || 0;
      const nCols = Number(meta.n_cols) || 0;
      const vmin = Number(meta.vmin);
      const vmax = Number(meta.vmax);
      const mode = String(meta.grid_mode || "cappi");

      const plotW = width - marginLeft - marginRight;
      const plotH = height - marginTop - marginBottom;

      if (nRows > 0 && nCols > 0 && grid.length >= nRows * nCols) {
        drawGridImage(ctx, grid, nRows, nCols, vmin, vmax, marginLeft, marginTop, plotW, plotH);
      }

      // Axes
      drawAxes(ctx, meta, marginLeft, marginTop, plotW, plotH, width, height);

      // Mode-specific guides
      if (mode === "cappi") {
        drawCappiGuides(ctx, meta, marginLeft, marginTop, plotW, plotH);
      } else {
        drawXsecGuides(ctx, meta, marginLeft, marginTop, plotW, plotH);
      }

      // Overlay diagrams
      drawModeOverlay(ctx, meta, marginLeft);
      drawSweepInfo(ctx, meta, marginLeft);
      drawElevationDiagram(ctx, meta, width, height);
      drawAzimuthIndicator(ctx, meta, width);
    }

    function hideTooltip() {
      tooltip.style.display = "none";
      if (lastHover !== "") {
        lastHover = "";
        model.set("hover", {});
        model.save_changes();
      }
    }

    function onMove(event) {
      const meta = model.get("meta") || {};
      const nRows = Number(meta.n_rows) || 0;
      const nCols = Number(meta.n_cols) || 0;
      if (nRows === 0 || nCols === 0 || grid.length === 0) {
        hideTooltip();
        return;
      }

      const rect = canvas.getBoundingClientRect();
      const px = event.clientX - rect.left;
      const py = event.clientY - rect.top;
      const width = canvas.width;
      const height = canvas.height;
      const plotW = width - marginLeft - marginRight;
      const plotH = height - marginTop - marginBottom;

      // Check if inside plot area
      if (px < marginLeft || px > marginLeft + plotW || py < marginTop || py > marginTop + plotH) {
        hideTooltip();
        return;
      }

      const col = Math.floor(((px - marginLeft) / plotW) * nCols);
      const row = Math.floor(((py - marginTop) / plotH) * nRows);
      if (row < 0 || row >= nRows || col < 0 || col >= nCols) {
        hideTooltip();
        return;
      }

      const val = grid[row * nCols + col];
      if (!Number.isFinite(val)) {
        hideTooltip();
        return;
      }

      const xMin = Number(meta.x_min);
      const xMax = Number(meta.x_max);
      const yMin = Number(meta.y_min);
      const yMax = Number(meta.y_max);
      const worldX = xMin + (col + 0.5) / nCols * (xMax - xMin);
      const worldY = yMax - (row + 0.5) / nRows * (yMax - yMin);
      const mode = String(meta.grid_mode || "cappi");

      let tooltipText;
      if (mode === "cappi") {
        const eastKm = (worldX / 1000).toFixed(2);
        const northKm = (worldY / 1000).toFixed(2);
        const rangeKm = (Math.sqrt(worldX * worldX + worldY * worldY) / 1000).toFixed(2);
        tooltipText = `value=${val.toFixed(2)} east=${eastKm}km north=${northKm}km range=${rangeKm}km`;
      } else {
        const grndKm = (worldX / 1000).toFixed(2);
        const altKm = (worldY / 1000).toFixed(2);
        tooltipText = `value=${val.toFixed(2)} ground_range=${grndKm}km altitude=${altKm}km`;
      }

      tooltip.style.left = `${Math.max(8, Math.round(px + 10))}px`;
      tooltip.style.top = `${Math.max(8, Math.round(py + 10))}px`;
      tooltip.style.display = "block";
      tooltip.textContent = tooltipText;

      const hoverKey = `${row},${col}`;
      if (hoverKey !== lastHover) {
        lastHover = hoverKey;
        const hover = { row, col, value: Number(val) };
        if (mode === "cappi") {
          hover.east_m = Number(worldX);
          hover.north_m = Number(worldY);
          hover.range_m = Math.sqrt(worldX * worldX + worldY * worldY);
        } else {
          hover.ground_range_m = Number(worldX);
          hover.altitude_m = Number(worldY);
        }
        model.set("hover", hover);
        model.save_changes();
      }
    }

    function onLeave() {
      hideTooltip();
    }

    const watched = ["width", "height", "meta", "grid_bytes"];
    for (const key of watched) {
      model.on(`change:${key}`, redraw);
    }

    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("mouseleave", onLeave);

    redraw();

    return () => {
      for (const key of watched) {
        model.off(`change:${key}`, redraw);
      }
      canvas.removeEventListener("mousemove", onMove);
      canvas.removeEventListener("mouseleave", onLeave);
    };
  },
};
"""

_WATERFALL_WIDGET_ESM: Final[str] = r"""
function toArrayBuffer(raw) {
  if (!raw) return new ArrayBuffer(0);
  if (raw instanceof ArrayBuffer) return raw;
  if (ArrayBuffer.isView(raw)) {
    return raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
  }
  return new Uint8Array(raw).buffer;
}

function decodeArray(raw, ctor) {
  const buffer = toArrayBuffer(raw);
  if (buffer.byteLength === 0) return new ctor(0);
  const bytesPerElement = ctor.BYTES_PER_ELEMENT;
  const trimmed = buffer.byteLength - (buffer.byteLength % bytesPerElement);
  return new ctor(buffer.slice(0, trimmed));
}

function clamp01(x) {
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function colorMap(t) {
  const x = clamp01(t);
  const stops = [
    [68, 1, 84],
    [59, 82, 139],
    [33, 145, 140],
    [94, 201, 98],
    [253, 231, 37],
  ];
  const scaled = x * (stops.length - 1);
  const lo = Math.floor(scaled);
  const hi = Math.min(stops.length - 1, lo + 1);
  const local = scaled - lo;
  return [
    Math.round(lerp(stops[lo][0], stops[hi][0], local)),
    Math.round(lerp(stops[lo][1], stops[hi][1], local)),
    Math.round(lerp(stops[lo][2], stops[hi][2], local)),
  ];
}

function niceTicks(lo, hi, maxTicks) {
  const range = hi - lo;
  if (range <= 0) return [];
  const rough = range / maxTicks;
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  let step = mag;
  if (rough / mag >= 5) step = mag * 5;
  else if (rough / mag >= 2) step = mag * 2;
  const ticks = [];
  const start = Math.ceil(lo / step) * step;
  for (let v = start; v <= hi; v += step) {
    ticks.push(v);
  }
  return ticks;
}

/** Map elevation to blue-red colour. */
function elevationColor(t) {
  const x = clamp01(t);
  return [
    Math.round(lerp(50, 220, x)),
    Math.round(lerp(50, 50, x)),
    Math.round(lerp(220, 50, 1 - x)),
  ];
}

export default {
  render({ model, el }) {
    const root = document.createElement("div");
    root.className = "radrs-viz-root";
    const canvas = document.createElement("canvas");
    canvas.className = "radrs-viz-canvas";
    const tooltip = document.createElement("div");
    tooltip.className = "radrs-viz-tooltip";
    tooltip.style.display = "none";

    root.appendChild(canvas);
    root.appendChild(tooltip);
    el.appendChild(root);

    let grid = new Float32Array(0);
    let azimuth = new Float32Array(0);
    let elevation = new Float32Array(0);
    let returnTimeMs = new Float64Array(0);
    let sweepNumber = new Uint16Array(0);
    let sweepBoundaries = new Int32Array(0);

    function formatTimeUTC(ms) {
      const d = new Date(ms);
      const hh = String(d.getUTCHours()).padStart(2, "0");
      const mm = String(d.getUTCMinutes()).padStart(2, "0");
      const ss = String(d.getUTCSeconds()).padStart(2, "0");
      return `${hh}:${mm}:${ss}`;
    }
    let lastHoverKey = "";

    const marginLeft = 60;
    const marginTop = 28;
    const marginRight = 40;
    const marginBottom = 40;
    const elevStripW = 30;

    function redraw() {
      const width = Number(model.get("width")) || 760;
      const height = Number(model.get("height")) || 760;
      canvas.width = width;
      canvas.height = height;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, width, height);

      const meta = model.get("meta") || {};
      grid = decodeArray(model.get("grid_bytes"), Float32Array);
      azimuth = decodeArray(model.get("azimuth_bytes"), Float32Array);
      elevation = decodeArray(model.get("elevation_bytes"), Float32Array);
      returnTimeMs = decodeArray(model.get("return_time_ms_bytes"), Float64Array);
      sweepNumber = decodeArray(model.get("sweep_number_bytes"), Uint16Array);
      sweepBoundaries = decodeArray(model.get("sweep_boundary_bytes"), Int32Array);

      const nReturns = Number(meta.n_returns) || 0;
      const nRange = Number(meta.n_range) || 0;
      const vmin = Number(meta.vmin);
      const vmax = Number(meta.vmax);

      const plotW = width - marginLeft - marginRight - elevStripW;
      const plotH = height - marginTop - marginBottom;

      // --- Main heatmap ---
      if (nReturns > 0 && nRange > 0 && grid.length >= nReturns * nRange) {
        const denom = vmax > vmin ? (vmax - vmin) : 1.0;
        const image = ctx.createImageData(plotW, plotH);
        const pixels = image.data;

        for (let py = 0; py < plotH; py++) {
          const row = Math.floor((py / plotH) * nReturns);
          for (let px = 0; px < plotW; px++) {
            const col = Math.floor((px / plotW) * nRange);
            const val = grid[row * nRange + col];
            const off = (py * plotW + px) * 4;
            if (!Number.isFinite(val)) {
              pixels[off + 3] = 0;
              continue;
            }
            const [r, g, b] = colorMap((val - vmin) / denom);
            pixels[off] = r;
            pixels[off + 1] = g;
            pixels[off + 2] = b;
            pixels[off + 3] = 255;
          }
        }
        ctx.putImageData(image, marginLeft, marginTop);
      }

      // --- Sweep boundary lines ---
      if (nReturns > 0) {
        ctx.save();
        ctx.strokeStyle = "rgba(255, 255, 255, 0.5)";
        ctx.lineWidth = 1;
        for (let i = 0; i < sweepBoundaries.length; i++) {
          const bndIdx = sweepBoundaries[i];
          const py = marginTop + (bndIdx / nReturns) * plotH;
          ctx.beginPath();
          ctx.moveTo(marginLeft, py);
          ctx.lineTo(marginLeft + plotW, py);
          ctx.stroke();
        }
        ctx.restore();
      }

      // --- Elevation side strip ---
      if (nReturns > 0 && elevation.length >= nReturns) {
        let elMin = Infinity, elMax = -Infinity;
        for (let i = 0; i < nReturns; i++) {
          const e = elevation[i];
          if (Number.isFinite(e)) {
            if (e < elMin) elMin = e;
            if (e > elMax) elMax = e;
          }
        }
        const elRange = elMax > elMin ? elMax - elMin : 1.0;
        const stripX = marginLeft + plotW + 2;
        const stripImage = ctx.createImageData(elevStripW - 4, plotH);
        const sp = stripImage.data;
        const sw = elevStripW - 4;
        for (let py = 0; py < plotH; py++) {
          const row = Math.floor((py / plotH) * nReturns);
          const t = (elevation[row] - elMin) / elRange;
          const [r, g, b] = elevationColor(t);
          for (let px = 0; px < sw; px++) {
            const off = (py * sw + px) * 4;
            sp[off] = r;
            sp[off + 1] = g;
            sp[off + 2] = b;
            sp[off + 3] = 255;
          }
        }
        ctx.putImageData(stripImage, stripX, marginTop);

        // Elevation ticks
        ctx.save();
        ctx.fillStyle = "rgba(17, 24, 39, 0.85)";
        ctx.font = "9px ui-monospace, SFMono-Regular, Menlo, Monaco, monospace";
        ctx.fillText(`${elMax.toFixed(1)}°`, stripX, marginTop - 2);
        ctx.fillText(`${elMin.toFixed(1)}°`, stripX, marginTop + plotH + 10);
        ctx.fillText("El", stripX + sw / 2 - 4, marginTop + plotH + 20);
        ctx.restore();
      }

      // --- X-axis: Range in km ---
      const rangeStartM = Number(meta.range_start_m) || 0;
      const rangeStepM = Number(meta.range_step_m) || 1;
      const nRangeOrig = Number(meta.n_range_orig) || nRange;
      const rangeStride = nRange > 0 ? Math.max(1, Math.ceil(nRangeOrig / nRange)) : 1;

      ctx.save();
      ctx.strokeStyle = "rgba(110, 118, 129, 0.5)";
      ctx.fillStyle = "rgba(17, 24, 39, 0.85)";
      ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, Monaco, monospace";
      ctx.lineWidth = 1;

      const rangeMinKm = rangeStartM / 1000;
      const rangeMaxKm = (rangeStartM + (nRange - 1) * rangeStepM * rangeStride) / 1000;
      const xTicks = niceTicks(rangeMinKm, rangeMaxKm, 8);
      for (const km of xTicks) {
        const col = ((km * 1000 - rangeStartM) / (rangeStepM * rangeStride)) / nRange;
        const px = marginLeft + col * plotW;
        if (px < marginLeft || px > marginLeft + plotW) continue;
        ctx.beginPath();
        ctx.moveTo(px, marginTop + plotH);
        ctx.lineTo(px, marginTop + plotH + 4);
        ctx.stroke();
        ctx.fillText(km.toFixed(0), px - 8, marginTop + plotH + 14);
      }
      ctx.fillText("Range (km)", marginLeft + plotW / 2 - 30, height - 4);

      // --- Y-axis: Time with sweep labels ---
      const nReturnsOrig = Number(meta.n_returns_orig) || nReturns;
      const returnStride = nReturns > 0 ? Math.max(1, Math.ceil(nReturnsOrig / nReturns)) : 1;

      // Tick at each sweep boundary
      for (let i = 0; i < sweepBoundaries.length; i++) {
        const bndIdx = sweepBoundaries[i];
        const py = marginTop + (bndIdx / nReturns) * plotH;
        ctx.beginPath();
        ctx.moveTo(marginLeft - 4, py);
        ctx.lineTo(marginLeft, py);
        ctx.stroke();
        const swpNum = bndIdx < sweepNumber.length ? sweepNumber[bndIdx] : "?";
        ctx.fillText(`S${swpNum}`, 4, py + 4);
      }
      // First sweep label
      if (sweepNumber.length > 0) {
        ctx.fillText(`S${sweepNumber[0]}`, 4, marginTop + 10);
      }

      // Time ticks on y-axis
      if (returnTimeMs.length >= nReturns && nReturns > 0) {
        const t0 = returnTimeMs[0];
        const t1 = returnTimeMs[nReturns - 1];
        const tRange = t1 - t0;
        if (tRange > 0) {
          // Pick nice tick interval in seconds
          const tRangeSec = tRange / 1000;
          const niceIntervals = [1, 2, 5, 10, 15, 30, 60, 120, 300];
          let stepSec = 60;
          for (const s of niceIntervals) {
            if (tRangeSec / s <= 8) { stepSec = s; break; }
          }
          const stepMs = stepSec * 1000;
          const startTick = Math.ceil(t0 / stepMs) * stepMs;
          for (let t = startTick; t <= t1; t += stepMs) {
            const frac = (t - t0) / tRange;
            const py = marginTop + frac * plotH;
            if (py < marginTop || py > marginTop + plotH) continue;
            ctx.beginPath();
            ctx.moveTo(marginLeft - 4, py);
            ctx.lineTo(marginLeft, py);
            ctx.stroke();
            ctx.fillText(formatTimeUTC(t), 14, py + 4);
          }
        }
        // Always label first and last time at edges
        ctx.fillText(formatTimeUTC(t0), 14, marginTop + 10);
        if (t1 !== t0) {
          ctx.fillText(formatTimeUTC(t1), 14, marginTop + plotH - 2);
        }
      }

      // Border around plot area
      ctx.strokeStyle = "rgba(110, 118, 129, 0.7)";
      ctx.strokeRect(marginLeft, marginTop, plotW, plotH);

      ctx.restore();

      // --- Overlay text ---
      const moment = String(meta.moment || "");
      const label = `Waterfall | moment=${moment} | ${nReturns}\u00d7${nRange} (from ${nReturnsOrig}\u00d7${nRangeOrig})`;
      ctx.save();
      const tw = ctx.measureText(label).width;
      ctx.fillStyle = "rgba(17, 24, 39, 0.75)";
      ctx.fillRect(marginLeft, 4, tw + 12, 20);
      ctx.fillStyle = "#f9fafb";
      ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, Monaco, monospace";
      ctx.fillText(label, marginLeft + 6, 18);
      ctx.restore();
    }

    function hideTooltip() {
      tooltip.style.display = "none";
      if (lastHoverKey !== "") {
        lastHoverKey = "";
        model.set("hover", {});
        model.save_changes();
      }
    }

    function onMove(event) {
      const meta = model.get("meta") || {};
      const nReturns = Number(meta.n_returns) || 0;
      const nRange = Number(meta.n_range) || 0;
      if (nReturns === 0 || nRange === 0 || grid.length === 0) {
        hideTooltip();
        return;
      }

      const rect = canvas.getBoundingClientRect();
      const px = event.clientX - rect.left;
      const py = event.clientY - rect.top;
      const width = canvas.width;
      const height = canvas.height;
      const plotW = width - marginLeft - marginRight - elevStripW;
      const plotH = height - marginTop - marginBottom;

      if (px < marginLeft || px > marginLeft + plotW || py < marginTop || py > marginTop + plotH) {
        hideTooltip();
        return;
      }

      const col = Math.floor(((px - marginLeft) / plotW) * nRange);
      const row = Math.floor(((py - marginTop) / plotH) * nReturns);
      if (row < 0 || row >= nReturns || col < 0 || col >= nRange) {
        hideTooltip();
        return;
      }

      const val = grid[row * nRange + col];
      const az = row < azimuth.length ? azimuth[row] : NaN;
      const el = row < elevation.length ? elevation[row] : NaN;
      const swp = row < sweepNumber.length ? sweepNumber[row] : -1;

      const rangeStartM = Number(meta.range_start_m) || 0;
      const rangeStepM = Number(meta.range_step_m) || 1;
      const nRangeOrig = Number(meta.n_range_orig) || nRange;
      const rangeStride = Math.max(1, Math.ceil(nRangeOrig / nRange));
      const rangeKm = (rangeStartM + col * rangeStepM * rangeStride) / 1000;

      const nReturnsOrig = Number(meta.n_returns_orig) || nReturns;
      const returnStride = Math.max(1, Math.ceil(nReturnsOrig / nReturns));
      const origRetIdx = row * returnStride;

      const tMs = row < returnTimeMs.length ? returnTimeMs[row] : NaN;
      const timeStr = Number.isFinite(tMs) ? formatTimeUTC(tMs) : "?";
      const valStr = Number.isFinite(val) ? val.toFixed(2) : "NaN";
      tooltip.style.left = `${Math.max(8, Math.round(px + 10))}px`;
      tooltip.style.top = `${Math.max(8, Math.round(py + 10))}px`;
      tooltip.style.display = "block";
      tooltip.textContent =
        `value=${valStr} time=${timeStr}Z ` +
        `az=${az.toFixed(2)}° el=${el.toFixed(2)}° sweep=${swp} range=${rangeKm.toFixed(2)}km`;

      const hoverKey = `${row},${col}`;
      if (hoverKey !== lastHoverKey) {
        lastHoverKey = hoverKey;
        model.set("hover", {
          row,
          col,
          value: Number(val),
          return_index: origRetIdx,
          gate_index: col * rangeStride,
          azimuth_deg: Number(az),
          elevation_deg: Number(el),
          sweep_number: Number(swp),
          range_km: rangeKm,
          time_utc: timeStr,
        });
        model.save_changes();
      }
    }

    function onLeave() {
      hideTooltip();
    }

    const watched = [
      "width", "height", "meta", "grid_bytes",
      "azimuth_bytes", "elevation_bytes", "return_time_ms_bytes",
      "sweep_number_bytes", "sweep_boundary_bytes",
    ];
    for (const key of watched) {
      model.on(`change:${key}`, redraw);
    }

    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("mouseleave", onLeave);

    redraw();

    return () => {
      for (const key of watched) {
        model.off(`change:${key}`, redraw);
      }
      canvas.removeEventListener("mousemove", onMove);
      canvas.removeEventListener("mouseleave", onLeave);
    };
  },
};
"""

_WIDGET_CSS: Final[str] = """
.radrs-viz-root {
  position: relative;
  display: inline-block;
  border: 1px solid #d0d7de;
  border-radius: 8px;
  overflow: hidden;
  background: #ffffff;
}

.radrs-viz-canvas {
  display: block;
  background: #f6f8fa;
}

.radrs-viz-deck-host {
  width: 100%;
  height: 100%;
}

.radrs-viz-overlay {
  position: absolute;
  top: 8px;
  left: 8px;
  pointer-events: none;
  background: rgba(17, 24, 39, 0.75);
  color: #f9fafb;
  font: 11px ui-monospace, SFMono-Regular, Menlo, Monaco, monospace;
  padding: 4px 6px;
  border-radius: 4px;
  z-index: 3;
}

.radrs-viz-tooltip {
  position: absolute;
  pointer-events: none;
  background: rgba(17, 24, 39, 0.95);
  color: #f9fafb;
  font: 11px ui-monospace, SFMono-Regular, Menlo, Monaco, monospace;
  padding: 6px 8px;
  border-radius: 4px;
  max-width: 640px;
  white-space: nowrap;
  z-index: 2;
}

.radrs-viz-inset {
  position: absolute;
  pointer-events: none;
  background: rgba(17, 24, 39, 0.80);
  border-radius: 4px;
  padding: 4px;
}
"""


def _payload_to_html(
    state: dict[str, object],
    esm_str: str,
    css_str: str,
    width: int,
    height: int,
    extra_state: dict[str, object] | None = None,
) -> str:
    """Render a payload's widget state as a self-contained HTML document."""

    # Build JS state entries: base64-encode bytes, JSON-encode everything else.
    js_entries: list[str] = []
    for key, val in state.items():
        if isinstance(val, (bytes, bytearray, memoryview)):
            raw = bytes(val) if not isinstance(val, bytes) else val
            b64 = base64.b64encode(raw).decode("ascii")
            js_entries.append(f"{json.dumps(key)}: _b64ToAB({json.dumps(b64)})")
        else:
            js_entries.append(f"{json.dumps(key)}: {json.dumps(val)}")

    # Merge width, height, and any extra_state scalars.
    js_entries.append(f'"width": {json.dumps(width)}')
    js_entries.append(f'"height": {json.dumps(height)}')
    if extra_state:
        for key, val in extra_state.items():
            js_entries.append(f"{json.dumps(key)}: {json.dumps(val)}")

    js_state_body = ", ".join(js_entries)

    return (
        "<!DOCTYPE html>\n"
        '<html><head><meta charset="utf-8">\n'
        f"<style>{css_str}\n"
        f"#widget-root {{ width: {width}px; height: {height}px; }}\n"
        "</style></head>\n"
        '<body><div id="widget-root"></div>\n'
        '<script type="module">\n'
        "function _b64ToAB(b64) {\n"
        "  const bin = atob(b64);\n"
        "  const u8 = new Uint8Array(bin.length);\n"
        "  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);\n"
        "  return u8.buffer;\n"
        "}\n"
        f"const _S = {{{js_state_body}}};\n"
        "const model = {\n"
        "  get(k) { return _S[k]; },\n"
        "  set() {},\n"
        "  on() {},\n"
        "};\n"
        f"const _esm = {json.dumps(esm_str)};\n"
        'const _blob = new Blob([_esm], {type:"text/javascript"});\n'
        "const _url = URL.createObjectURL(_blob);\n"
        "const _mod = await import(_url);\n"
        "URL.revokeObjectURL(_url);\n"
        '_mod.default.render({ model, el: document.getElementById("widget-root") });\n'
        "</script></body></html>"
    )


try:
    import anywidget as _anywidget
    import traitlets as _traitlets
except ImportError:
    _anywidget = None
    _traitlets = None


def _state_bytes(state: dict[str, object], key: str) -> bytes:
    raw = state.get(key)
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, bytearray):
        return bytes(raw)
    if isinstance(raw, memoryview):
        return raw.tobytes()
    raise TypeError(f"widget state key '{key}' must be bytes-like")


if _anywidget is not None and _traitlets is not None:

    class PolarWidget(_anywidget.AnyWidget):
        """Binary anywidget renderer for quick polar inspection."""

        _esm = _WIDGET_ESM
        _css = _WIDGET_CSS

        width = _traitlets.Int(760).tag(sync=True)
        height = _traitlets.Int(760).tag(sync=True)

        azimuth_bytes = _traitlets.Bytes(b"").tag(sync=True)
        range_bytes = _traitlets.Bytes(b"").tag(sync=True)
        value_bytes = _traitlets.Bytes(b"").tag(sync=True)
        elevation_bytes = _traitlets.Bytes(b"").tag(sync=True)
        gate_index_bytes = _traitlets.Bytes(b"").tag(sync=True)
        return_index_bytes = _traitlets.Bytes(b"").tag(sync=True)
        return_time_ms_bytes = _traitlets.Bytes(b"").tag(sync=True)

        meta = _traitlets.Dict(default_value={}).tag(sync=True)
        hover = _traitlets.Dict(default_value={}).tag(sync=True)

        def __init__(self, width: int = 760, height: int = 760):
            super().__init__()
            self.width = int(width)
            self.height = int(height)

        def set_payload(self, payload: PolarPayload) -> None:
            state = payload.to_widget_state()
            self.azimuth_bytes = _state_bytes(state, "azimuth_bytes")
            self.range_bytes = _state_bytes(state, "range_bytes")
            self.value_bytes = _state_bytes(state, "value_bytes")
            self.elevation_bytes = _state_bytes(state, "elevation_bytes")
            self.gate_index_bytes = _state_bytes(state, "gate_index_bytes")
            self.return_index_bytes = _state_bytes(state, "return_index_bytes")
            self.return_time_ms_bytes = _state_bytes(state, "return_time_ms_bytes")

            meta = state.get("meta")
            if not isinstance(meta, dict):
                raise TypeError("widget state meta must be a dict")
            self.meta = meta

        def clear(self) -> None:
            self.azimuth_bytes = b""
            self.range_bytes = b""
            self.value_bytes = b""
            self.elevation_bytes = b""
            self.gate_index_bytes = b""
            self.return_index_bytes = b""
            self.return_time_ms_bytes = b""
            self.meta = {"point_count": 0, "moment": "", "sweep_number": -1}
            self.hover = {}

        @classmethod
        def from_datatree(
            cls,
            dt: xr.DataTree,
            moment: str,
            *,
            sweep_index: int = 0,
            max_points: int | None = None,
            width: int = 760,
            height: int = 760,
        ) -> "PolarWidget":
            returns, sweeps = get_returns_and_sweeps(dt)
            payload = prepare_polar_payload(
                returns, sweeps, sweep_index, moment, max_points=max_points
            )
            w = cls(width=width, height=height)
            w.set_payload(payload)
            return w

    class VolumeWidget(_anywidget.AnyWidget):
        """Binary anywidget renderer for volume-wide ray points."""

        _esm = _VOLUME_WIDGET_ESM
        _css = _WIDGET_CSS

        width = _traitlets.Int(760).tag(sync=True)
        height = _traitlets.Int(760).tag(sync=True)
        yaw_deg = _traitlets.Float(35.0).tag(sync=True)
        pitch_deg = _traitlets.Float(30.0).tag(sync=True)

        x_bytes = _traitlets.Bytes(b"").tag(sync=True)
        y_bytes = _traitlets.Bytes(b"").tag(sync=True)
        z_bytes = _traitlets.Bytes(b"").tag(sync=True)
        value_bytes = _traitlets.Bytes(b"").tag(sync=True)
        gate_index_bytes = _traitlets.Bytes(b"").tag(sync=True)
        return_index_bytes = _traitlets.Bytes(b"").tag(sync=True)

        meta = _traitlets.Dict(default_value={}).tag(sync=True)
        hover = _traitlets.Dict(default_value={}).tag(sync=True)

        def __init__(
            self,
            width: int = 760,
            height: int = 760,
            yaw_deg: float = 35.0,
            pitch_deg: float = 30.0,
        ):
            super().__init__()
            self.width = int(width)
            self.height = int(height)
            self.yaw_deg = float(yaw_deg)
            self.pitch_deg = float(pitch_deg)

        def set_payload(self, payload: VolumePayload) -> None:
            state = payload.to_widget_state()
            self.x_bytes = _state_bytes(state, "x_bytes")
            self.y_bytes = _state_bytes(state, "y_bytes")
            self.z_bytes = _state_bytes(state, "z_bytes")
            self.value_bytes = _state_bytes(state, "value_bytes")
            self.gate_index_bytes = _state_bytes(state, "gate_index_bytes")
            self.return_index_bytes = _state_bytes(state, "return_index_bytes")

            meta = state.get("meta")
            if not isinstance(meta, dict):
                raise TypeError("widget state meta must be a dict")
            self.meta = meta

        def clear(self) -> None:
            self.x_bytes = b""
            self.y_bytes = b""
            self.z_bytes = b""
            self.value_bytes = b""
            self.gate_index_bytes = b""
            self.return_index_bytes = b""
            self.meta = {"point_count": 0, "moment": "", "render_mode": "points"}
            self.hover = {}

        @classmethod
        def from_datatree(
            cls,
            dt: xr.DataTree,
            moment: str,
            *,
            max_points: int = 125_000,
            render_mode: str = "points",
            width: int = 760,
            height: int = 760,
        ) -> "VolumeWidget":
            returns, _sweeps = get_returns_and_sweeps(dt)
            if render_mode == "rays":
                payload = prepare_ray_payload(returns, moment, max_points=max_points)
            else:
                payload = prepare_volume_payload(returns, moment, max_points=max_points)
            w = cls(width=width, height=height)
            w.set_payload(payload)
            return w

    class GridWidget(_anywidget.AnyWidget):
        """Binary anywidget renderer for 2D gridded CAPPI / cross-section views."""

        _esm = _GRID_WIDGET_ESM
        _css = _WIDGET_CSS

        width = _traitlets.Int(760).tag(sync=True)
        height = _traitlets.Int(760).tag(sync=True)
        grid_bytes = _traitlets.Bytes(b"").tag(sync=True)
        meta = _traitlets.Dict(default_value={}).tag(sync=True)
        hover = _traitlets.Dict(default_value={}).tag(sync=True)

        def __init__(self, width: int = 760, height: int = 760):
            super().__init__()
            self.width = int(width)
            self.height = int(height)

        def set_payload(self, payload: GridPayload) -> None:
            state = payload.to_widget_state()
            self.grid_bytes = _state_bytes(state, "grid_bytes")

            meta = state.get("meta")
            if not isinstance(meta, dict):
                raise TypeError("widget state meta must be a dict")
            self.meta = meta

        def clear(self) -> None:
            self.grid_bytes = b""
            self.meta = {"grid_mode": "cappi", "n_rows": 0, "n_cols": 0, "moment": ""}
            self.hover = {}

        @classmethod
        def from_cappi(
            cls,
            dt: xr.DataTree,
            moment: str,
            *,
            altitude_m: float = 2000.0,
            tolerance_m: float = 500.0,
            grid_size: int = 500,
            width: int = 760,
            height: int = 760,
        ) -> "GridWidget":
            returns, sweeps = get_returns_and_sweeps(dt)
            payload = prepare_cappi_payload(
                returns, sweeps, moment,
                altitude_m=altitude_m, tolerance_m=tolerance_m,
                grid_size=grid_size,
            )
            w = cls(width=width, height=height)
            w.set_payload(payload)
            return w

        @classmethod
        def from_xsec(
            cls,
            dt: xr.DataTree,
            moment: str,
            *,
            azimuth_deg: float = 0.0,
            azimuth_tolerance_deg: float = 2.0,
            grid_size: int = 500,
            width: int = 760,
            height: int = 760,
        ) -> "GridWidget":
            returns, sweeps = get_returns_and_sweeps(dt)
            payload = prepare_xsec_payload(
                returns, sweeps, moment,
                azimuth_deg=azimuth_deg,
                azimuth_tolerance_deg=azimuth_tolerance_deg,
                grid_size=grid_size,
            )
            w = cls(width=width, height=height)
            w.set_payload(payload)
            return w

    class WaterfallWidget(_anywidget.AnyWidget):
        """Binary anywidget renderer for waterfall (return_time x range) heatmaps."""

        _esm = _WATERFALL_WIDGET_ESM
        _css = _WIDGET_CSS

        width = _traitlets.Int(760).tag(sync=True)
        height = _traitlets.Int(760).tag(sync=True)
        grid_bytes = _traitlets.Bytes(b"").tag(sync=True)
        azimuth_bytes = _traitlets.Bytes(b"").tag(sync=True)
        elevation_bytes = _traitlets.Bytes(b"").tag(sync=True)
        return_time_ms_bytes = _traitlets.Bytes(b"").tag(sync=True)
        sweep_number_bytes = _traitlets.Bytes(b"").tag(sync=True)
        sweep_boundary_bytes = _traitlets.Bytes(b"").tag(sync=True)
        meta = _traitlets.Dict(default_value={}).tag(sync=True)
        hover = _traitlets.Dict(default_value={}).tag(sync=True)

        def __init__(self, width: int = 760, height: int = 760):
            super().__init__()
            self.width = int(width)
            self.height = int(height)

        def set_payload(self, payload: WaterfallPayload) -> None:
            state = payload.to_widget_state()
            self.grid_bytes = _state_bytes(state, "grid_bytes")
            self.azimuth_bytes = _state_bytes(state, "azimuth_bytes")
            self.elevation_bytes = _state_bytes(state, "elevation_bytes")
            self.return_time_ms_bytes = _state_bytes(state, "return_time_ms_bytes")
            self.sweep_number_bytes = _state_bytes(state, "sweep_number_bytes")
            self.sweep_boundary_bytes = _state_bytes(state, "sweep_boundary_bytes")

            meta = state.get("meta")
            if not isinstance(meta, dict):
                raise TypeError("widget state meta must be a dict")
            self.meta = meta

        def clear(self) -> None:
            self.grid_bytes = b""
            self.azimuth_bytes = b""
            self.elevation_bytes = b""
            self.return_time_ms_bytes = b""
            self.sweep_number_bytes = b""
            self.sweep_boundary_bytes = b""
            self.meta = {"n_returns": 0, "n_range": 0, "moment": ""}
            self.hover = {}

        @classmethod
        def from_datatree(
            cls,
            dt: xr.DataTree,
            moment: str,
            *,
            max_returns: int = 2048,
            max_range: int = 1024,
            width: int = 760,
            height: int = 760,
        ) -> "WaterfallWidget":
            returns, _sweeps = get_returns_and_sweeps(dt)
            payload = prepare_waterfall_payload(
                returns, moment, max_returns=max_returns, max_range=max_range
            )
            w = cls(width=width, height=height)
            w.set_payload(payload)
            return w

else:

    class PolarWidget:  # pragma: no cover - runtime guard for optional deps
        def __init__(self, *args: object, **kwargs: object):
            raise ImportError(
                "PolarWidget requires optional dependencies: anywidget and traitlets"
            )

        @classmethod
        def from_datatree(cls, *args: object, **kwargs: object) -> "PolarWidget":
            raise ImportError(
                "PolarWidget requires optional dependencies: anywidget and traitlets"
            )

    class VolumeWidget:  # pragma: no cover - runtime guard for optional deps
        def __init__(self, *args: object, **kwargs: object):
            raise ImportError(
                "VolumeWidget requires optional dependencies: anywidget and traitlets"
            )

        @classmethod
        def from_datatree(cls, *args: object, **kwargs: object) -> "VolumeWidget":
            raise ImportError(
                "VolumeWidget requires optional dependencies: anywidget and traitlets"
            )

    class GridWidget:  # pragma: no cover - runtime guard for optional deps
        def __init__(self, *args: object, **kwargs: object):
            raise ImportError(
                "GridWidget requires optional dependencies: anywidget and traitlets"
            )

        @classmethod
        def from_cappi(cls, *args: object, **kwargs: object) -> "GridWidget":
            raise ImportError(
                "GridWidget requires optional dependencies: anywidget and traitlets"
            )

        @classmethod
        def from_xsec(cls, *args: object, **kwargs: object) -> "GridWidget":
            raise ImportError(
                "GridWidget requires optional dependencies: anywidget and traitlets"
            )

    class WaterfallWidget:  # pragma: no cover - runtime guard for optional deps
        def __init__(self, *args: object, **kwargs: object):
            raise ImportError(
                "WaterfallWidget requires optional dependencies: anywidget and traitlets"
            )

        @classmethod
        def from_datatree(cls, *args: object, **kwargs: object) -> "WaterfallWidget":
            raise ImportError(
                "WaterfallWidget requires optional dependencies: anywidget and traitlets"
            )


Payload = PolarPayload | VolumePayload | GridPayload | WaterfallPayload

__all__ = [
    "MOMENT_NAMES",
    "SweepInfo",
    "PolarPayload",
    "VolumePayload",
    "GridPayload",
    "WaterfallPayload",
    "Payload",
    "available_moments",
    "get_returns_and_sweeps",
    "sweep_offsets",
    "sweep_infos",
    "prepare_polar_payload",
    "prepare_volume_payload",
    "prepare_ray_payload",
    "prepare_cappi_payload",
    "prepare_xsec_payload",
    "prepare_waterfall_payload",
    "PolarWidget",
    "VolumeWidget",
    "GridWidget",
    "WaterfallWidget",
]
