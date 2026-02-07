"""Quicklook helpers for interactive raystack visualization.

This module provides:

- sweep and moment selectors over raystack DataTree nodes
- a vectorized adapter from sweep data to polar point buffers
- an optional anywidget polar renderer for marimo notebooks
"""

from __future__ import annotations

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
                "max_abs_m": float(self.max_abs_m),
                "vmin": float(self.vmin),
                "vmax": float(self.vmax),
            },
        }


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
    flat_values = moment_matrix.reshape(-1)
    finite = np.isfinite(flat_values)

    if n_returns == 0 or n_range == 0 or not np.any(finite):
        return VolumePayload(
            x_m=np.empty(0, dtype=np.float32),
            y_m=np.empty(0, dtype=np.float32),
            z_m=np.empty(0, dtype=np.float32),
            values=np.empty(0, dtype=np.float32),
            gate_index=np.empty(0, dtype=np.uint16),
            return_index=np.empty(0, dtype=np.uint32),
            moment=moment,
            max_abs_m=0.0,
            vmin=0.0,
            vmax=1.0,
        )

    azimuth = np.asarray(returns["azimuth"].values, dtype=np.float32)
    elevation = np.asarray(returns["elevation"].values, dtype=np.float32)
    base_range = np.asarray(returns["base_range"].values, dtype=np.float32)
    range_step = np.asarray(returns["range_step"].values, dtype=np.float32)

    gate_index_full = np.tile(np.arange(n_range, dtype=np.uint16), n_returns)
    return_index_full = np.repeat(np.arange(n_returns, dtype=np.uint32), n_range)
    azimuth_full = np.repeat(azimuth, n_range)
    elevation_full = np.repeat(elevation, n_range)
    base_range_full = np.repeat(base_range, n_range)
    range_step_full = np.repeat(range_step, n_range)

    gate_index_sel = gate_index_full[finite]
    return_index_sel = return_index_full[finite]
    azimuth_sel = azimuth_full[finite]
    elevation_sel = elevation_full[finite]
    values_sel = flat_values[finite]
    range_sel = base_range_full[finite] + gate_index_sel.astype(np.float32) * range_step_full[finite]

    azimuth_rad = np.deg2rad(azimuth_sel.astype(np.float64))
    elevation_rad = np.deg2rad(elevation_sel.astype(np.float64))
    horizontal = range_sel.astype(np.float64) * np.cos(elevation_rad)

    # x=east, y=north, z=up
    x_sel = horizontal * np.sin(azimuth_rad)
    y_sel = horizontal * np.cos(azimuth_rad)
    z_sel = range_sel.astype(np.float64) * np.sin(elevation_rad)

    n_points = int(values_sel.size)
    if max_points is not None and max_points > 0 and n_points > max_points:
        keep = _sample_indices(n_points, max_points)
        x_sel = x_sel[keep]
        y_sel = y_sel[keep]
        z_sel = z_sel[keep]
        range_sel = range_sel[keep]
        azimuth_sel = azimuth_sel[keep]
        elevation_sel = elevation_sel[keep]
        values_sel = values_sel[keep]
        gate_index_sel = gate_index_sel[keep]
        return_index_sel = return_index_sel[keep]

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
        max_abs_m=max_abs_m,
        vmin=vmin,
        vmax=vmax,
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
    root.className = "radrs-quicklook-root";
    const canvas = document.createElement("canvas");
    canvas.className = "radrs-quicklook-canvas";
    const tooltip = document.createElement("div");
    tooltip.className = "radrs-quicklook-tooltip";
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
import { COORDINATE_SYSTEM, Deck, OrbitView } from "https://esm.sh/@deck.gl/core@9.2.2?bundle";
import { PointCloudLayer } from "https://esm.sh/@deck.gl/layers@9.2.2?bundle";

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

export default {
  render({ model, el }) {
    const root = document.createElement("div");
    root.className = "radrs-quicklook-root";
    const deckHost = document.createElement("div");
    deckHost.className = "radrs-quicklook-deck-host";
    const overlay = document.createElement("div");
    overlay.className = "radrs-quicklook-overlay";
    const tooltip = document.createElement("div");
    tooltip.className = "radrs-quicklook-tooltip";
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
          rotationOrbit: Number(model.get("yaw_deg")) || 35,
          rotationX: Number(model.get("pitch_deg")) || 30,
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
      overlay.textContent = (
        `Volume 3D (deck.gl) | points=${n.toLocaleString()} ` +
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
        rotationOrbit: Number(model.get("yaw_deg")) || viewState.rotationOrbit,
        rotationX: Number(model.get("pitch_deg")) || viewState.rotationX,
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

    return () => {
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

_WIDGET_CSS: Final[str] = """
.radrs-quicklook-root {
  position: relative;
  display: inline-block;
  border: 1px solid #d0d7de;
  border-radius: 8px;
  overflow: hidden;
  background: #ffffff;
}

.radrs-quicklook-canvas {
  display: block;
  background: #f6f8fa;
}

.radrs-quicklook-deck-host {
  width: 100%;
  height: 100%;
}

.radrs-quicklook-overlay {
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

.radrs-quicklook-tooltip {
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
"""


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

    class QuicklookPolarWidget(_anywidget.AnyWidget):
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

    class QuicklookVolumeWidget(_anywidget.AnyWidget):
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
            self.meta = {"point_count": 0, "moment": ""}
            self.hover = {}

else:

    class QuicklookPolarWidget:  # pragma: no cover - runtime guard for optional deps
        def __init__(self, *args: object, **kwargs: object):
            raise ImportError(
                "QuicklookPolarWidget requires optional dependencies: anywidget and traitlets"
            )

    class QuicklookVolumeWidget:  # pragma: no cover - runtime guard for optional deps
        def __init__(self, *args: object, **kwargs: object):
            raise ImportError(
                "QuicklookVolumeWidget requires optional dependencies: anywidget and traitlets"
            )


__all__ = [
    "MOMENT_NAMES",
    "SweepInfo",
    "PolarPayload",
    "VolumePayload",
    "available_moments",
    "get_returns_and_sweeps",
    "sweep_offsets",
    "sweep_infos",
    "prepare_polar_payload",
    "prepare_volume_payload",
    "QuicklookPolarWidget",
    "QuicklookVolumeWidget",
]
