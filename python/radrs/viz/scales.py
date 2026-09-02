"""Per-moment colour scales and the fallback percentile stretch."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
class ColorScale:
    """Fixed colour mapping for one moment.

    ``name`` selects a palette implemented in the widget JS (see ``PALETTES``
    in ``static/shared.js``); ``vmin``/``vmax`` are absolute data bounds, so the
    same value is the same colour in every view.
    """

    name: str
    vmin: float
    vmax: float
    units: str = ""


#: Palette used when a moment has no entry in :data:`COLOR_SCALES`.
FALLBACK_COLORMAP: Final[str] = "viridis"

#: Default absolute scale per moment. Moments absent here (QC arrays, anything
#: unrecognized) fall back to a 5th–95th percentile stretch over ``viridis``.
COLOR_SCALES: Final[dict[str, ColorScale]] = {
    # NWS reflectivity ramp: grey noise floor below ~5 dBZ, then the familiar
    # cyan/green/yellow/red/magenta steps up to the 75 dBZ top of the scale.
    "DBZH": ColorScale("nws_reflectivity", -30.0, 75.0, "dBZ"),
    # Symmetric about zero so inbound/outbound read as opposite hues. Replaced
    # by ±Nyquist when the dataset carries one.
    "VRADH": ColorScale("nws_velocity", -32.0, 32.0, "m/s"),
    "WRADH": ColorScale("magma", 0.0, 15.0, "m/s"),
    "ZDR": ColorScale("nws_zdr", -2.0, 6.0, "dB"),
    # The parser emits PHIDP on 0..360 (NEXRAD encodes the full turn), so the
    # scale is cyclic and wraps back to its starting colour at the top.
    "PHIDP": ColorScale("cyclic", 0.0, 360.0, "deg"),
    # Tops out above 1.0: values there are unphysical and worth seeing.
    "RHOHV": ColorScale("nws_rhohv", 0.7, 1.05, ""),
    "CCORH": ColorScale("viridis", 0.0, 1.0, ""),
}

#: Variable names searched, in order, for a Nyquist velocity.
_NYQUIST_NAMES: Final[tuple[str, ...]] = (
    "nyquist_velocity",
    "nyquist_vel",
    "nyquist",
)


def _nyquist_velocity(
    returns: xr.Dataset | None, sweeps: xr.Dataset | None
) -> float | None:
    """Find a positive Nyquist velocity in either dataset, if one is recorded."""

    for dataset in (returns, sweeps):
        if dataset is None:
            continue
        for name in _NYQUIST_NAMES:
            if name not in dataset.variables:
                continue
            values = np.asarray(dataset[name].values, dtype=np.float64).ravel()
            values = values[np.isfinite(values) & (values > 0.0)]
            if values.size:
                return float(np.median(values))
    return None


def color_scale_for(
    moment: str,
    *,
    returns: xr.Dataset | None = None,
    sweeps: xr.Dataset | None = None,
) -> ColorScale | None:
    """Look up the fixed scale for ``moment``, or ``None`` for a stretch.

    ``None`` means the moment has no meteorological default and callers should
    fall back to the percentile stretch.
    """

    scale = COLOR_SCALES.get(moment)
    if scale is None:
        return None
    if moment == "VRADH":
        nyquist = _nyquist_velocity(returns, sweeps)
        if nyquist is not None:
            return replace(scale, vmin=-nyquist, vmax=nyquist)
    return scale


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


def _resolve_color_scale(
    moment: str,
    values: np.ndarray,
    override: ColorScale | None = None,
    returns: xr.Dataset | None = None,
    sweeps: xr.Dataset | None = None,
) -> tuple[float, float, str, str]:
    """Resolve ``(vmin, vmax, colormap, units)`` for a moment's values.

    With nothing finite to draw the bounds stay at the neutral 0..1 placeholder
    the payloads have always used; the palette and units still name the moment.
    """

    scale = (
        override
        if override is not None
        else color_scale_for(moment, returns=returns, sweeps=sweeps)
    )
    colormap = scale.name if scale is not None else FALLBACK_COLORMAP
    units = scale.units if scale is not None else ""

    array = np.asarray(values)
    if array.size == 0 or not np.any(np.isfinite(array)):
        return 0.0, 1.0, colormap, units
    if scale is not None:
        return float(scale.vmin), float(scale.vmax), colormap, units
    vmin, vmax = _value_bounds(array)
    return vmin, vmax, colormap, units
