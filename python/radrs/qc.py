"""Quality control tools for radar data.

The module exposes two layers:

- **Low-level array ops** (fast, implemented in Rust):
  ``rhohv_threshold``, ``sun_spike``, ``vradh_winding_number``.
- **High-level QC steps** for raystack parsing:
  ``RhohvThreshold``, ``SunSpike``, ``VradhWindingNumber``.

Mask values are:

- ``1`` — valid data
- ``0`` — invalid data (e.g., below threshold or sun spike)
- ``-1`` — missing data (NaN in input)

Examples
--------
Apply an RHOHV threshold during raystack parsing:

```python
import radrs.raystack as rrs
import radrs.qc as qc

rs = rrs.parse(file_bytes, qc=[qc.RhohvThreshold(threshold=0.8)])
rs["returns"]["qc.rhohv_threshold_mask"].shape
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

# Import from the Rust extension
try:
    from radrs._radrs import qc as _qc
except ImportError:
    _qc = None

if _qc is None:
    def rhohv_threshold(rhohv, threshold=None):
        """Apply RHOHV threshold mask."""
        raise NotImplementedError("radrs.qc module not available")

    def sun_spike(dbzh, dbzh_threshold=None, fill_threshold=None, corr_threshold=None):
        """Detect sun spikes in reflectivity data."""
        raise NotImplementedError("radrs.qc module not available")

    def vradh_winding_number(
        vradh,
        dbzh=None,
        nyquist=None,
        wind_size=None,
        velocity_texture_threshold=None,
        reflectivity_threshold=None,
        interval_splits=None,
        skip_between_rays=None,
        skip_along_ray=None,
        centered=None,
        rays_wrap_around=None,
        fill_value=None,
        fill_tolerance=None,
    ):
        """Compute VRADH winding number from region-based dealiasing."""
        raise NotImplementedError("radrs.qc module not available")
else:
    rhohv_threshold = _qc.rhohv_threshold
    sun_spike = _qc.sun_spike
    vradh_winding_number = _qc.vradh_winding_number


class QCStep:
    """Base class for built-in QC steps."""

    def _spec(self) -> dict:
        raise NotImplementedError


@dataclass(frozen=True)
class RhohvThreshold(QCStep):
    """RHOHV threshold mask."""

    threshold: float = 0.8
    vname: str = "rhohv_threshold_mask"

    def _spec(self) -> dict:
        return {
            "name": "rhohv_threshold",
            "threshold": float(self.threshold),
            "vname": self.vname,
        }


@dataclass(frozen=True)
class SunSpike(QCStep):
    """Sun spike mask from DBZH."""

    dbzh_threshold: float = 0.0
    fill_threshold: float = 0.9
    corr_threshold: float = 0.8
    vname: str = "sun_spike_mask"

    def _spec(self) -> dict:
        return {
            "name": "sun_spike",
            "dbzh_threshold": float(self.dbzh_threshold),
            "fill_threshold": float(self.fill_threshold),
            "corr_threshold": float(self.corr_threshold),
            "vname": self.vname,
        }


@dataclass(frozen=True)
class VradhWindingNumber(QCStep):
    """VRADH winding number from region-based dealiasing."""

    nyquist: float | None = None
    wind_size: int = 3
    velocity_texture_threshold: float = 4.0
    reflectivity_threshold: float = 0.0
    interval_splits: int = 3
    skip_between_rays: int = 100
    skip_along_ray: int = 100
    centered: bool = True
    rays_wrap_around: bool = True
    fill_value: float | None = -64.5
    fill_tolerance: float = 1.0
    vname: str = "vradh_winding_number"

    def _spec(self) -> dict:
        return {
            "name": "vradh_winding_number",
            "nyquist": self.nyquist,
            "wind_size": int(self.wind_size),
            "velocity_texture_threshold": float(self.velocity_texture_threshold),
            "reflectivity_threshold": float(self.reflectivity_threshold),
            "interval_splits": int(self.interval_splits),
            "skip_between_rays": int(self.skip_between_rays),
            "skip_along_ray": int(self.skip_along_ray),
            "centered": bool(self.centered),
            "rays_wrap_around": bool(self.rays_wrap_around),
            "fill_value": self.fill_value,
            "fill_tolerance": float(self.fill_tolerance),
            "vname": self.vname,
        }


def compile_qc_steps(qc: Optional[Iterable[QCStep] | QCStep]) -> Optional[List[dict]]:
    """Compile QC steps into a compact spec for the Rust parser."""
    if qc is None:
        return None

    if isinstance(qc, QCStep):
        steps = [qc]
    else:
        steps = list(qc)

    specs = []
    for step in steps:
        if not isinstance(step, QCStep):
            raise TypeError(
                f"QC steps must be QCStep instances, got {type(step)!r}"
            )
        specs.append(step._spec())
    return specs

__all__ = [
    "rhohv_threshold",
    "sun_spike",
    "vradh_winding_number",
    "QCStep",
    "RhohvThreshold",
    "SunSpike",
    "VradhWindingNumber",
    "compile_qc_steps",
]
