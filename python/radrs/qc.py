"""
radrs.qc - Quality control tools for radar data

This module provides two layers of QC:
1) Low-level array ops (fast, implemented in Rust):
   - rhohv_threshold(...)
   - sun_spike(...)
2) High-level QC steps for raystack parsing:
   - RhohvThreshold(...)
   - SunSpike(...)

Mask values:
- 1 = valid data
- 0 = invalid data (e.g., below threshold or sun spike)
- -1 = missing data (NaN in input)

Example
-------
>>> import radrs.raystack as rrs
>>> import radrs.qc as qc
>>> rs = rrs.parse(file_bytes, qc=[qc.RhohvThreshold(threshold=0.8)])
>>> rs["returns"]["rhohv_threshold_mask"].shape
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
else:
    rhohv_threshold = _qc.rhohv_threshold
    sun_spike = _qc.sun_spike


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
    "QCStep",
    "RhohvThreshold",
    "SunSpike",
    "compile_qc_steps",
]
