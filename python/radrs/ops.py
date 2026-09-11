"""
radrs.ops - Low-level array ops

This module provides fast alignment and texture utilities implemented in Rust.

Notes
-----
- Arrays are expected to be float32 and C-contiguous for zero-copy performance.
- Alignment plans map destination indices to source indices and can be reused
  across multiple fields.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from radrs._radrs import ops as _ops

AzimuthPlan = _ops.AzimuthPlan
RangePlan = _ops.RangePlan

def align_azimuth(
    src_azimuth,
    dst_azimuth,
    tolerance_deg: Optional[float] = None,
    wrap: bool = True,
) -> AzimuthPlan:
    """Create an azimuth alignment plan.

    Parameters
    ----------
    src_azimuth : array-like
        Source azimuths (degrees).
    dst_azimuth : array-like
        Destination azimuths (degrees).
    tolerance_deg : float, optional
        Maximum angular difference allowed for a match.
    wrap : bool, default True
        If True, treat azimuths modulo 360.
    """
    src = np.ascontiguousarray(src_azimuth, dtype=np.float32)
    dst = np.ascontiguousarray(dst_azimuth, dtype=np.float32)
    return _ops.align_azimuth(src, dst, tolerance_deg=tolerance_deg, wrap=wrap)

def align_range(
    src_start_m: float,
    src_step_m: float,
    src_len: int,
    dst_start_m: float,
    dst_step_m: float,
    dst_len: int,
    tolerance_m: Optional[float] = None,
) -> RangePlan:
    """Create a range alignment plan.

    Parameters
    ----------
    src_start_m : float
        Source gate start (meters).
    src_step_m : float
        Source gate spacing (meters).
    src_len : int
        Number of source gates.
    dst_start_m : float
        Destination gate start (meters).
    dst_step_m : float
        Destination gate spacing (meters).
    dst_len : int
        Number of destination gates.
    tolerance_m : float, optional
        Maximum range difference allowed for a match.
    """
    return _ops.align_range(
        float(src_start_m),
        float(src_step_m),
        int(src_len),
        float(dst_start_m),
        float(dst_step_m),
        int(dst_len),
        tolerance_m=tolerance_m,
    )

def velocity_texture(vradh, nyquist: Optional[float] = None, wind_size: Optional[int] = None):
    """Compute velocity texture for a single sweep (2D array)."""
    data = np.ascontiguousarray(vradh, dtype=np.float32)
    return _ops.velocity_texture(data, nyquist=nyquist, wind_size=wind_size)


__all__ = [
    "AzimuthPlan",
    "RangePlan",
    "align_azimuth",
    "align_range",
    "velocity_texture",
]
