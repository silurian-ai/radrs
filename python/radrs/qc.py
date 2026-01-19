"""
radrs.qc - Quality control functions for radar data

This module provides QC functions that operate on numpy arrays
and return mask arrays indicating valid/invalid data.

Mask values:
- 1 = valid data
- 0 = invalid data (e.g., below threshold)
- -1 = missing data (NaN in input)

Example
-------
>>> import radrs.qc as qc
>>> import numpy as np
>>>
>>> # RHOHV threshold mask
>>> rhohv = np.array(...)  # (time, range)
>>> mask = qc.rhohv_threshold(rhohv, threshold=0.8)
>>> # Returns: int8 array, 1=valid, 0=invalid, -1=missing
>>>
>>> # Sun spike detection
>>> dbzh = np.array(...)
>>> mask = qc.sun_spike(dbzh, fill_threshold=0.9, corr_threshold=0.8)
"""

# Import from the Rust extension
try:
    from radrs.radrs.qc import (
        rhohv_threshold,
        sun_spike,
    )
except ImportError:
    # Fallback import path
    try:
        from radrs.radrs import qc as _qc
        rhohv_threshold = _qc.rhohv_threshold
        sun_spike = _qc.sun_spike
    except (ImportError, AttributeError):
        def rhohv_threshold(rhohv, threshold=None):
            """Apply RHOHV threshold mask."""
            raise NotImplementedError("radrs.qc module not available")

        def sun_spike(dbzh, fill_threshold=None, corr_threshold=None):
            """Detect sun spikes in reflectivity data."""
            raise NotImplementedError("radrs.qc module not available")

__all__ = [
    "rhohv_threshold",
    "sun_spike",
]
