"""
radrs.raystack - Raystack format tools for ML training

This module provides tools for converting NEXRAD data to the raystack format,
which is optimized for machine learning training pipelines.

Example
-------
>>> import radrs.raystack as rrs
>>>
>>> # Parse directly to raystack format (faster - single pass)
>>> rs = rrs.parse(file_bytes, fold_size=128)
>>>
>>> # rs is a dict with vcps/sweeps/returns
>>> print(rs.keys())  # ['vcps', 'sweeps', 'returns']
>>> print(rs['returns']['DBZH'].shape)  # (n_returns, 128)
>>>
>>> # Convert DataTree to raystack format
>>> import radrs.xradar as rxr
>>> dt = rxr.open_datatree(file_path)
>>> rs = rrs.from_datatree(dt, fold_size=128)
>>>
>>> # Write raystack to Zarr
>>> rrs.to_zarr(rs, "output.zarr")
"""

# Import from the Rust extension
try:
    from radrs.radrs.raystack import (
        parse,
        from_datatree,
        to_datatree,
    )
except ImportError:
    # Fallback import path
    try:
        from radrs.radrs import raystack as _raystack
        parse = _raystack.parse
        from_datatree = _raystack.from_datatree
        to_datatree = _raystack.to_datatree
    except (ImportError, AttributeError):
        def parse(data, fold_size=None, qc=None):
            """Parse NEXRAD data to raystack format."""
            raise NotImplementedError("radrs.raystack module not available")

        def from_datatree(datatree, fold_size=None):
            """Convert DataTree to raystack format."""
            raise NotImplementedError("radrs.raystack module not available")

        def to_datatree(raystack):
            """Convert raystack back to DataTree."""
            raise NotImplementedError("radrs.raystack module not available")

# Optional zarr functions
try:
    from radrs.radrs.raystack import to_zarr, open_zarr
except ImportError:
    try:
        from radrs.radrs import raystack as _raystack
        to_zarr = getattr(_raystack, 'to_zarr', None)
        open_zarr = getattr(_raystack, 'open_zarr', None)
    except (ImportError, AttributeError):
        to_zarr = None
        open_zarr = None

    if to_zarr is None:
        def to_zarr(raystack, path, mode=None):
            """Write raystack to Zarr (requires zarr feature)."""
            raise NotImplementedError("Zarr feature not enabled")

    if open_zarr is None:
        def open_zarr(path, mode=None):
            """Open Zarr store (requires zarr feature)."""
            raise NotImplementedError("Zarr feature not enabled")

__all__ = [
    "parse",
    "from_datatree",
    "to_datatree",
    "to_zarr",
    "open_zarr",
]
