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
>>> rs = rrs.from_xradar_datatree(dt, fold_size=128)
>>>
>>> # Write raystack to Zarr via xarray
>>> rrs.to_raystack_datatree(rs).to_zarr("output.zarr")
"""

# Import from the Rust extension
try:
    from radrs._radrs import raystack as _raystack
except ImportError:
    _raystack = None

if _raystack is None:
    def parse(data, fold_size=None, qc=None):
        """Parse NEXRAD data to raystack format."""
        raise NotImplementedError("radrs.raystack module not available")

    def from_xradar_datatree(datatree, fold_size=None):
        """Convert xradar DataTree to raystack format."""
        raise NotImplementedError("radrs.raystack module not available")

    def to_xradar_datatree(raystack):
        """Convert raystack back to xradar DataTree."""
        raise NotImplementedError("radrs.raystack module not available")

    def to_raystack_datatree(raystack):
        """Convert raystack dict to raystack DataTree."""
        raise NotImplementedError("radrs.raystack module not available")

    def open_datatree(source, fold_size=None, qc=None):
        """Open NEXRAD data and return raystack DataTree."""
        raise NotImplementedError("radrs.raystack module not available")

    async def open_datatree_async(source, fold_size=None, qc=None):
        """Open NEXRAD data and return raystack DataTree (async)."""
        raise NotImplementedError("radrs.raystack module not available")

else:
    parse = _raystack.parse
    from_xradar_datatree = _raystack.from_xradar_datatree
    to_xradar_datatree = _raystack.to_xradar_datatree
    to_raystack_datatree = _raystack.to_raystack_datatree
    open_datatree = _raystack.open_datatree
    open_datatree_async = _raystack.open_datatree_async

__all__ = [
    "parse",
    "from_xradar_datatree",
    "to_xradar_datatree",
    "to_raystack_datatree",
    "open_datatree",
    "open_datatree_async",
]
