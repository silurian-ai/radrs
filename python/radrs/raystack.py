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
>>> # rs is a dict with vcps/sweeps/returns/activity (and qc when requested)
>>> print(rs.keys())  # ['vcps', 'sweeps', 'returns', 'activity']
>>> print(rs['returns']['DBZH'].shape)  # (n_returns, 128)
>>>
>>> # Convert DataTree to raystack format
>>> import radrs.xradar as rxr
>>> dt = rxr.open_datatree(file_path)
>>> rs = rrs.from_xradar_datatree(dt, fold_size=128)
>>>
>>> # Apply built-in QC during parse
>>> import radrs.qc as qc
>>> rs = rrs.parse(file_bytes, qc=[qc.RhohvThreshold(threshold=0.8)])
>>> rs["qc"]["rhohv_threshold_mask"].shape
>>>
>>> # Write raystack to Zarr via xarray
>>> rrs.to_raystack_datatree(rs).to_zarr("output.zarr")

Note: activity metrics (if enabled) are computed over the rays present in the
raystack window. If you time-slice or otherwise create partial raystacks, sweep
activity reflects only observed rays (normalized by observed rays × fold_size).
"""

from radrs.qc import compile_qc_steps

# Import from the Rust extension
try:
    from radrs._radrs import raystack as _raystack
except ImportError:
    _raystack = None

if _raystack is None:
    def parse(data, fold_size=None, qc=None, include_activity=True):
        """Parse NEXRAD data to raystack format."""
        raise NotImplementedError("radrs.raystack module not available")

    def from_xradar_datatree(datatree, fold_size=None, include_activity=True):
        """Convert xradar DataTree to raystack format."""
        raise NotImplementedError("radrs.raystack module not available")

    def to_xradar_datatree(raystack):
        """Convert raystack back to xradar DataTree."""
        raise NotImplementedError("radrs.raystack module not available")

    def to_raystack_datatree(raystack):
        """Convert raystack dict to raystack DataTree."""
        raise NotImplementedError("radrs.raystack module not available")

    def open_datatree(source, fold_size=None, qc=None, include_activity=True):
        """Open NEXRAD data and return raystack DataTree."""
        raise NotImplementedError("radrs.raystack module not available")

    async def open_datatree_async(source, fold_size=None, qc=None, include_activity=True):
        """Open NEXRAD data and return raystack DataTree (async)."""
        raise NotImplementedError("radrs.raystack module not available")

else:
    def parse(data, fold_size=None, qc=None, include_activity=True):
        qc_spec = compile_qc_steps(qc)
        return _raystack.parse(
            data, fold_size=fold_size, qc=qc_spec, include_activity=include_activity
        )

    def from_xradar_datatree(datatree, fold_size=None, include_activity=True):
        return _raystack.from_xradar_datatree(
            datatree, fold_size=fold_size, include_activity=include_activity
        )
    to_xradar_datatree = _raystack.to_xradar_datatree
    to_raystack_datatree = _raystack.to_raystack_datatree

    def open_datatree(source, fold_size=None, qc=None, include_activity=True):
        qc_spec = compile_qc_steps(qc)
        return _raystack.open_datatree(
            source,
            fold_size=fold_size,
            qc=qc_spec,
            include_activity=include_activity,
        )

    async def open_datatree_async(source, fold_size=None, qc=None, include_activity=True):
        qc_spec = compile_qc_steps(qc)
        return await _raystack.open_datatree_async(
            source,
            fold_size=fold_size,
            qc=qc_spec,
            include_activity=include_activity,
        )

__all__ = [
    "parse",
    "from_xradar_datatree",
    "to_xradar_datatree",
    "to_raystack_datatree",
    "open_datatree",
    "open_datatree_async",
]
