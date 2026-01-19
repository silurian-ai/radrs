"""
radrs.xradar - xradar-compatible interface

This module provides an interface that matches xradar's API for easy comparison
and drop-in replacement.

Example
-------
>>> import radrs.xradar as rxr
>>>
>>> # Exactly like xradar.io.open_nexradlevel2_datatree
>>> dt = rxr.open_datatree("s3://unidata-nexrad-level2/2024/03/15/KTLX/KTLX20240315_000000_V06")
>>> dt = rxr.open_datatree("/path/to/local/file.ar2v")
>>> dt = rxr.open_datatree(file_bytes)  # From bytes
>>>
>>> # Returns xarray.DataTree with same structure as xradar
>>> print(dt)
"""

# Import from the Rust extension
try:
    from radrs._radrs import xradar as _xradar
except ImportError:
    _xradar = None

if _xradar is None:
    def open_datatree(source):
        """Open a NEXRAD Level 2 file as xarray DataTree."""
        raise NotImplementedError("radrs.xradar module not available")

    async def open_datatree_async(source):
        """Open a NEXRAD Level 2 file as xarray DataTree (async)."""
        raise NotImplementedError("radrs.xradar module not available")
else:
    open_datatree = _xradar.open_datatree
    open_datatree_async = _xradar.open_datatree_async

__all__ = ["open_datatree", "open_datatree_async"]
