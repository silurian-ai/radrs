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
    from radrs.radrs.xradar import open_datatree
except ImportError:
    # Fallback import path
    try:
        from radrs.radrs import xradar as _xradar
        open_datatree = _xradar.open_datatree
    except (ImportError, AttributeError):
        def open_datatree(source):
            """Open a NEXRAD Level 2 file as xarray DataTree."""
            raise NotImplementedError("radrs.xradar module not available")

__all__ = ["open_datatree"]
