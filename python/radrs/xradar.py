"""xradar-compatible interface.

Drop-in replacement for ``xradar.io.open_nexradlevel2_datatree`` —
returns an ``xarray.DataTree`` with the same structure as xradar.

Examples
--------
Open a NEXRAD volume from S3, a local path, or in-memory bytes:

```python
import radrs.xradar as rxr

dt = rxr.open_datatree("s3://unidata-nexrad-level2/2024/03/15/KTLX/KTLX20240315_000000_V06")
dt = rxr.open_datatree("/path/to/local/file.ar2v")
dt = rxr.open_datatree(file_bytes)
```
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
