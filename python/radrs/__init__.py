"""
radrs - Rust-Native NEXRAD Processing Pipeline

A modular Rust-native NEXRAD processing pipeline with:
- radrs.xradar - xradar-compatible interface for easy comparison
- radrs.raystack - raystack format tools for ML training
- radrs.qc - Quality control functions
- High-level iterators built on top of these primitives

Example
-------
>>> import radrs
>>> import radrs.xradar as rxr
>>> import radrs.raystack as rrs
>>>
>>> # Load a NEXRAD file as xarray DataTree
>>> dt = rxr.open_datatree("path/to/file.ar2v")
>>>
>>> # Or parse directly to raystack format for ML
>>> rs = rrs.parse(file_bytes, fold_size=128)
>>>
>>> # Iterate over S3 archive
>>> for dt in radrs.iter_volumes("KTLX", start="2024-03-15"):
...     process(dt)
"""

from radrs.radrs import (
    list_volumes,
    iter_volumes,
    stream_realtime,
    stream_archive,
)

# Import submodules
from radrs import xradar
from radrs import raystack
from radrs import qc

# Optional cache function
try:
    from radrs.radrs import open_cache
except ImportError:
    def open_cache(path: str):
        """Open a cache store (requires cache feature)."""
        raise NotImplementedError("Cache feature not enabled. Rebuild with --features cache")

__all__ = [
    "list_volumes",
    "iter_volumes",
    "stream_realtime",
    "stream_archive",
    "open_cache",
    "xradar",
    "raystack",
    "qc",
]
