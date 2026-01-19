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


from radrs import _radrs
from radrs import qc, raystack, xradar

list_volumes = _radrs.list_volumes
iter_volumes = _radrs.iter_volumes
iter_volumes_async = _radrs.iter_volumes_async
stream_realtime = _radrs.stream_realtime
stream_archive = _radrs.stream_archive


__all__ = [
    "list_volumes",
    "iter_volumes",
    "iter_volumes_async",
    "stream_realtime",
    "stream_archive",
    "xradar",
    "raystack",
    "qc",
]
