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
>>> source = radrs.VolumeSource.nexrad("KTLX", start="2024-03-15")
>>> for dt in radrs.iter_volumes(source):
...     process(dt)
"""


import os

from radrs import _radrs
from radrs import archive, ops, qc, quicklook, raystack, xradar

list_volumes = _radrs.list_volumes
peek_volume = _radrs.peek_volume
VolumeSource = _radrs.VolumeSource
VolumeInfo = _radrs.VolumeInfo
VolumeMeta = _radrs.VolumeMeta
iter_volumes = _radrs.iter_volumes
iter_volumes_async = _radrs.iter_volumes_async
iter_meta_urls = _radrs.iter_meta_urls
iter_meta_urls_async = _radrs.iter_meta_urls_async
iter_meta_candidates = _radrs.iter_meta_candidates
iter_meta_candidates_async = _radrs.iter_meta_candidates_async
stream_realtime = _radrs.stream_realtime
stream_archive = _radrs.stream_archive

# L2 Archive iterators
NexradL2ArchiveIter = _radrs.NexradL2ArchiveIter
NexradL2ArchiveInfo = _radrs.NexradL2ArchiveInfo
list_nexrad_l2_archive_volumes = _radrs.list_nexrad_l2_archive_volumes_py

# Logging (opt-in, zero overhead by default)
set_log_filter = _radrs.set_log_filter

# Auto-initialize logging if RADRS_LOG env var is set
if "RADRS_LOG" in os.environ:
    _radrs.initialize_logs()


__all__ = [
    "list_volumes",
    "peek_volume",
    "VolumeSource",
    "VolumeInfo",
    "VolumeMeta",
    "iter_volumes",
    "iter_volumes_async",
    "iter_meta_urls",
    "iter_meta_urls_async",
    "iter_meta_candidates",
    "iter_meta_candidates_async",
    "stream_realtime",
    "stream_archive",
    "NexradL2ArchiveIter",
    "NexradL2ArchiveInfo",
    "list_nexrad_l2_archive_volumes",
    "set_log_filter",
    "xradar",
    "raystack",
    "qc",
    "ops",
    "archive",
    "quicklook",
]
