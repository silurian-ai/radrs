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
>>> from datetime import datetime
>>>
>>> # Load a NEXRAD file as xarray DataTree
>>> dt = rxr.open_datatree("path/to/file.ar2v")
>>>
>>> # Or parse directly to raystack format for ML
>>> rs = rrs.parse(file_bytes, fold_size=128)
>>>
>>> # Iterate over S3 archive
>>> archive = radrs.NexradL2ArchiveIter(
...     base_uri="s3://unidata-nexrad-level2",
...     start_time=datetime(2024, 3, 15),
...     end_time=datetime(2024, 3, 16),
...     storage_options={"anon": "true"},
...     site_filter=["KTLX"],
... )
>>> for info in archive:
...     process(info.uri)
"""


import os

from radrs import _radrs
from radrs import ops, qc, viz, raystack, xradar

peek_volume = _radrs.peek_volume
VolumeMeta = _radrs.VolumeMeta
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
    "peek_volume",
    "VolumeMeta",
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
    "viz",
]
