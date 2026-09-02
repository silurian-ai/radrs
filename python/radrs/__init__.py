"""Fast NEXRAD Level 2 processing for Python.

radrs is a modular NEXRAD processing pipeline:

- ``radrs.xradar`` — xradar-compatible DataTree interface
- ``radrs.raystack`` — flat raystack layout for ML training
- ``radrs.qc`` — quality control functions
- High-level iterators built on top of these primitives

Examples
--------
Load a NEXRAD volume as an xarray DataTree:

```python
import radrs.xradar as rxr

dt = rxr.open_datatree("s3://unidata-nexrad-level2/2024/03/15/KTLX/KTLX20240315_120000_V06")
```

Or in the flat raystack layout for ML pipelines:

```python
import radrs.raystack as rrs

rdt = rrs.open_datatree(src, fold_size=128)
```

Iterate over a time-bounded archive on S3:

```python
from datetime import datetime, timezone
import radrs

archive = radrs.NexradL2ArchiveIter(
    base_uri="s3://unidata-nexrad-level2",
    start_time=datetime(2024, 3, 15, tzinfo=timezone.utc),
    end_time=datetime(2024, 3, 16, tzinfo=timezone.utc),
    storage_options={"anon": "true"},
    site_filter=["KTLX"],
)
for info in archive:
    process(info.uri)
```

Archive bounds use UTC. Aware datetimes are converted by instant; naive
datetimes are interpreted as UTC for compatibility with older examples. The
end bound is exclusive, so volumes are selected from ``[start_time, end_time)``.
"""


import os

from radrs import _radrs, ops, qc, raystack, viz, xradar
from radrs._radrs import (
    NexradL2ArchiveInfo,
    NexradL2ArchiveIter,
    VolumeMeta,
    list_nexrad_l2_archive_volumes_py as list_nexrad_l2_archive_volumes,
    peek_volume,
    set_log_filter,
    stream_archive,
    stream_realtime,
)

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
