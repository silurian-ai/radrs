# S3 archive iteration

`NexradL2ArchiveIter` scans the date-partitioned NEXRAD Level 2 archive and
returns volumes in a UTC time interval. It supports S3, GCS, Azure, and local
filesystem URIs.

Use timezone-aware UTC bounds in new code:

```python
from datetime import datetime, timezone
import radrs

archive = radrs.NexradL2ArchiveIter(
    base_uri="s3://unidata-nexrad-level2",
    start_time=datetime(2024, 3, 15, 10, tzinfo=timezone.utc),
    end_time=datetime(2024, 3, 15, 14, tzinfo=timezone.utc),
    storage_options={"anon": "true"},
    site_filter=["KTLX"],
)
for info in archive:
    print(info.vcp_time, info.uri)
```

The time contract is explicit and stable across host timezones:

- Aware bounds are interpreted by instant and normalized to UTC.
- Naive bounds are interpreted as UTC for compatibility with existing code.
- The interval is half-open, `[start_time, end_time)`: `start_time` is
  included and `end_time` is excluded.
- `NexradL2ArchiveInfo.vcp_time` is an aware Python datetime with
  `tzinfo=timezone.utc`.

See the [`radrs` API reference](../reference/radrs.md) for the current signature.
