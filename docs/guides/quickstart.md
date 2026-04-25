# Quickstart

Fast NEXRAD Level 2 processing for Python. This page takes you from
install to a working `open_datatree` call.

## Install

```bash
uv add radrs
```

See [Install](../install.md) for pip, optional extras, and from-source
builds.

## Open a volume as an xradar DataTree

`radrs.xradar.open_datatree` returns an `xarray.DataTree` with the same
structure as `xradar.io.open_nexradlevel2_datatree`. The source can be a
cloud URL, a local path, or in-memory bytes.

```python
import radrs.xradar as rxr

src = "s3://unidata-nexrad-level2/2024/03/15/KTLX/KTLX20240315_120000_V06"
dt = rxr.open_datatree(src)

# Local path
dt = rxr.open_datatree("/data/nexrad/KTLX20240315_120000_V06")

# Already-loaded bytes
with open("/data/nexrad/KTLX20240315_120000_V06", "rb") as f:
    dt = rxr.open_datatree(f.read())
```

The public Unidata and NOAA NEXRAD buckets are accessed anonymously — no
AWS credentials needed.

## Open the same volume in raystack format

`radrs.raystack.open_datatree` returns a flat layout suited to ML
pipelines and tensor batching. `fold_size` chunks each radial into
fixed-width segments so every moment array is rectangular.

```python
import radrs.raystack as rrs

rdt = rrs.open_datatree(src, fold_size=128)
rdt["returns"]["DBZH"].shape  # (n_returns, 128)
```

## Iterate a time-bounded archive

For multi-volume workloads, `NexradL2ArchiveIter` walks the
`YYYY/MM/DD/SITE/` layout on S3, GCS, Azure, or local filesystems. Pass
`storage_options={"anon": "true"}` for unsigned access to public
buckets.

```python
from datetime import datetime
import radrs

archive = radrs.NexradL2ArchiveIter(
    base_uri="s3://unidata-nexrad-level2",
    start_time=datetime(2024, 3, 15),
    end_time=datetime(2024, 3, 16),
    storage_options={"anon": "true"},
    site_filter=["KTLX"],
)
for info in archive:
    dt = rxr.open_datatree(info.uri)
```

## What's next

- [S3 archive iteration](s3-archive.md) — `NexradL2ArchiveIter` for
  time-bounded slices across S3, GCS, Azure, and local filesystems.
- [Raystack format](raystack-format.md) — what `vcps`, `sweeps`,
  `returns`, and `activity` actually contain.
- [Quality control](qc.md) — `RhohvThreshold`, `SunSpike`, and
  `VradhWindingNumber` applied at parse time.
- [xradar interop](xradar-interop.md) — sweep ordering, NaN semantics
  for below-threshold and range-folded gates, and `sort_by_azimuth`.
