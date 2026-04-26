# Quickstart

Fast NEXRAD Level 2 processing for Python. This page goes from a fresh
environment to a working `open_datatree` call against a public S3 file.

## Install

=== "uv"

    ```bash
    uv add radrs
    ```

=== "pip"

    ```bash
    pip install radrs
    ```

Python 3.11+ is required. See [Install](../install.md) for source builds and
platform notes.

## Open a volume as an xradar DataTree

`radrs.xradar.open_datatree` is a drop-in replacement for
`xradar.io.open_nexradlevel2_datatree`. It returns an `xarray.DataTree` with
the same sweep-per-node structure.

```python
import radrs.xradar as rxr

src = "s3://unidata-nexrad-level2/2024/03/15/KTLX/KTLX20240315_120000_V06"
dt = rxr.open_datatree(src)
```

The `src` argument also accepts a local filesystem path or a `bytes` buffer
of raw NEXRAD data.

## Open the same volume as a raystack DataTree

The raystack format is a flat `vcps` / `sweeps` / `returns` / `activity`
layout designed for ML training pipelines that want fixed-shape tensors
rather than nested xradar groups.

```python
import radrs.raystack as rrs

rdt = rrs.open_datatree(src, fold_size=128)
rdt["returns"]["DBZH"].shape  # (n_returns, 128)
```

`fold_size` chunks each radial along the range axis: a radial with
`n_gates > fold_size` becomes multiple returns of `fold_size` gates each. See
[Raystack format](raystack-format.md) for the array layout.

## Anonymous S3 access

The single-volume `open_datatree` path fetches from S3 without signing, so
public buckets like `unidata-nexrad-level2` and `noaa-nexrad-level2` work
out of the box even without AWS credentials.

For multi-volume iteration over an archive, pass
`storage_options={"anon": "true"}` to `NexradL2ArchiveIter`:

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
```

See [S3 archive iteration](s3-archive.md) for the full surface.

## What's next

- [S3 archive iteration](s3-archive.md) — `NexradL2ArchiveIter` for
  time-bounded multi-cloud slices.
- [Raystack format](raystack-format.md) — what `vcps`, `sweeps`, `returns`,
  and `activity` actually contain.
- [Quality control](qc.md) — `RhohvThreshold`, `SunSpike`, and
  `VradhWindingNumber` applied during raystack parsing.
- [xradar interop](xradar-interop.md) — sweep ordering, NaN semantics for
  below-threshold gates, and the `sort_by_azimuth` flag.
