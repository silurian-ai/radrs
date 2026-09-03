# Batch iteration and folding

`open_datatree` handles one volume at a time. To build a training set you
usually want a whole time range in one array. `BatchedRaystack` pre-allocates
flat `vcps` / `sweeps` / `returns` buffers and fills them incrementally from an
archive iterator, so a hundred volumes land in a single contiguous returns
matrix instead of a hundred separate DataTrees.

Two things need care. Capacity is reserved when the batch is constructed, so
`max_returns` has to be estimated before the first fetch. And a volume that
does not fit is skipped rather than raising, so the count of added volumes has
to be checked afterwards. The sections below walk through the process in order
and end with a complete example.

## Iterating an archive

A batch is fed by `NexradL2ArchiveIter`, which walks the standard NEXRAD
Level 2 archive layout, `YYYY/MM/DD/SITE/SITE<YYYYMMDD>_<HHMMSS>_V06`, and
yields one entry per volume in the requested window:

```python
from datetime import datetime, timezone
import radrs

archive = radrs.NexradL2ArchiveIter(
    "s3://unidata-nexrad-level2",
    start_time=datetime(2024, 7, 2, 0, 0, tzinfo=timezone.utc),
    end_time=datetime(2024, 7, 2, 2, 0, tzinfo=timezone.utc),
    storage_options={"anon": "true"},
    site_filter=["KABR"],
)
for info in archive:
    print(info.instrument_name, info.vcp_time, info.uri)
```

| Parameter | Meaning |
|---|---|
| `base_uri` | Archive root: `s3://`, `gs://`, `az://`, or a local path. Not a path to one volume. |
| `start_time`, `end_time` | Window bounds, half-open `[start, end)`. `end_time < start_time` raises. |
| `storage_options` | Backend config, forwarded to object_store. |
| `site_filter` | List of 4-letter ICAO codes. Omit to take every site in range, which is usually far more than you want. |
| `max_concurrent_ls` | Parallel directory listings, default 10. Raise it for wide date ranges, where listing dominates. |

Bounds are UTC. An aware datetime is interpreted by instant and normalized to
UTC; a naive one is interpreted as UTC rather than as the machine's local
timezone, so the same code selects the same volumes on any host. `vcp_time`
comes back as an aware UTC datetime, which is why it can be fed straight back
in as a bound.

Iteration is lazy. Directories are listed as it goes, and nothing is fetched
until a volume is actually consumed. `radrs.list_nexrad_l2_archive_volumes`
takes the same arguments and returns the whole listing eagerly as a list, which
is what you want when you need a count or a sorted range up front.

Each entry is a `NexradL2ArchiveInfo` with `uri`, `instrument_name`,
`vcp_time`, and `size`. Only listing metadata is populated and no volume bytes
are read, so it is cheap to enumerate a day and then decide what to fetch.
`size` is optional: it is whatever the backend reported during listing, and
not every store returns it, so guard before doing arithmetic on it.

The public buckets `s3://unidata-nexrad-level2` and `s3://noaa-nexrad-level2`
both need `storage_options={"anon": "true"}` for unsigned access. Other
backends take their own keys:

=== "S3"

    ```python
    radrs.NexradL2ArchiveIter(
        "s3://unidata-nexrad-level2",
        start_time=start, end_time=end,
        storage_options={"anon": "true", "region": "us-east-1"},
    )
    ```

=== "GCS"

    ```python
    radrs.NexradL2ArchiveIter(
        "gs://my-bucket/nexrad",
        start_time=start, end_time=end,
        storage_options={"service_account_path": "/path/to/key.json"},
    )
    ```

=== "Azure"

    ```python
    radrs.NexradL2ArchiveIter(
        "az://my-container/nexrad",
        start_time=start, end_time=end,
        storage_options={"account_name": "...", "access_key": "..."},
    )
    ```

=== "Local"

    ```python
    radrs.NexradL2ArchiveIter(
        "/data/nexrad",
        start_time=start, end_time=end,
    )
    ```

!!! warning "`vcp_time` is naive local time"

    `info.vcp_time` carries no `tzinfo` and is expressed in the host's local
    timezone, even though NEXRAD names volumes in UTC. Under
    `TZ=America/Chicago` the volume `KABR20240702_000016_V06` reports
    `2024-07-01 19:00:16`.

    Naive bounds passed to `NexradL2ArchiveIter` are interpreted the same way,
    so feeding `vcp_time` straight back in as `start_time` or `end_time`
    round-trips correctly on any host. To display it, or compare it against
    UTC bounds, convert rather than relabel:

    ```python
    info.vcp_time.astimezone(timezone.utc)     # correct: shifts local to UTC
    info.vcp_time.replace(tzinfo=timezone.utc) # wrong: off by the UTC offset
    ```

## Folding

[Raystack format](raystack-format.md#returns-and-folding) covers what folding
does to a single volume. `fold_size` chunks each radial along the range axis,
so a radial with `n_gates` becomes `ceil(n_gates / fold_size)` returns of
exactly `fold_size` gates each, and `returns` is a 2-D `(return_time, range)`
matrix whose `range` dimension equals `fold_size`.

What matters for a batch is the row count that falls out of that choice. The
gate count is fixed by the radar, so folding only trades rows against columns.
A smaller `fold_size` means more, narrower returns:

| `fold_size` | returns per volume | compacted | `range` dim |
|---|---:|---:|---:|
| 128 | 129,960 | 41,026 | 128 |
| 256 | 67,320 | 25,138 | 256 |
| 512 | 36,720 | 17,215 | 512 |
| 1832 | 11,520 | 11,520 | 1832 |

Measured on one KTLX VCP-12 volume (2024-03-15 00:02Z): 20 sweeps, 11,520
radials, 15.7M physical gate cells. The "compacted" column is
`drop_empty_returns=True`, described in the next section. At
`fold_size=1832` every radial fits in a single return, so the row count equals
the radial count. Nothing is dropped at that size because the first fold of a
radial always holds real echoes close to the radar; it is the far folds that
come back empty.

Two folding details matter more once you batch, and both are covered in the
format guide. The `range` coordinate is a
[gate index rather than a distance](raystack-format.md#returns-and-folding),
so rows from different folds and sweeps are not comparable along that axis.
And the last fold of each radial is NaN-padded past the final real gate, which
is part of why the compacted column above is so much smaller.

## Dropping empty returns

`drop_empty_returns=True` discards any return whose every gate is NaN across
every moment. Fine folding produces a lot of these, because a fold far past the
storm top or beyond the last real gate carries no data at all. The rate climbs
steeply with range. Same volume, `fold_size=256`:

| fold | starts at | returns | all-NaN | rate |
|---:|---:|---:|---:|---:|
| 0 | 2 km | 11,520 | 0 | 0.0% |
| 1 | 66 km | 11,520 | 6,491 | 56.3% |
| 2 | 130 km | 11,160 | 7,863 | 70.5% |
| 3 | 194 km | 10,440 | 7,523 | 72.1% |
| 4 | 258 km | 9,720 | 7,947 | 81.8% |
| 5 | 322 km | 5,040 | 4,601 | 91.3% |
| 6 | 386 km | 4,320 | 4,157 | 96.2% |
| 7 | 450 km | 3,600 | 3,600 | 100.0% |

Overall 62.7% of returns are empty here, which is why compaction cuts the row
count from 67,320 to 25,138. That ratio is a property of the scene, not of the
format. A widespread precipitation event fills the near folds and drops far
fewer rows, so do not use it to size `max_returns`.

Compaction removes whole rows, so what remains is emptiness within surviving
returns. Split-cut VCPs record reflectivity and Doppler moments on separate
sweeps, so a return from a surveillance cut has no `VRADH` at any gate and
still counts as non-empty on the strength of its `DBZH`.

Sweep-to-return slicing survives compaction either way. `sweeps["num_returns"]`
is written after returns are dropped, so it always sums to the actual row count
and `viz.sweep_offsets(sweeps)` stays valid. Leave compaction off only when you
need a row per radial per fold: a rigid layout you can index positionally, or
reconstruct a full sweep geometry from.

## Sizing the batch

`BatchedRaystack` takes three hard capacities, reserved when the object is
constructed. Undershooting silently drops volumes, so the safe direction is to
overshoot, and the [cost section below](#what-a-reservation-costs) explains why
that is cheaper than it looks.

```python
import radrs.raystack as rrs

n_volumes = 20
fold_size = 256

batch = rrs.BatchedRaystack(
    max_vcps=n_volumes,
    max_sweeps=n_volumes * 32,
    max_returns=n_volumes * 90_000,
    fold_size=fold_size,
    truncate=True,
    drop_empty_returns=True,
)
```

`max_vcps` is one slot per volume. `max_sweeps` is cheap metadata, and 32 per
volume comfortably covers the operational VCPs, including their SAILS and MRLE
supplemental cuts. `max_returns` is the one worth computing, because it also
sizes the moment arrays: `max_returns * fold_size` floats per moment, seven
moments.

The check the batch actually applies, per volume, is:

```
reserved = Σ over sweeps of  n_radials * ceil(max_gates / fold_size)
```

For the VCP-12 volume above that is exactly the uncompacted column in the fold
table: 67,320 returns at `fold_size=256`. A quick estimate is gate cells
divided by `fold_size`, which lands about 10% low because of the `ceil()`, so
add 20–50% headroom on top for that and for wider VCPs and denser scenes. The
[worked example below](#worked-example) uses 16M cells and 40%.

!!! note "Reservation ignores compaction"

    The capacity check runs against `reserved`, the count before
    `drop_empty_returns` removes anything. Compaction happens per return as the
    volume is written, so it lowers the row count in the output but never buys
    you room for another volume. Size `max_returns` against the uncompacted
    figure regardless of whether compaction is on.

### What a reservation costs

`max_returns` sizes the moment arrays, so the ceiling is:

```
max_returns * fold_size * 7 moments * 4 bytes
```

That number gets large fast. The 20-volume batch in the
[worked example](#worked-example) reserves 1.75M returns at `fold_size=256`,
a 12.5 GB ceiling. In practice the
reservation is lazy: constructing that batch moves resident memory by about a
megabyte, and pages are only touched as returns are written, so what you
actually pay tracks the rows that survive compaction (0.18 GB for the first
volume above).

Overshooting `max_returns` therefore costs address space rather than resident
memory, which is why generous headroom is the right default. When deciding how
large a batch a machine can hold, estimate the memory of the volumes you expect
to load, not the reservation ceiling.

## Filling from an archive

`add_volumes_from_l2` takes an archive iterator and fetches volumes
concurrently, adding each one as it arrives:

```python
n_added = batch.add_volumes_from_l2(archive, prefetch=8)
```

`prefetch` is how many volumes are in flight at once. 8 saturates a typical
link without much memory cost, since only the fetched bytes are held.

!!! warning "Check the return value"

    `add_volumes_from_l2` returns the number of volumes it actually added, and
    it does not raise when the batch fills up. A volume whose reservation does
    not fit is rejected whole (adds are atomic, so a partial volume never
    lands) and iteration continues, so a later, smaller volume can still be
    admitted: an undersized batch quietly yields a time range with holes in it
    rather than a clean prefix. Iteration stops early only once a dimension is
    exactly full.

    ```python
    if n_added < len(expected_volumes):
        raise RuntimeError(f"only {n_added} volumes fit; raise max_returns")
    ```

    Fetch failures are skipped the same way, as are parse failures on the
    default `include_sweeps=True` path. Set `RADRS_LOG=warn` to see the reason
    for each skip. With `include_sweeps=False`, a volume that cannot be peeked
    raises instead of being skipped.

Archive bounds are half-open, `[start_time, end_time)`, so pad past the last
volume's start time when you are driving the range off a listing from
`radrs.list_nexrad_l2_archive_volumes`. Feeding `vcp_time` back in as a bound
is safe because both sides are UTC-aware, as described
[above](#iterating-an-archive).

## Watching capacity

`progress()` reports fill against capacity:

```python
prog = batch.progress()
# patterns_filled / patterns_capacity   (one "pattern" per volume)
# sweeps_filled   / sweeps_capacity
# returns_filled  / returns_capacity
# fill_fraction
```

Read it before `finalize_to_dict()` or `finalize_to_rs_dt()`, which consume
the batch — `progress()` panics once the buffers have been handed to Python.
Plain `finalize()` leaves these numbers untouched: it pads the arrays out to
capacity when `truncate=False` and is a no-op otherwise.

`returns_filled` counts the rows that survived compaction, while
`returns_capacity` is the uncompacted reservation. With
`drop_empty_returns=True` the two are not comparable: the volume above reports
a `fill_fraction` of 0.37 at the exact point where the next volume no longer
fits. Treat `fill_fraction` as a lower bound on real pressure, and use
`n_added` as the authoritative signal that everything fit.

## Finalizing

```python
batch.add_qc_outputs([qc.RhohvThreshold(threshold=0.8, vname="rhohv_mask")])

rs_dt = batch.finalize_to_rs_dt()   # xarray.DataTree
rs_dict = batch.finalize_to_dict()  # zero-copy numpy arrays
```

Call `add_qc_outputs` before finalizing. QC runs over everything accumulated so
far and lands in `returns` under a `qc.` prefix. The batch is single-use: once
finalized it accepts no further volumes.

`truncate` decides the output shape. The default `truncate=True` trims to what
was filled. `truncate=False` keeps the arrays at full capacity, NaN- and
zero-filled, which is what you want for fixed-shape tensors:

```python
batch = rrs.BatchedRaystack(
    max_vcps=10, max_sweeps=140, max_returns=75_000,
    fold_size=128, truncate=False,
)
batch.finalize_to_dict()["returns"]["azimuth"].shape  # (75000,) always
```

With `truncate=False`, padding rows carry NaN moments and NaN azimuth,
elevation, `base_range` and `range_step`, which makes them indistinguishable
from a real all-NaN return on those variables alone. The time coordinates are
the reliable discriminator, because padding is `NaT`:

```python
import numpy as np

returns = rs_dt["returns"].dataset
real = returns.isel(return_time=~np.isnat(returns["return_time"].values))
```

In the `finalize_to_dict` output the same rows hold the raw `int64` sentinel
`np.iinfo(np.int64).min` instead, since that layout keeps times unconverted.

## Worked example

Everything above, end to end: list two hours of KABR volumes, size a batch for
them, fill it, and finalize to a DataTree.

```python
from datetime import datetime, timedelta, timezone

import radrs
import radrs.raystack as rrs
import radrs.viz as viz

STATION = "KABR"
BASE_URI = "s3://unidata-nexrad-level2"
STORAGE = {"anon": "true"}
FOLD_SIZE = 256

infos = radrs.list_nexrad_l2_archive_volumes(
    base_uri=BASE_URI,
    start_time=datetime(2024, 7, 2, 0, 0, tzinfo=timezone.utc),
    end_time=datetime(2024, 7, 2, 2, 0, tzinfo=timezone.utc),
    storage_options=STORAGE,
    site_filter=[STATION],
)
selected = sorted(infos, key=lambda info: info.vcp_time)[:20]

# ~16M gate cells per volume (15.7M measured on a VCP-12 volume), plus 40%
# headroom for wider VCPs and the ceil() rounding folding adds.
returns_per_volume = int(16_000_000 / FOLD_SIZE * 1.4)

batch = rrs.BatchedRaystack(
    max_vcps=len(selected),
    max_sweeps=len(selected) * 32,
    max_returns=len(selected) * returns_per_volume,
    fold_size=FOLD_SIZE,
    truncate=True,
    drop_empty_returns=True,
)

n_added = batch.add_volumes_from_l2(
    radrs.NexradL2ArchiveIter(
        BASE_URI,
        start_time=selected[0].vcp_time,
        # Bounds are half-open, so pad past the last volume's start.
        end_time=selected[-1].vcp_time + timedelta(seconds=1),
        storage_options=STORAGE,
        site_filter=[STATION],
    ),
    prefetch=8,
)
if n_added < len(selected):
    raise RuntimeError(f"only {n_added} of {len(selected)} volumes fit; raise max_returns")

prog = batch.progress()  # read before finalize_to_rs_dt consumes the batch
rs_dt = batch.finalize_to_rs_dt()
returns, sweeps = viz.get_returns_and_sweeps(rs_dt)

print(f"{n_added}/{len(selected)} volumes, {returns.sizes['return_time']:,} returns")
print(f"reserved {prog['returns_filled']:,}/{prog['returns_capacity']:,}")
```

## What's next

- [Raystack format](raystack-format.md) — what `vcps`, `sweeps`, `returns`,
  and `activity` contain, and the folding walkthrough.
- [Quality control](qc.md) — the QC steps `add_qc_outputs` accepts.
- [Visualization](visualization.md) — `batched_raystack_viz.py`, an interactive
  viewer built on this API with capacity and compaction wired to live controls.
