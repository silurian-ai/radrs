# Batch iteration and folding

`open_datatree` handles one volume at a time. To build a training set you
usually want a whole time range in one array: `BatchedRaystack` pre-allocates
flat `vcps` / `sweeps` / `returns` buffers and fills them incrementally from an
archive iterator, so a hundred volumes land in a single contiguous returns
matrix instead of a hundred separate DataTrees.

Two things drive everything else on this page: **folding**, which sets how many
returns a volume produces, and **capacity**, which is reserved up front and
therefore has to be estimated before the first fetch.

## Folding

`fold_size` chunks each radial along the range axis. A radial with `n_gates`
becomes `ceil(n_gates / fold_size)` returns of exactly `fold_size` gates each:

```python
import radrs.raystack as rrs

rdt = rrs.open_datatree(src, fold_size=256)
rdt["returns"]["DBZH"].shape  # (n_returns, 256)
```

The `returns` node is always a 2-D `(return_time, range)` matrix whose `range`
dimension equals `fold_size`. Folding trades rows against columns — the gate
count is fixed by the radar, so a smaller `fold_size` means more, narrower
returns:

| `fold_size` | returns per volume | compacted | `range` dim |
|---|---:|---:|---:|
| 128 | 129,960 | 41,026 | 128 |
| 256 | 67,320 | 25,138 | 256 |
| 512 | 36,720 | 17,215 | 512 |
| 1832 | 11,520 | 11,520 | 1832 |

Measured on one KTLX VCP-12 volume (2024-03-15 00:02Z): 20 sweeps, 11,520
radials, 15.7M physical gate cells. The "compacted" column is
`drop_empty_returns=True`, [described below](#dropping-empty-returns). At
`fold_size=1832` every radial fits in a single return, which is why nothing is
dropped and why the row count equals the radial count.

!!! warning "`range` is a gate index, not a distance"

    The `range` coordinate runs `0 … fold_size - 1` — it is the gate's offset
    *within its fold*, not a physical range. Two returns at the same `range`
    index sit at completely different distances if they came from different
    folds or different sweeps.

    Physical range lives on the per-return `base_range` and `range_step`
    variables (both metres), so reconstruct it explicitly:

    ```python
    returns = rdt["returns"].dataset
    gate_index = returns["range"].values                       # (fold_size,)
    range_m = (
        returns["base_range"].values[:, None]
        + gate_index[None, :] * returns["range_step"].values[:, None]
    )                                                          # (n_returns, fold_size)
    ```

    `base_range` already accounts for the fold offset, so this is correct for
    trailing folds too.

Because `fold_size` rarely divides the gate count evenly, the last fold of each
radial is padded with NaN past the final real gate. Those tail gates are
indistinguishable from below-threshold gates, which are also NaN — both are
"no data", so most pipelines do not need to tell them apart.

## Pre-allocating a batch

`BatchedRaystack` takes three hard capacities, reserved when the object is
constructed. Undershooting silently drops volumes, so the safe direction is to
overshoot — see [the memory note below](#what-a-reservation-costs) for why that
is cheaper than it looks:

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

`max_vcps` is one slot per volume and `max_sweeps` is cheap metadata — 32 per
volume clears every operational VCP. `max_returns` is the one worth computing,
because it also sizes the moment arrays (`max_returns * fold_size` floats per
moment, seven moments).

The check the batch actually applies, per volume, is:

```
reserved = Σ over sweeps of  n_radials * ceil(max_gates / fold_size)
```

For the VCP-12 volume above that is exactly the uncompacted column in the fold
table — 67,320 returns at `fold_size=256`. Scale that inversely with
`fold_size`, then add 20–50% headroom for wider VCPs and denser scenes.

!!! note "Reservation ignores compaction"

    The capacity check runs against `reserved` — the count *before*
    `drop_empty_returns` removes anything. Compaction happens per return as the
    volume is written, so it lowers the row count in the output but never buys
    you room for another volume. Size `max_returns` against the uncompacted
    figure regardless of whether compaction is on.

### What a reservation costs

`max_returns` sizes the moment arrays, so the ceiling is:

```
max_returns * fold_size * 7 moments * 4 bytes
```

That number gets large fast — 20 volumes at `fold_size=256` with 40% headroom
is 1.88M returns, or a 13.5 GB ceiling. In practice the reservation is lazy:
constructing that batch moves resident memory by about a megabyte, and pages
are only touched as returns are written, so what you actually pay tracks the
rows that survive compaction (0.18 GB for the first volume above).

Overshooting `max_returns` therefore costs address space rather than resident
memory, which is why generous headroom is the right default — undershooting
loses data, overshooting mostly does not. Size it so the *filled* result fits
in RAM, not so the ceiling does.

## Filling from an archive

`add_volumes_from_l2` takes a `NexradL2ArchiveIter` and fetches volumes
concurrently, adding each one as it arrives:

```python
from datetime import datetime, timezone
import radrs

n_added = batch.add_volumes_from_l2(
    radrs.NexradL2ArchiveIter(
        "s3://unidata-nexrad-level2",
        start_time=datetime(2024, 7, 2, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2024, 7, 2, 2, 0, tzinfo=timezone.utc),
        storage_options={"anon": "true"},
        site_filter=["KABR"],
    ),
    prefetch=8,
)
```

`prefetch` is how many volumes are in flight at once; 8 saturates a typical
link without much memory cost, since only the fetched bytes are held.

!!! warning "Check the return value"

    `add_volumes_from_l2` returns the number of volumes it actually added, and
    it does not raise when the batch fills up. A volume whose reservation does
    not fit is rejected whole — adds are atomic, so a partial volume never
    lands — and iteration simply continues, so an undersized batch quietly
    yields a truncated time range.

    ```python
    if n_added < len(expected_volumes):
        raise RuntimeError(f"only {n_added} volumes fit — raise max_returns")
    ```

    Fetch and parse failures are skipped the same way. Set `RADRS_LOG=warn` to
    see the reason for each skip.

Archive bounds are half-open, `[start_time, end_time)`, so pad past the last
volume's start time when you are driving the range off a listing from
`radrs.list_nexrad_l2_archive_volumes`.

## Watching capacity

`progress()` reports fill against capacity:

```python
prog = batch.progress()
# patterns_filled / patterns_capacity
# sweeps_filled   / sweeps_capacity
# returns_filled  / returns_capacity
# fill_fraction
```

Read it *before* finalizing — `finalize()` trims the arrays, after which the
capacity numbers no longer describe what was reserved.

`returns_filled` counts the rows that survived compaction, while
`returns_capacity` is the uncompacted reservation. With
`drop_empty_returns=True` the two are not comparable: the volume above reports
`fill_fraction` of 0.37 at the exact point where the next volume no longer
fits. Treat `fill_fraction` as a lower bound on real pressure, and use
`n_added` as the authoritative signal that everything fit.

## Dropping empty returns

`drop_empty_returns=True` discards any return whose every gate is NaN across
every moment. Fine folding produces a lot of these, because a fold far past the
storm top or beyond the last real gate carries no data at all. The rate climbs
steeply with range — same volume, `fold_size=256`:

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
count from 67,320 to 25,138. That ratio is a property of the *scene*, not of
the format — a widespread precipitation event fills the near folds and drops
far fewer rows. Do not use it to size `max_returns`.

Compaction removes whole rows, so what remains is emptiness *within* surviving
returns: split-cut VCPs record reflectivity and Doppler moments on separate
sweeps, so a return from a surveillance cut has no `VRADH` at any gate and
still counts as non-empty on the strength of its `DBZH`.

Sweep-to-return slicing survives compaction either way: `sweeps["num_returns"]`
is written after returns are dropped, so it always sums to the actual row count
and `viz.sweep_offsets(sweeps)` stays valid. Leave compaction off only when you
need a row per radial per fold — a rigid layout you can index positionally, or
reconstruct a full sweep geometry from.

## Finalizing

```python
batch.add_qc_outputs([qc.RhohvThreshold(threshold=0.8, vname="rhohv_mask")])

rs_dt = batch.finalize_to_rs_dt()   # xarray.DataTree
rs_dict = batch.finalize_to_dict()  # zero-copy numpy arrays
```

Call `add_qc_outputs` before finalizing; QC runs over everything accumulated so
far and lands in `returns` under a `qc.` prefix. The batch is single-use — once
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
the reliable discriminator — padding is `NaT`:

```python
import numpy as np

returns = rs_dt["returns"].dataset
real = returns.isel(return_time=~np.isnat(returns["return_time"].values))
```

In the `finalize_to_dict` output the same rows hold the raw `int64` sentinel
`np.iinfo(np.int64).min` instead, since that layout keeps times unconverted.

## Worked example

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

# ~16M gate cells per volume, plus 40% headroom for wider VCPs.
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

prog = batch.progress()  # read before finalize trims
rs_dt = batch.finalize_to_rs_dt()
returns, sweeps = viz.get_returns_and_sweeps(rs_dt)

print(f"{n_added}/{len(selected)} volumes, {returns.sizes['return_time']:,} returns")
print(f"reserved {prog['returns_filled']:,}/{prog['returns_capacity']:,}")
```

!!! warning "`vcp_time` is naive local time"

    `info.vcp_time` carries no `tzinfo` and is expressed in the *local*
    timezone, even though NEXRAD names volumes in UTC. Under `TZ=America/Chicago`
    the volume `KABR20240702_000016_V06` reports `2024-07-01 19:00:16`.

    Naive bounds passed to `NexradL2ArchiveIter` are read the same way, so
    feeding `vcp_time` straight back in as `start_time`/`end_time` — as above —
    round-trips correctly on any host. But to display it, or compare it against
    the UTC bounds you started from, convert rather than relabel:

    ```python
    info.vcp_time.astimezone(timezone.utc)     # correct — shifts local to UTC
    info.vcp_time.replace(tzinfo=timezone.utc) # wrong — off by the UTC offset
    ```

## What's next

- [S3 archive iteration](s3-archive.md) — the `NexradL2ArchiveIter` surface
  that feeds a batch.
- [Raystack format](raystack-format.md) — what `vcps`, `sweeps`, `returns`,
  and `activity` contain.
- [Quality control](qc.md) — the QC steps `add_qc_outputs` accepts.
- `notebooks/batched_raystack_viz.py` — an interactive marimo viewer built on
  this API, with capacity and compaction wired to live controls.
