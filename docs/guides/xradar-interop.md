# XRadar interop

`radrs.xradar.open_datatree` is a drop-in replacement for
`xradar.io.open_nexradlevel2_datatree`. It returns an `xarray.DataTree` with
the same `sweep_N` children, the same variable names, and the same dimensions,
so code written against xradar generally works by changing the import.

```python
# before
import xradar as xd
dt = xd.io.open_nexradlevel2_datatree(path)

# after
import radrs.xradar as rxr
dt = rxr.open_datatree(path)
```

Where the two libraries differ, they differ deliberately, in three places:
radial ordering, the representation of gates with no valid measurement, and how
cloud sources are reached. Everything else — sweep count, sweep naming,
elevation angles, range geometry, variable sets, and decoded moment values — is
intended to match, and is checked against xradar in the test suite.

## What matches

Comparing a KTLX VCP-12 volume against xradar 0.12, aligned by azimuth:

| Property | Result |
|---|---|
| Sweeps | 20 vs 20, same `sweep_N` names |
| Radials per sweep | Identical on every sweep (720) |
| Range gates per sweep | Identical on every sweep |
| `sweep_fixed_angle` | Identical to within float representation |
| Variable sets | Identical — no radrs-only or xradar-only variables |
| Moment values | **Exact agreement** on all 104 moment/sweep pairs, wherever both libraries report a valid measurement |

"Exact" is literal for the integer-scaled moments: mean, p95, and p99 absolute
difference are all 0.0 for `DBZH`. `RHOHV` differs by up to 5e-8, which is
float32 round-off in the scale/offset decode, not a difference in
interpretation.

## Radial ordering and `sort_by_azimuth`

xradar sorts the radials within each sweep by ascending azimuth. radrs
preserves the order the radials appear in the file, which is the order the
radar actually collected them — a sweep starts wherever the antenna was.

```python
rxr.open_datatree(path)["sweep_0"]["azimuth"].values[:5]
# array([167.27, 167.75, 168.27, 168.67, 169.21])   file order

rxr.open_datatree(path, sort_by_azimuth=True)["sweep_0"]["azimuth"].values[:5]
# array([0.19, 0.71, 1.19, 1.76, 2.25])             xradar convention
```

Pass `sort_by_azimuth=True` to match xradar. On the volume above the sorted
azimuths agree with xradar's to the bit — `max|Δazimuth| == 0.0`.

Which default you want depends on the work. Sorted is right for plotting and
for anything that indexes by azimuth. File order is right when collection
sequence matters — spotting antenna irregularities, or reasoning about the time
axis, since azimuth-sorting scrambles it.

!!! warning "Sort before comparing anything positionally"

    Without `sort_by_azimuth=True`, row *i* of a radrs sweep and row *i* of an
    xradar sweep are different radials. A cell-by-cell diff of the two will
    show large, meaningless differences.

## Gates with no valid measurement

NEXRAD encodes each gate as a byte, reserving the two lowest codes: one for
*below threshold* (the return was too weak to measure) and one for *range
folded* (the return is ambiguous because the target lies beyond the unambiguous
range).

radrs decodes both to **NaN** — semantically, "no measurement here". xradar
runs them through the moment's scale and offset like any other code, so they
come back as numbers at the bottom of that moment's scale:

| Moment | xradar value, below threshold | xradar value, range folded |
|---|---:|---:|
| `DBZH` | -33.0 | -32.5 |
| `VRADH` | -64.5 | -64.0 |
| `WRADH` | -64.5 | -64.0 |
| `ZDR` | -13.062 | -13.031 |
| `PHIDP` | -0.705 | -0.353 |
| `RHOHV` | 0.202 | 0.205 |
| `CCORH` | -8.0 | -6.0, -7.0 |

In every case the below-threshold value is the minimum of that moment's decoded
scale and the range-folded value is the next step up — they are ordinary
decodes of the two reserved codes, not out-of-band markers. (`CCORH` is the odd
one out, with a third value; radrs masks slightly more of it than the two
reserved codes alone account for.)

Both readings are defensible — radrs is cleaner for analysis, xradar retains
which of the two codes it was. But the practical consequence is sharp: on
`sweep_0` of the volume above, **78.5% of `DBZH` cells** are below threshold.
Under xradar those cells are the value -33.0, which is a physically plausible
reflectivity, so any mean, histogram, or gradient computed over the raw array
is dominated by them. `RHOHV`'s 0.202 is worse — it looks like an ordinary
low-correlation measurement.

!!! danger "Do not mask by literal value"

    The values above are the *decoded scale minimum* for each moment, so they
    depend on the scale and offset recorded in the file and vary by moment and
    by sweep. Hard-coding `-33.0` works for `DBZH` on this volume and silently
    fails on `RHOHV`, on other moments, and potentially on other volumes.

    To compare the two libraries, restrict to gates where both report
    something finite:

    ```python
    import numpy as np

    valid = np.isfinite(rs_vals) & np.isfinite(xr_vals)
    diff = np.abs(rs_vals[valid] - xr_vals[valid])
    ```

    To clean an xradar array on its own terms, derive the two reserved values
    from the data rather than hard-coding them — they are always the two lowest
    decoded values present:

    ```python
    reserved = np.unique(ds[moment].values)[:2]
    cleaned = ds[moment].where(~np.isin(ds[moment].values, reserved))
    ```

    On Doppler sweeps that reproduces radrs's NaN mask exactly. But it is
    still a heuristic, and it over-masks where range folding never occurred:
    on the surveillance cut of the volume above the two lowest `DBZH` values
    are `-33.0` and `-24.0`, and that `-24.0` is a real measurement. For
    `RHOHV` the same recipe agrees with radrs on 99.93% of cells rather than
    100% — the shortfall is genuine low-correlation gates being thrown away.

    Reading the volume with radrs avoids the guesswork, since the distinction
    is made during decode. The cost is that radrs collapses both codes to NaN,
    so which one a gate was is not recoverable from its output.

Nothing is lost in the other direction: there are **zero** cells that xradar
reports as NaN while radrs reports a value.

## Radial counts

Sweep and radial counts matched exactly on the volume above, but small
differences of one or two radials per sweep are possible, from different
handling of duplicate or incomplete radials at the sweep boundary. Align by
azimuth rather than assuming equal lengths if you are comparing across
libraries, and expect `min(n_rs, n_xr)` rows of overlap.

## Cloud sources

radrs takes a URI directly and fetches it itself:

```python
rxr.open_datatree(
    "s3://unidata-nexrad-level2/2024/03/15/KTLX/KTLX20240315_000217_V06"
)
```

`s3://` defaults to anonymous access, so the public buckets work with no
credentials. `gs://`, `az://`, `file://`, and plain local paths are accepted
too, along with raw `bytes`; pass `storage_options` for private buckets. Only
single-volume keys work — the GCS mirror's 6-minute tar bundles are not
unpacked and fail at parse.

`xradar.io.open_nexradlevel2_datatree` needs a local file or an open file
object, so reaching the same object means staging it first:

```python
import fsspec, xradar as xd

local = fsspec.open_local(f"simplecache::{uri}", s3={"anon": True})
dt = xd.io.open_nexradlevel2_datatree(local)
```

## Verifying it yourself

Two tools in the repo compare the libraries directly. Both need the `dev`
group, which brings in xradar:

```bash
uv sync --group dev
```

**`scripts/xradar_radrs_bench.py`** loads one volume through both libraries and
prints, per sweep, the radial and range counts, both fixed angles, and for each
moment the fraction of gates where both are valid plus the mean, p95, and p99
absolute difference. Moments that agree everywhere collapse to
`(all obs equal)`.

```bash
uv run python scripts/xradar_radrs_bench.py /path/to/KTLX20240315_000217_V06

# or compare a remote volume, letting each library reach it its own way
uv run python scripts/xradar_radrs_bench.py \
  --radrs-source s3://unidata-nexrad-level2/2024/03/15/KTLX/KTLX20240315_000217_V06 \
  --xradar-source https://unidata-nexrad-level2.s3.amazonaws.com/2024/03/15/KTLX/KTLX20240315_000217_V06
```

It aligns by azimuth internally, so `sort_by_azimuth` does not need to be set.

**`notebooks/xradar_radrs_compare.py`** is the interactive version — pick a
volume, sweep, and moment, and get the two arrays side by side with a
difference map, a statistics table (NaN agreement, mean and max absolute
difference, percentage within tolerance), an azimuth alignment plot, and the
raystack `activity` summary for the same volume.

```bash
uv run marimo run notebooks/xradar_radrs_compare.py
```

Note that it loads radrs with `sort_by_azimuth=True` so the side-by-side images
line up; the NaN-disagreement row in its statistics table is measuring the
below-threshold difference described above, not an error.

!!! note "Timings need a release build"

    The benchmark also reports load times for both libraries, and prints a
    banner naming the build profile it measured. An editable install from
    `uv sync` is a **debug** build — `opt-level 0`, no LTO, roughly 5–10×
    slower than release — so its timings say nothing about radrs performance.

    Build the extension in release mode before reading the numbers:

    ```bash
    maturin develop --release
    ```

    See [Install](../install.md) for the full source-build steps.

## What's next

- [Quickstart](quickstart.md) — the `open_datatree` basics for both interfaces.
- [Raystack format](raystack-format.md) — the flat alternative to the
  sweep-tree layout, for when fixed-shape tensors suit the work better.
- [`radrs.xradar` API reference](../reference/xradar.md) — the full signature.
