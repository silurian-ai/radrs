# Visualization

`radrs.viz` renders raystack data directly, with no Py-ART or Cartopy in the
path. It has two halves: `prepare_*_payload` functions that reduce a `returns`
dataset to a compact, JSON-serializable payload, and
[anywidget](https://anywidget.dev)-based widgets that draw it in the browser
with hover readout and 3D interaction.

Two marimo notebooks wire that surface to live controls. Both pull volumes
straight from the public NEXRAD archives, so they need no local data.

```bash
uv sync --group dev
uv run marimo run notebooks/raystack_viz.py
```

`marimo run` opens the notebook as an app: controls only, no source. Use
`marimo edit` instead to see and change the cells.

## `raystack_viz.py`: single volume

Opens one volume with `rrs.open_datatree` and offers five ways to look at it:

| Mode | What it draws |
|---|---|
| **Ray 3D** | One endpoint per return ray, so the sampling geometry of the whole VCP is visible at once. Sweep-independent. |
| **Gate cloud 3D** | Every finite gate as a point in radar-relative space, downsampled to the sample cap. |
| **CAPPI** | Constant-altitude horizontal slice, gridded from the gates within an altitude tolerance. |
| **Cross-section** | Vertical slice along a target azimuth, within an azimuth tolerance. |
| **Waterfall** | The raw `(return_time, range)` moment matrix as a 2-D heatmap. This is the raystack layout itself, unprojected. |

Pick a station, archive, and time window; the notebook lists the volumes it
finds and you choose one from a dropdown. `fold_size` is a control too, so you
can watch the waterfall's row count change as folding changes.

The 3D modes rotate on drag, zoom on wheel, and reset on double-click. The 2-D
modes report the value under the cursor along with its azimuth, elevation,
sweep, and range. CAPPI and cross-section add their own altitude/azimuth and
tolerance sliders, and grid resolution.

## `batched_raystack_viz.py`: a whole time range

Accumulates many volumes into one `BatchedRaystack` and renders the result as a
single returns matrix, so you can follow a storm's evolution down one image.
See [Batch iteration and folding](batching.md) for the API underneath.

| Mode | What it draws |
|---|---|
| **Folded waterfall** | The batch's `(return_time, range)` matrix drawn one pixel-row per return, so volumes stack along the time axis. Rows are exact, not resampled. |
| **Gate cloud 3D** | Every finite gate in the entire batch, overlaid in radar-relative space. |

The controls map onto the batch parameters:

- **Fold size** and **volume range** set what goes in.
- **Prefetch** sets how many fetches run concurrently.
- **Capacity headroom** is the multiplier on the estimated returns per volume.
  Because capacity is reserved up front, undershooting drops volumes off the
  end of the range; the notebook reports how many of the selected volumes
  actually fit.
- **Drop all-NaN returns** toggles compaction, on by default.

After loading it prints a metrics table: volumes added versus selected, sweep
and return counts, the NaN share of the surviving cells, and returns filled
against returns capacity. That last figure is reported as spare capacity, and
because reservation happens before compaction, real spare is lower than shown
whenever dropping is on.

The folded waterfall gets tall, since a batch is hundreds of thousands of
returns and each one is a pixel row, so the notebook windows it with row offset
and row count sliders sized to keep the payload under marimo's output limit.

## Building your own

The payload functions are usable on their own; the widgets are optional. Each
takes a `returns` dataset and a moment name and returns a dataclass of plain
arrays:

```python
import radrs.raystack as rrs
import radrs.viz as viz

src = "s3://unidata-nexrad-level2/2024/07/02/KABR/KABR20240702_000016_V06"
rdt = rrs.open_datatree(src, fold_size=256)
returns, sweeps = viz.get_returns_and_sweeps(rdt)

viz.available_moments(returns)             # what's renderable, QC included
payload = viz.prepare_volume_payload(returns=returns, moment="DBZH", max_points=125_000)
payload.point_count, payload.max_abs_m
```

`prepare_volume_payload` and the other 3-D paths take `max_points` and
downsample deterministically, so the same inputs always yield the same picture.
`viz.sweep_offsets(sweeps)` gives the cumulative return offsets if you want to
slice a single sweep out of a volume or a batch first.

The widgets need `anywidget` and `traitlets` (both in the `dev` group). Without
them the payload functions still work; only the `*Widget` classes raise.
