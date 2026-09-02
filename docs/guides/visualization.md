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

Opens one volume with `rrs.open_datatree` and offers six ways to look at it:

| Mode | What it draws |
|---|---|
| **PPI** | One sweep in radar-native polar coordinates: azimuth around, slant range out, with range rings. The default. |
| **Ray 3D** | One endpoint per return ray, so the sampling geometry of the whole VCP is visible at once. Sweep-independent. |
| **Gate cloud 3D** | Every finite gate as a point in radar-relative space, sampled down to fit the payload's byte budget or the sample cap, whichever is tighter. |
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

CAPPI and cross-section altitudes come from the standard 4/3
effective-earth-radius model (`viz.beam_geometry`), so a gate's height
accounts for earth curvature and mean refraction rather than flat-earth
`r * sin(el)`. At 150 km that is a difference of about 1.3 km. Heights are
above the antenna, not mean sea level; add the site altitude for MSL. Ducting
and strong inversions are not corrected for.

## `batched_raystack_viz.py`: a whole time range

Accumulates many volumes into one `BatchedRaystack` and renders the result as a
single returns matrix, so you can follow a storm's evolution down one image.
See [Batch iteration and folding](batching.md) for the API underneath.

| Mode | What it draws |
|---|---|
| **PPI** | One sweep of one volume; the sweep picker labels each sweep with its time so volumes are distinguishable. |
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
returns and each one is a pixel row, so it is drawn through a window. How many
rows that window holds is the payload's decision, not the notebook's:
`prepare_folded_waterfall_payload` fills its byte budget with as many rows as
the fold size leaves room for, so a narrow fold buys more of them. The only
control left is where the window sits.

## Color scales

Every widget colors its data on a fixed, absolute scale keyed by moment name,
and draws a labelled colorbar. The same value is the same color in every
sweep, every view, and every volume, so you can read a number off the picture.

| Moment | Palette | Range | Units |
| --- | --- | --- | --- |
| `DBZH` | `nws_reflectivity` | -30 to 75 | dBZ |
| `VRADH` | `nws_velocity` | ±Nyquist, else ±32 | m/s |
| `WRADH` | `magma` | 0 to 15 | m/s |
| `ZDR` | `nws_zdr` | -2 to 6 | dB |
| `PHIDP` | `cyclic` | 0 to 360 | deg |
| `RHOHV` | `nws_rhohv` | 0.7 to 1.05 | |
| `CCORH` | `viridis` | 0 to 1 | |

`VRADH` is symmetric about zero so inbound and outbound read as opposite hues;
if the `returns` or `sweeps` dataset carries a `nyquist_velocity`, the bounds
follow it. `PHIDP` wraps back to its starting color at 360 because the parser
emits the full turn. `RHOHV` runs past 1.0 on purpose: values there are
unphysical and worth seeing.

Moments outside the table, such as QC arrays or anything unrecognized, keep
the old behavior: a 5th to 95th percentile stretch over `viridis`.

Pass `color_scale=` to any `prepare_*_payload` function to override:

```python
from radrs.viz import ColorScale, prepare_polar_payload

# Zoom the reflectivity ramp onto a light-rain volume.
payload = prepare_polar_payload(
    returns, sweeps, sweep_index=0, moment="DBZH",
    color_scale=ColorScale("nws_reflectivity", vmin=-10.0, vmax=40.0, units="dBZ"),
)
```

Passing a `ColorScale` whose `name` is not a known palette falls back to
`viridis` with the bounds you gave.

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
payload = viz.prepare_volume_payload(returns=returns, moment="DBZH")
payload.point_count, payload.max_abs_m, payload.nbytes
```

`viz.sweep_offsets(sweeps)` gives the cumulative return offsets if you want to
slice a single sweep out of a volume or a batch first.

### The byte budget

Every `prepare_*_payload` function takes a `byte_budget` and sizes itself to
fit, so nothing downstream has to guess a sample count. The default,
`viz.DEFAULT_BYTE_BUDGET`, is 3.5 MB, which sits just under marimo's 5 MB
output limit. `payload.nbytes` reports what the payload actually costs on the
wire, counting the buffers as they ship rather than the arrays it holds.

Each view spends the budget on whatever reduction suits its shape:

| Payload | What the budget buys |
| --- | --- |
| `prepare_polar_payload` | return and gate strides, decimated together |
| `prepare_volume_payload`, `prepare_ray_payload` | sample count |
| `prepare_waterfall_payload` | row and column strides |
| `prepare_folded_waterfall_payload` | rows in the window |
| `prepare_cappi_payload`, `prepare_xsec_payload` | a ceiling on `grid_size` |

The knobs that were there before stay as explicit overrides, and only ever
tighten the result: `max_points` on the 3-D paths (sampled points) and the PPI
(drawn grid cells), `max_returns`/`max_range` on the waterfall, `row_count` on
the folded waterfall, `grid_size` on CAPPI and cross-section. Pass
`byte_budget=None` to ship a payload whole and take care of the size yourself.

Sampling is deterministic, so the same inputs always yield the same picture. It
is a fixed-seed permutation rather than a fixed stride, because the buffers it
indexes are return-major: striding them lands every sample on gates that are
multiples of one number, which collapses a PPI onto a few concentric rings.

### What crosses the wire

Payloads ship structure, not coordinates. A gate's azimuth, elevation, range
and time all follow from its row and column, so the polar and waterfall views
send one entry per return plus a dense value grid and let the widget rebuild
the rest. The 3-D views need an explicit positions buffer for deck.gl, so they
send per-return geometry plus a return slot and a gate index per point, and
build that buffer in JS with the same 4/3 effective-earth model
`viz.beam_geometry` uses.

Values are quantized to `uint16` across the payload's `[vmin, vmax]`, with one
code reserved for missing data. On the reflectivity scale that is a step of
0.0016 dBZ, far below what the instrument resolves, and hover readouts
dequantize back to physical units, so a hovered number is still good enough to
do QC against. Values outside `[vmin, vmax]` clamp to the ends of the scale
rather than wrapping or dropping out.

Between them, those two choices take a full 360 x 1832 PPI sweep from roughly
19 MB of float32 point buffers down to about 1.3 MB, which is why a whole sweep
now fits the default budget with nothing thrown away.

The widgets need `anywidget` and `traitlets` (both in the `dev` group). Without
them the payload functions still work; only the `*Widget` classes raise.

The front-end is plain JavaScript in `python/radrs/viz/static/`: one file per
widget (`polar.js`, `volume.js`, `grid.js`, `waterfall.js`,
`folded_waterfall.js`), a `shared.js` of helpers they all use (binary decoding,
dequantization, the palette table and LUT, the colorbar, axis ticks, beam
height and ground range), and
`widget.css`. There is no bundler: `radrs.viz.assets.assemble_esm` concatenates
`shared.js` with one widget file at import time, because anywidget loads the
module from a blob URL where a relative import has nothing to resolve against.
Anything a widget file needs must therefore be declared in its own file or in
`shared.js`. Set `RADRS_VIZ_DEV=1` before starting the notebook and the sources
are re-read on every widget instantiation, so editing a `.js` file and
re-running the cell is enough to see the change.
