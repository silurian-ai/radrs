import marimo

__generated_with = "0.19.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr

    import radrs.xradar as rxr
    import radrs.raystack as rrs
    return mo, np, plt, rrs, rxr


@app.cell
def _(mo):
    mo.md("""
    # xradar vs radrs Comparison

    This notebook compares NEXRAD data loaded via radrs against xradar to verify correctness.
    """)
    return


@app.cell
def _(mo):
    default_source = "s3://unidata-nexrad-level2/2024/07/02/KABR/KABR20240702_000016_V06"
    source_input = mo.ui.text(
        value=default_source,
        label="NEXRAD file (local path or S3 URL)",
        full_width=True,
    )
    source_input
    return (source_input,)


@app.cell
def _(mo, source_input):
    mo.stop(not source_input.value, mo.md("*Enter a file path above*"))
    return


@app.cell
def _(mo):
    mo.md("""
    ## Load with both libraries
    """)
    return


@app.cell
def _(mo, rrs, rxr, source_input):
    # Use sort_by_azimuth=True to match xradar's sorted output
    with mo.status.spinner("Loading with radrs.xradar..."):
        dt_radrs = rxr.open_datatree(source_input.value, sort_by_azimuth=True)

    with mo.status.spinner("Loading with radrs.raystack..."):
        dt_raystack = rrs.open_datatree(source_input.value)
    return dt_radrs, dt_raystack


@app.cell
def _(mo):
    try:
        import xradar as xd

        _xradar_available = True
    except ImportError:
        _xradar_available = False

    mo.stop(
        not _xradar_available,
        mo.md("**xradar not installed** - install with `uv pip install xradar` to enable comparison"),
    )
    return (xd,)


@app.cell
def _(mo, source_input, xd):
    import fsspec

    _source = source_input.value
    with mo.status.spinner("Loading with xradar..."):
        if _source.startswith("s3://"):
            # xradar doesn't handle S3 URLs directly, use fsspec to get local path
            _local_path = fsspec.open_local(
                f"simplecache::{_source}",
                s3={"anon": True},
            )
            dt_xradar = xd.io.open_nexradlevel2_datatree(_local_path)
        else:
            dt_xradar = xd.io.open_nexradlevel2_datatree(_source)
    return (dt_xradar,)


@app.cell
def _(dt_radrs, dt_xradar, mo):
    radrs_sweeps = sorted([k for k in dt_radrs.children if k.startswith("sweep_")])
    xradar_sweeps = sorted([k for k in dt_xradar.children if k.startswith("sweep_")])

    mo.md(f"""
    ## Structure Comparison

    | | radrs | xradar |
    |---|---|---|
    | **Sweeps** | {len(radrs_sweeps)} | {len(xradar_sweeps)} |
    | **Sweep names** | {radrs_sweeps[:3]}... | {xradar_sweeps[:3]}... |
    """)
    return (radrs_sweeps,)


@app.cell
def _(mo, radrs_sweeps):
    sweep_selector = mo.ui.dropdown(
        options=radrs_sweeps,
        value=radrs_sweeps[0] if radrs_sweeps else None,
        label="Select sweep",
    )
    sweep_selector
    return (sweep_selector,)


@app.cell
def _(dt_radrs, dt_xradar, mo, sweep_selector):
    sweep_name = sweep_selector.value
    ds_radrs = dt_radrs[sweep_name].dataset if sweep_name else None
    ds_xradar = dt_xradar[sweep_name].dataset if sweep_name else None

    _output = None
    if ds_radrs is not None and ds_xradar is not None:
        radrs_vars = set(ds_radrs.data_vars)
        xradar_vars = set(ds_xradar.data_vars)
        common_vars = radrs_vars & xradar_vars
        radrs_only = radrs_vars - xradar_vars
        xradar_only = xradar_vars - radrs_vars

        _output = mo.md(f"""
    ### Variables in {sweep_name}

    | Category | Variables |
    |----------|-----------|
    | **Common** | {sorted(common_vars)} |
    | **radrs only** | {sorted(radrs_only)} |
    | **xradar only** | {sorted(xradar_only)} |
    """)
    else:
        common_vars = set()
        radrs_vars = set()
        xradar_vars = set()
        radrs_only = set()
        xradar_only = set()

    _output
    return common_vars, ds_radrs, ds_xradar


@app.cell
def _(common_vars, mo):
    moments = sorted([v for v in common_vars if v in {"DBZH", "VRADH", "WRADH", "ZDR", "PHIDP", "RHOHV", "CCORH"}])
    moment_selector = mo.ui.dropdown(
        options=moments if moments else ["(none)"],
        value=moments[0] if moments else None,
        label="Select moment",
    )
    moment_selector
    return (moment_selector,)


@app.cell
def _(mo):
    mo.md("""
    ## Side-by-Side Visualization
    """)
    return


@app.cell
def _(ds_radrs, ds_xradar, mo, moment_selector, np, plt):
    moment = moment_selector.value
    _output = None
    if moment and moment != "(none)" and ds_radrs is not None and ds_xradar is not None:
        data_radrs = ds_radrs[moment].values
        data_xradar = ds_xradar[moment].values

        # Handle shape differences
        shape_match = data_radrs.shape == data_xradar.shape
        if not shape_match:
            _note = f"Shape mismatch: radrs {data_radrs.shape} vs xradar {data_xradar.shape}"
        else:
            _note = f"Shape: {data_radrs.shape}"

        # Compute common color limits
        vmin = np.nanmin([np.nanmin(data_radrs), np.nanmin(data_xradar)])
        vmax = np.nanmax([np.nanmax(data_radrs), np.nanmax(data_xradar)])

        _fig, _axes = plt.subplots(1, 3, figsize=(15, 5))

        _im0 = _axes[0].imshow(data_radrs, aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
        _axes[0].set_title(f"radrs - {moment}")
        _axes[0].set_xlabel("range")
        _axes[0].set_ylabel("azimuth")
        _fig.colorbar(_im0, ax=_axes[0])

        _im1 = _axes[1].imshow(data_xradar, aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
        _axes[1].set_title(f"xradar - {moment}")
        _axes[1].set_xlabel("range")
        _axes[1].set_ylabel("azimuth")
        _fig.colorbar(_im1, ax=_axes[1])

        if shape_match:
            _diff = data_radrs - data_xradar
            _im2 = _axes[2].imshow(_diff, aspect="auto", cmap="RdBu_r")
            _axes[2].set_title("Difference (radrs - xradar)")
            _axes[2].set_xlabel("range")
            _axes[2].set_ylabel("azimuth")
            _fig.colorbar(_im2, ax=_axes[2])
        else:
            _axes[2].text(0.5, 0.5, "Shape mismatch\nCannot compute diff", ha="center", va="center", transform=_axes[2].transAxes)
            _axes[2].set_title("Difference")

        _fig.suptitle(_note)
        _fig.tight_layout()
        _output = _fig
    else:
        _output = mo.md("*Select a moment above*")
    _output
    return


@app.cell
def _(mo):
    mo.md("""
    ## Numerical Comparison
    """)
    return


@app.cell
def _(ds_radrs, ds_xradar, mo, moment_selector, np):
    _moment = moment_selector.value
    _output = None
    if _moment and _moment != "(none)" and ds_radrs is not None and ds_xradar is not None:
        _data_radrs = ds_radrs[_moment].values
        _data_xradar = ds_xradar[_moment].values

        if _data_radrs.shape == _data_xradar.shape:
            _diff = _data_radrs - _data_xradar
            _abs_diff = np.abs(_diff)

            # Mask for valid comparisons (both finite)
            _valid_mask = np.isfinite(_data_radrs) & np.isfinite(_data_xradar)
            _valid_diff = _diff[_valid_mask]
            _valid_abs_diff = _abs_diff[_valid_mask]

            # NaN comparison
            _radrs_nan = np.isnan(_data_radrs)
            _xradar_nan = np.isnan(_data_xradar)
            _nan_agree = np.sum(_radrs_nan == _xradar_nan)
            _nan_disagree = np.sum(_radrs_nan != _xradar_nan)

            _output = mo.md(f"""
    ### {_moment} Statistics

    | Metric | Value |
    |--------|-------|
    | **Total cells** | {_data_radrs.size:,} |
    | **Both valid** | {_valid_mask.sum():,} |
    | **NaN agreement** | {_nan_agree:,} ({100*_nan_agree/_data_radrs.size:.1f}%) |
    | **NaN disagreement** | {_nan_disagree:,} ({100*_nan_disagree/_data_radrs.size:.1f}%) |
    | **Mean diff** | {np.mean(_valid_diff):.6f} |
    | **Std diff** | {np.std(_valid_diff):.6f} |
    | **Max abs diff** | {np.max(_valid_abs_diff):.6f} |
    | **Median abs diff** | {np.median(_valid_abs_diff):.6f} |
    | **% within 0.01** | {100*np.mean(_valid_abs_diff < 0.01):.2f}% |
    | **% within 0.1** | {100*np.mean(_valid_abs_diff < 0.1):.2f}% |
    | **% exact match** | {100*np.mean(_valid_diff == 0):.2f}% |
    """)
        else:
            _output = mo.md(f"Shape mismatch: radrs {_data_radrs.shape} vs xradar {_data_xradar.shape}")
    _output
    return


@app.cell
def _(mo):
    mo.md("""
    ## Raystack Activity Metrics

    Activity metrics show data availability per moment.
    """)
    return


@app.cell
def _(dt_raystack, mo, plt):
    _output = None
    if "activity" in dt_raystack:
        _activity = dt_raystack["activity"].dataset
        _moments = [str(m) for m in _activity["moment"].values]

        _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))

        # Volume-level summary
        _vol_frac = _activity["volume_valid_fraction"].values.flatten()
        _axes[0].barh(_moments, _vol_frac)
        _axes[0].set_xlabel("Valid fraction")
        _axes[0].set_title("Volume-level data availability")
        _axes[0].set_xlim(0, 1)

        # Sweep-level heatmap
        _sweep_frac = _activity["sweep_valid_fraction"].values
        _im = _axes[1].imshow(_sweep_frac, aspect="auto", vmin=0, vmax=1, cmap="YlGn")
        _axes[1].set_yticks(range(len(_moments)))
        _axes[1].set_yticklabels(_moments)
        _axes[1].set_xlabel("Sweep index")
        _axes[1].set_title("Sweep-level data availability")
        _fig.colorbar(_im, ax=_axes[1], label="Valid fraction")

        _fig.tight_layout()
        _output = _fig
    else:
        _output = mo.md("*Activity metrics not available*")
    _output
    return


@app.cell
def _(mo):
    mo.md("""
    ## Azimuth Alignment Check

    Radar data should have matching azimuths between libraries.
    """)
    return


@app.cell
def _(ds_radrs, ds_xradar, mo, np, plt):
    _output = None
    if ds_radrs is not None and ds_xradar is not None:
        _az_radrs = ds_radrs["azimuth"].values if "azimuth" in ds_radrs else None
        _az_xradar = ds_xradar["azimuth"].values if "azimuth" in ds_xradar else None

        if _az_radrs is not None and _az_xradar is not None:
            _len_match = len(_az_radrs) == len(_az_xradar)

            _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))

            _axes[0].plot(_az_radrs, label="radrs", alpha=0.7)
            _axes[0].plot(_az_xradar, label="xradar", alpha=0.7)
            _axes[0].set_xlabel("Ray index")
            _axes[0].set_ylabel("Azimuth (deg)")
            _axes[0].set_title("Azimuth values")
            _axes[0].legend()

            if _len_match:
                _az_diff = _az_radrs - _az_xradar
                # Handle wraparound at 360
                _az_diff = np.where(_az_diff > 180, _az_diff - 360, _az_diff)
                _az_diff = np.where(_az_diff < -180, _az_diff + 360, _az_diff)
                _axes[1].plot(_az_diff)
                _axes[1].set_xlabel("Ray index")
                _axes[1].set_ylabel("Difference (deg)")
                _axes[1].set_title(f"Azimuth diff (max: {np.max(np.abs(_az_diff)):.4f})")
            else:
                _axes[1].text(0.5, 0.5, f"Length mismatch\nradrs: {len(_az_radrs)}\nxradar: {len(_az_xradar)}",
                           ha="center", va="center", transform=_axes[1].transAxes)

            _fig.tight_layout()
            _output = _fig
        else:
            _output = mo.md("*Azimuth not available in one or both datasets*")
    _output
    return


if __name__ == "__main__":
    app.run()
