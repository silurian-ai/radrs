import marimo

__generated_with = "0.19.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    import radrs.raystack as rrs
    return mo, np, plt, rrs


@app.cell
def _(mo):
    mo.md("""
    # Raystack Explorer

    Explore the raystack data format and activity metrics.
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
def _(mo, rrs, source_input):
    with mo.status.spinner("Loading raystack..."):
        dt = rrs.open_datatree(source_input.value)
    return (dt,)


@app.cell
def _(dt, mo):
    mo.md(f"""
    ## DataTree Structure

    ```
    {dt}
    ```
    """)
    return


@app.cell
def _(dt, mo):
    returns = dt["returns"].dataset
    sweeps = dt["sweeps"].dataset

    n_returns = returns.sizes["return_time"]
    n_sweeps = sweeps.sizes["sweep_time"]
    fold_size = returns.sizes["range"]

    mo.md(f"""
    ## Dimensions

    | Dimension | Size |
    |-----------|------|
    | **n_returns** (rays) | {n_returns:,} |
    | **n_sweeps** | {n_sweeps} |
    | **fold_size** (range bins) | {fold_size} |
    | **Total cells** | {n_returns * fold_size:,} |
    """)
    return returns, sweeps


@app.cell
def _(mo):
    mo.md("""
    ## Sweep Structure
    """)
    return


@app.cell
def _(plt, returns, sweeps):
    _starts = sweeps["start_index"].values
    _n_radials = sweeps["n_radials"].values
    _elevations = sweeps["sweep_fixed_angle"].values

    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4))

    # Radials per sweep
    _axes[0].bar(range(len(_n_radials)), _n_radials)
    _axes[0].set_xlabel("Sweep index")
    _axes[0].set_ylabel("Number of radials")
    _axes[0].set_title("Radials per sweep")

    # Elevation angles
    _axes[1].bar(range(len(_elevations)), _elevations)
    _axes[1].set_xlabel("Sweep index")
    _axes[1].set_ylabel("Elevation angle (deg)")
    _axes[1].set_title("Elevation angles")

    # Sweep boundaries on azimuth
    _azimuths = returns["azimuth"].values
    _axes[2].plot(_azimuths, lw=0.5)
    for _i, (_start, _n) in enumerate(zip(_starts, _n_radials)):
        if _i < 5:  # Only show first few for clarity
            _axes[2].axvline(_start, color="red", alpha=0.5, lw=0.5)
    _axes[2].set_xlabel("Return index")
    _axes[2].set_ylabel("Azimuth (deg)")
    _axes[2].set_title("Azimuth with sweep boundaries (red)")

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Activity Metrics
    """)
    return


@app.cell
def _(dt, mo):
    if "activity" not in dt:
        mo.stop(True, mo.md("*Activity metrics not available*"))

    activity = dt["activity"].dataset
    moments = [str(m) for m in activity["moment"].values]

    mo.md(f"**Moments tracked:** {moments}")
    return activity, moments


@app.cell
def _(activity, mo, moments):
    _vol_count = activity["volume_valid_count"].values.flatten()
    _vol_frac = activity["volume_valid_fraction"].values.flatten()

    _rows = []
    for _i, _m in enumerate(moments):
        _rows.append(f"| {_m} | {_vol_count[_i]:,} | {_vol_frac[_i]:.2%} |")

    mo.md(f"""
    ### Volume-Level Summary

    | Moment | Valid Count | Valid Fraction |
    |--------|-------------|----------------|
    {chr(10).join(_rows)}
    """)
    return


@app.cell
def _(activity, moments, np, plt):
    _fig, _axes = plt.subplots(2, 2, figsize=(14, 10))

    # Volume fractions bar chart
    _vol_frac = activity["volume_valid_fraction"].values.flatten()
    _colors = plt.cm.tab10(np.linspace(0, 1, len(moments)))
    _axes[0, 0].barh(moments, _vol_frac, color=_colors)
    _axes[0, 0].set_xlabel("Valid fraction")
    _axes[0, 0].set_title("Volume-level data availability")
    _axes[0, 0].set_xlim(0, 1)

    # Sweep fractions heatmap
    _sweep_frac = activity["sweep_valid_fraction"].values
    _im1 = _axes[0, 1].imshow(_sweep_frac, aspect="auto", vmin=0, vmax=1, cmap="YlGn")
    _axes[0, 1].set_yticks(range(len(moments)))
    _axes[0, 1].set_yticklabels(moments)
    _axes[0, 1].set_xlabel("Sweep index")
    _axes[0, 1].set_title("Sweep-level valid fraction")
    _fig.colorbar(_im1, ax=_axes[0, 1])

    # Ray fractions - show distribution
    _ray_frac = activity["ray_valid_fraction"].values
    for _i, _m in enumerate(moments):
        _frac = _ray_frac[_i]
        _frac_valid = _frac[np.isfinite(_frac)]
        if len(_frac_valid) > 0:
            _axes[1, 0].hist(_frac_valid, bins=50, alpha=0.5, label=_m)
    _axes[1, 0].set_xlabel("Valid fraction per ray")
    _axes[1, 0].set_ylabel("Count")
    _axes[1, 0].set_title("Distribution of ray-level valid fractions")
    _axes[1, 0].legend(fontsize=8)

    # Ray counts over time (shows sweep structure)
    _ray_count = activity["ray_valid_count"].values
    for _i, _m in enumerate(["DBZH", "VRADH"]):
        if _m in moments:
            _idx = moments.index(_m)
            _axes[1, 1].plot(_ray_count[_idx], label=_m, alpha=0.7, lw=0.5)
    _axes[1, 1].set_xlabel("Return index")
    _axes[1, 1].set_ylabel("Valid count per ray")
    _axes[1, 1].set_title("Ray-level valid counts (shows sweep structure)")
    _axes[1, 1].legend()

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md("""
    ## Moment Data
    """)
    return


@app.cell
def _(mo, returns):
    _moment_vars = [v for v in returns.data_vars if v in {"DBZH", "VRADH", "WRADH", "ZDR", "PHIDP", "RHOHV", "CCORH"}]
    moment_selector = mo.ui.dropdown(
        options=_moment_vars,
        value=_moment_vars[0] if _moment_vars else None,
        label="Select moment",
    )
    moment_selector
    return (moment_selector,)


@app.cell
def _(moment_selector, np, plt, returns):
    _moment = moment_selector.value
    _output = None
    if _moment:
        _data = returns[_moment].values

        _fig, _axes = plt.subplots(1, 2, figsize=(14, 5))

        # Full image
        _im0 = _axes[0].imshow(_data, aspect="auto", cmap="viridis")
        _axes[0].set_xlabel("Range bin")
        _axes[0].set_ylabel("Return index")
        _axes[0].set_title(f"{_moment} - Full volume")
        _fig.colorbar(_im0, ax=_axes[0])

        # Histogram of values
        _valid = _data[np.isfinite(_data)]
        _axes[1].hist(_valid, bins=100, edgecolor="none")
        _axes[1].set_xlabel(_moment)
        _axes[1].set_ylabel("Count")
        _axes[1].set_title(f"{_moment} value distribution (n={len(_valid):,})")

        _fig.tight_layout()
        _output = _fig
    _output
    return


@app.cell
def _(mo):
    mo.md("""
    ## Compare with/without Activity
    """)
    return


@app.cell
def _(mo, rrs, source_input):
    import time as _time

    _output = None
    # For S3, we need to fetch first - skip timing comparison
    if source_input.value.startswith("s3://"):
        _output = mo.md("*Timing comparison requires local file (S3 fetch dominates timing)*")
    else:
        with mo.status.spinner("Parsing with activity..."):
            _t0 = _time.perf_counter()
            with open(source_input.value, "rb") as _f:
                _rs_with = rrs.parse(_f.read())
            _t_with = _time.perf_counter() - _t0

        with mo.status.spinner("Parsing without activity..."):
            _t0 = _time.perf_counter()
            with open(source_input.value, "rb") as _f:
                _rs_without = rrs.parse(_f.read(), include_activity=False)
            _t_without = _time.perf_counter() - _t0

        _output = mo.md(f"""
    ### Parse Timing

    | Mode | Time |
    |------|------|
    | With activity | {_t_with*1000:.1f} ms |
    | Without activity | {_t_without*1000:.1f} ms |
    | Overhead | {(_t_with - _t_without)*1000:.1f} ms ({100*(_t_with - _t_without)/_t_without:.1f}%) |
    """)
    _output
    return


if __name__ == "__main__":
    app.run()
