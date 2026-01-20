import marimo

__generated_with = "0.19.4"
app = marimo.App()


@app.cell
def _():
    import os
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    import marimo as mo

    import radrs.qc as qc
    import radrs.xradar as rxr

    import pyart
    return mo, np, os, plt, pyart, qc, rxr, xr


@app.cell
def _(mo, os):
    default_path = "/Users/rejuvyesh/src/silurian/nexrad/downloads/KDMX20220305_232324_V06"
    if not os.path.exists(default_path):
        default_path = ""
    file_path = mo.ui.text(
        value=default_path,
        label="NEXRAD level2 file path",
        full_width=True,
    )
    file_path
    return (file_path,)


@app.cell
def _(file_path, rxr):
    dt = rxr.open_datatree(file_path.value)
    return (dt,)


@app.cell
def _(dt, np):
    sweep_name = None
    sweep_ds = None
    for name in dt.children:
        if not name.startswith("sweep_"):
            continue
        ds = dt[name].dataset
        if "VRADH" in ds and "DBZH" in ds:
            sweep_name = name
            sweep_ds = ds
            break
    if sweep_ds is None:
        raise ValueError("No sweep with VRADH/DBZH found")
    sweep_ds = sweep_ds.assign(VRADH=sweep_ds["VRADH"].where(
        ~np.isclose(sweep_ds["VRADH"], -64.5, atol=1.0), np.nan
    ))
    return (sweep_ds,)


@app.cell
def _(dt, sweep_ds, xr):
    _sweep_ds = sweep_ds.copy()
    _sweep_ds["sweep_number"] = 0
    root = dt.dataset.copy()
    for coord in ("latitude", "longitude", "altitude"):
        if coord not in root:
            root = root.assign({coord: 0.0})
    single_sweep_dt = xr.DataTree.from_dict({".": root, "sweep_0": _sweep_ds})
    return (single_sweep_dt,)


@app.cell
def _(single_sweep_dt):
    radar = single_sweep_dt.pyart.to_radar()
    return (radar,)


@app.cell
def _(np, pyart, radar, sweep_ds):
    nyq = np.nanmax(np.abs(sweep_ds["VRADH"].values))
    if not np.isfinite(nyq) or nyq <= 0:
        raise ValueError("Invalid Nyquist velocity")

    vel_texture = pyart.retrieve.calculate_velocity_texture(
        radar, vel_field="VRADH", nyq=nyq
    )
    radar.add_field("velocity_texture", vel_texture, replace_existing=True)

    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_above("velocity_texture", 4.0)
    gatefilter.exclude_below("DBZH", 0.0)

    velocity_dealiased = pyart.correct.dealias_region_based(
        radar,
        vel_field="VRADH",
        nyquist_vel=nyq,
        centered=True,
        gatefilter=gatefilter,
    )

    original = np.ma.filled(radar["sweep_0"]["VRADH"].values, np.nan)
    dealiased = np.ma.filled(velocity_dealiased["data"], np.nan)
    expected = (dealiased - original) / (2 * nyq)
    return expected, nyq, original


@app.cell
def _(nyq, qc, sweep_ds):
    actual = qc.vradh_winding_number(
        sweep_ds["VRADH"].values,
        sweep_ds["DBZH"].values,
        nyquist=nyq,
        wind_size=3,
        velocity_texture_threshold=4.0,
        reflectivity_threshold=0.0,
        interval_splits=3,
        skip_between_rays=100,
        skip_along_ray=100,
        centered=True,
        rays_wrap_around=True,
        fill_value=-64.5,
        fill_tolerance=1.0,
    )
    return (actual,)


@app.cell
def _(actual, expected, np):
    diff = np.nan_to_num(np.abs(actual - expected), nan=0.0)
    mismatch = np.count_nonzero(diff > 0)
    mismatch_ratio = mismatch / diff.size
    mismatch_ratio
    return (diff,)


@app.cell
def _(actual, diff, expected, original, plt):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax = axes[0, 0]
    im0 = ax.imshow(original, aspect="auto")
    ax.set_title("Original VRADH")
    fig.colorbar(im0, ax=ax)

    ax = axes[0, 1]
    im1 = ax.imshow(expected, aspect="auto")
    ax.set_title("Py-ART winding number")
    fig.colorbar(im1, ax=ax)

    ax = axes[1, 0]
    im2 = ax.imshow(actual, aspect="auto")
    ax.set_title("radrs winding number")
    fig.colorbar(im2, ax=ax)

    ax = axes[1, 1]
    im3 = ax.imshow(diff, aspect="auto")
    ax.set_title("Absolute diff")
    fig.colorbar(im3, ax=ax)

    fig.tight_layout()
    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
