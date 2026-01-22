import marimo

__generated_with = "0.19.4"
app = marimo.App()


@app.cell
def _():
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo as mo
    import radrs.ops as ops
    import radrs.xradar as rxr

    return mo, np, ops, os, plt, rxr


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

    az = np.asarray(sweep_ds["azimuth"].values)
    return az, sweep_ds, sweep_name


@app.cell
def _(az, np, ops, sweep_ds):
    sorted_az = np.sort(az)
    plan = ops.align_azimuth(az, sorted_az)
    dbzh = np.asarray(sweep_ds["DBZH"].values)
    dbzh_sorted = plan.apply_2d(dbzh)
    return dbzh, dbzh_sorted, plan, sorted_az


@app.cell
def _(ops, sweep_ds):
    vel = sweep_ds["VRADH"].values
    texture = ops.velocity_texture(vel, wind_size=3)
    return (texture,)


@app.cell
def _(az, dbzh, dbzh_sorted, np, plt, sorted_az, texture):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    im0 = ax.imshow(dbzh, aspect="auto")
    ax.set_title("DBZH (original)")
    fig.colorbar(im0, ax=ax)

    ax = axes[0, 1]
    im1 = ax.imshow(dbzh_sorted, aspect="auto")
    ax.set_title("DBZH (azimuth-aligned)")
    fig.colorbar(im1, ax=ax)

    ax = axes[1, 0]
    im2 = ax.imshow(texture, aspect="auto")
    ax.set_title("VRADH velocity texture")
    fig.colorbar(im2, ax=ax)

    ax = axes[1, 1]
    ax.plot(az, np.arange(len(az)), ".", label="src az")
    ax.plot(sorted_az, np.arange(len(sorted_az)), ".", label="dst az")
    ax.set_title("Azimuth ordering")
    ax.set_xlabel("azimuth (deg)")
    ax.set_ylabel("ray index")
    ax.legend()

    fig.tight_layout()
    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
