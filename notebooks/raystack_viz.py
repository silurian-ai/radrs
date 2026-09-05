import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    import datetime as dt

    import marimo as mo

    import radrs
    import radrs.raystack as rrs
    import radrs.viz as viz
    return dt, mo, radrs, rrs, viz


@app.cell
def _(mo):
    mo.md("""
    # Raystack Viz

    **Part 1** builds one raystack from an archive time range. **Part 2** plots it.

    Editing any Part 1 control re-fetches. A volume costs ~10 s to pull from S3,
    so the defaults load a single one; widen the range once you know what you want.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 1 — Data
    """)
    return


@app.cell
def _(dt, mo):
    archive_url = mo.ui.text(
        value="s3://unidata-nexrad-level2",
        label="Archive URL",
        full_width=True,
        debounce=True,
    )
    instrument_name = mo.ui.text(value="KABR", label="Instrument", debounce=True)
    start_time = mo.ui.datetime(
        value=dt.datetime(2024, 7, 2, 0, 0, 0), precision="minute", label="Start (UTC)"
    )
    end_time = mo.ui.datetime(
        value=dt.datetime(2024, 7, 2, 0, 40, 0), precision="minute", label="End (UTC)"
    )
    fold_size = mo.ui.slider(
        start=128, stop=1920, step=64, value=256, label="Fold size (gates)", show_value=True
    )
    # Capacity is pre-allocated, so an unbounded range would allocate itself to death.
    max_volumes = mo.ui.number(start=1, stop=200, step=1, value=4, label="Max volumes")

    mo.vstack(
        [
            mo.hstack([archive_url, instrument_name], widths=[6, 2]),
            mo.hstack([start_time, end_time, fold_size, max_volumes], widths=[3, 3, 3, 1]),
        ],
        align="stretch",
    )
    return (
        archive_url,
        end_time,
        fold_size,
        instrument_name,
        max_volumes,
        start_time,
    )


@app.cell
def _(
    archive_url,
    dt,
    end_time,
    fold_size,
    instrument_name,
    max_volumes,
    mo,
    radrs,
    rrs,
    start_time,
):
    site = instrument_name.value.strip().upper()
    mo.stop(len(site) != 4, mo.md("*Instrument must be a 4-letter ICAO code, e.g. `KTLX`.*"))
    mo.stop(start_time.value >= end_time.value, mo.md("*Start must be before end.*"))

    start_utc = start_time.value.replace(tzinfo=dt.timezone.utc)
    end_utc = end_time.value.replace(tzinfo=dt.timezone.utc)

    with mo.status.spinner("Listing archive volumes..."):
        volumes = sorted(
            radrs.list_nexrad_l2_archive_volumes(
                base_uri=archive_url.value,
                start_time=start_utc,
                end_time=end_utc,
                storage_options={"anon": "true"},
                site_filter=[site],
            ),
            key=lambda info: info.vcp_time,
        )[: int(max_volumes.value)]

    mo.stop(
        not volumes,
        mo.md(f"*No volumes for `{site}` in `{start_utc}` .. `{end_utc}` at `{archive_url.value}`.*"),
    )

    # A volume holds ~21M gate cells however it is folded; 1.5x covers uneven VCPs.
    returns_per_volume = int(21_000_000 / int(fold_size.value) * 1.5)
    batch = rrs.BatchedRaystack(
        max_vcps=len(volumes),
        max_sweeps=len(volumes) * 32,
        max_returns=len(volumes) * returns_per_volume,
        fold_size=int(fold_size.value),
        truncate=True,
        drop_empty_returns=True,
    )

    with mo.status.spinner(f"Fetching {len(volumes)} volumes from {site}..."):
        n_added = batch.add_volumes_from_l2(
            radrs.NexradL2ArchiveIter(
                archive_url.value,
                start_time=volumes[0].vcp_time,
                # Archive bounds are [start, end), so pad past the last volume.
                end_time=volumes[-1].vcp_time + dt.timedelta(seconds=1),
                storage_options={"anon": "true"},
                site_filter=[site],
            ),
            prefetch=8,
        )
        rs_dt = batch.finalize_to_rs_dt()

    mo.stop(n_added == 0, mo.md("*No volumes were added — widen the time range.*"))
    rs_dt
    return n_added, rs_dt, volumes


@app.cell
def _(mo, n_added, rs_dt, viz, volumes):
    mo.md(f"""
    Loaded **{n_added} of {len(volumes)}** volumes ·
    **{rs_dt["returns"].sizes["return_time"]:,}** returns ·
    **{rs_dt["sweeps"].sizes["sweep_time"]:,}** sweeps ·
    fold **{rs_dt["returns"].sizes["range"]}** gates ·
    moments `{", ".join(viz.available_moments(rs_dt))}`
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 2 — Plots
    """)
    return


@app.cell
def _(mo, rs_dt, viz):
    mo.md("""### Sweeps — `viz.plot_sweeps`""")

    sweep_vcp = mo.ui.dropdown(
        options={info.label: info.index for info in viz.vcp_infos(rs_dt)},
        value=viz.vcp_infos(rs_dt)[0].label,
        label="VCP",
    )
    # Sweep numbers repeat across volumes of the same pattern, so VCP 0 names them all.
    sweep_nums = mo.ui.multiselect(
        options={info.label: info.sweep_number for info in viz.sweep_infos(rs_dt)},
        value=[viz.sweep_infos(rs_dt)[0].label],
        label="Sweeps (overlaid in order)",
    )
    sweep_moment = mo.ui.dropdown(
        options=viz.available_moments(rs_dt), value="DBZH", label="Moment"
    )
    sweep_alpha = mo.ui.slider(
        start=0.1, stop=1.0, step=0.05, value=0.6, label="Overlay alpha", show_value=True
    )

    mo.vstack(
        [
            mo.hstack([sweep_vcp, sweep_moment, sweep_alpha], widths=[4, 2, 3]),
            sweep_nums,
        ],
        align="stretch",
    )
    return sweep_alpha, sweep_moment, sweep_nums, sweep_vcp


@app.cell
def _(mo, rs_dt, sweep_alpha, sweep_moment, sweep_nums, sweep_vcp, viz):
    mo.stop(not sweep_nums.value, mo.md("*Pick at least one sweep.*"))

    sweep_fig, _ax = viz.plot_sweeps(
        rs_dt,
        sweep_nums.value,
        moment_name=sweep_moment.value,
        vcp_num=sweep_vcp.value,
        alpha=float(sweep_alpha.value),
    )
    sweep_fig
    return


@app.cell
def _(mo, rs_dt, viz):
    mo.md("""### Waterfall — `viz.plot_waterfall`""")

    _n_returns = int(rs_dt["returns"].sizes["return_time"])
    wf_moment = mo.ui.dropdown(options=viz.available_moments(rs_dt), value="DBZH", label="Moment")
    wf_reduce = mo.ui.dropdown(options=["max", "mean"], value="max", label="Downsample")
    wf_offset = mo.ui.slider(
        start=0, stop=max(1, _n_returns - 1), step=64, value=0, label="Row offset", show_value=True
    )
    # Zoom in below a couple thousand rows and the red radial ticks separate.
    wf_rows = mo.ui.slider(
        start=512,
        stop=_n_returns,
        step=512,
        value=min(5120*2, _n_returns),
        label="Rows in window",
        show_value=True,
    )

    mo.vstack(
        [
            mo.hstack([wf_moment, wf_reduce], widths=[2, 2]),
            mo.hstack([wf_offset, wf_rows], widths=[1, 1]),
        ],
        align="stretch",
    )
    return wf_moment, wf_offset, wf_reduce, wf_rows


@app.cell
def _(rs_dt, viz, wf_moment, wf_offset, wf_reduce, wf_rows):
    wf_fig, _ax = viz.plot_waterfall(
        rs_dt,
        moment_name=wf_moment.value,
        row_offset=int(wf_offset.value),
        row_count=int(wf_rows.value),
        reduce=wf_reduce.value,
    )
    wf_fig
    return


@app.cell
def _(mo, rs_dt, viz):
    mo.md("""### Geographic point cloud — `viz.plot_geo`""")

    geo_vcp = mo.ui.dropdown(
        options={"all volumes": None} | {i.label: i.index for i in viz.vcp_infos(rs_dt)},
        value="all volumes",
        label="VCP",
    )
    geo_moment = mo.ui.dropdown(options=viz.available_moments(rs_dt), value="DBZH", label="Moment")
    # Without a floor the weak-return haze fills the map and hides the storm.
    geo_min = mo.ui.number(value=20.0, label="Min value")
    # deck.gl data ships as JSON, at roughly 90 bytes a point.
    geo_points = mo.ui.slider(
        start=5_000, stop=60_000, step=5_000, value=30_000, label="Max points", show_value=True
    )
    geo_size = mo.ui.slider(
        start=1, stop=30, step=1, value=1, label="Point size (m)", show_value=True
    )

    mo.vstack(
        [
            mo.hstack([geo_vcp, geo_moment, geo_min], widths=[4, 2, 2]),
            mo.hstack([geo_points, geo_size], widths=[1, 1]),
        ],
        align="stretch",
    )
    return geo_min, geo_moment, geo_points, geo_size, geo_vcp


@app.cell
def _(geo_min, geo_moment, geo_points, geo_size, geo_vcp, mo, rs_dt, viz):
    with mo.status.spinner("Geolocating gates..."):
        geo_deck = viz.plot_geo(
            rs_dt,
            moment_name=geo_moment.value,
            vcp_num=geo_vcp.value,
            min_value=float(geo_min.value),
            max_points=int(geo_points.value),
            point_size=float(geo_size.value),
        )
    geo_deck
    return


if __name__ == "__main__":
    app.run()
