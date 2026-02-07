import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    import datetime as dt

    import marimo as mo

    import radrs
    import radrs.quicklook as ql
    import radrs.raystack as rrs
    return dt, mo, ql, radrs, rrs


@app.cell
def _(mo):
    mo.md("""
    # Raystack Quicklook

    Ray-centric viewer with three modes:
    - **Ray 3D**: one endpoint per return ray (sweep-independent)
    - **Gate cloud 3D**: all finite gates in a rotating 3D projection
    - **Sweep polar**: single sweep polar view for focused inspection

    Volume controls: drag rotate, wheel zoom, double-click reset.

    **Sample cap** means deterministic downsampling of rendered items.
    """)
    return


@app.cell
def _(dt, mo):
    station_input = mo.ui.text(value="KABR", label="Station (4-letter ICAO)")
    base_uri = mo.ui.dropdown(
        options=["s3://unidata-nexrad-level2", "s3://noaa-nexrad-level2"],
        value="s3://unidata-nexrad-level2",
        label="Archive base",
    )
    start_time = mo.ui.datetime(
        value=dt.datetime(2024, 7, 2, 0, 0, 0),
        precision="minute",
        label="Start (UTC)",
    )
    end_time = mo.ui.datetime(
        value=dt.datetime(2024, 7, 2, 0, 40, 0),
        precision="minute",
        label="End (UTC)",
    )
    max_volumes = mo.ui.slider(
        start=1,
        stop=50,
        step=1,
        value=20,
        label="Max listed volumes",
        show_value=True,
    )

    mo.vstack(
        [
            mo.hstack([station_input, base_uri], widths=[2, 6]),
            mo.hstack([start_time, end_time, max_volumes], widths=[3, 3, 2]),
        ],
        align="stretch",
    )
    return base_uri, end_time, max_volumes, start_time, station_input


@app.cell
def _(base_uri, dt, end_time, max_volumes, mo, radrs, start_time, station_input):
    station = station_input.value.strip().upper()
    mo.stop(len(station) != 4, mo.md("*Station must be a 4-letter ICAO code (e.g., KABR).*"))
    mo.stop(start_time.value is None or end_time.value is None, mo.md("*Start/end time required.*"))
    mo.stop(start_time.value > end_time.value, mo.md("*Start time must be before end time.*"))

    start_utc = start_time.value.replace(tzinfo=dt.timezone.utc)
    end_utc = end_time.value.replace(tzinfo=dt.timezone.utc)

    with mo.status.spinner("Listing archive volumes..."):
        infos = radrs.list_nexrad_l2_archive_volumes(
            base_uri=base_uri.value,
            start_time=start_utc,
            end_time=end_utc,
            storage_options={"anon": "true"},
            site_filter=[station],
        )

    infos = sorted(infos, key=lambda item: item.vcp_time)[: int(max_volumes.value)]
    return end_utc, infos, start_utc, station


@app.cell
def _(base_uri, end_utc, infos, mo, start_utc, station):
    if len(infos) == 0:
        mo.stop(
            True,
            mo.md(
                f"*No volumes found for `{station}` between `{start_utc}` and `{end_utc}` from `{base_uri.value}`.*"
            ),
        )

    options: dict[str, str] = {}
    for idx, info in enumerate(infos):
        uri = str(info.uri)
        source_url = f"{base_uri.value.rstrip('/')}/{uri.lstrip('/')}"
        ts = info.vcp_time.strftime("%Y-%m-%d %H:%M:%S")
        size_mb = info.size / 1_000_000.0
        label = f"{idx:02d} | {ts} | {size_mb:6.1f} MB | {uri.split('/')[-1]}"
        options[label] = source_url

    volume_selector = mo.ui.dropdown(
        options=options,
        value=next(iter(options.keys())),
        label="Volume",
        searchable=True,
        full_width=True,
    )
    volume_selector
    return (volume_selector,)


@app.cell
def _(mo, ql, rrs, volume_selector):
    with mo.status.spinner("Loading selected volume..."):
        dtree = rrs.open_datatree(volume_selector.value, include_activity=False)
        returns, sweeps = ql.get_returns_and_sweeps(dtree)
    return returns, sweeps


@app.cell
def _(mo, returns, sweeps, volume_selector):
    n_returns = int(returns.sizes.get("return_time", 0))
    n_sweeps = int(sweeps.sizes.get("sweep_time", 0))
    fold_size = int(returns.sizes.get("range", 0))
    mo.md(
        f"""
    **Source:** `{volume_selector.value}`

    | Metric | Value |
    |---|---:|
    | returns | {n_returns:,} |
    | sweeps | {n_sweeps:,} |
    | fold_size | {fold_size:,} |
    | total cells | {n_returns * fold_size:,} |
    """
    )
    return


@app.cell
def _(mo, ql, returns, sweeps):
    mode_selector = mo.ui.dropdown(
        options=["Ray 3D", "Gate cloud 3D", "Sweep polar"],
        value="Ray 3D",
        label="Mode",
    )

    moment_names = ql.available_moments(returns, include_qc=True)
    mo.stop(len(moment_names) == 0, mo.md("*No moment fields in returns dataset.*"))
    moment_selector = mo.ui.dropdown(
        options=moment_names,
        value=moment_names[0],
        label="Moment",
    )

    sample_cap = mo.ui.slider(
        start=25_000,
        stop=300_000,
        step=25_000,
        value=125_000,
        label="Sample cap (items)",
        show_value=True,
    )
    canvas_size = mo.ui.slider(
        start=500,
        stop=1100,
        step=20,
        value=820,
        label="Canvas size",
        show_value=True,
    )

    sweep_info = ql.sweep_infos(sweeps)
    mo.stop(len(sweep_info) == 0, mo.md("*No sweeps available in this file.*"))
    sweep_selector = mo.ui.dropdown(
        options={info.label: info.index for info in sweep_info},
        value=sweep_info[0].label,
        label="Sweep (sweep mode)",
        searchable=True,
        full_width=True,
    )

    mo.vstack(
        [
            mo.hstack([mode_selector, moment_selector], widths=[2, 2]),
            mo.hstack([sample_cap, canvas_size], widths=[4, 3]),
            sweep_selector,
        ],
        align="stretch",
    )
    return canvas_size, mode_selector, moment_selector, sample_cap, sweep_selector


@app.cell
def _(
    canvas_size,
    mo,
    mode_selector,
    moment_selector,
    ql,
    returns,
    sample_cap,
    sweep_selector,
    sweeps,
):
    mode = str(mode_selector.value)
    moment = str(moment_selector.value)
    requested_points = int(sample_cap.value)
    # Keep widget output under marimo's default byte limit.
    if mode == "Gate cloud 3D":
        max_points = min(requested_points, 180_000)
    elif mode == "Ray 3D":
        max_points = min(requested_points, 180_000)
    else:
        max_points = min(requested_points, 180_000)

    try:
        if mode == "Sweep polar":
            payload = ql.prepare_polar_payload(
                returns=returns,
                sweeps=sweeps,
                sweep_index=int(sweep_selector.value),
                moment=moment,
                max_points=max_points,
            )
            widget = ql.QuicklookPolarWidget(
                width=int(canvas_size.value),
                height=int(canvas_size.value),
            )
            widget.set_payload(payload)
        elif mode == "Ray 3D":
            payload = ql.prepare_ray_payload(
                returns=returns,
                moment=moment,
                max_points=max_points,
            )
            widget = ql.QuicklookVolumeWidget(
                width=int(canvas_size.value),
                height=int(canvas_size.value),
            )
            widget.set_payload(payload)
        else:
            payload = ql.prepare_volume_payload(
                returns=returns,
                moment=moment,
                max_points=max_points,
            )
            widget = ql.QuicklookVolumeWidget(
                width=int(canvas_size.value),
                height=int(canvas_size.value),
            )
            widget.set_payload(payload)
    except ImportError as exc:
        mo.stop(
            True,
            mo.md(
                f"*Quicklook widget dependencies are missing: `{exc}`. "
                "Install `anywidget` + `traitlets` in the environment.*"
            ),
        )
        raise RuntimeError("unreachable")

    widget_ui = mo.ui.anywidget(widget)
    widget_ui
    return max_points, mode, payload, requested_points, widget_ui


@app.cell
def _(max_points, mo, mode, payload, requested_points, widget_ui):
    hover = widget_ui.hover if isinstance(widget_ui.hover, dict) else {}

    if mode == "Sweep polar":
        header = (
            f"`mode={mode}`  `points={payload.point_count:,}`  "
            f"`moment={payload.moment}`  `max_range={payload.max_range_m / 1000.0:.2f} km`"
        )
    else:
        render_mode = getattr(payload, "render_mode", "points")
        item_label = "rays" if render_mode == "rays" else "gates"
        header = (
            f"`mode={mode}`  `{item_label}={payload.point_count:,}`  "
            f"`moment={payload.moment}`  `max_abs={payload.max_abs_m / 1000.0:.2f} km`"
        )
    cap_note = (
        f"`requested_cap={requested_points:,}`  `effective_cap={max_points:,}`"
        if requested_points != max_points
        else f"`cap={max_points:,}`"
    )

    if hover:
        hover_block = (
            f"hover idx={hover.get('index', -1)} "
            f"value={hover.get('value', float('nan')):.2f} "
            f"az={hover.get('azimuth_deg', float('nan')):.2f}deg "
            f"el={hover.get('elevation_deg', float('nan')):.2f}deg "
            f"range={hover.get('range_m', 0.0) / 1000.0:.2f}km "
            f"ret={hover.get('return_index', -1)} gate={hover.get('gate_index', -1)}"
        )
    else:
        hover_block = "hover: move cursor over a point"

    mo.md(
        f"""
    {header}
    
    {cap_note}

    {hover_block}
    """
    )
    return


if __name__ == "__main__":
    app.run()
