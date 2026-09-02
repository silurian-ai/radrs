import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    import datetime as dt

    import marimo as mo

    import radrs
    import radrs.viz as viz
    import radrs.raystack as rrs
    return dt, mo, radrs, rrs, viz


@app.cell
def _(mo):
    mo.md("""
    # Raystack Viz

    Volume viewer with multiple visualization modes:
    - **PPI**: one sweep in radar-native polar coordinates (default)
    - **Ray 3D**: one endpoint per return ray (sweep-independent)
    - **Gate cloud 3D**: all finite gates in a rotating 3D projection
    - **CAPPI**: constant-altitude horizontal slice through the volume
    - **Cross-section**: vertical slice along a target azimuth
    - **Waterfall**: raw (return_time, range) moment matrix as a 2D heatmap

    3D modes: drag rotate, wheel zoom, double-click reset.
    2D modes: hover for coordinates and values.

    Every payload sizes itself to `viz.DEFAULT_BYTE_BUDGET`, so the widget stays
    under marimo's output limit without any tuning here. **Sample cap** is a
    ceiling on top of that for the 3D modes, where cost is per point.
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
    fold_size_input = mo.ui.slider(
        start=128,
        stop=1920,
        step=64,
        value=1832,
        label="Fold size (range gates)",
        show_value=True,
    )

    mo.vstack(
        [
            mo.hstack([station_input, base_uri], widths=[2, 6]),
            mo.hstack([start_time, end_time, max_volumes, fold_size_input], widths=[3, 3, 1, 1]),
        ],
        align="stretch",
    )
    return (
        base_uri,
        end_time,
        fold_size_input,
        max_volumes,
        start_time,
        station_input,
    )


@app.cell
def _(
    base_uri,
    dt,
    end_time,
    max_volumes,
    mo,
    radrs,
    start_time,
    station_input,
):
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
    infos
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
def _(fold_size_input, mo, rrs, viz, volume_selector):
    with mo.status.spinner("Loading selected volume..."):
        dtree = rrs.open_datatree(
            volume_selector.value,
            fold_size=int(fold_size_input.value),
            include_activity=False,
        )
        returns, sweeps = viz.get_returns_and_sweeps(dtree)
    dtree
    return dtree, returns, sweeps


@app.cell
def _(dtree, mo, returns, sweeps, volume_selector):
    n_returns = int(returns.sizes.get("return_time", 0))
    n_sweeps = int(sweeps.sizes.get("sweep_time", 0))
    fold_size = int(returns.sizes.get("range", 0))

    _vcp_number = "N/A"
    try:
        if "vcps" in dtree.children and "vcp_number" in dtree["vcps"].dataset:
            _vcp_vals = dtree["vcps"].dataset["vcp_number"].values
            if _vcp_vals.size > 0:
                _vcp_number = str(int(_vcp_vals[0]))
        if _vcp_number == "N/A" and "volume_coverage_pattern" in dtree.attrs:
            _vcp_number = str(int(dtree.attrs["volume_coverage_pattern"]))
    except (KeyError, AttributeError, TypeError, ValueError):
        pass

    mo.md(
        f"""
    **Source:** `{volume_selector.value}`

    | Metric | Value |
    |---|---:|
    | VCP | {_vcp_number} |
    | returns | {n_returns:,} |
    | sweeps | {n_sweeps:,} |
    | fold_size | {fold_size:,} |
    | total cells | {n_returns * fold_size:,} |
    """
    )
    return


@app.cell
def _(mo, returns, sweeps, viz):
    mode_selector = mo.ui.dropdown(
        options=["PPI", "Ray 3D", "Gate cloud 3D", "CAPPI", "Cross-section", "Waterfall"],
        value="PPI",
        label="Mode",
    )

    moment_names = viz.available_moments(returns, include_qc=True)
    mo.stop(len(moment_names) == 0, mo.md("*No moment fields in returns dataset.*"))
    moment_selector = mo.ui.dropdown(
        options=moment_names,
        value=moment_names[0],
        label="Moment",
    )

    canvas_size = mo.ui.slider(
        start=500,
        stop=1100,
        step=20,
        value=820,
        label="Canvas size",
        show_value=True,
    )

    # PPI controls — one sweep at a time, so the sweep picker lives here.
    ppi_sweep_infos = viz.sweep_infos(sweeps)
    mo.stop(len(ppi_sweep_infos) == 0, mo.md("*No sweeps in the sweeps dataset.*"))
    ppi_sweep = mo.ui.dropdown(
        options={info.label: info.index for info in ppi_sweep_infos},
        value=ppi_sweep_infos[0].label,
        label="Sweep",
        searchable=True,
    )

    # 3D mode controls. The PPI has none: a whole sweep fits the byte budget,
    # so there is nothing left for a slider to express.
    sample_cap = mo.ui.slider(
        start=25_000,
        stop=300_000,
        step=25_000,
        value=125_000,
        label="Sample cap (items)",
        show_value=True,
    )

    # CAPPI controls
    cappi_altitude = mo.ui.slider(
        start=0.5,
        stop=15.0,
        step=0.1,
        value=2.0,
        label="Altitude (km)",
        show_value=True,
    )
    cappi_tolerance = mo.ui.slider(
        start=0.1,
        stop=3.0,
        step=0.1,
        value=0.5,
        label="Tolerance (km)",
        show_value=True,
    )
    grid_resolution = mo.ui.slider(
        start=200,
        stop=800,
        step=50,
        value=500,
        label="Grid resolution",
        show_value=True,
    )

    # Cross-section controls
    xsec_azimuth = mo.ui.slider(
        start=0,
        stop=359,
        step=1,
        value=0,
        label="Azimuth (deg)",
        show_value=True,
    )
    xsec_tolerance = mo.ui.slider(
        start=0.5,
        stop=5.0,
        step=0.5,
        value=2.0,
        label="Az tolerance (deg)",
        show_value=True,
    )

    # Waterfall controls
    wf_max_returns = mo.ui.slider(
        start=256,
        stop=2048,
        step=256,
        value=1024,
        label="Max returns",
        show_value=True,
    )
    wf_max_range = mo.ui.slider(
        start=256,
        stop=1024,
        step=128,
        value=512,
        label="Max range bins",
        show_value=True,
    )
    return (
        canvas_size,
        cappi_altitude,
        cappi_tolerance,
        grid_resolution,
        mode_selector,
        moment_selector,
        ppi_sweep,
        ppi_sweep_infos,
        sample_cap,
        wf_max_range,
        wf_max_returns,
        xsec_azimuth,
        xsec_tolerance,
    )


@app.cell
def _(
    canvas_size,
    cappi_altitude,
    cappi_tolerance,
    grid_resolution,
    mo,
    mode_selector,
    moment_selector,
    ppi_sweep,
    sample_cap,
    wf_max_range,
    wf_max_returns,
    xsec_azimuth,
    xsec_tolerance,
):
    _mode = str(mode_selector.value)
    if _mode == "PPI":
        controls_row = mo.hstack([ppi_sweep, canvas_size], widths=[6, 2])
    elif _mode in ("Ray 3D", "Gate cloud 3D"):
        controls_row = mo.hstack([sample_cap, canvas_size], widths=[4, 3])
    elif _mode == "CAPPI":
        controls_row = mo.hstack(
            [cappi_altitude, cappi_tolerance, grid_resolution, canvas_size],
            widths=[2, 2, 2, 2],
        )
    elif _mode == "Waterfall":
        controls_row = mo.hstack(
            [wf_max_returns, wf_max_range, canvas_size],
            widths=[3, 3, 2],
        )
    else:
        controls_row = mo.hstack(
            [xsec_azimuth, xsec_tolerance, grid_resolution, canvas_size],
            widths=[2, 2, 2, 2],
        )

    mo.vstack(
        [
            mo.hstack([mode_selector, moment_selector], widths=[2, 2]),
            controls_row,
        ],
        align="stretch",
    )
    return


@app.cell
def _(
    cappi_altitude,
    cappi_tolerance,
    grid_resolution,
    mo,
    mode_selector,
    moment_selector,
    ppi_sweep,
    ppi_sweep_infos,
    returns,
    sample_cap,
    sweeps,
    viz,
    wf_max_range,
    wf_max_returns,
    xsec_azimuth,
    xsec_tolerance,
):
    mode = str(mode_selector.value)
    moment = str(moment_selector.value)
    # A drawing ceiling for the 3D modes, not a transport limit: every payload
    # already sizes itself to viz.DEFAULT_BYTE_BUDGET.
    sample_ceiling = int(sample_cap.value)
    # PolarPayload carries no fixed elevation, so keep the selected sweep around.
    ppi_info = ppi_sweep_infos[int(ppi_sweep.value)]

    try:
        if mode == "PPI":
            payload = viz.prepare_polar_payload(
                returns=returns,
                sweeps=sweeps,
                sweep_index=ppi_info.index,
                moment=moment,
            )
        elif mode == "Ray 3D":
            payload = viz.prepare_ray_payload(
                returns=returns,
                moment=moment,
                max_points=sample_ceiling,
            )
        elif mode == "Gate cloud 3D":
            payload = viz.prepare_volume_payload(
                returns=returns,
                moment=moment,
                max_points=sample_ceiling,
            )
        elif mode == "CAPPI":
            payload = viz.prepare_cappi_payload(
                returns=returns,
                sweeps=sweeps,
                moment=moment,
                altitude_m=float(cappi_altitude.value) * 1000.0,
                tolerance_m=float(cappi_tolerance.value) * 1000.0,
                grid_size=int(grid_resolution.value),
            )
        elif mode == "Waterfall":
            payload = viz.prepare_waterfall_payload(
                returns=returns,
                moment=moment,
                max_returns=int(wf_max_returns.value),
                max_range=int(wf_max_range.value),
            )
        else:
            payload = viz.prepare_xsec_payload(
                returns=returns,
                sweeps=sweeps,
                moment=moment,
                azimuth_deg=float(xsec_azimuth.value),
                azimuth_tolerance_deg=float(xsec_tolerance.value),
                grid_size=int(grid_resolution.value),
            )
    except ImportError as exc:
        mo.stop(
            True,
            mo.md(
                f"*Viz widget dependencies are missing: `{exc}`. "
                "Install `anywidget` + `traitlets` in the environment.*"
            ),
        )
        raise RuntimeError("unreachable")
    return mode, payload, ppi_info, sample_ceiling


@app.cell
def _(canvas_size, mo, payload, viz):
    size = int(canvas_size.value)
    if isinstance(payload, viz.VolumePayload):
        widget = viz.VolumeWidget(width=size, height=size)
    elif isinstance(payload, viz.PolarPayload):
        widget = viz.PolarWidget(width=size, height=size)
    elif isinstance(payload, viz.WaterfallPayload):
        widget = viz.WaterfallWidget(width=size, height=size)
    else:
        widget = viz.GridWidget(width=size, height=size)
    widget.set_payload(payload)
    widget_ui = mo.ui.anywidget(widget)
    widget_ui
    return (widget_ui,)


@app.cell
def _(mo, mode, payload, ppi_info, sample_ceiling, viz, widget_ui):
    hover = widget_ui.hover if isinstance(widget_ui.hover, dict) else {}

    # The library decides how much to ship; this makes that decision visible.
    size_note = (
        f"`payload={payload.nbytes / 1e6:.2f} MB`  "
        f"`budget={viz.DEFAULT_BYTE_BUDGET / 1e6:.2f} MB`"
    )

    if isinstance(payload, viz.PolarPayload):
        header = (
            f"`mode={mode}`  `sweep={payload.sweep_number}`  "
            f"`elev={ppi_info.elevation_deg:.2f}deg`  "
            f"`grid={payload.n_returns}x{payload.n_gates}`  "
            f"`gates={payload.point_count:,}`  `moment={payload.moment}`  "
            f"`max_range={payload.max_range_m / 1000.0:.2f} km`"
        )
        if payload.return_stride > 1 or payload.gate_stride > 1:
            size_note += (
                f"  `stride={payload.return_stride}x{payload.gate_stride}"
                f" of {payload.n_returns_orig}x{payload.n_gates_orig}`"
            )
    elif isinstance(payload, viz.WaterfallPayload):
        header = (
            f"`mode={mode}`  `grid={payload.n_returns}x{payload.n_range}`  "
            f"`moment={payload.moment}`  "
            f"`from {payload.n_returns_orig}x{payload.n_range_orig}`"
        )
    elif isinstance(payload, viz.GridPayload):
        header = (
            f"`mode={mode}`  `grid={payload.n_rows}x{payload.n_cols}`  "
            f"`moment={payload.moment}`"
        )
        if payload.grid_mode == "cappi":
            header += f"  `alt={payload.cappi_altitude_m / 1000.0:.1f}km`  `tol={payload.cappi_tolerance_m / 1000.0:.1f}km`"
        else:
            header += f"  `az={payload.xsec_azimuth_deg:.1f}deg`"
    else:
        render_mode = getattr(payload, "render_mode", "points")
        item_label = "rays" if render_mode == "rays" else "gates"
        header = (
            f"`mode={mode}`  `{item_label}={payload.point_count:,}`  "
            f"`moment={payload.moment}`  `max_abs={payload.max_abs_m / 1000.0:.2f} km`"
        )
        size_note += f"  `cap={sample_ceiling:,}`"

    if hover:
        if isinstance(payload, viz.PolarPayload):
            hover_block = (
                f"hover idx={hover.get('index', -1)} "
                f"value={hover.get('value', float('nan')):.2f} "
                f"az={hover.get('azimuth_deg', float('nan')):.2f}deg "
                f"el={hover.get('elevation_deg', float('nan')):.2f}deg "
                f"range={hover.get('range_m', 0.0) / 1000.0:.2f}km "
                f"ret={hover.get('return_index', -1)} gate={hover.get('gate_index', -1)} "
                f"time={hover.get('return_time_iso', '')}"
            )
        elif isinstance(payload, viz.WaterfallPayload):
            hover_block = (
                f"hover value={hover.get('value', float('nan')):.2f} "
                f"ret={hover.get('return_index', -1)} gate={hover.get('gate_index', -1)} "
                f"az={hover.get('azimuth_deg', float('nan')):.2f}deg "
                f"el={hover.get('elevation_deg', float('nan')):.2f}deg "
                f"sweep={hover.get('sweep_number', -1)} "
                f"range={hover.get('range_km', 0.0):.2f}km"
            )
        elif isinstance(payload, viz.GridPayload):
            if payload.grid_mode == "cappi":
                hover_block = (
                    f"hover value={hover.get('value', float('nan')):.2f} "
                    f"east={hover.get('east_m', 0.0) / 1000.0:.2f}km "
                    f"north={hover.get('north_m', 0.0) / 1000.0:.2f}km "
                    f"range={hover.get('range_m', 0.0) / 1000.0:.2f}km"
                )
            else:
                hover_block = (
                    f"hover value={hover.get('value', float('nan')):.2f} "
                    f"ground_range={hover.get('ground_range_m', 0.0) / 1000.0:.2f}km "
                    f"altitude={hover.get('altitude_m', 0.0) / 1000.0:.2f}km"
                )
        else:
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

    info_parts = [header, size_note, hover_block]

    mo.md("\n\n".join(f"    {p}" for p in info_parts))
    return


if __name__ == "__main__":
    app.run()
