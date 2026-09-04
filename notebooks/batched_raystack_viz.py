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
    # Batched Raystack Viz

    Multi-volume viewer: a whole time range is accumulated into one
    `BatchedRaystack` and rendered as a single returns matrix.

    - **Waterfall**: raw (return_time, range) matrix, so volumes stack along the time axis
    - **Gate cloud 3D**: every finite gate in the batch, overlaid in radar-relative space

    3D: drag rotate, wheel zoom, double-click reset.
    2D: hover for coordinates and values.

    **Sample cap** applies to the gate cloud (deterministic downsampling).
    The waterfall defaults to a canvas tall enough to label every sweep — expect to scroll.
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
        value=dt.datetime(2024, 7, 2, 2, 0, 0),
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
        value=256,
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

    # info.vcp_time is already timezone-aware UTC. Normalize defensively so the
    # notebook also runs against older radrs builds that returned naive local.
    def to_utc(value):
        return value.astimezone(dt.timezone.utc)

    infos = sorted(infos, key=lambda item: item.vcp_time)[: int(max_volumes.value)]
    infos
    return end_utc, infos, start_utc, station, to_utc


@app.cell
def _(base_uri, end_utc, infos, mo, start_utc, station, to_utc):
    if len(infos) == 0:
        mo.stop(
            True,
            mo.md(
                f"*No volumes found for `{station}` between `{start_utc}` and `{end_utc}` from `{base_uri.value}`.*"
            ),
        )

    listing = "\n".join(
        f"    {idx:02d} | {to_utc(info.vcp_time).strftime('%Y-%m-%d %H:%M:%S')} | "
        f"{info.size / 1_000_000.0:6.1f} MB | {str(info.uri).split('/')[-1]}"
        for idx, info in enumerate(infos)
    )

    volume_range = mo.ui.range_slider(
        start=0,
        stop=max(1, len(infos) - 1),
        step=1,
        value=[0, min(3, len(infos) - 1)],
        label="Volume range (index)",
        show_value=True,
        full_width=True,
    )
    prefetch = mo.ui.slider(
        start=1,
        stop=16,
        step=1,
        value=8,
        label="Prefetch",
        show_value=True,
    )
    # Capacity is pre-allocated up front, so overshoot costs memory and undershoot drops volumes.
    headroom = mo.ui.slider(
        start=1.0,
        stop=3.0,
        step=0.1,
        value=1.4,
        label="Capacity headroom",
        show_value=True,
    )
    # All-NaN returns are ~46% of a volume at fold 256 (split cuts leave a moment
    # absent for whole sweeps, and trailing folds sit past the last gate).
    drop_empty = mo.ui.checkbox(value=True, label="Drop all-NaN returns (compaction)")
    load_button = mo.ui.run_button(label="Load batch", kind="success", full_width=True)

    mo.vstack(
        [
            mo.md(f"```\n{listing}\n```"),
            volume_range,
            mo.hstack([prefetch, headroom, drop_empty, load_button], widths=[2, 2, 2, 3]),
        ],
        align="stretch",
    )
    return drop_empty, headroom, load_button, prefetch, volume_range


@app.cell
def _(
    base_uri,
    drop_empty,
    dt,
    fold_size_input,
    headroom,
    infos,
    load_button,
    mo,
    prefetch,
    radrs,
    rrs,
    station,
    viz,
    volume_range,
):
    mo.stop(not load_button.value, mo.md("*Pick a volume range, then press **Load batch**.*"))

    _lo, _hi = (int(v) for v in volume_range.value)
    selected = infos[_lo : min(_hi, len(infos) - 1) + 1]
    fold_size = int(fold_size_input.value)

    # Archive bounds are [start, end), so pad past the last volume's start time.
    batch_start = selected[0].vcp_time
    batch_end = selected[-1].vcp_time + dt.timedelta(seconds=1)

    # A volume holds ~21M gate cells however it is folded, so returns scale as cells/fold_size.
    returns_per_vcp = int(21_000_000 / fold_size * float(headroom.value))

    batch = rrs.BatchedRaystack(
        max_vcps=len(selected),
        max_sweeps=len(selected) * 32,
        max_returns=len(selected) * returns_per_vcp,
        fold_size=fold_size,
        truncate=True,
        drop_empty_returns=bool(drop_empty.value),
    )

    with mo.status.spinner(f"Fetching {len(selected)} volumes..."):
        n_added = batch.add_volumes_from_l2(
            radrs.NexradL2ArchiveIter(
                base_uri.value,
                start_time=batch_start,
                end_time=batch_end,
                storage_options={"anon": "true"},
                site_filter=[station],
            ),
            prefetch=int(prefetch.value),
        )
        # progress() reports fill against capacity, so read it before finalize trims.
        progress = batch.progress()
        rs_dt = batch.finalize_to_rs_dt()
        returns, sweeps = viz.get_returns_and_sweeps(rs_dt)

    mo.stop(n_added == 0, mo.md("*No volumes were added — check the range and try again.*"))
    (rs_dt, returns["DBZH"])
    return fold_size, n_added, progress, returns, selected, sweeps


@app.cell
def _(
    drop_empty,
    fold_size,
    mo,
    n_added,
    progress,
    returns,
    selected,
    sweeps,
    to_utc,
    viz,
):
    n_returns = int(returns.sizes.get("return_time", 0))
    n_sweeps = int(sweeps.sizes.get("sweep_time", 0))

    # Share of NaN cells across every moment. With compaction on, all-NaN rows are
    # already gone, so what is left is emptiness *within* surviving returns: split
    # cuts leave a whole moment absent on the sweeps that never recorded it.
    _moments = viz.available_moments(returns, include_qc=False)
    _cells = len(_moments) * n_returns * fold_size
    _nans = sum(int(returns[m].isnull().sum()) for m in _moments)
    return_nans = 100.0 * _nans / max(1, _cells)

    # Capacity is pre-allocated, so spare capacity is slack we never needed — more is
    # safer. Reported as spare rather than used so the polarity reads the right way:
    # 0% spare is the bad case, because that is where volumes start being dropped.
    _filled = int(progress["returns_filled"])
    _capacity = int(progress["returns_capacity"])
    spare = 100.0 * (1.0 - _filled / max(1, _capacity))

    # batch.rs reserves n_radials * n_folds before compaction runs, but returns_filled
    # counts what survived it, so spare reads high when dropping is on (~1.8x at fold 256).
    capacity_caveat = (
        "*Capacity is reserved against uncompacted returns, so usable spare is lower than shown.*"
        if bool(drop_empty.value)
        else ""
    )

    # add_volumes_from_l2 stops quietly once capacity runs out.
    # Keep the indent — an unindented line here would break mo.md's dedent.
    shortfall = (
        f"*Only {n_added} of {len(selected)} volumes fit — raise capacity headroom.*"
        if n_added < len(selected)
        else ""
    )

    mo.md(
        f"""
    **Batch:** `{to_utc(selected[0].vcp_time):%Y-%m-%d %H:%M:%S}` → `{to_utc(selected[-1].vcp_time):%Y-%m-%d %H:%M:%S}` UTC

    {shortfall}

    | Metric | Value |
    |---|---:|
    | volumes | {n_added:,} / {len(selected):,} |
    | sweeps | {n_sweeps:,} |
    | returns | {n_returns:,} |
    | return NaNs | {return_nans:.1f}% of {_cells:,} cells |
    | fold_size | {fold_size:,} |
    | total cells | {n_returns * fold_size:,} |
    | returns capacity | {_filled:,} / {_capacity:,} — {spare:.0f}% spare |

    {capacity_caveat}
    """
    )
    return


@app.cell
def _(mo, returns, sweeps, viz):
    mode_selector = mo.ui.dropdown(
        options=["Folded waterfall", "Gate cloud 3D"],
        value="Folded waterfall",
        label="Mode",
    )

    moment_names = viz.available_moments(returns, include_qc=True)
    mo.stop(len(moment_names) == 0, mo.md("*No moment fields in returns dataset.*"))
    moment_selector = mo.ui.dropdown(
        options=moment_names,
        value=moment_names[0],
        label="Moment",
    )

    # The waterfall's y-axis is return index, not time, and sweeps differ ~16x in return
    # count (360..5760 here), so the *smallest* sweep sets label spacing: its share of the
    # axis must be ~12 px wide for its "S<n>" label to clear the next one. 68 px covers the
    # top/bottom margins. Batches get very tall as a result — that is the point, scroll it.
    _min_share = float(sweeps["num_returns"].min()) / max(1.0, float(sweeps["num_returns"].sum()))
    _wanted_h = 12.0 / _min_share + 68.0
    _suggested_h = min(16000, max(600, round(_wanted_h / 100) * 100))

    # Past the cap the tightest sweeps still collide; narrowing the volume range is the fix.
    wf_height_note = (
        f"*Sweeps this dense need a ~{int(_wanted_h):,} px canvas to label cleanly — "
        f"capped at 16,000. Narrow the volume range to separate them.*"
        if _wanted_h > 16000
        else ""
    )

    canvas_width = mo.ui.slider(
        start=500,
        stop=2000,
        step=20,
        value=900,
        label="Canvas width",
        show_value=True,
    )
    canvas_height = mo.ui.slider(
        start=500,
        stop=16000,
        step=100,
        value=_suggested_h,
        label="Canvas height (waterfall)",
        show_value=True,
    )

    # Gate cloud controls
    sample_cap = mo.ui.slider(
        start=25_000,
        stop=300_000,
        step=25_000,
        value=125_000,
        label="Sample cap (items)",
        show_value=True,
    )

    # Waterfall controls — the batch is many volumes deep, so allow more rows than a single volume.
    # Rows resample to the canvas height, so there is no gain past roughly that many.
    wf_max_returns = mo.ui.slider(
        start=256,
        stop=4096,
        step=256,
        value=2048,
        label="Max returns",
        show_value=True,
    )
    wf_max_range = mo.ui.slider(
        start=128,
        stop=1024,
        step=128,
        value=256,
        label="Max range bins",
        show_value=True,
    )

    # Folded-waterfall controls. Rows are drawn verbatim, so the window is bounded by
    # payload size rather than pixels: fold_size * 4 B of grid plus ~40 B of per-row
    # metadata, kept under ~3 MB (marimo's output limit is 5 MB).
    _n_rows_total = int(returns.sizes.get("return_time", 0))
    _fold = int(returns.sizes.get("range", 1)) or 1
    _budget_rows = max(256, int(3_000_000 / (_fold * 4 + 40)))
    fw_row_count = mo.ui.slider(
        start=256,
        stop=max(512, min(16384, _budget_rows)),
        step=256,
        value=max(256, min(_budget_rows, _n_rows_total)),
        label="Rows in window",
        show_value=True,
    )
    fw_row_offset = mo.ui.slider(
        start=0,
        stop=max(1, _n_rows_total - 1),
        step=64,
        value=0,
        label="Row offset",
        show_value=True,
    )
    return (
        canvas_height,
        canvas_width,
        fw_row_count,
        fw_row_offset,
        mode_selector,
        moment_selector,
        sample_cap,
        wf_height_note,
        wf_max_range,
        wf_max_returns,
    )


@app.cell
def _(
    canvas_height,
    canvas_width,
    fw_row_count,
    fw_row_offset,
    mo,
    mode_selector,
    moment_selector,
    sample_cap,
    wf_height_note,
    wf_max_range,
    wf_max_returns,
):
    rows = [mo.hstack([mode_selector, moment_selector], widths=[2, 2])]
    if str(mode_selector.value) == "Folded waterfall":
        # Height is driven by the row count (1 px per row), so no height slider here.
        rows.append(mo.hstack([fw_row_offset, fw_row_count, canvas_width], widths=[3, 3, 2]))
    elif str(mode_selector.value) == "Waterfall":
        rows.append(
            mo.hstack(
                [wf_max_returns, wf_max_range, canvas_width, canvas_height],
                widths=[2, 2, 2, 2],
            )
        )
        if wf_height_note:
            rows.append(mo.md(wf_height_note))
    else:
        # The 3D projection assumes a square viewport, so height is width there.
        rows.append(mo.hstack([sample_cap, canvas_width], widths=[4, 3]))

    mo.vstack(rows, align="stretch")
    return


@app.cell
def _(
    fw_row_count,
    fw_row_offset,
    mo,
    mode_selector,
    moment_selector,
    returns,
    sample_cap,
    viz,
):
    mode = str(mode_selector.value)
    moment = str(moment_selector.value)
    requested_points = int(sample_cap.value)
    # Keep widget output under marimo's default byte limit.
    max_points = min(requested_points, 180_000)

    try:
        if mode == "Folded waterfall":
            payload = viz.prepare_folded_waterfall_payload(
                returns,
                moment,
                row_offset=int(fw_row_offset.value),
                row_count=int(fw_row_count.value),
            )
        else:
            payload = viz.prepare_volume_payload(
                returns=returns,
                moment=moment,
                max_points=max_points,
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
    return max_points, mode, payload, requested_points


@app.cell
def _(canvas_height, canvas_width, mo, payload, viz):
    width = int(canvas_width.value)
    if isinstance(payload, viz.FoldedWaterfallPayload):
        # One pixel per row, so every fold is drawn; 68 px covers the margins.
        widget = viz.FoldedWaterfallWidget(width=width, height=payload.n_rows + 68)
    elif isinstance(payload, viz.WaterfallPayload):
        widget = viz.WaterfallWidget(width=width, height=int(canvas_height.value))
    else:
        widget = viz.VolumeWidget(width=width, height=width)
    widget.set_payload(payload)
    widget_ui = mo.ui.anywidget(widget)
    widget_ui   
    return (widget_ui,)


@app.cell
def _(max_points, mo, mode, payload, requested_points, viz, widget_ui):
    hover = widget_ui.hover if isinstance(widget_ui.hover, dict) else {}

    if isinstance(payload, viz.FoldedWaterfallPayload):
        header = (
            f"`mode={mode}`  `rows={payload.row_offset}-{payload.row_offset + payload.n_rows - 1}"
            f" of {payload.row_total:,}`  `fold_sets={payload.n_groups:,}`  "
            f"`moment={payload.moment}`"
        )
        cap_note = ""
        hover_block = (
            f"hover value={hover.get('value', float('nan')):.2f} "
            f"vcp={hover.get('vcp_index', -1)} sweep={hover.get('sweep_number', -1)} "
            f"fold={hover.get('fold_index', -1)} gate={hover.get('gate_index', -1)} "
            f"range={hover.get('range_km', 0.0):.2f}km "
            f"az={hover.get('azimuth_deg', float('nan')):.2f}deg "
            f"time={hover.get('time_utc', '')}"
        )
    elif isinstance(payload, viz.WaterfallPayload):
        header = (
            f"`mode={mode}`  `grid={payload.n_returns}x{payload.n_range}`  "
            f"`moment={payload.moment}`  "
            f"`from {payload.n_returns_orig}x{payload.n_range_orig}`"
        )
        cap_note = ""
        hover_block = (
            f"hover value={hover.get('value', float('nan')):.2f} "
            f"ret={hover.get('return_index', -1)} gate={hover.get('gate_index', -1)} "
            f"az={hover.get('azimuth_deg', float('nan')):.2f}deg "
            f"el={hover.get('elevation_deg', float('nan')):.2f}deg "
            f"sweep={hover.get('sweep_number', -1)} "
            f"range={hover.get('range_km', 0.0):.2f}km"
        )
    else:
        header = (
            f"`mode={mode}`  `gates={payload.point_count:,}`  "
            f"`moment={payload.moment}`  `max_abs={payload.max_abs_m / 1000.0:.2f} km`"
        )
        cap_note = (
            f"`requested_cap={requested_points:,}`  `effective_cap={max_points:,}`"
            if requested_points != max_points
            else f"`cap={max_points:,}`"
        )
        hover_block = (
            f"hover idx={hover.get('index', -1)} "
            f"value={hover.get('value', float('nan')):.2f} "
            f"az={hover.get('azimuth_deg', float('nan')):.2f}deg "
            f"el={hover.get('elevation_deg', float('nan')):.2f}deg "
            f"range={hover.get('range_m', 0.0) / 1000.0:.2f}km "
            f"ret={hover.get('return_index', -1)} gate={hover.get('gate_index', -1)}"
        )

    if not hover:
        hover_block = "hover: move cursor over a point"

    info_parts = [header]
    if cap_note:
        info_parts.append(cap_note)
    info_parts.append(hover_block)

    mo.md("\n\n".join(f"    {p}" for p in info_parts))
    return


if __name__ == "__main__":
    app.run()
