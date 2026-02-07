import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    import radrs.quicklook as ql
    import radrs.raystack as rrs
    return mo, ql, rrs


@app.cell
def _(mo):
    mo.md("""
    # Raystack Quicklook

    Daily-driver polar quicklook for raystack data:
    - sweep selector
    - moment selector
    - gate hover (value, azimuth, range, return index, time)
    """)
    return


@app.cell
def _(mo):
    source_input = mo.ui.text(
        value="s3://unidata-nexrad-level2/2024/07/02/KABR/KABR20240702_000016_V06",
        label="NEXRAD source (local file path or cloud URL)",
        full_width=True,
    )
    source_input
    return (source_input,)


@app.cell
def _(mo, source_input):
    mo.stop(not source_input.value, mo.md("*Enter a source above.*"))
    return


@app.cell
def _(mo, ql, rrs, source_input):
    with mo.status.spinner("Loading raystack DataTree..."):
        dt = rrs.open_datatree(source_input.value, include_activity=False)
        returns, sweeps = ql.get_returns_and_sweeps(dt)
    return returns, sweeps


@app.cell
def _(mo, returns, sweeps):
    n_returns = int(returns.sizes.get("return_time", 0))
    n_sweeps = int(sweeps.sizes.get("sweep_time", 0))
    fold_size = int(returns.sizes.get("range", 0))
    mo.md(
        f"""
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
    sweep_info = ql.sweep_infos(sweeps)
    moment_names = ql.available_moments(returns, include_qc=True)

    mo.stop(len(sweep_info) == 0, mo.md("*No sweeps available in this file.*"))
    mo.stop(len(moment_names) == 0, mo.md("*No moment fields available in returns dataset.*"))

    sweep_selector = mo.ui.dropdown(
        options={info.label: info.index for info in sweep_info},
        value=sweep_info[0].label,
        label="Sweep",
        full_width=True,
        searchable=True,
    )
    moment_selector = mo.ui.dropdown(
        options=moment_names,
        value=moment_names[0],
        label="Moment",
    )
    max_points = mo.ui.slider(
        start=25_000,
        stop=700_000,
        step=25_000,
        value=300_000,
        label="Max finite gates",
        show_value=True,
    )
    canvas_size = mo.ui.slider(
        start=500,
        stop=1100,
        step=20,
        value=780,
        label="Canvas size",
        show_value=True,
    )

    mo.vstack(
        [
            sweep_selector,
            mo.hstack([moment_selector, max_points, canvas_size], widths=[2, 3, 3]),
        ],
        align="stretch",
    )
    return canvas_size, max_points, moment_selector, sweep_selector


@app.cell
def _(
    canvas_size,
    max_points,
    mo,
    moment_selector,
    ql,
    returns,
    sweep_selector,
    sweeps,
):
    payload = ql.prepare_polar_payload(
        returns=returns,
        sweeps=sweeps,
        sweep_index=int(sweep_selector.value),
        moment=str(moment_selector.value),
        max_points=int(max_points.value),
    )

    try:
        widget = ql.QuicklookPolarWidget(width=int(canvas_size.value), height=int(canvas_size.value))
    except ImportError as exc:
        mo.stop(
            True,
            mo.md(
                f"*Quicklook widget dependencies are missing: `{exc}`. "
                "Install `anywidget` + `traitlets` in the environment.*"
            ),
        )
        raise RuntimeError("unreachable")

    widget.set_payload(payload)
    widget_ui = mo.ui.anywidget(widget)
    widget_ui
    return payload, widget_ui


@app.cell
def _(mo, payload, widget_ui):
    hover = widget_ui.hover if isinstance(widget_ui.hover, dict) else {}
    point_count = payload.point_count
    max_range_km = payload.max_range_m / 1000.0

    if hover:
        hover_block = (
            f"hover idx={hover.get('index', -1)} "
            f"value={hover.get('value', float('nan')):.2f} "
            f"az={hover.get('azimuth_deg', float('nan')):.2f}deg "
            f"range={hover.get('range_m', 0.0) / 1000.0:.2f}km "
            f"ret={hover.get('return_index', -1)} "
            f"gate={hover.get('gate_index', -1)}"
        )
    else:
        hover_block = "hover: move cursor over a gate"

    mo.md(
        f"""
        `points={point_count:,}`  `max_range={max_range_km:.2f} km`  `moment={payload.moment}`

        {hover_block}
        """
    )
    return


if __name__ == "__main__":
    app.run()
