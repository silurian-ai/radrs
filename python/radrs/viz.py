"""Raystack plots: PPI sweeps, unfolded waterfalls, and 3-D geographic point clouds.

Needs the ``viz`` extra: ``uv add 'radrs[viz]'``.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

import numpy as np
import xarray as xr

MOMENT_NAMES: Final[tuple[str, ...]] = (
    "DBZH",
    "VRADH",
    "WRADH",
    "ZDR",
    "PHIDP",
    "RHOHV",
    "CCORH",
)


# --------------------------------------------------------------------------
# Sweeps
# --------------------------------------------------------------------------


def plot_sweep(
    rs_dt: xr.DataTree,
    sweep_num: int,
    moment_name: str = "DBZH",
    vcp_num: int = 0,
    plot_kwargs: Mapping[str, Any] = {},
):
    """Plot one sweep as a PPI. See :func:`plot_sweeps` for the parameters."""

    return plot_sweeps(
        rs_dt, [sweep_num], moment_name=moment_name, vcp_num=vcp_num, plot_kwargs=plot_kwargs
    )


def plot_sweeps(
    rs_dt: xr.DataTree,
    sweep_nums: Sequence[int],
    moment_name: str = "DBZH",
    vcp_num: int = 0,
    *,
    cmaps: Sequence[Any] | None = None,
    alpha: float = 0.6,
    figsize: tuple[float, float] = (9.0, 8.0),
    plot_kwargs: Mapping[str, Any] = {},
):
    """Plot one or more sweeps as overlaid PPIs, in km from the radar.

    Sweeps are drawn in the order given, each over the last at ``alpha``
    transparency (the first is opaque) and each in its own colormap: yellow to
    green, then blue to purple, then orange to red, then matplotlib defaults.
    NaN gates are transparent, so lower sweeps show through the gaps.

    Parameters
    ----------
    rs_dt : xr.DataTree
        Raystack tree, from ``open_datatree`` or ``BatchedRaystack.finalize_to_rs_dt``.
    sweep_nums : sequence of int
        ``sweep_number`` values, as reported by :func:`sweep_infos`.
    moment_name : str, default "DBZH"
        Moment variable in ``returns``.
    vcp_num : int, default 0
        Index of the VCP in time order; 0 is the first volume in the raystack.
    cmaps : sequence, optional
        Per-sweep colormaps (names or ``Colormap``) replacing the defaults.
    alpha : float, default 0.6
        Transparency of every sweep after the first.
    plot_kwargs : mapping
        Extra arguments for ``Axes.pcolormesh``. ``vmin``/``vmax`` here pin the
        color scale across all sweeps instead of autoscaling each one.

    Returns
    -------
    (Figure, Axes)
    """

    import matplotlib.pyplot as plt

    sweep_nums = [int(n) for n in sweep_nums]
    if not sweep_nums:
        raise ValueError("sweep_nums is empty")

    vcp = vcp_infos(rs_dt)[vcp_num]
    infos = {info.sweep_number: info for info in sweep_infos(rs_dt, vcp_num)}
    returns = _node(rs_dt, "returns")
    _require_moment(returns, moment_name)

    kwargs = dict(plot_kwargs)
    pinned_vmin = kwargs.pop("vmin", None)
    pinned_vmax = kwargs.pop("vmax", None)
    kwargs.pop("cmap", None)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.set_aspect("equal")

    meshes = []
    for position, sweep_num in enumerate(sweep_nums):
        if sweep_num not in infos:
            raise KeyError(f"sweep {sweep_num} not in VCP {vcp.vcp_number} at index {vcp_num}")

        grid, azimuth_deg = _sweep_grid(returns, vcp.vcp_time, sweep_num, moment_name)
        if grid.values.size == 0:
            continue

        vmin, vmax = value_bounds(grid.values)
        vmin = float(pinned_vmin) if pinned_vmin is not None else vmin
        vmax = float(pinned_vmax) if pinned_vmax is not None else vmax

        cmap = _resolve_cmap(cmaps[position]) if cmaps is not None else _sweep_cmap(position)
        cmap.set_bad((0.0, 0.0, 0.0, 0.0))

        x_km, y_km = _ppi_mesh(azimuth_deg, grid.range_m, grid.range_step_m)
        meshes.append(
            (
                sweep_num,
                ax.pcolormesh(
                    x_km,
                    y_km,
                    np.ma.masked_invalid(grid.values),
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    alpha=1.0 if position == 0 else alpha,
                    shading="flat",
                    **kwargs,
                ),
            )
        )

    if not meshes:
        raise ValueError(f"no returns for sweeps {sweep_nums} in VCP index {vcp_num}")

    ax.axhline(0.0, color="#94a3b8", linewidth=0.5, zorder=0)
    ax.axvline(0.0, color="#94a3b8", linewidth=0.5, zorder=0)
    ax.plot(0.0, 0.0, marker="+", color="#0f172a", markersize=10, zorder=5)
    ax.set_xlabel("X distance from radar (km)")
    ax.set_ylabel("Y distance from radar (km)")

    # Colorbars stack on the right; past a handful they squeeze the plot flat.
    # Each colorbar lands left of the last, so add in reverse to read in sweep order.
    shown = meshes[:8]
    for sweep_num, mesh in reversed(shown):
        fig.colorbar(
            mesh,
            ax=ax,
            fraction=0.046 / len(shown),
            pad=0.02,
            label=f"sweep {sweep_num} ({infos[sweep_num].elevation_deg:.2f} deg)",
        )

    elevations = ", ".join(f"{infos[n].elevation_deg:.2f}" for n in sweep_nums)
    ax.set_title(
        f"VCP {vcp.vcp_number} | sweep {', '.join(str(n) for n in sweep_nums)} | {moment_name}\n"
        f"{vcp.instrument_name} at {_format_site(vcp)} | elevation {elevations} deg\n"
        f"{_format_time(vcp.vcp_time)} UTC",
        fontsize=10,
    )
    return fig, ax


#: Endpoint pairs for the first sweeps drawn; later ones fall back to these maps.
_SWEEP_COLOR_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("#fde047", "#15803d"),  # yellow -> green
    ("#93c5fd", "#6b21a8"),  # blue -> purple
    ("#fdba74", "#b91c1c"),  # orange -> red
)
_SWEEP_FALLBACK_CMAPS: Final[tuple[str, ...]] = ("viridis", "cividis", "magma", "cool", "winter")


def _sweep_cmap(position: int):
    from matplotlib.colors import LinearSegmentedColormap

    if position < len(_SWEEP_COLOR_PAIRS):
        return LinearSegmentedColormap.from_list(
            f"radrs_sweep{position}", _SWEEP_COLOR_PAIRS[position]
        )
    offset = position - len(_SWEEP_COLOR_PAIRS)
    return _resolve_cmap(_SWEEP_FALLBACK_CMAPS[offset % len(_SWEEP_FALLBACK_CMAPS)])


def _sweep_grid(
    returns: xr.Dataset, vcp_time: np.datetime64, sweep_num: int, moment_name: str
) -> tuple[_UnfoldedGrid, np.ndarray]:
    """Unfold one sweep into a ``(n_radials, n_gates)`` grid sorted by azimuth."""

    rows = np.flatnonzero(
        (np.asarray(returns["vcp_time"].values) == vcp_time)
        & (np.asarray(returns["sweep_number"].values).astype(np.int64) == sweep_num)
    )
    if rows.size == 0:
        return _UnfoldedGrid.empty(), np.empty(0)

    return_time = np.asarray(returns["return_time"].values[rows])
    radial_index, first_row = _radial_index(return_time)
    grid = _unfold(
        np.asarray(returns[moment_name].values[rows], dtype=np.float32),
        np.asarray(returns["base_range"].values[rows], dtype=np.float64),
        np.asarray(returns["range_step"].values[rows], dtype=np.float64),
        radial_index,
        int(radial_index.max()) + 1,
    )

    azimuth_deg = np.asarray(returns["azimuth"].values[rows], dtype=np.float64)[first_row]
    order = np.argsort(azimuth_deg)
    return grid.take_rows(order), azimuth_deg[order]


def _ppi_mesh(
    azimuth_deg: np.ndarray, range_m: np.ndarray, range_step_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Cell-corner mesh in km for ``pcolormesh``, from ray centers."""

    range_edges = np.r_[range_m - range_step_m / 2.0, range_m[-1] + range_step_m / 2.0] / 1000.0
    azimuth_rad = np.deg2rad(_edges(azimuth_deg))[:, None]
    # Radar convention: azimuth 0 is north, increasing clockwise.
    return range_edges * np.sin(azimuth_rad), range_edges * np.cos(azimuth_rad)


def _edges(centers: np.ndarray) -> np.ndarray:
    if centers.size == 1:
        return np.r_[centers[0] - 0.5, centers[0] + 0.5]
    deltas = np.diff(centers)
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = centers[:-1] + deltas / 2.0
    edges[0] = centers[0] - deltas[0] / 2.0
    edges[-1] = centers[-1] + deltas[-1] / 2.0
    return edges


# --------------------------------------------------------------------------
# Waterfall
# --------------------------------------------------------------------------


def plot_waterfall(
    rs_dt: xr.DataTree,
    moment_name: str = "DBZH",
    *,
    row_offset: int = 0,
    row_count: int | None = None,
    max_rows: int = 1500,
    max_cols: int = 1200,
    reduce: str = "max",
    figsize: tuple[float, float] = (12.0, 9.0),
    plot_kwargs: Mapping[str, Any] = {},
):
    """Plot every return against true range, stacked in time.

    One image row per return row, unfolded onto an absolute range axis, so each
    radial's folds read as a staircase and dotted vertical lines mark where each
    fold window begins. Black is range no fold ever covered; white is a gate
    that was sampled and came back NaN.

    Two strips run down the left, aligned with the rows: VCP number in tab20b
    (ruled at each volume boundary, since a batch from one site usually repeats
    a single VCP number) and sweep number in tab20c. Red ticks inside the axis
    mark each distinct return time, exposing radial boundaries and sparse folds.

    Parameters
    ----------
    rs_dt : xr.DataTree
        Raystack tree.
    moment_name : str, default "DBZH"
        Moment variable in ``returns``.
    row_offset, row_count : int, optional
        Window of return rows to draw. Zoom in far enough and the red radial
        ticks stop merging, which is the only way to read fold structure on a
        raystack of more than a couple thousand radials.
    max_rows, max_cols : int
        Rows and range bins are block-reduced to fit these caps; a raystack has
        far more returns than a figure has pixels.
    reduce : {"max", "mean"}, default "max"
        Reducer for that block-reduction. ``max`` keeps echoes crisp; ``mean``
        suits signed moments like VRADH.
    plot_kwargs : mapping
        Extra arguments for ``Axes.imshow`` (``vmin``, ``vmax``, ``cmap``, ...).

    Returns
    -------
    (Figure, Axes)
        The axes is the returns panel, not the side strips.
    """

    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch

    if reduce not in ("max", "mean"):
        raise ValueError(f"reduce must be 'max' or 'mean', got {reduce!r}")

    returns = _node(rs_dt, "returns")
    _require_moment(returns, moment_name)

    total_rows = int(returns.sizes["return_time"])
    if total_rows == 0:
        raise ValueError("raystack has no returns")
    start = min(max(0, row_offset), total_rows - 1)
    window = slice(start, total_rows if row_count is None else min(total_rows, start + row_count))

    moment = np.asarray(returns[moment_name].values[window], dtype=np.float32)
    n_rows = int(moment.shape[0])
    return_time = np.asarray(returns["return_time"].values[window])
    sweep_number = np.asarray(returns["sweep_number"].values[window]).astype(np.int64)
    vcp_time = np.asarray(returns["vcp_time"].values[window])

    grid = _unfold(
        moment,
        np.asarray(returns["base_range"].values[window], dtype=np.float64),
        np.asarray(returns["range_step"].values[window], dtype=np.float64),
        np.arange(n_rows),
        n_rows,
    )
    row_step = max(1, int(np.ceil(n_rows / max_rows)))
    col_step = max(1, int(np.ceil(grid.values.shape[1] / max_cols)))
    values = _block_reduce(grid.values, row_step, col_step, reduce)
    covered = _block_reduce_any(grid.covered, row_step, col_step)

    kwargs = dict(plot_kwargs)
    auto_min, auto_max = value_bounds(grid.values)
    vmin = float(kwargs.pop("vmin", auto_min))
    vmax = float(kwargs.pop("vmax", auto_max))

    cmap = _resolve_cmap(kwargs.pop("cmap", None) or _sweep_cmap(0))
    cmap.set_bad("white")
    norm = Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(np.ma.masked_invalid(values)))
    rgba[~covered] = (0.0, 0.0, 0.0, 1.0)

    fig, (ax_vcp, ax_sweep, ax) = plt.subplots(
        1,
        3,
        figsize=figsize,
        width_ratios=[0.022, 0.022, 1.0],
        sharey=True,
        constrained_layout=True,
    )

    n_plot_rows = rgba.shape[0]
    half_gate_km = grid.range_step_m / 2000.0
    ax.imshow(
        rgba,
        aspect="auto",
        origin="upper",
        extent=(
            float(grid.range_km[0] - half_gate_km),
            float(grid.range_km[-1] + half_gate_km),
            float(n_plot_rows) - 0.5,
            -0.5,
        ),
        interpolation="nearest",
        **kwargs,
    )
    ax.set_facecolor("black")
    ax.set_xlabel("range from radar (km)")
    for edge_m in grid.fold_edges_m:
        ax.axvline(edge_m / 1000.0, color="#64748b", linewidth=0.6, linestyle=":", alpha=0.9)

    # Past this many, radial ticks merge into a solid bar and stop saying anything.
    _, first_row = _radial_index(return_time)
    marks_drawn = first_row.size <= 2000
    if marks_drawn:
        ax.hlines(
            first_row / row_step - 0.5,
            0.0,
            0.012,
            transform=ax.get_yaxis_transform(),
            color="red",
            linewidth=0.6,
        )

    vcp_numbers = _vcp_numbers_per_return(rs_dt, vcp_time)
    _draw_row_strip(
        ax_vcp, np.array([_vcp_color(v)[:3] for v in vcp_numbers]), row_step, n_plot_rows, "VCP"
    )
    _draw_row_strip(
        ax_sweep,
        np.array([_sweep_color(s)[:3] for s in sweep_number]),
        row_step,
        n_plot_rows,
        "sweep",
    )
    for boundary in np.flatnonzero(np.r_[False, vcp_time[1:] != vcp_time[:-1]]):
        ax_vcp.axhline(boundary / row_step - 0.5, color="black", linewidth=1.0)

    ticks = np.linspace(0, n_plot_rows - 1, min(10, n_plot_rows)).astype(np.int64)
    ax_vcp.set_yticks(ticks)
    ax_vcp.set_yticklabels(
        [_format_time(return_time[min(n_rows - 1, int(t) * row_step)]) for t in ticks], fontsize=8
    )
    ax_vcp.set_ylabel(
        f"return time (rows {start:,}-{start + n_rows - 1:,} of {total_rows:,}, "
        + (f"{first_row.size:,} radials marked red)" if marks_drawn else "radial marks omitted)")
    )

    fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.035, pad=0.02, label=moment_name
    )
    fig.legend(
        handles=[
            Patch(facecolor=_vcp_color(v), label=f"VCP {v}")
            for v in sorted({int(v) for v in vcp_numbers})
        ],
        loc="outside lower left",
        ncol=6,
        fontsize=8,
        title="VCP",
        title_fontsize=8,
        frameon=False,
    )
    fig.legend(
        handles=[
            Patch(facecolor=_sweep_color(s), label=str(s))
            for s in sorted({int(s) for s in np.unique(sweep_number)})
        ],
        loc="outside lower right",
        ncol=12,
        fontsize=8,
        title="sweep number",
        title_fontsize=8,
        frameon=False,
    )
    fig.suptitle(
        f"Returns {_format_time(return_time[0])} to {_format_time(return_time[-1])} UTC | "
        f"{moment_name} | fold {grid.fold_size} gates",
        fontsize=11,
    )
    return fig, ax


def _vcp_color(vcp_number: int):
    return _resolve_cmap("tab20b")(int(vcp_number) % 20)


def _sweep_color(sweep_number: int):
    return _resolve_cmap("tab20c")(int(sweep_number) % 20)


def _vcp_numbers_per_return(rs_dt: xr.DataTree, vcp_time: np.ndarray) -> np.ndarray:
    """Map each return's ``vcp_time`` back to its VCP number."""

    vcps = _node(rs_dt, "vcps")
    order = np.argsort(np.asarray(vcps["vcp_time"].values))
    positions = np.searchsorted(np.asarray(vcps["vcp_time"].values)[order], vcp_time)
    numbers = np.asarray(vcps["vcp_number"].values).astype(np.int64)[order]
    return numbers[np.clip(positions, 0, order.size - 1)]


def _draw_row_strip(ax, colors: np.ndarray, row_step: int, n_plot_rows: int, label: str) -> None:
    picked = np.clip(np.arange(n_plot_rows) * row_step, 0, colors.shape[0] - 1)
    ax.imshow(colors[picked][:, None, :], aspect="auto", origin="upper", interpolation="nearest")
    ax.set_xticks([])
    ax.set_xlabel(label, fontsize=8, rotation=90)


def _block_reduce(values: np.ndarray, row_step: int, col_step: int, reduce: str) -> np.ndarray:
    if row_step == 1 and col_step == 1:
        return values
    blocks = _to_blocks(_pad_to_blocks(values, row_step, col_step, np.nan), row_step, col_step)
    with warnings.catch_warnings():
        # All-NaN blocks are expected: range no fold ever covered.
        warnings.simplefilter("ignore", RuntimeWarning)
        return (np.nanmean if reduce == "mean" else np.nanmax)(blocks, axis=(1, 3))


def _block_reduce_any(covered: np.ndarray, row_step: int, col_step: int) -> np.ndarray:
    if row_step == 1 and col_step == 1:
        return covered
    return _to_blocks(
        _pad_to_blocks(covered, row_step, col_step, False), row_step, col_step
    ).any(axis=(1, 3))


def _to_blocks(values: np.ndarray, row_step: int, col_step: int) -> np.ndarray:
    return values.reshape(
        values.shape[0] // row_step, row_step, values.shape[1] // col_step, col_step
    )


def _pad_to_blocks(values: np.ndarray, row_step: int, col_step: int, fill: Any) -> np.ndarray:
    pad = ((0, (-values.shape[0]) % row_step), (0, (-values.shape[1]) % col_step))
    return values if pad == ((0, 0), (0, 0)) else np.pad(values, pad, constant_values=fill)


# --------------------------------------------------------------------------
# Geographic point cloud
# --------------------------------------------------------------------------


def plot_geo(
    rs_dt: xr.DataTree,
    moment_name: str = "DBZH",
    *,
    vcp_num: int | None = None,
    min_value: float | None = None,
    max_points: int = 30_000,
    point_size: float = 30.0,
    opacity: float = 0.85,
    pitch: float = 50.0,
    cmap: Any = None,
    map_style: str | None = None,
    plot_kwargs: Mapping[str, Any] = {},
):
    """Render finite gates as a 3-D point cloud on a real-world map.

    Returns a ``pydeck.Deck``, which marimo and Jupyter display directly. The
    view rotates, so the vertical structure of the volume is legible from the
    side; colors match :func:`plot_sweep`'s default yellow-to-green.

    Parameters
    ----------
    rs_dt : xr.DataTree
        Raystack tree.
    moment_name : str, default "DBZH"
        Moment variable in ``returns``.
    vcp_num : int, optional
        Index of the VCP in time order. Default plots every return in the
        raystack, geolocated against the first VCP's site.
    min_value : float, optional
        Drop gates below this value. Without it a volume of weak returns fogs
        the map, so set it (20 dBZ or so for DBZH).
    max_points : int, default 30000
        Cap on rendered gates, bounding notebook output size — deck.gl data
        travels as JSON, at roughly 90 bytes a point. Excess gates are dropped
        by strided subsampling.
    point_size : float, default 30
        Point radius in metres.
    cmap : str or Colormap, optional
        Replaces the default yellow-to-green ramp.
    map_style : str, optional
        pydeck basemap style. Defaults to Carto dark, which needs no API token.
    plot_kwargs : mapping
        Extra arguments for ``pydeck.Deck``.

    Returns
    -------
    pydeck.Deck
    """

    import pydeck as pdk
    from matplotlib.colors import Normalize

    site = vcp_infos(rs_dt)[0 if vcp_num is None else vcp_num]
    points = gate_positions(
        rs_dt, moment_name, vcp_num=vcp_num, min_value=min_value, max_points=max_points
    )
    if points["value"].size == 0:
        raise ValueError(f"no finite {moment_name} gates left after filtering")

    ramp = _resolve_cmap(cmap) if cmap is not None else _sweep_cmap(0)
    vmin, vmax = value_bounds(points["value"])
    colors = (ramp(Normalize(vmin=vmin, vmax=vmax)(points["value"]))[:, :3] * 255).astype(np.uint8)

    # Short keys and rounded values: every point is serialized into the output.
    data = [
        {
            "p": [round(float(lon), 5), round(float(lat), 5), round(float(alt), 1)],
            "c": [int(r), int(g), int(b)],
            "v": round(float(val), 1),
        }
        for lon, lat, alt, val, (r, g, b) in zip(
            points["longitude"], points["latitude"], points["altitude"], points["value"], colors
        )
    ]

    return _compact_deck(
        layers=[
            pdk.Layer(
                "PointCloudLayer",
                data=data,
                get_position="p",
                get_color="c",
                get_normal=[0, 0, 1],
                point_size=point_size,
                opacity=opacity,
                pickable=True,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=[{"p": [site.longitude, site.latitude], "v": site.instrument_name}],
                get_position="p",
                get_fill_color=[255, 0, 0],
                get_radius=2000,
                pickable=True,
            ),
        ],
        initial_view_state=pdk.ViewState(
            latitude=site.latitude, longitude=site.longitude, zoom=6.5, pitch=pitch, bearing=0.0
        ),
        map_style=map_style or pdk.map_styles.CARTO_DARK,
        tooltip={"text": f"{moment_name} {{v}}"},
        **dict(plot_kwargs),
    )


def gate_positions(
    rs_dt: xr.DataTree,
    moment_name: str = "DBZH",
    *,
    vcp_num: int | None = None,
    min_value: float | None = None,
    max_points: int = 30_000,
) -> dict[str, np.ndarray]:
    """Geolocate finite gates to ``longitude``/``latitude``/``altitude``/``value`` arrays.

    Beam height uses the standard 4/3-earth-radius refraction model, so
    altitudes are above mean sea level rather than above the radar.
    """

    returns = _node(rs_dt, "returns")
    _require_moment(returns, moment_name)

    site = vcp_infos(rs_dt)[0 if vcp_num is None else vcp_num]
    if vcp_num is None:
        rows = np.arange(int(returns.sizes["return_time"]))
    else:
        rows = np.flatnonzero(np.asarray(returns["vcp_time"].values) == site.vcp_time)

    moment = np.asarray(returns[moment_name].values[rows], dtype=np.float32)
    base_range = np.asarray(returns["base_range"].values[rows], dtype=np.float64)
    range_step = np.asarray(returns["range_step"].values[rows], dtype=np.float64)
    gate_index = np.arange(moment.shape[1], dtype=np.float64)
    range_m = base_range[:, None] + gate_index[None, :] * range_step[:, None]

    keep = np.isfinite(moment)
    if min_value is not None:
        keep &= moment >= float(min_value)
    flat = np.flatnonzero(keep.reshape(-1))
    if flat.size > max_points:
        # Strided, so the same raystack always renders the same cloud.
        flat = flat[:: int(np.ceil(flat.size / max_points))]

    row_of = flat // moment.shape[1]
    ranges = range_m.reshape(-1)[flat]
    az_rad = np.deg2rad(np.asarray(returns["azimuth"].values[rows], dtype=np.float64)[row_of])
    el_rad = np.deg2rad(np.asarray(returns["elevation"].values[rows], dtype=np.float64)[row_of])

    earth = _EFFECTIVE_EARTH_RADIUS_M
    above_radar = np.sqrt(ranges**2 + earth**2 + 2.0 * ranges * earth * np.sin(el_rad)) - earth
    ground = earth * np.arcsin(ranges * np.cos(el_rad) / (earth + above_radar))

    return {
        "longitude": site.longitude
        + np.rad2deg(ground * np.sin(az_rad) / (earth * np.cos(np.deg2rad(site.latitude)))),
        "latitude": site.latitude + np.rad2deg(ground * np.cos(az_rad) / earth),
        "altitude": above_radar + site.altitude,
        "value": moment.reshape(-1)[flat].astype(np.float64),
    }


#: Mean earth radius scaled by 4/3, the standard beam-refraction approximation.
_EFFECTIVE_EARTH_RADIUS_M: Final[float] = 4.0 / 3.0 * 6_371_000.0


def _compact_deck(**kwargs):
    """Build a Deck whose JSON drops pydeck's hardcoded indent — a 2.5x smaller payload."""

    import pydeck as pdk
    from pydeck.bindings.json_tools import default_serialize

    class CompactDeck(pdk.Deck):
        def to_json(self):
            return json.dumps(
                self, sort_keys=True, default=default_serialize, separators=(",", ":")
            )

    return CompactDeck(**kwargs)


# --------------------------------------------------------------------------
# Selectors
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VcpInfo:
    """One volume coverage pattern in a raystack, for selector controls."""

    index: int
    vcp_number: int
    vcp_time: np.datetime64
    instrument_name: str
    latitude: float
    longitude: float
    altitude: float
    num_sweeps: int
    label: str


@dataclass(frozen=True)
class SweepInfo:
    """One sweep within a VCP, for selector controls."""

    index: int
    sweep_number: int
    elevation_deg: float
    num_returns: int
    label: str


def vcp_infos(rs_dt: xr.DataTree) -> list[VcpInfo]:
    """Describe every VCP in the raystack, ordered in time."""

    vcps = _node(rs_dt, "vcps")
    times = np.asarray(vcps["vcp_time"].values)
    sweep_vcp_times = np.asarray(_node(rs_dt, "sweeps")["vcp_time"].values)

    infos = []
    for index, pos in enumerate(np.argsort(times)):
        vcp_number = int(_at(vcps, "vcp_number", pos, 0))
        infos.append(
            VcpInfo(
                index=index,
                vcp_number=vcp_number,
                vcp_time=times[pos],
                instrument_name=str(_at(vcps, "instrument_name", pos, "")),
                latitude=float(_at(vcps, "latitude", pos, np.nan)),
                longitude=float(_at(vcps, "longitude", pos, np.nan)),
                altitude=float(_at(vcps, "altitude", pos, np.nan)),
                num_sweeps=int(np.count_nonzero(sweep_vcp_times == times[pos])),
                label=f"{index:02d} | VCP {vcp_number} | {_format_time(times[pos])}",
            )
        )
    return infos


def sweep_infos(rs_dt: xr.DataTree, vcp_num: int = 0) -> list[SweepInfo]:
    """Describe the sweeps of one VCP, ordered by sweep number.

    ``vcp_num`` indexes the VCPs in time order; 0 is the first volume.
    """

    sweeps = _node(rs_dt, "sweeps")
    mask = np.asarray(sweeps["vcp_time"].values) == vcp_infos(rs_dt)[vcp_num].vcp_time
    numbers = np.asarray(sweeps["sweep_number"].values)[mask].astype(np.int64)
    elevations = np.asarray(sweeps["elevation_angle"].values)[mask].astype(np.float64)
    counts = np.asarray(sweeps["num_returns"].values)[mask].astype(np.int64)

    return [
        SweepInfo(
            index=index,
            sweep_number=int(numbers[pos]),
            elevation_deg=float(elevations[pos]),
            num_returns=int(counts[pos]),
            label=(
                f"sweep {int(numbers[pos]):2d} | elev {float(elevations[pos]):5.2f} deg | "
                f"{int(counts[pos]):,} returns"
            ),
        )
        for index, pos in enumerate(np.argsort(numbers))
    ]


def available_moments(rs_dt: xr.DataTree, include_qc: bool = True) -> list[str]:
    """List the plottable moment variables present in ``returns``."""

    returns = _node(rs_dt, "returns")
    moments = [name for name in MOMENT_NAMES if name in returns.data_vars]
    if include_qc:
        moments.extend(sorted(n for n in returns.data_vars if n.startswith("qc.")))
    return moments


def value_bounds(values: np.ndarray) -> tuple[float, float]:
    """Robust ``(vmin, vmax)`` for a moment, from its 2nd and 98th percentiles."""

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = (float(v) for v in np.percentile(finite, (2.0, 98.0)))
    if low == high:
        low, high = float(finite.min()), float(finite.max())
    return (low, low + 1.0) if low == high else (low, high)


def _node(rs_dt: xr.DataTree, name: str) -> xr.Dataset:
    if name not in rs_dt.children:
        raise KeyError(f"raystack DataTree missing '{name}' node")
    return rs_dt[name].dataset


def _require_moment(returns: xr.Dataset, moment_name: str) -> None:
    if moment_name not in returns.data_vars:
        raise KeyError(f"moment '{moment_name}' not found in returns dataset")


def _at(ds: xr.Dataset, name: str, pos: int, default: Any) -> Any:
    return np.asarray(ds[name].values)[pos] if name in ds else default


def _format_time(value: np.datetime64) -> str:
    return str(np.datetime64(value, "s")).replace("T", " ")


def _format_site(vcp: VcpInfo) -> str:
    return (
        f"X {abs(vcp.longitude):.4f} deg {'E' if vcp.longitude >= 0 else 'W'}, "
        f"Y {abs(vcp.latitude):.4f} deg {'N' if vcp.latitude >= 0 else 'S'}, "
        f"elevation {vcp.altitude:.0f} m"
    )


def _resolve_cmap(spec: Any):
    from matplotlib.colors import Colormap
    from matplotlib.pyplot import get_cmap

    return spec.copy() if isinstance(spec, Colormap) else get_cmap(spec).copy()


# --------------------------------------------------------------------------
# Unfolding
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _UnfoldedGrid:
    """Folded returns placed back onto an absolute range axis.

    ``covered`` marks the cells a fold actually wrote to, so a NaN inside a fold
    (below threshold, or trailing padding) stays distinguishable from range that
    was never sampled.
    """

    values: np.ndarray
    covered: np.ndarray
    range_m: np.ndarray
    range_step_m: float
    fold_size: int
    fold_edges_m: np.ndarray

    @classmethod
    def empty(cls) -> _UnfoldedGrid:
        return cls(
            values=np.empty((0, 0), dtype=np.float32),
            covered=np.empty((0, 0), dtype=bool),
            range_m=np.empty(0),
            range_step_m=1.0,
            fold_size=0,
            fold_edges_m=np.empty(0),
        )

    @property
    def range_km(self) -> np.ndarray:
        return self.range_m / 1000.0

    def take_rows(self, order: np.ndarray) -> _UnfoldedGrid:
        return replace(self, values=self.values[order], covered=self.covered[order])


def _unfold(
    moment: np.ndarray,
    base_range: np.ndarray,
    range_step: np.ndarray,
    row_index: np.ndarray,
    n_rows: int,
) -> _UnfoldedGrid:
    """Scatter folded rows into an absolute-range grid of ``n_rows`` rows."""

    fold_size = int(moment.shape[1])
    usable = range_step[np.isfinite(range_step) & (range_step > 0)]
    step = float(np.median(usable)) if usable.size else 1.0

    # Folds start a whole number of gates out, so base_range / step is the gate index.
    col0 = np.rint(base_range / step).astype(np.int64)
    col0 -= int(col0.min())
    n_cols = int(col0.max()) + fold_size

    values = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    covered = np.zeros((n_rows, n_cols), dtype=bool)
    rows = np.repeat(row_index.astype(np.int64), fold_size)
    cols = (col0[:, None] + np.arange(fold_size, dtype=np.int64)[None, :]).ravel()
    values[rows, cols] = moment.reshape(-1)
    covered[rows, cols] = True

    return _UnfoldedGrid(
        values=values,
        covered=covered,
        range_m=float(np.min(base_range)) + np.arange(n_cols, dtype=np.float64) * step,
        range_step_m=step,
        fold_size=fold_size,
        fold_edges_m=np.unique(base_range).astype(np.float64),
    )


def _radial_index(return_time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Group consecutive folds of one radial: ``(radial per row, first row per radial)``."""

    if return_time.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    is_new = np.r_[True, return_time[1:] != return_time[:-1]]
    return (np.cumsum(is_new) - 1).astype(np.int64), np.flatnonzero(is_new).astype(np.int64)
