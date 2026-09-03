"""Generate a four-panel raystack explainer figure from a real NEXRAD volume.

Regenerate the docs assets:

    uv sync --group dev
    uv run --with matplotlib python docs/examples/raystack_explainer.py \\
        --source s3://unidata-nexrad-level2/2024/07/02/KABR/KABR20240702_000016_V06 \\
        --fold-size 128 \\
        --moment DBZH \\
        --output docs/assets/raystack-explainer.png \\
        --panel-output-dir docs/assets/raystack-explainer-panels
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import radrs.raystack as rrs
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Rectangle
from numpy.typing import NDArray


DEFAULT_SOURCE = "s3://unidata-nexrad-level2/2024/07/02/KABR/KABR20240702_000016_V06"
MOMENT_NAMES = ("DBZH", "VRADH", "WRADH", "ZDR", "PHIDP", "RHOHV", "CCORH")
PAD_COLOR = "#b45309"


@dataclass(frozen=True)
class RaystackParts:
    vcps: xr.Dataset
    sweeps: xr.Dataset
    returns: xr.Dataset
    moment: str
    offsets: NDArray[np.int64]


def sweep_return_offsets(sweeps: xr.Dataset) -> NDArray[np.int64]:
    """Return the start offset for each sweep plus the final stop offset."""
    counts = np.asarray(sweeps["num_returns"].values, dtype=np.int64)
    offsets = np.zeros(counts.size + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    return offsets


def first_available_moment(returns: xr.Dataset, preferred: str = "DBZH") -> str:
    """Choose a moment variable present in the raystack returns dataset."""
    if preferred in returns.data_vars:
        return preferred
    for name in MOMENT_NAMES:
        if name in returns.data_vars:
            return name
    raise ValueError("raystack returns dataset does not contain a supported moment")


def draw_box(
    ax: Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    label: str,
    *,
    facecolor: str,
    edgecolor: str = "#315452",
    fontsize: int = 9,
) -> None:
    """Draw a labeled rectangle in axis coordinates."""
    rect = Rectangle(
        xy,
        width,
        height,
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
        joinstyle="round",
    )
    ax.add_patch(rect)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#203332",
    )


def draw_arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#315452",
) -> None:
    """Draw a compact arrow in axis coordinates."""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.2,
        color=color,
    )
    ax.add_patch(arrow)


def draw_physical_scan(
    ax: Axes,
    vcps: xr.Dataset,
    sweeps: xr.Dataset,
    *,
    show_title: bool = True,
) -> None:
    """Draw the VCP as a side-view fan of beams at real elevation angles."""
    if show_title:
        ax.set_title("1. Volume coverage", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.15, 1.05)
    ax.axis("off")

    elevations = np.sort(np.asarray(sweeps["elevation_angle"].values, dtype=np.float32))
    vcp_number = int(np.asarray(vcps["vcp_number"].values)[0])
    n_sweeps = elevations.size
    lo = float(elevations[0])
    hi = float(elevations[-1])

    ax.plot([-0.02, 1.02], [0, 0], color="#a8b3b1", linewidth=1.0, zorder=1)
    ax.scatter([0], [0], s=140, color="#315452", zorder=5)
    ax.text(0.0, -0.06, "radar", ha="center", va="top", fontsize=8, color="#203332")

    cmap = plt.get_cmap("viridis")
    visual_scale = 60.0 / max(hi, 1.0)
    beam_length = 0.95

    for i, angle in enumerate(elevations):
        plot_angle = np.deg2rad(angle * visual_scale)
        x_end = beam_length * np.cos(plot_angle)
        y_end = beam_length * np.sin(plot_angle)
        color = cmap(0.18 + 0.72 * (i / max(n_sweeps - 1, 1)))
        ax.plot([0, x_end], [0, y_end], color=color, linewidth=1.6, alpha=0.92, zorder=3)

    for angle, label in ((lo, f"{lo:.1f}°"), (hi, f"{hi:.1f}°")):
        plot_angle = np.deg2rad(angle * visual_scale)
        ax.text(
            beam_length * np.cos(plot_angle) + 0.02,
            beam_length * np.sin(plot_angle),
            label,
            ha="left",
            va="center",
            fontsize=8,
            color="#203332",
        )

    ax.text(0.02, 1.0, f"VCP {vcp_number}", fontsize=11, fontweight="bold", color="#203332")
    ax.text(
        0.02,
        0.92,
        f"{n_sweeps} sweeps from {lo:.1f}° to {hi:.1f}° elevation",
        fontsize=9,
        color="#50615f",
    )
    ax.text(
        0.5,
        -0.12,
        "side view; vertical scale exaggerated for clarity",
        ha="center",
        va="top",
        fontsize=7,
        color="#7a8a87",
        style="italic",
    )


def draw_radial_folding(
    ax: Axes,
    sweeps: xr.Dataset,
    returns: xr.Dataset,
    moment: str,
    offsets: NDArray[np.int64],
    fold_size: int,
    *,
    show_title: bool = True,
) -> None:
    """Show a real radial as a flattened strip, then folded into return rows."""
    if show_title:
        ax.set_title("2. Fold one radial", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    sweep_max_gates = np.asarray(sweeps["max_gates"].values, dtype=np.int64)
    moment_data = np.asarray(returns[moment].values, dtype=np.float32)

    target_sweep = 0
    best_score = -1.0
    for s_idx in range(sweep_max_gates.size):
        chunks = int(np.ceil(sweep_max_gates[s_idx] / fold_size))
        if chunks < 2:
            continue
        s_start = int(offsets[s_idx])
        s_stop = int(offsets[s_idx + 1])
        if s_stop <= s_start:
            continue
        valid_count = float(np.sum(~np.isnan(moment_data[s_start:s_stop])))
        if valid_count > best_score:
            best_score = valid_count
            target_sweep = s_idx

    n_chunks = max(1, int(np.ceil(sweep_max_gates[target_sweep] / fold_size)))
    s_start = int(offsets[target_sweep])
    s_stop = int(offsets[target_sweep + 1])
    n_radials_in_sweep = max(1, (s_stop - s_start) // n_chunks)
    sweep_grid = moment_data[s_start : s_start + n_radials_in_sweep * n_chunks].reshape(
        n_radials_in_sweep, n_chunks, fold_size
    )
    valid_per_radial = np.sum(~np.isnan(sweep_grid), axis=(1, 2))
    radial = sweep_grid[int(np.argmax(valid_per_radial))]
    flattened = radial.reshape(1, -1)
    actual_gates = int(sweep_max_gates[target_sweep])

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#e6ebe9")

    n_total = flattened.shape[1]
    # Gates >= actual_gates exist only to pad the final return row to fold_size.
    pad_start_col = actual_gates % fold_size
    has_padding = actual_gates < n_total

    strip_y0, strip_h = 0.78, 0.10
    x_strip_pad = 0.05 + actual_gates / n_total * 0.9
    ax.imshow(
        flattened[:, :actual_gates],
        extent=(0.05, x_strip_pad, strip_y0, strip_y0 + strip_h),
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        zorder=2,
    )
    for chunk in range(1, n_chunks):
        x_pos = 0.05 + (chunk * fold_size) / n_total * 0.9
        if x_pos >= x_strip_pad:
            continue
        ax.plot(
            [x_pos, x_pos],
            [strip_y0, strip_y0 + strip_h],
            color="white",
            linewidth=0.6,
            alpha=0.85,
            zorder=3,
        )
    if has_padding:
        ax.add_patch(
            Rectangle(
                (x_strip_pad, strip_y0),
                0.95 - x_strip_pad,
                strip_h,
                linewidth=1.0,
                edgecolor=PAD_COLOR,
                facecolor="none",
                hatch="///",
                zorder=4,
            )
        )
        ax.text(
            (x_strip_pad + 0.95) / 2,
            strip_y0 - 0.04,
            "pad",
            fontsize=7,
            color=PAD_COLOR,
            ha="center",
            va="top",
        )
    ax.add_patch(
        Rectangle(
            (0.05, strip_y0),
            0.9,
            strip_h,
            linewidth=1.0,
            edgecolor="#315452",
            facecolor="none",
            zorder=5,
        )
    )
    ax.text(
        0.05,
        strip_y0 + strip_h + 0.04,
        f"one radial: {actual_gates:,} range gates of {moment}",
        fontsize=9,
        color="#203332",
    )
    ax.text(0.05, strip_y0 - 0.04, "gate 0", fontsize=7, color="#50615f", ha="left", va="top")
    ax.text(
        x_strip_pad,
        strip_y0 - 0.04,
        f"gate {actual_gates - 1:,}",
        fontsize=7,
        color="#50615f",
        ha="right",
        va="top",
    )

    draw_arrow(ax, (0.5, 0.72), (0.5, 0.55))
    ax.text(
        0.52,
        0.635,
        f"split into chunks of fold_size = {fold_size}",
        fontsize=8,
        color="#50615f",
        va="center",
    )

    stack_top, stack_h = 0.50, 0.36
    stack_bottom = stack_top - stack_h
    ax.imshow(
        radial,
        extent=(0.05, 0.95, stack_bottom, stack_top),
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        zorder=2,
    )
    cell_h = stack_h / n_chunks
    for chunk in range(n_chunks):
        y_top = stack_top - chunk * cell_h
        ax.add_patch(
            Rectangle(
                (0.05, y_top - cell_h),
                0.9,
                cell_h,
                linewidth=0.5,
                edgecolor="#315452",
                facecolor="none",
                zorder=3,
            )
        )
    for chunk_label_idx in (0, n_chunks - 1):
        y_center = stack_top - (chunk_label_idx + 0.5) * cell_h
        ax.text(
            0.045,
            y_center,
            f"return {chunk_label_idx}",
            ha="right",
            va="center",
            fontsize=7,
            color="#50615f",
        )
    ax.text(
        0.05,
        stack_top + 0.03,
        f"stored as {n_chunks} return rows × fold_size = {fold_size}",
        fontsize=9,
        color="#203332",
    )

    if has_padding:
        x_pad = 0.05 + pad_start_col / fold_size * 0.9
        ax.add_patch(
            Rectangle(
                (x_pad, stack_bottom),
                0.95 - x_pad,
                cell_h,
                linewidth=1.0,
                edgecolor=PAD_COLOR,
                facecolor="none",
                hatch="///",
                zorder=5,
            )
        )
        ax.annotate(
            f"padding: last {n_total - actual_gates} cells of return {n_chunks - 1}",
            xy=((x_pad + 0.95) / 2, stack_bottom),
            xytext=((x_pad + 0.95) / 2, stack_bottom - 0.06),
            ha="center",
            va="top",
            fontsize=7,
            color=PAD_COLOR,
            arrowprops={"arrowstyle": "-|>", "color": PAD_COLOR, "linewidth": 0.9},
        )
    ax.text(
        0.05,
        stack_bottom - 0.115,
        "hatched: trailing padding past the last real gate; "
        "gray: NaN gates (missing or below threshold)",
        fontsize=7,
        color="#7a8a87",
        ha="left",
        va="top",
        style="italic",
    )


def draw_flat_schema(
    ax: Axes,
    vcps: xr.Dataset,
    sweeps: xr.Dataset,
    returns: xr.Dataset,
    *,
    show_title: bool = True,
) -> None:
    """Draw raystack datasets as linked flat tables."""
    if show_title:
        ax.set_title("3. Flat schema", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    n_vcps = vcps.sizes["vcp_time"]
    n_sweeps = sweeps.sizes["sweep_time"]
    n_returns = returns.sizes["return_time"]
    fold_size = returns.sizes["range"]

    draw_box(ax, (0.08, 0.68), 0.28, 0.16, f"vcps[P]\nP = {n_vcps}", facecolor="#e0f2fe")
    draw_box(
        ax,
        (0.08, 0.42),
        0.34,
        0.16,
        f"sweeps[S]\nS = {n_sweeps}",
        facecolor="#dcfce7",
    )
    draw_box(
        ax,
        (0.08, 0.13),
        0.46,
        0.18,
        f"returns[R, G]\nR = {n_returns:,}, G = {fold_size}",
        facecolor="#fef3c7",
    )

    draw_arrow(ax, (0.24, 0.68), (0.24, 0.58))
    ax.text(0.28, 0.62, "vcp_time", fontsize=8, color="#50615f")
    draw_arrow(ax, (0.28, 0.42), (0.28, 0.31))
    ax.text(0.32, 0.35, "sweep_number", fontsize=8, color="#50615f")

    fields = [
        ("vcps", "vcp_time, vcp_number, site"),
        ("sweeps", "elevation, range_step, num_returns"),
        ("returns", "azimuth, time, base_range, moments"),
    ]
    for idx, (name, values) in enumerate(fields):
        ax.text(0.62, 0.74 - idx * 0.21, name, fontsize=9, fontweight="bold", color="#203332")
        ax.text(0.62, 0.68 - idx * 0.21, values, fontsize=8, color="#50615f")


def draw_ml_tensor(
    ax: Axes,
    sweeps: xr.Dataset,
    returns: xr.Dataset,
    moment: str,
    offsets: NDArray[np.int64],
    *,
    show_title: bool = True,
    window_rows: int = 320,
) -> None:
    """Draw a real moment matrix window straddling a sweep boundary."""
    if show_title:
        ax.set_title("4. ML tensor", loc="left", fontsize=12, fontweight="bold")
    data = np.asarray(returns[moment].values, dtype=np.float32)
    n_rows = data.shape[0]
    fold_size = min(returns.sizes["range"], data.shape[1])

    # Centre the window on a real sweep boundary so the panel shows one.
    boundary = next(
        (int(off) for off in offsets[1:-1] if 0 < int(off) < n_rows),
        None,
    )
    window = min(window_rows, n_rows)
    if boundary is None:
        row_start = 0
    else:
        row_start = min(max(0, boundary - window // 2), n_rows - window)
    row_stop = row_start + window
    sample = data[row_start:row_stop, :fold_size]

    im = ax.imshow(
        sample,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        extent=(-0.5, fold_size - 0.5, row_stop - 0.5, row_start - 0.5),
    )

    # Faint lines every ceil(max_gates / fold_size) rows: one physical radial.
    max_gates = np.asarray(sweeps["max_gates"].values, dtype=np.int64)
    for s_idx in range(max_gates.size):
        chunks = max(1, int(np.ceil(max_gates[s_idx] / fold_size)))
        s_start, s_stop = int(offsets[s_idx]), int(offsets[s_idx + 1])
        first = s_start + chunks * int(np.ceil(max(row_start - s_start, 0) / chunks))
        for row in range(first, min(s_stop, row_stop), chunks):
            if row_start < row < row_stop:
                ax.axhline(row - 0.5, color="#94a3b8", linewidth=0.4, alpha=0.55)

    sweep_idx = None
    if boundary is not None:
        sweep_idx = int(np.searchsorted(offsets, boundary)) - 1
        ax.axhline(boundary - 0.5, color="#f97316", linewidth=1.8)
        for label, row_offset, valign in (
            (f"sweep {sweep_idx}", -3, "bottom"),
            (f"sweep {sweep_idx + 1}", 3, "top"),
        ):
            ax.text(
                fold_size - 3,
                boundary + row_offset,
                label,
                ha="right",
                va=valign,
                fontsize=8,
                fontweight="bold",
                color="#7c2d12",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none"},
            )

    ax.set_xlabel("gate within fold_size")
    ax.set_ylabel("return row (absolute index)")
    ax.text(
        1.02,
        0.92,
        "aligned metadata:\nazimuth[R]\nelevation[R]\nreturn_time[R]\nbase_range[R]",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#203332",
    )
    boundary_note = (
        f"orange line: sweep {sweep_idx} | sweep {sweep_idx + 1} boundary; "
        if sweep_idx is not None
        else ""
    )
    ax.text(
        0.0,
        -0.16,
        f"window: rows {row_start:,}–{row_stop - 1:,} of {n_rows:,}; "
        f"{boundary_note}faint lines: radial boundaries",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="#7a8a87",
        style="italic",
    )
    ax.text(
        0.02,
        0.96,
        f"{moment}[R, fold_size]",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="white",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#203332", "edgecolor": "none"},
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def load_raystack_parts(
    source: str,
    *,
    fold_size: int = 128,
    preferred_moment: str = "DBZH",
) -> RaystackParts:
    """Load a source volume and return the datasets needed for the figure."""
    tree = rrs.open_datatree(source, fold_size=fold_size)
    vcps = tree["vcps"].dataset
    sweeps = tree["sweeps"].dataset
    returns = tree["returns"].dataset
    moment = first_available_moment(returns, preferred=preferred_moment)
    offsets = sweep_return_offsets(sweeps)
    return RaystackParts(
        vcps=vcps,
        sweeps=sweeps,
        returns=returns,
        moment=moment,
        offsets=offsets,
    )


def create_raystack_explainer(parts: RaystackParts, *, fold_size: int = 128) -> Figure:
    """Build the full four-panel explainer figure."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    flat_axes: Sequence[Axes] = axes.ravel()
    draw_physical_scan(flat_axes[0], parts.vcps, parts.sweeps)
    draw_radial_folding(
        flat_axes[1],
        parts.sweeps,
        parts.returns,
        parts.moment,
        parts.offsets,
        fold_size,
    )
    draw_flat_schema(flat_axes[2], parts.vcps, parts.sweeps, parts.returns)
    draw_ml_tensor(flat_axes[3], parts.sweeps, parts.returns, parts.moment, parts.offsets)
    fig.suptitle("Raystack: from NEXRAD L2 volume to flat training arrays", fontsize=16)
    return fig


def create_panel_figures(parts: RaystackParts, *, fold_size: int = 128) -> list[tuple[str, Figure]]:
    """Build one figure per explainer panel."""
    panel_specs = [
        (
            "physical-scan",
            lambda ax: draw_physical_scan(ax, parts.vcps, parts.sweeps, show_title=False),
        ),
        (
            "fold-radial",
            lambda ax: draw_radial_folding(
                ax,
                parts.sweeps,
                parts.returns,
                parts.moment,
                parts.offsets,
                fold_size,
                show_title=False,
            ),
        ),
        (
            "flat-schema",
            lambda ax: draw_flat_schema(
                ax,
                parts.vcps,
                parts.sweeps,
                parts.returns,
                show_title=False,
            ),
        ),
        (
            "ml-tensor",
            lambda ax: draw_ml_tensor(
                ax,
                parts.sweeps,
                parts.returns,
                parts.moment,
                parts.offsets,
                show_title=False,
            ),
        ),
    ]

    figures: list[tuple[str, Figure]] = []
    for slug, draw in panel_specs:
        fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        draw(ax)
        figures.append((slug, fig))
    return figures


def save_panel_figures(
    parts: RaystackParts,
    output_dir: Path,
    *,
    fold_size: int = 128,
) -> None:
    """Write individual panel images for docs pages that explain each step."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for slug, fig in create_panel_figures(parts, fold_size=fold_size):
        fig.savefig(output_dir / f"{slug}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="NEXRAD L2 file path or URI")
    parser.add_argument("--fold-size", default=128, type=int, help="Range gates per return row")
    parser.add_argument("--moment", default="DBZH", help="Preferred moment to show")
    parser.add_argument(
        "--output",
        default="docs/assets/raystack-explainer.png",
        help="Output image path",
    )
    parser.add_argument(
        "--panel-output-dir",
        default="docs/assets/raystack-explainer-panels",
        help="Directory for individual panel images",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    parts = load_raystack_parts(
        args.source,
        fold_size=args.fold_size,
        preferred_moment=args.moment,
    )
    fig = create_raystack_explainer(parts, fold_size=args.fold_size)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    save_panel_figures(parts, Path(args.panel_output_dir), fold_size=args.fold_size)
    print(f"wrote {output}")
    print(f"wrote panels to {args.panel_output_dir}")


if __name__ == "__main__":
    main()
