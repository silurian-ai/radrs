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
    """Draw the VCP/sweep/radial/gate hierarchy using real sweep counts."""
    if show_title:
        ax.set_title("1. Physical scan", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    vcp_number = int(np.asarray(vcps["vcp_number"].values)[0])
    elevations = np.asarray(sweeps["elevation_angle"].values, dtype=np.float32)
    counts = np.asarray(sweeps["num_returns"].values, dtype=np.int64)
    shown = min(5, elevations.size)

    radar = (0.15, 0.22)
    ax.scatter([radar[0]], [radar[1]], s=130, color="#315452", zorder=3)
    ax.text(radar[0], 0.08, "radar", ha="center", va="center", fontsize=8)

    colors = ["#6ea8fe", "#72bf78", "#f3b95f", "#a990d6", "#e56b6f"]
    for idx in range(shown):
        y = 0.34 + idx * 0.1
        ax.plot(
            [radar[0], 0.85],
            [radar[1], y],
            color=colors[idx % len(colors)],
            linewidth=2.0,
            alpha=0.85,
        )
        ax.scatter(
            [0.46, 0.56, 0.66, 0.76],
            np.interp([0.46, 0.56, 0.66, 0.76], [radar[0], 0.85], [radar[1], y]),
            s=18,
            color=colors[idx % len(colors)],
            alpha=0.9,
        )
        ax.text(
            0.88,
            y,
            f"sweep {idx}: {elevations[idx]:.1f} deg, {counts[idx]:,} returns",
            ha="left",
            va="center",
            fontsize=8,
        )

    ax.text(0.05, 0.91, f"VCP {vcp_number}", fontsize=10, fontweight="bold", color="#203332")
    ax.text(0.05, 0.82, f"{elevations.size} sweeps", fontsize=9, color="#50615f")
    ax.text(0.42, 0.17, "radials carry range gates", fontsize=9, color="#50615f")


def draw_radial_folding(
    ax: Axes,
    sweeps: xr.Dataset,
    fold_size: int,
    *,
    show_title: bool = True,
) -> None:
    """Draw how a radial is split into fixed-width return chunks."""
    if show_title:
        ax.set_title("2. Fold one radial", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    max_gates = int(np.nanmax(np.asarray(sweeps["max_gates"].values, dtype=np.float32)))
    chunks = max(1, int(np.ceil(max_gates / fold_size)))
    shown_chunks = min(chunks, 4)
    gate_rows = 12
    gate_cols = 8
    gate_count_shown = gate_rows * gate_cols
    chunk_width = 0.18
    x0 = 0.08
    y0 = 0.45
    colors = ["#6ea8fe", "#72bf78", "#f3b95f", "#a990d6"]

    for chunk in range(shown_chunks):
        start_x = x0 + chunk * (chunk_width + 0.035)
        draw_box(
            ax,
            (start_x, y0 - 0.05),
            chunk_width,
            0.28,
            "",
            facecolor="#ffffff",
            edgecolor=colors[chunk % len(colors)],
        )
        for gate in range(gate_count_shown):
            row = gate // gate_cols
            col = gate % gate_cols
            ax.add_patch(
                Rectangle(
                    (start_x + 0.012 + col * 0.018, y0 + 0.165 - row * 0.018),
                    0.012,
                    0.012,
                    linewidth=0,
                    facecolor=colors[chunk % len(colors)],
                    alpha=0.75,
                )
            )
        label = f"return {chunk}"
        if chunk == shown_chunks - 1 and chunks > shown_chunks:
            label = f"return {chunk}..."
        ax.text(start_x + chunk_width / 2, y0 - 0.11, label, ha="center", fontsize=8)

    ax.text(0.08, 0.82, f"max gates in volume: {max_gates:,}", fontsize=9, color="#203332")
    ax.text(0.08, 0.74, f"fold_size: {fold_size:,} gates", fontsize=9, color="#203332")
    ax.text(0.08, 0.28, f"one physical radial can become {chunks} return rows", fontsize=9)
    draw_arrow(ax, (0.78, 0.58), (0.92, 0.58))
    draw_box(ax, (0.93, 0.48), 0.06, 0.2, "R,G", facecolor="#dbeafe", fontsize=8)


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
    returns: xr.Dataset,
    moment: str,
    offsets: NDArray[np.int64],
    *,
    show_title: bool = True,
) -> None:
    """Draw a real moment matrix sample and its aligned metadata."""
    if show_title:
        ax.set_title("4. ML tensor", loc="left", fontsize=12, fontweight="bold")
    data = np.asarray(returns[moment].values, dtype=np.float32)
    sample_rows = min(320, data.shape[0])
    sample_cols = min(returns.sizes["range"], data.shape[1])
    sample = data[:sample_rows, :sample_cols]

    im = ax.imshow(sample, aspect="auto", interpolation="nearest", cmap="viridis")
    for offset in offsets:
        if 0 < offset < sample_rows:
            ax.axhline(offset - 0.5, color="white", linewidth=0.6, alpha=0.8)

    ax.set_xlabel("gate within fold_size")
    ax.set_ylabel("return row")
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
    draw_radial_folding(flat_axes[1], parts.sweeps, fold_size)
    draw_flat_schema(flat_axes[2], parts.vcps, parts.sweeps, parts.returns)
    draw_ml_tensor(flat_axes[3], parts.returns, parts.moment, parts.offsets)
    fig.suptitle("Raystack: from NEXRAD L2 volume to flat training arrays", fontsize=16)
    return fig


def create_panel_figures(parts: RaystackParts, *, fold_size: int = 128) -> list[tuple[str, Figure]]:
    """Build one figure per explainer panel."""
    panel_specs = [
        (
            "physical-scan",
            "1. Physical scan",
            lambda ax: draw_physical_scan(ax, parts.vcps, parts.sweeps, show_title=False),
        ),
        (
            "fold-radial",
            "2. Fold one radial",
            lambda ax: draw_radial_folding(ax, parts.sweeps, fold_size, show_title=False),
        ),
        (
            "flat-schema",
            "3. Flat schema",
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
            "4. ML tensor",
            lambda ax: draw_ml_tensor(
                ax,
                parts.returns,
                parts.moment,
                parts.offsets,
                show_title=False,
            ),
        ),
    ]

    figures: list[tuple[str, Figure]] = []
    for slug, _title, draw in panel_specs:
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
