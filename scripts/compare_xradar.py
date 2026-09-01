#!/usr/bin/env python3
"""Compare radrs.xradar output against xradar for a single file.

Examples:
  python python/tools/compare_xradar.py /path/to/file.ar2v
  python python/tools/compare_xradar.py \
    --radrs-source s3://unidata-nexrad-level2/2024/03/15/KTLX/KTLX20240315_000217_V06 \
    --xradar-source https://unidata-nexrad-level2.s3.amazonaws.com/2024/03/15/KTLX/KTLX20240315_000217_V06
"""

from __future__ import annotations

import argparse
import time
from typing import Iterable
import re
import fsspec

import numpy as np


MOMENTS = ["DBZH", "VRADH", "WRADH", "ZDR", "PHIDP", "RHOHV", "CCORH"]

# sweep_fixed_angle is a nominal VCP angle, so the two libraries should agree
# exactly; the tolerance only absorbs float32/float64 representation.
ELEV_TOL = 1e-4


def _as_float_array(values) -> np.ndarray:
    if np.ma.isMaskedArray(values):
        return values.filled(np.nan).astype(np.float32, copy=False)
    return np.asarray(values, dtype=np.float32)


def _fixed_angle(ds) -> float:
    """Nominal (fixed) elevation angle of a sweep, in degrees."""
    if "sweep_fixed_angle" in ds:
        return float(np.ravel(ds["sweep_fixed_angle"].values)[0])
    if "elevation" in ds:
        return float(np.mean(ds["elevation"].values))
    return float("nan")


def _sweep_keys(dt) -> list[str]:
    return [k for k in dt.children.keys() if k.startswith("sweep_")]


def _align_by_azimuth(
    rs_vals: np.ndarray,
    rs_az: np.ndarray,
    xr_vals: np.ndarray,
    xr_az: np.ndarray,
    decimals: int = 2,
) -> tuple[np.ndarray, np.ndarray] | None:
    rs_az_r = np.round(rs_az, decimals)
    xr_az_r = np.round(xr_az, decimals)
    common = np.intersect1d(rs_az_r, xr_az_r)
    if common.size == 0:
        return None

    rs_mask = np.isin(rs_az_r, common)
    xr_mask = np.isin(xr_az_r, common)

    rs_vals = rs_vals[rs_mask]
    xr_vals = xr_vals[xr_mask]
    rs_az_r = rs_az_r[rs_mask]
    xr_az_r = xr_az_r[xr_mask]

    rs_order = np.argsort(rs_az_r)
    xr_order = np.argsort(xr_az_r)

    rs_vals = rs_vals[rs_order]
    xr_vals = xr_vals[xr_order]

    n = min(rs_vals.shape[0], xr_vals.shape[0])
    if n == 0:
        return None
    return rs_vals[:n], xr_vals[:n]


def _summarize_diff(rs_vals: np.ndarray, xr_vals: np.ndarray) -> dict[str, float]:
    n_gates = min(rs_vals.shape[1], xr_vals.shape[1])
    rs_vals = rs_vals[:, :n_gates]
    xr_vals = xr_vals[:, :n_gates]

    mask = np.isfinite(rs_vals) & np.isfinite(xr_vals)
    overlap = float(mask.mean()) if mask.size else 0.0

    if not np.any(mask):
        return {
            "overlap": overlap,
            "nan_rs": float(np.isnan(rs_vals).mean()),
            "nan_xr": float(np.isnan(xr_vals).mean()),
            "mean_abs": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
        }

    diff = np.abs(rs_vals[mask] - xr_vals[mask])
    return {
        "overlap": overlap,
        "nan_rs": float(np.isnan(rs_vals).mean()),
        "nan_xr": float(np.isnan(xr_vals).mean()),
        "mean_abs": float(np.mean(diff)),
        "p95": float(np.quantile(diff, 0.95)),
        "p99": float(np.quantile(diff, 0.99)),
    }


def _print_metrics(label: str, metrics: dict[str, float]) -> None:
    print(
        f"{label:<6} valid_obs={metrics['overlap']:.3f} "
        + (
            f"mean={metrics['mean_abs']:.3f} p95={metrics['p95']:.3f} p99={metrics['p99']:.3f}"
            if (abs(metrics["mean_abs"]) + abs(metrics["p99"]) + abs(metrics["p95"]))
            != 0.0
            else "(all obs equal)"
        )
    )


def _get_build_profile() -> str | None:
    try:
        from radrs import _radrs
    except ImportError:
        return None
    return getattr(_radrs, "__profile__", None)


def _warn_about_build(profile: str | None) -> None:
    """Print a banner describing the build the timings below came from."""
    if profile == "release":
        headline = "radrs is a RELEASE build."
        detail = (
            "A release build should show significant speedup over xradar data loading."
        )
    elif profile == "debug":
        headline = "WARNING: radrs is a DEBUG build."
        detail = (
            "A debug build compiles at opt-level 0 with no LTO and runs roughly\n"
            "  5-10x slower than release."
        )
    else:
        headline = "WARNING: Could not determine the radrs build profile."
        detail = (
            "The installed extension predates `_radrs.__profile__`. If it was built\n"
            "  by `uv sync` or a plain `maturin develop`, it is a debug build."
        )

    print("=" * 78)
    print(f"  {headline}")
    print(f"  {detail}")
    print("=" * 78)
    print()


def _time_call(fn) -> tuple[float, object]:
    start = time.perf_counter()
    result = fn()
    return time.perf_counter() - start, result


def _is_remote_url(url):
    return re.match(r"^[a-zA-z][^:]*://.+", url) is not None


def compare(radrs_source: str, xradar_source: str | None) -> None:
    import radrs.xradar as rxr

    try:
        import xradar as xd
    except ImportError:
        raise SystemExit("xradar not installed")

    if xradar_source is None:
        xradar_source = radrs_source

    profile = _get_build_profile()
    _warn_about_build(profile)

    if not _is_remote_url(xradar_source):
        xradar_source_call = lambda: xradar_source
    else:

        def xradar_source_call():
            with fsspec.open(xradar_source, "rb") as f:
                return f.read()

    t_rs, rs_dt = _time_call(lambda: rxr.open_datatree(radrs_source))
    t_xr, xr_dt = _time_call(
        lambda: xd.io.open_nexradlevel2_datatree(xradar_source_call())
    )

    suffix = "" if profile == "release" else f"  [{profile or 'unknown'} build]"
    print(f"radrs load: {t_rs:.2f}s{suffix}")
    print(f"xradar load: {t_xr:.2f}s")

    rs_sweeps = _sweep_keys(rs_dt)
    xr_sweeps = _sweep_keys(xr_dt)
    print(f"sweeps: radrs={len(rs_sweeps)} xradar={len(xr_sweeps)}")

    for key in rs_sweeps:
        if key not in xr_dt.children:
            continue
        rs_ds = rs_dt[key].dataset
        xr_ds = xr_dt[key].dataset

        rs_n = len(rs_ds["azimuth"])
        xr_n = len(xr_ds["azimuth"])
        rs_r = rs_ds.sizes.get("range", 0)
        xr_r = xr_ds.sizes.get("range", 0)

        rs_el = _fixed_angle(rs_ds)
        xr_el = _fixed_angle(xr_ds)
        el_flag = "" if abs(rs_el - xr_el) <= ELEV_TOL else "  <-- MISMATCH"

        print(
            f"\n{key}: radials={rs_n} vs {xr_n}, range={rs_r} vs {xr_r}, "
            f"elevation={rs_el:.4f} vs {xr_el:.4f}{el_flag}"
        )

        rs_az = _as_float_array(rs_ds["azimuth"].values)
        xr_az = _as_float_array(xr_ds["azimuth"].values)

        for moment in MOMENTS:
            if moment not in rs_ds or moment not in xr_ds:
                continue
            rs_vals = _as_float_array(rs_ds[moment].values)
            xr_vals = _as_float_array(xr_ds[moment].values)

            aligned = _align_by_azimuth(rs_vals, rs_az, xr_vals, xr_az)
            if aligned is None:
                print(f"{moment:<6} no azimuth overlap")
                continue

            metrics = _summarize_diff(*aligned)
            _print_metrics(moment, metrics)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        help="Local file path to compare (used for both radrs and xradar if --radrs-source not set).",
    )
    parser.add_argument("--radrs-source", help="Path or s3:// URL for radrs.")
    parser.add_argument("--xradar-source", help="Path or URL for xradar.")
    args = parser.parse_args(argv)

    radrs_source = args.radrs_source or args.path
    if not radrs_source:
        raise SystemExit("Provide a file path or --radrs-source.")

    compare(radrs_source, args.xradar_source)


if __name__ == "__main__":
    main()
