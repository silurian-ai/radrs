"""Benchmark raystack activity computation (Rust vs Python)."""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Dict

import numpy as np

import radrs.raystack as rrs

MOMENT_NAMES = ["DBZH", "VRADH", "WRADH", "ZDR", "PHIDP", "RHOHV", "KDP"]
DEFAULT_SOURCE = (
    "s3://unidata-nexrad-level2/2024/07/02/KABR/"
    "KABR20240702_000016_V06"
)


def compute_activity_python(returns_ds, sweeps_ds, moments=None) -> Dict[str, np.ndarray]:
    if moments is None:
        moments = [m for m in MOMENT_NAMES if m in returns_ds]

    n_returns = int(returns_ds.sizes["return_time"])
    fold_size = int(returns_ds.sizes["range"])
    n_sweeps = int(sweeps_ds.sizes["sweep_time"])

    ray_valid_count = np.zeros((len(moments), n_returns), dtype=np.uint32)
    ray_valid_fraction = np.full((len(moments), n_returns), np.nan, dtype=np.float32)

    if fold_size > 0 and n_returns > 0:
        for idx, moment in enumerate(moments):
            data = np.asarray(returns_ds[moment].values)
            counts = np.isfinite(data).sum(axis=1).astype(np.uint32)
            ray_valid_count[idx] = counts
            ray_valid_fraction[idx] = counts.astype(np.float32) / float(fold_size)

    sweep_valid_count = np.zeros((len(moments), n_sweeps), dtype=np.uint32)
    sweep_valid_fraction = np.full((len(moments), n_sweeps), np.nan, dtype=np.float32)

    starts = np.asarray(sweeps_ds["start_index"].values, dtype=np.int64)
    n_radials = np.asarray(sweeps_ds["n_radials"].values, dtype=np.int64)
    for s_idx, (start, count) in enumerate(zip(starts, n_radials, strict=False)):
        end = min(start + count, n_returns)
        denom = max(0, end - start) * fold_size
        sweep_counts = ray_valid_count[:, start:end].sum(axis=1).astype(np.uint32)
        sweep_valid_count[:, s_idx] = sweep_counts
        if denom > 0:
            sweep_valid_fraction[:, s_idx] = sweep_counts.astype(np.float32) / float(denom)

    volume_valid_count = ray_valid_count.sum(axis=1).astype(np.uint32)
    volume_valid_fraction = np.full((len(moments), 1), np.nan, dtype=np.float32)
    denom = n_returns * fold_size
    if denom > 0:
        volume_valid_fraction[:, 0] = volume_valid_count.astype(np.float32) / float(denom)

    return {
        "moment": moments,
        "ray_valid_count": ray_valid_count,
        "ray_valid_fraction": ray_valid_fraction,
        "sweep_valid_count": sweep_valid_count,
        "sweep_valid_fraction": sweep_valid_fraction,
        "volume_valid_count": volume_valid_count.reshape(-1, 1),
        "volume_valid_fraction": volume_valid_fraction,
    }


def timed(fn, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return samples


def summarize(samples: list[float]) -> tuple[float, float]:
    median = statistics.median(samples)
    p90 = statistics.quantiles(samples, n=10)[-1]
    return median, p90


def main() -> None:
    parser = argparse.ArgumentParser(description="Raystack activity benchmark")
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    print("=" * 60)
    print("radrs activity benchmark")
    print("=" * 60)
    print(f"Source: {args.source}")

    dt = rrs.open_datatree(args.source)
    activity = dt["activity"].dataset
    moments = [str(m) for m in activity["moment"].values.tolist()]

    def rust_access():
        _ = activity["ray_valid_count"].values
        _ = activity["sweep_valid_fraction"].values

    def python_compute():
        _ = compute_activity_python(dt["returns"].dataset, dt["sweeps"].dataset, moments)

    rust_samples = timed(rust_access, args.repeats)
    py_samples = timed(python_compute, args.repeats)

    rust_med, rust_p90 = summarize(rust_samples)
    py_med, py_p90 = summarize(py_samples)

    print("\n--- Activity timings ---")
    print(f"Rust activity access: {rust_med*1000:.1f} ms (p90 {rust_p90*1000:.1f} ms)")
    print(f"Python recompute:    {py_med*1000:.1f} ms (p90 {py_p90*1000:.1f} ms)")

    expected = compute_activity_python(dt["returns"].dataset, dt["sweeps"].dataset, moments)
    print("\n--- Consistency ---")
    max_diff = float(np.nanmax(np.abs(activity["ray_valid_fraction"].values - expected["ray_valid_fraction"])) )
    print(f"Max abs diff (ray_valid_fraction): {max_diff:.6f}")


if __name__ == "__main__":
    main()
