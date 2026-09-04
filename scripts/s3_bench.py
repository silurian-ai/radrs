"""S3 benchmark suite for radrs and optional xradar comparison.

Usage examples:
  uv run python scripts/s3_bench.py
  uv run python scripts/s3_bench.py --mode full
  uv run python scripts/s3_bench.py --site KTLX --date 2024-03-15 --n 3
  uv run python scripts/s3_bench.py --xradar
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import radrs
import radrs.xradar as rxr

DEFAULT_SITE = "KTLX"
DEFAULT_DATE = "2024-03-15"
DEFAULT_N = 3
DEFAULT_N_FULL = 10

BASE_ARCHIVE_URL="s3://unidata-nexrad-level2"

def _build_profile() -> str | None:
    try:
        from radrs import _radrs
    except ImportError:
        return None
    return getattr(_radrs, "__profile__", None)


def _warn_about_build() -> None:
    """Every number below is meaningless on a debug build -- say so loudly."""
    profile = _build_profile()
    if profile == "release":
        return

    if profile == "debug":
        headline = "WARNING: radrs is a DEBUG build."
        detail = (
            "Parsing runs ~13x slower than release, so CPU swamps the S3 fetch\n"
            "  and the async section cannot show any speedup. Rebuild with\n"
            "  `uv run maturin develop --release` before reading these numbers."
        )
    else:
        headline = "WARNING: Could not determine the radrs build profile."
        detail = (
            "The installed extension predates `_radrs.__profile__`. If it was built\n"
            "  by `uv sync` or a plain `maturin develop`, it is a debug build."
        )

    print("=" * 60)
    print(f"  {headline}")
    print(f"  {detail}")
    print("=" * 60)


def get_test_urls(site: str, date: str, n: int) -> list[str]:
    day = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    archive = radrs.NexradL2ArchiveIter(
        base_uri=BASE_ARCHIVE_URL,
        start_time=day,
        end_time=day + timedelta(days=1),
        storage_options={"anon": "true", "region": "us-east-1"},
        site_filter=[site],
    )
    urls = []
    for info in archive:
        urls.append(BASE_ARCHIVE_URL + "/" + info.uri)
        if len(urls) >= n:
            break
    return urls


def timed(fn: Callable[[], Any], warmup: int = 0) -> tuple[float, Any]:
    for _ in range(warmup):
        fn()

    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    return elapsed, result


def _maybe_import_xradar():
    try:
        import fsspec
        import xradar as xd
    except Exception:
        return None, None
    return fsspec, xd


def _xradar_local(url: str, tmpdir: str):
    fsspec, xd = _maybe_import_xradar()
    if fsspec is None or xd is None:
        return None
    local = fsspec.open_local(
        f"simplecache::{url}",
        s3={"anon": True},
        filecache={"cache_storage": tmpdir, "same_names": True},
    )
    return xd.io.open_nexradlevel2_datatree(local)


def benchmark_single_file(urls: list[str], use_xradar: bool) -> None:
    print("\n--- Single File Fetch ---")
    t1, _ = timed(lambda: rxr.open_datatree(urls[0]))
    print(f"radrs (cold):  {t1:.2f}s")

    t2, _ = timed(lambda: rxr.open_datatree(urls[1]))
    print(f"radrs (warm):  {t2:.2f}s")

    if use_xradar:
        fsspec, xd = _maybe_import_xradar()
        if fsspec is None or xd is None:
            print("xradar/fsspec not installed, skipping xradar comparison")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                t3, _ = timed(lambda: _xradar_local(urls[2], tmpdir))
            print(f"xradar:        {t3:.2f}s")
            print(f"Speedup (warm): {t3 / t2:.1f}x")


def benchmark_sequential(urls: list[str], use_xradar: bool) -> None:
    print(f"\n--- Sequential Fetch ({len(urls)} files) ---")
    start = time.perf_counter()
    for url in urls:
        rxr.open_datatree(url)
    radrs_seq = time.perf_counter() - start
    print(f"radrs: {radrs_seq:.2f}s ({len(urls) / radrs_seq:.2f} vol/s)")

    if use_xradar:
        fsspec, xd = _maybe_import_xradar()
        if fsspec is None or xd is None:
            print("xradar/fsspec not installed, skipping xradar comparison")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                start = time.perf_counter()
                for url in urls:
                    _xradar_local(url, tmpdir)
                xradar_seq = time.perf_counter() - start
            print(f"xradar: {xradar_seq:.2f}s ({len(urls) / xradar_seq:.2f} vol/s)")
            print(f"Speedup: {xradar_seq / radrs_seq:.1f}x")


def benchmark_async_concurrent(urls: list[str]) -> None:
    print(f"\n--- Async Concurrent ({len(urls)} files) ---")

    async def _run():
        start = time.perf_counter()
        tasks = [rxr.open_datatree_async(url) for url in urls]
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start
        print(f"radrs async concurrent: {elapsed:.2f}s ({len(urls) / elapsed:.2f} vol/s)")

    asyncio.run(_run())


def benchmark_connection_reuse(urls: list[str]) -> None:
    print("\n--- Connection Reuse ---")
    t1, _ = timed(lambda: rxr.open_datatree(urls[0]))
    t2, _ = timed(lambda: rxr.open_datatree(urls[1]))
    t3, _ = timed(lambda: rxr.open_datatree(urls[2]))

    avg_warm = (t2 + t3) / 2
    print(f"1st call (cold): {t1:.2f}s")
    print(f"2nd call (warm): {t2:.2f}s")
    print(f"3rd call (warm): {t3:.2f}s")
    print(f"Speedup: {t1 / avg_warm:.1f}x (cold/warm)")


def run(mode: str, site: str, date: str, n: int, n_full: int, use_xradar: bool) -> None:
    print("=" * 60)
    print("radrs S3 Benchmark Suite")
    print("=" * 60)
    _warn_about_build()
    print(f"Site: {site}  Date: {date}")

    urls = get_test_urls(site, date, max(n, 3))
    print(f"Testing with {len(urls)} volume(s)")

    if mode in {"quick", "all"}:
        benchmark_single_file(urls[: max(3, n)], use_xradar=use_xradar)
        benchmark_sequential(urls[:n], use_xradar=use_xradar)
        benchmark_async_concurrent(urls[:n])
        benchmark_connection_reuse(urls[: max(3, n)])

    if mode in {"full", "all"}:
        urls_full = get_test_urls(site, date, n_full)
        print(f"\nFull run: {len(urls_full)} volume(s)")
        benchmark_sequential(urls_full, use_xradar=use_xradar)
        benchmark_async_concurrent(urls_full)
        benchmark_connection_reuse(urls_full[:3])

    print("\n" + "=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="S3 benchmark suite for radrs")
    parser.add_argument("--site", default=DEFAULT_SITE)
    parser.add_argument("--date", default=DEFAULT_DATE)
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="Quick run volume count")
    parser.add_argument(
        "--n-full", type=int, default=DEFAULT_N_FULL, help="Full run volume count"
    )
    parser.add_argument(
        "--mode", choices=["quick", "full", "all"], default="quick"
    )
    parser.add_argument(
        "--xradar",
        action="store_true",
        help="Enable xradar comparisons (requires xradar + fsspec)",
    )

    args = parser.parse_args()
    run(args.mode, args.site, args.date, args.n, args.n_full, args.xradar)


if __name__ == "__main__":
    main()
