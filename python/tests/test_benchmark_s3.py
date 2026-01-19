"""Benchmark tests comparing radrs vs xradar S3 fetch performance.

Run with:
    pytest python/tests/test_benchmark_s3.py -v -s --benchmark-only

Or for quick comparison without pytest-benchmark:
    python python/tests/test_benchmark_s3.py
"""

import asyncio
import tempfile
import time
from typing import Any, Callable

import pytest

import radrs
import radrs.xradar as rxr

# Test parameters
SITE = "KTLX"
DATE = "2024-03-15"
N_VOLUMES_QUICK = 3
N_VOLUMES_FULL = 10


def get_test_urls(n: int = N_VOLUMES_QUICK) -> list[str]:
    """Get first N volume URLs for test date."""
    volumes = radrs.list_volumes(SITE, DATE)[:n]
    return [
        f"s3://unidata-nexrad-level2/2024/03/15/{SITE}/{v}"
        for v in volumes
    ]


def timed(fn: Callable[[], Any], label: str, warmup: int = 0) -> tuple[float, Any]:
    """Time a function call, optionally with warmup runs."""
    for _ in range(warmup):
        fn()

    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    return elapsed, result


class TestS3SingleFileBenchmark:
    """Benchmark single file fetch latency."""

    @pytest.mark.slow
    @pytest.mark.network
    def test_radrs_single_file_cold(self):
        """First call latency (cold start)."""
        urls = get_test_urls(1)
        elapsed, dt = timed(lambda: rxr.open_datatree(urls[0]), "radrs cold")

        print(f"\nradrs single file (cold): {elapsed:.2f}s")
        assert len(list(dt.children.keys())) > 0

    @pytest.mark.slow
    @pytest.mark.network
    def test_radrs_single_file_warm(self):
        """Second call latency (warm, connection reused)."""
        urls = get_test_urls(2)
        # Warmup
        rxr.open_datatree(urls[0])

        elapsed, dt = timed(lambda: rxr.open_datatree(urls[1]), "radrs warm")

        print(f"\nradrs single file (warm): {elapsed:.2f}s")
        assert len(list(dt.children.keys())) > 0

    @pytest.mark.slow
    @pytest.mark.network
    @pytest.mark.xfail(reason="xradar needs local file access, direct HTTPS doesn't work")
    def test_xradar_single_file(self):
        """xradar baseline for comparison."""
        xradar = pytest.importorskip("xradar")

        urls = get_test_urls(1)
        # Convert to HTTPS for xradar/fsspec
        https_url = urls[0].replace(
            "s3://unidata-nexrad-level2",
            "https://unidata-nexrad-level2.s3.amazonaws.com"
        )

        def fetch():
            # xradar typically uses fsspec for remote files
            return xradar.io.open_nexradlevel2_datatree(https_url)

        elapsed, dt = timed(fetch, "xradar")

        print(f"\nxradar single file: {elapsed:.2f}s")
        assert len(list(dt.children.keys())) > 0


class TestS3SequentialBenchmark:
    """Benchmark sequential multi-file fetch."""

    @pytest.mark.slow
    @pytest.mark.network
    def test_radrs_sequential(self):
        """radrs sequential fetch (sync API)."""
        urls = get_test_urls(N_VOLUMES_QUICK)

        start = time.perf_counter()
        results = [rxr.open_datatree(url) for url in urls]
        elapsed = time.perf_counter() - start

        volumes_per_sec = len(urls) / elapsed
        print(f"\nradrs sequential ({len(urls)} files): {elapsed:.2f}s ({volumes_per_sec:.2f} vol/s)")
        assert len(results) == len(urls)

    @pytest.mark.slow
    @pytest.mark.network
    @pytest.mark.xfail(reason="xradar needs local file access, direct HTTPS doesn't work")
    def test_xradar_sequential(self):
        """xradar sequential fetch for comparison."""
        xradar = pytest.importorskip("xradar")

        urls = get_test_urls(N_VOLUMES_QUICK)
        https_urls = [
            url.replace("s3://unidata-nexrad-level2",
                       "https://unidata-nexrad-level2.s3.amazonaws.com")
            for url in urls
        ]

        start = time.perf_counter()
        results = [xradar.io.open_nexradlevel2_datatree(url) for url in https_urls]
        elapsed = time.perf_counter() - start

        volumes_per_sec = len(urls) / elapsed
        print(f"\nxradar sequential ({len(urls)} files): {elapsed:.2f}s ({volumes_per_sec:.2f} vol/s)")
        assert len(results) == len(urls)


class TestS3IteratorBenchmark:
    """Benchmark iterator performance."""

    @pytest.mark.slow
    @pytest.mark.network
    def test_radrs_iter_sync(self):
        """radrs sync iterator (no prefetch)."""
        start = time.perf_counter()
        results = []
        for i, dt in enumerate(radrs.iter_volumes(SITE, start=DATE, end=DATE)):
            results.append(dt)
            if i >= N_VOLUMES_QUICK - 1:
                break
        elapsed = time.perf_counter() - start

        volumes_per_sec = len(results) / elapsed
        print(f"\nradrs iter_volumes sync ({len(results)} files): {elapsed:.2f}s ({volumes_per_sec:.2f} vol/s)")
        assert len(results) == N_VOLUMES_QUICK

    @pytest.mark.slow
    @pytest.mark.network
    async def test_radrs_iter_async_prefetch(self):
        """radrs async iterator with prefetch."""
        start = time.perf_counter()
        results = []
        async for dt in radrs.iter_volumes_async(SITE, start=DATE, end=DATE, prefetch=3):
            results.append(dt)
            if len(results) >= N_VOLUMES_QUICK:
                break
        elapsed = time.perf_counter() - start

        volumes_per_sec = len(results) / elapsed
        print(f"\nradrs iter_volumes_async prefetch=3 ({len(results)} files): {elapsed:.2f}s ({volumes_per_sec:.2f} vol/s)")
        assert len(results) == N_VOLUMES_QUICK


class TestS3AsyncBenchmark:
    """Benchmark async concurrent fetch."""

    @pytest.mark.slow
    @pytest.mark.network
    async def test_radrs_async_concurrent(self):
        """radrs async concurrent fetch."""
        urls = get_test_urls(N_VOLUMES_QUICK)

        start = time.perf_counter()
        tasks = [rxr.open_datatree_async(url) for url in urls]
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start

        volumes_per_sec = len(urls) / elapsed
        print(f"\nradrs async concurrent ({len(urls)} files): {elapsed:.2f}s ({volumes_per_sec:.2f} vol/s)")
        assert len(results) == len(urls)


class TestConnectionReuse:
    """Verify connection reuse is working."""

    @pytest.mark.slow
    @pytest.mark.network
    def test_connection_reuse_speedup(self):
        """Second call should be faster due to connection reuse."""
        urls = get_test_urls(3)

        # First call (cold)
        t1, _ = timed(lambda: rxr.open_datatree(urls[0]), "cold")

        # Second call (warm - same connection)
        t2, _ = timed(lambda: rxr.open_datatree(urls[1]), "warm")

        # Third call (still warm)
        t3, _ = timed(lambda: rxr.open_datatree(urls[2]), "warm2")

        print(f"\nConnection reuse test:")
        print(f"  1st call (cold): {t1:.2f}s")
        print(f"  2nd call (warm): {t2:.2f}s")
        print(f"  3rd call (warm): {t3:.2f}s")

        # Warm calls should generally be faster (no TCP/TLS setup)
        # Allow some variance due to network conditions
        avg_warm = (t2 + t3) / 2
        print(f"  Speedup: {t1/avg_warm:.1f}x (cold/warm)")


def run_quick_comparison():
    """Quick comparison script (run without pytest)."""
    import fsspec
    import xradar as xd

    print("=" * 60)
    print("radrs vs xradar S3 Performance Comparison")
    print("=" * 60)

    # Get test URLs
    print(f"\nFetching volume list for {SITE} on {DATE}...")
    urls = get_test_urls(N_VOLUMES_QUICK)
    print(f"Testing with {len(urls)} volumes")

    # radrs single file
    print("\n--- Single File Fetch ---")
    t1, _ = timed(lambda: rxr.open_datatree(urls[0]), "radrs cold")
    print(f"radrs (cold):  {t1:.2f}s")

    t2, _ = timed(lambda: rxr.open_datatree(urls[1]), "radrs warm")
    print(f"radrs (warm):  {t2:.2f}s")

    # xradar comparison (needs simplecache - downloads then parses)
    try:
        def fetch_xradar(url, tmpdir):
            # xradar needs local file access (random seeks), so use simplecache
            local = fsspec.open_local(
                f"simplecache::{url}",
                s3={"anon": True},
                filecache={"cache_storage": tmpdir, "same_names": True}
            )
            return xd.io.open_nexradlevel2_datatree(local)

        with tempfile.TemporaryDirectory() as tmpdir:
            t3, _ = timed(lambda: fetch_xradar(urls[2], tmpdir), "xradar")
        print(f"xradar:        {t3:.2f}s")
        print(f"Speedup (warm): {t3/t2:.1f}x")
    except ImportError as e:
        print(f"xradar/fsspec not installed, skipping comparison: {e}")

    # Sequential fetch
    print("\n--- Sequential Fetch ({} files) ---".format(len(urls)))
    start = time.perf_counter()
    for url in urls:
        rxr.open_datatree(url)
    radrs_seq = time.perf_counter() - start
    print(f"radrs: {radrs_seq:.2f}s ({len(urls)/radrs_seq:.2f} vol/s)")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            start = time.perf_counter()
            for url in urls:
                local = fsspec.open_local(
                    f"simplecache::{url}",
                    s3={"anon": True},
                    filecache={"cache_storage": tmpdir, "same_names": True}
                )
                xd.io.open_nexradlevel2_datatree(local)
            xradar_seq = time.perf_counter() - start
        print(f"xradar: {xradar_seq:.2f}s ({len(urls)/xradar_seq:.2f} vol/s)")
        print(f"Speedup: {xradar_seq/radrs_seq:.1f}x")
    except ImportError:
        pass

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_quick_comparison()
