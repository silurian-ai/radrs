# Benchmarks

## S3 benchmark

Compares radrs S3 fetch performance (and optionally xradar) using real NEXRAD data.

Run a quick pass (default):

```bash
uv run python benchmarks/s3_benchmark.py
```

Full run (more volumes):

```bash
uv run python benchmarks/s3_benchmark.py --mode full
```

Enable xradar comparison (requires `xradar` and `fsspec`):

```bash
uv run python benchmarks/s3_benchmark.py --xradar
```

Customize site/date/volume count:

```bash
uv run python benchmarks/s3_benchmark.py --site KTLX --date 2024-03-15 --n 3
```

Notes:
- These benchmarks hit S3 and are sensitive to network conditions.
- Expect variance between runs; use them for relative comparisons.
