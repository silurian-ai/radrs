# radrs

A high-performance Rust-native NEXRAD Level 2 processing library with Python bindings.

radrs provides fast parsing of NEXRAD weather radar data with direct S3 access, connection pooling, and async support. It offers both xradar-compatible output (xarray DataTree) and a flat "raystack" format optimized for ML pipelines.

## Features

- **Fast parsing**: 10-15x faster than xradar for NEXRAD Level 2 files
- **Direct S3 access**: Stream data from `unidata-nexrad-level2` with connection pooling
- **Async support**: Native async/await API with prefetch pipelines
- **Multiple output formats**: xarray DataTree (xradar-compatible) or raystack (ML-optimized)
- **Quality control**: Built-in RHOHV threshold and sun spike detection
- **Parallel decompression**: Multi-threaded bzip2 decompression

## Installation

```bash
# From source (requires Rust toolchain and uv)
git clone <repo>
cd radrs
uv sync
maturin develop --release
```

## Quick Start

```python
import radrs
import radrs.xradar as rxr
import radrs.raystack as rrs

# Open a single file (local or S3)
dt = rxr.open_datatree("s3://unidata-nexrad-level2/2024/03/15/KTLX/KTLX20240315_120000_V06")
dt = rxr.open_datatree("/path/to/local/file.ar2v")

# Iterate over a date range with prefetch
for dt in radrs.iter_volumes("KTLX", start="2024-03-15", end="2024-03-16", prefetch=5):
    process(dt)

# Async iteration
async for dt in radrs.iter_volumes_async("KTLX", start="2024-03-15", prefetch=5):
    await process(dt)

# Parse directly to raystack format (faster for ML)
rs = rrs.parse(file_bytes, fold_size=128)
print(rs["returns"]["DBZH"].shape)  # (n_returns, 128)
```

## API Reference

### radrs (top-level)

| Function | Description |
|----------|-------------|
| `list_volumes(site, date)` | List available volumes for a site and date |
| `iter_volumes(site, start, end, schema, prefetch)` | Iterate over volumes with optional prefetch |
| `iter_volumes_async(site, start, end, schema, prefetch)` | Async iterator with prefetch |
| `stream_archive(site, poll_interval)` | Poll archive for new volumes (~5 min delay) |

### radrs.xradar

| Function | Description |
|----------|-------------|
| `open_datatree(source)` | Open file/URL/bytes as xarray DataTree |
| `open_datatree_async(source)` | Async version of open_datatree |

### radrs.raystack

| Function | Description |
|----------|-------------|
| `parse(data, fold_size)` | Parse bytes directly to raystack dict (fastest) |
| `from_datatree(dt, fold_size)` | Convert xradar-style DataTree to raystack dict |
| `to_datatree(rs)` | Convert raystack dict to xradar-style DataTree |
| `open_datatree(source, fold_size)` | Open file as raystack-style DataTree (flat layout) |

### radrs.qc

| Function | Description |
|----------|-------------|
| `rhohv_threshold(rhohv, threshold=0.8)` | Mask by correlation coefficient |
| `sun_spike(dbzh, fill_threshold, corr_threshold)` | Detect sun spike contamination |

## Output Formats

### xarray DataTree (xradar-compatible)

```
DataTree('root')
├── DataTree('sweep_0')
│   └── Dataset: DBZH, VRADH, RHOHV, ZDR, ... (time, range)
├── DataTree('sweep_1')
│   └── ...
```

### Raystack (ML-optimized)

```python
{
    "vcps": {"pattern_number": 215},
    "sweeps": [
        {"elevation_number": 1, "elevation_angle": 0.5, "n_radials": 720, "start_index": 0},
        ...
    ],
    "returns": {
        "azimuth": ndarray(n_returns,),
        "elevation": ndarray(n_returns,),
        "time": ndarray(n_returns,),
        "sweep_idx": ndarray(n_returns,),
        "DBZH": ndarray(n_returns, fold_size),
        "VRADH": ndarray(n_returns, fold_size),
        ...
    }
}
```

## Performance

Benchmarks on Apple M2, single NEXRAD volume (~12 MB):

| Operation | radrs | xradar | Speedup |
|-----------|-------|--------|---------|
| Parse only | 0.22s | 3.25s | 14.6x |
| S3 fetch + parse (warm) | 1.3s | 3.0s | 2.3x |
| Sequential 3 files | 4.2s | 7.5s | 1.8x |
| With prefetch=5 | 7.4s | - | 3.9x vs no prefetch |

Network latency dominates S3 operations; the prefetch pipeline overlaps I/O with parsing.

## Development

```bash
# Setup
uv sync

# Build
maturin develop --release

# Test
uv run pytest python/tests/ -v

# Benchmark S3 performance
uv run pytest python/tests/test_benchmark_s3.py -v -s
```

## Architecture

```
radrs/
├── src/
│   ├── xradar/        # xarray DataTree output
│   ├── raystack/      # ML-optimized flat format
│   ├── qc/            # Quality control functions
│   ├── iter/          # Volume iterators with prefetch
│   └── fetch/         # S3 access with connection pooling
└── python/
    ├── radrs/         # Python package
    └── tests/         # Test suite
```

## Known Differences from xradar

| Behavior | radrs | xradar |
|----------|-------|--------|
| Radial ordering | File order | Sorted by azimuth |
| Below-threshold | NaN | Raw value |
| Range-folded | NaN | Raw value |

Both approaches are valid; radrs uses NaN for cleaner downstream analysis.

## License

[Add license information]
