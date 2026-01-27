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
source = radrs.VolumeSource.nexrad("KTLX", start="2024-03-15", end="2024-03-16")
for dt in radrs.iter_volumes(source, prefetch=5):
    process(dt)

# Async iteration
async for dt in radrs.iter_volumes_async(source, prefetch=5):
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
| `VolumeSource.nexrad(site, start, end)` | Create a NEXRAD archive volume source |
| `iter_volumes(source, output, qc, fold_size, prefetch, sort_by_azimuth)` | Iterate over volumes from a source |
| `iter_volumes_async(source, output, qc, fold_size, prefetch, sort_by_azimuth)` | Async iterator with prefetch |
| `stream_archive(site, poll_interval)` | Poll archive for new volumes (~5 min delay) |

### radrs.xradar

| Function | Description |
|----------|-------------|
| `open_datatree(source, sort_by_azimuth=False)` | Open file/URL/bytes as xarray DataTree |
| `open_datatree_async(source, sort_by_azimuth=False)` | Async version of open_datatree |

Set `sort_by_azimuth=True` to sort radials by azimuth angle (0°→360°), matching xradar's output order. By default, radrs preserves the original file order.

### radrs.raystack

| Function | Description |
|----------|-------------|
| `parse(data, fold_size)` | Parse bytes directly to raystack dict (fastest) |
| `from_xradar_datatree(dt, fold_size)` | Convert xradar-style DataTree to raystack dict |
| `to_xradar_datatree(rs)` | Convert raystack dict to xradar-style DataTree |
| `to_raystack_datatree(rs)` | Convert raystack dict to raystack-style DataTree |
| `open_datatree(source, fold_size)` | Open file/URL/bytes as raystack-style DataTree (flat layout) |
| `open_datatree_async(source, fold_size)` | Async version of open_datatree |

### radrs.qc

| Function | Description |
|----------|-------------|
| `rhohv_threshold(rhohv, threshold=0.8)` | Mask by correlation coefficient |
| `sun_spike(dbzh, dbzh_threshold=0.0, fill_threshold=0.9, corr_threshold=0.8)` | Detect sun spike contamination |
| `vradh_winding_number(vradh, dbzh=None, nyquist=None, wind_size=3, velocity_texture_threshold=4.0, reflectivity_threshold=0.0, ...)` | VRADH winding number (dealias) |
| `RhohvThreshold(threshold=0.8, vname="rhohv_threshold_mask")` | QC step for raystack parsing |
| `SunSpike(dbzh_threshold=0.0, fill_threshold=0.9, corr_threshold=0.8, vname="sun_spike_mask")` | QC step for raystack parsing |
| `VradhWindingNumber(..., vname="vradh_winding_number")` | QC step for raystack parsing |

## Output Formats

### xarray DataTree (xradar-compatible)

```
DataTree('root')
├── DataTree('sweep_0')
│   └── Dataset: DBZH, VRADH, RHOHV, ZDR, ... (time, range)
├── DataTree('sweep_1')
│   └── ...
```

### Raystack DataTree (flat layout)

```
DataTree('root')
├── DataTree('vcps')
│   └── Dataset: pattern_number, ...
├── DataTree('sweeps')
│   └── Dataset: elevation_number, elevation_angle, n_radials, start_index, ...
├── DataTree('qc')
│   └── Dataset: rhohv_threshold_mask, sun_spike_mask, vradh_winding_number, ...
├── DataTree('activity')
│   └── Dataset: ray_valid_count, ray_valid_fraction, sweep_valid_count, sweep_valid_fraction, volume_valid_count, volume_valid_fraction
└── DataTree('returns')
    └── Dataset: azimuth, elevation, time, sweep_idx, DBZH, VRADH, ... (n_returns, fold_size)
```

### Raystack dict (ML-optimized)

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
    },
    "qc": {
        "rhohv_threshold_mask": ndarray(n_returns, fold_size),
        "sun_spike_mask": ndarray(n_returns, fold_size),
        "vradh_winding_number": ndarray(n_returns, fold_size),
    },
    "activity": {
        "moment": ["DBZH", "VRADH", "WRADH", "ZDR", "PHIDP", "RHOHV", "CCORH"],
        "ray_valid_count": ndarray(n_moments, n_returns),
        "ray_valid_fraction": ndarray(n_moments, n_returns),
        "sweep_valid_count": ndarray(n_moments, n_sweeps),
        "sweep_valid_fraction": ndarray(n_moments, n_sweeps),
        "volume_valid_count": ndarray(n_moments, 1),
        "volume_valid_fraction": ndarray(n_moments, 1),
    },
}
```

### Activity Metrics

Activity metrics summarize data availability for each radar moment (DBZH, VRADH, WRADH, ZDR, PHIDP, RHOHV, CCORH) at three levels:

| Metric | Shape | Description |
|--------|-------|-------------|
| `ray_valid_count` | (n_moments, n_returns) | Count of finite values per ray |
| `ray_valid_fraction` | (n_moments, n_returns) | Fraction of valid gates per ray (count / fold_size) |
| `sweep_valid_count` | (n_moments, n_sweeps) | Total valid values per sweep |
| `sweep_valid_fraction` | (n_moments, n_sweeps) | Fraction valid per sweep (count / (n_radials * fold_size)) |
| `volume_valid_count` | (n_moments, 1) | Total valid values in the volume |
| `volume_valid_fraction` | (n_moments, 1) | Fraction valid across the volume |

Activity is computed during parsing and included by default. To disable:

```python
rs = rrs.parse(file_bytes, include_activity=False)
dt = rrs.open_datatree(source, include_activity=False)
```

Fractions are normalized by the rays actually present in the raystack. If processing a partial volume (e.g., time-sliced), sweep/volume fractions reflect only the observed data, not full-sweep geometry.

### QC masks

Raystack parsing can emit QC outputs under a dedicated `qc` node/dict:

## Compatibility notes (xradar / Py-ART)

- **Sweep ordering:** radrs preserves native sweep order from the file. Some VCPs reuse elevation angles, so pairing by elevation alone can misalign sweeps. When comparing against Py-ART/xradar, align by sweep index when sweep counts match.
- **Raystack folding semantics:** raystack folding uses physical range alignment (first gate + gate spacing) against the sweep grid, which can differ from simple gate-index folding if moments have different gate geometries.
- **Dual-pol decoding:** ZDR/PHIDP decoding depends on the upstream `nexrad` crate. If you see NaNs or mismatches for these moments, check the `nexrad` decode status; DBZH/VRADH/WRADH/RHOHV are expected to match.
- **Azimuth alignment in tests:** floating-point rounding can make exact azimuth equality brittle; radrs tests align by rounded azimuth or nearest-neighbor to avoid false mismatches.

```python
import radrs.raystack as rrs
import radrs.qc as qc

rs = rrs.parse(file_bytes, qc=[qc.RhohvThreshold(), qc.SunSpike()])
mask = rs["qc"]["rhohv_threshold_mask"]  # int8, same shape as DBZH/RHOHV

# Winding number from VRADH dealiasing
rs = rrs.parse(file_bytes, qc=[qc.VradhWindingNumber()])
winding = rs["qc"]["vradh_winding_number"]  # float32, same shape as VRADH
```

## Development

```bash
# Setup
uv sync

# Build
maturin develop --release

# Test
uv run pytest python/tests/ -v

# Benchmark S3 performance
uv run python benchmarks/s3_benchmark.py
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
| Radial ordering | File order (use `sort_by_azimuth=True` to match xradar) | Sorted by azimuth |
| Below-threshold | NaN | Raw value |
| Range-folded | NaN | Raw value |

Both approaches are valid; radrs uses NaN for cleaner downstream analysis.

## License

[Add license information]
