# radrs

[![PyPI](https://img.shields.io/pypi/v/radrs.svg)](https://pypi.org/project/radrs/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-silurian--ai.github.io-blue)](https://silurian-ai.github.io/radrs/dev/)

Fast NEXRAD Level 2 processing for Python.

radrs parses NEXRAD weather radar data with direct multi-cloud streaming,
async I/O, and two output formats: a CfRadial2 xarray DataTree (xradar-compatible)
or a flattened "raystack" xarray DataTree optimized for ML pipelines.

- Multi-cloud streaming (`s3://`, `gs://`, `az://`, local) with connection pooling
- Async/await API with prefetch
- Time-bounded archive iteration via `NexradL2ArchiveIter`
- Built-in QC: RHOHV thresholding, sun spike detection, VRADH dealiasing
- Multi-threaded bzip2 decompression

## Install

```bash
uv add radrs
```

Or with pip: `pip install radrs`. Requires Python 3.11+.

## Quick start

```python
import radrs.xradar as rxr
import radrs.raystack as rrs

src = "s3://unidata-nexrad-level2/2024/03/15/KTLX/KTLX20240315_000217_V06"

# xradar-compatible DataTree
dt = rxr.open_datatree(src)

# Raystack DataTree — flat layout, ML-friendly
rdt = rrs.open_datatree(src, fold_size=128)
```

For S3 archive iteration, async I/O, QC, and the raystack format reference, see
the [full documentation](https://silurian-ai.github.io/radrs/dev/).

## Documentation

- [User guide & API reference](https://silurian-ai.github.io/radrs/dev/)
- [Changelog](CHANGELOG.md)

## xarray engines

radrs registers two explicit xarray engines:

| Engine | Schema | Use case |
|--------|--------|----------|
| `radrs-xradar` | xradar/CfRadial-style sweep tree (`/sweep_0`, `/sweep_1`, ...) | Ecosystem compatibility with xarray, xradar, and Py-ART workflows |
| `radrs-raystack` | Flattened raystack tree (`/vcps`, `/sweeps`, `/returns`, `/activity`) | radrs-native analytics, batching, and ML-oriented storage |

```python
import xarray as xr

dt = xr.open_datatree(source, engine="radrs-xradar", sort_by_azimuth=True)
sweep = xr.open_dataset(source, engine="radrs-xradar", group="sweep_0")

rs_dt = xr.open_datatree(source, engine="radrs-raystack", fold_size=128)
returns = xr.open_dataset(
    source,
    engine="radrs-raystack",
    group="returns",
    fold_size=128,
)
```

Both engines accept `format="nexrad-level2"`, which is the current default and
the only supported source format.

The direct `radrs.xradar` and `radrs.raystack` APIs remain the canonical APIs.
The xarray engines are thin adapters for workflows that prefer xarray dispatch.

## License

Apache-2.0 — see [LICENSE](LICENSE).
