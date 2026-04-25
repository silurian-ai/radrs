# Install

Requires Python 3.11 or newer.

## With uv

```bash
uv add radrs
```

## With pip

```bash
pip install radrs
```

The package page on [PyPI](https://pypi.org/project/radrs/) lists every
released wheel.

## Optional extras

Install extras with `uv add 'radrs[<extra>]'` or
`pip install 'radrs[<extra>]'`.

- `zarr` — pulls in `zarr>=3` for writing raystack DataTrees to Zarr.
- `ray` — pulls in `ray` for distributed batch processing across a
  cluster.
- `test` — `pytest`, `pytest-asyncio`, `pytest-benchmark`, and `xradar`
  for running the test suite against the package.

## From source

For development, clone the repo and build the Rust extension in place
with [`maturin`](https://www.maturin.rs/).

```bash
git clone https://github.com/silurian-ai/radrs.git
cd radrs
uv sync
uv run maturin develop --release
```

`uv sync` installs the dev dependency group (including `maturin`).
`maturin develop --release` builds the optimized Rust extension and
links it into the active environment. The `cache-keys` entry in
`pyproject.toml` triggers a rebuild on the next `uv sync` whenever
`Cargo.toml`, `Cargo.lock`, or any `src/**/*.rs` file changes.
