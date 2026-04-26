# Install

radrs ships as a precompiled wheel on PyPI. Python 3.11+ is required.

=== "uv"

    ```bash
    uv add radrs
    ```

=== "pip"

    ```bash
    pip install radrs
    ```

Wheels are published for Linux (x86_64, aarch64) and macOS (x86_64, arm64).
On other platforms, pip will build from source and you will need a Rust
toolchain.

## From source

Clone the repository and build the extension with `maturin develop`:

```bash
git clone https://github.com/silurian-ai/radrs.git
cd radrs
uv sync
maturin develop --release
```

`uv sync` and `uv run` will also build a debug extension automatically; the
`maturin develop --release` step is only needed for release-mode performance.
Run the test suite with `uv run pytest python/tests/`.
