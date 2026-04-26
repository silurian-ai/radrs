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

Wheels are published for Linux, macOS, and Windows on x86_64 and aarch64.

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
