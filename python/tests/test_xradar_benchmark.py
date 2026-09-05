"""Check the benchmark's materialization boundary with real xarray readers."""

from collections.abc import Callable
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "xradar_radrs_bench.py"
_SPEC = importlib.util.spec_from_file_location("xradar_radrs_bench", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


def test_time_load_materializes_every_node(tmp_path: Path) -> None:
    """Root, child, and nested arrays must all outlive their backing files."""
    expected = xr.Dataset({"signal": ("gate", np.arange(6, dtype=np.float32))})
    paths = [tmp_path / f"sweep_{i}.nc" for i in range(3)]
    for path in paths:
        expected.to_netcdf(path, engine="scipy")

    tree = xr.DataTree.from_dict(
        {
            group: xr.open_dataset(path, engine="scipy", cache=False)
            for group, path in zip(("/", "/sweep_0", "/sweep_0/nested"), paths, strict=True)
        }
    )
    try:
        assert all(not node.dataset["signal"].variable._in_memory for node in tree.subtree)
        elapsed, loaded = benchmark._time_load(lambda: tree)
        assert loaded is tree
        assert elapsed >= 0
        assert all(node.dataset["signal"].variable._in_memory for node in loaded.subtree)
        tree.close()
        for path in paths:
            path.unlink()
        for node in loaded.subtree:
            xr.testing.assert_equal(node.dataset["signal"], expected["signal"])
    finally:
        tree.close()


def test_time_load_preserves_eager_arrays() -> None:
    values = np.arange(6, dtype=np.float32)
    tree = xr.DataTree(dataset=xr.Dataset({"signal": ("gate", values)}))
    _, loaded = benchmark._time_load(lambda: tree)
    assert loaded is tree
    assert loaded["signal"].data is values


@pytest.mark.slow
@pytest.mark.parametrize("reader", ["radrs", "xradar"])
def test_time_load_real_nexrad(reader: str, full_volume_path: str) -> None:
    import radrs.xradar as rxr
    import xradar as xd

    open_tree: Callable[[], xr.DataTree]
    if reader == "radrs":
        open_tree = lambda: rxr.open_datatree(full_volume_path)
    else:
        open_tree = lambda: xd.io.open_nexradlevel2_datatree(full_volume_path)

    _, tree = benchmark._time_load(open_tree)
    try:
        sweeps = [name for name in tree.children if name.startswith("sweep_")]
        assert len(sweeps) > 1
        assert tree["sweep_0"]["DBZH"].size > 0
        assert all(
            variable._in_memory
            for node in tree.subtree
            for variable in node.dataset.variables.values()
        )
    finally:
        tree.close()
