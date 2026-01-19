"""Ray integration tests for radrs."""

from pathlib import Path

import pytest

ray = pytest.importorskip("ray")


@pytest.mark.slow
@pytest.mark.network
def test_ray_worker_can_open_datatree():
    """Ensure a Ray worker can import and use radrs in a separate process."""

    url = "s3://unidata-nexrad-level2/2024/01/15/KTLX/KTLX20240115_000309_V06"

    try:
        import radrs  # Ensure local package is available for py_modules.
    except Exception as exc:  # pragma: no cover - environment setup issue
        pytest.skip(f"radrs not importable: {exc}")

    module_path = Path(radrs.__file__).parent

    ray.init(
        num_cpus=1,
        ignore_reinit_error=True,
        runtime_env={"py_modules": [str(module_path)]},
    )

    try:
        @ray.remote
        def load_vcp(target_url: str):
            import radrs.xradar as rxr

            dt = rxr.open_datatree(target_url)
            sweeps = [k for k in dt.children.keys() if k.startswith("sweep_")]
            return dt.attrs.get("volume_coverage_pattern"), len(sweeps)

        vcp, sweep_count = ray.get(load_vcp.remote(url))
    finally:
        ray.shutdown()

    assert vcp is not None, "Expected VCP in DataTree attrs"
    assert sweep_count > 0, "Expected sweeps in DataTree"
