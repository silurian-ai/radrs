"""Real-data fixtures for radrs tests.

Most behavioral tests use Py-ART's compact Message 31 archive. Tests which
need multiple sweeps, dual-polarization moments, or compatibility with the
upstream xradar reader opt into the full-volume fixtures explicitly.
"""

from collections.abc import Iterator, Mapping
from hashlib import sha256
import os
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TEST_DATA_DIR = _REPO_ROOT.parent / "nexrad" / "downloads"
TEST_DATA_DIR = Path(os.environ.get("RADRS_TEST_DATA_DIR", _DEFAULT_TEST_DATA_DIR))
FULL_VOLUME_PATHS = (
    TEST_DATA_DIR / "KDMX20220305_232324_V06",
    TEST_DATA_DIR / "KCRP20170826_044114_V06",
)


def _update_signature(digest, value: object) -> None:
    """Hash nested scientific objects to detect mutation of shared fixtures."""
    import numpy as np

    if isinstance(value, np.ma.MaskedArray):
        _update_signature(digest, np.asarray(value.data))
        _update_signature(digest, np.ma.getmaskarray(value))
    elif isinstance(value, np.ndarray):
        digest.update(str(value.dtype).encode())
        digest.update(repr(value.shape).encode())
        if value.dtype.hasobject:
            digest.update(repr(value.tolist()).encode())
        else:
            digest.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, Mapping):
        for key in sorted(value, key=repr):
            digest.update(repr(key).encode())
            _update_signature(digest, value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            _update_signature(digest, item)
    else:
        digest.update(repr(value).encode())


def _mapping_signature(value: Mapping[object, object]) -> str:
    digest = sha256()
    _update_signature(digest, value)
    return digest.hexdigest()


def _datatree_signature(tree) -> str:
    digest = sha256()
    for node in tree.subtree:
        digest.update(node.path.encode())
        dataset = node.dataset
        _update_signature(digest, dataset.attrs)
        for name in sorted(dataset.variables):
            digest.update(name.encode())
            variable = dataset[name]
            digest.update(repr(variable.dims).encode())
            _update_signature(digest, variable.values)
    return digest.hexdigest()


def _pyart_signature(radar) -> str:
    digest = sha256()
    for name in (
        "fields",
        "metadata",
        "time",
        "range",
        "azimuth",
        "elevation",
        "fixed_angle",
        "sweep_start_ray_index",
        "sweep_end_ray_index",
        "instrument_parameters",
    ):
        digest.update(name.encode())
        _update_signature(digest, getattr(radar, name, None))
    return digest.hexdigest()


def _compact_volume_path() -> Path:
    """Resolve the stable real NEXRAD sample shipped by the CI dependency."""
    try:
        import pyart.testing
    except ImportError:
        pytest.fail(
            "The compact NEXRAD fixtures require the declared arm-pyart dev dependency. "
            "Run `uv sync --locked --group dev` before running the test suite."
        )

    asset = getattr(pyart.testing, "NEXRAD_ARCHIVE_MSG31_COMPRESSED_FILE", None)
    if asset is None:
        pytest.fail(
            "pyart.testing.NEXRAD_ARCHIVE_MSG31_COMPRESSED_FILE is unavailable; "
            "the installed arm-pyart version does not provide the expected test asset."
        )

    path = Path(asset)
    if not path.is_file():
        pytest.fail(f"Py-ART's compact NEXRAD test asset is missing: {path}")
    return path


@pytest.fixture(scope="session")
def compact_volume_path() -> str:
    """Path to Py-ART's compact real Message 31 NEXRAD archive."""
    return os.fspath(_compact_volume_path())


@pytest.fixture(scope="session")
def compact_volume_bytes(compact_volume_path: str) -> bytes:
    """Bytes for the compact real Message 31 NEXRAD archive."""
    return Path(compact_volume_path).read_bytes()


@pytest.fixture(scope="session")
def full_volume_path() -> str:
    """Path to a local full multi-sweep, dual-polarization NEXRAD volume."""
    configured_path = os.environ.get("RADRS_FULL_TEST_FILE")
    if configured_path:
        path = Path(configured_path)
        if path.is_file():
            return os.fspath(path)
        pytest.fail(f"RADRS_FULL_TEST_FILE does not name a readable file: {path}")

    for path in FULL_VOLUME_PATHS:
        if path.is_file():
            return os.fspath(path)
    pytest.fail(
        "Full multi-sweep/dual-pol tests require RADRS_FULL_TEST_FILE or one of "
        "these local real volumes: "
        + ", ".join(os.fspath(path) for path in FULL_VOLUME_PATHS)
    )


@pytest.fixture(scope="session")
def full_volume_bytes(full_volume_path: str) -> bytes:
    """Bytes for the default full multi-sweep, dual-pol volume."""
    return Path(full_volume_path).read_bytes()


# Compatibility aliases intentionally point at the compact tier. Callers that
# depend on full-volume capabilities must request full_volume_* by name.
@pytest.fixture(scope="session")
def test_file_path(compact_volume_path: str) -> str:
    return compact_volume_path


@pytest.fixture(scope="session")
def test_file_bytes(compact_volume_bytes: bytes) -> bytes:
    return compact_volume_bytes


@pytest.fixture(scope="session")
def compact_raystack(
    compact_volume_bytes: bytes,
) -> Iterator[dict[str, dict[str, object]]]:
    """Mutation-guarded default parse for read-only characterization."""
    import radrs.raystack as rrs

    raystack = rrs.parse(compact_volume_bytes)
    signature = _mapping_signature(raystack)
    yield raystack
    assert _mapping_signature(raystack) == signature, "Shared compact Raystack was mutated"


@pytest.fixture(scope="session")
def compact_radrs_datatree(compact_volume_path: str) -> Iterator[object]:
    """Mutation-guarded radrs DataTree for read-only characterization."""
    import radrs.xradar as rxr

    tree = rxr.open_datatree(compact_volume_path)
    signature = _datatree_signature(tree)
    yield tree
    assert _datatree_signature(tree) == signature, "Shared compact radrs DataTree was mutated"


@pytest.fixture(scope="session")
def full_radrs_datatree(full_volume_path: str) -> Iterator[object]:
    """Mutation-guarded full radrs DataTree for read-only comparisons."""
    import radrs.xradar as rxr

    tree = rxr.open_datatree(full_volume_path)
    signature = _datatree_signature(tree)
    yield tree
    assert _datatree_signature(tree) == signature, "Shared full radrs DataTree was mutated"


@pytest.fixture(scope="session")
def full_xradar_datatree(full_volume_path: str) -> Iterator[object]:
    """Mutation-guarded upstream-xradar DataTree for read-only comparisons."""
    try:
        import xradar as xd
    except ImportError:
        pytest.fail("The declared xradar test dependency is not installed")
    tree = xd.io.open_nexradlevel2_datatree(full_volume_path)
    signature = _datatree_signature(tree)
    yield tree
    assert _datatree_signature(tree) == signature, "Shared upstream xradar DataTree was mutated"


@pytest.fixture(scope="session")
def full_pyart_radar(full_volume_path: str) -> Iterator[object]:
    """Mutation-guarded Py-ART Radar for read-only compatibility assertions."""
    try:
        import pyart
    except ImportError:
        pytest.fail("The declared arm-pyart dev dependency is not installed")
    read_fn = getattr(pyart.io, "read_nexrad_archive", None)
    if read_fn is None:
        pytest.fail("pyart.io.read_nexrad_archive is unavailable")
    radar = read_fn(full_volume_path)
    signature = _pyart_signature(radar)
    yield radar
    assert _pyart_signature(radar) == signature, "Shared Py-ART Radar was mutated"
