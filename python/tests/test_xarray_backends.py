"""Tests for radrs xarray backend entry points."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
import xarray as xr

from radrs import backends
from radrs._source_format import DEFAULT_SOURCE_FORMAT


def _clear_engine_cache() -> None:
    cache_clear = getattr(xr.backends.list_engines, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()


def _assert_dataset_content_equal(actual: xr.Dataset, expected: xr.Dataset) -> None:
    xr.testing.assert_equal(actual.drop_encoding(), expected.drop_encoding())


@pytest.fixture(autouse=True)
def clear_xarray_engine_cache() -> None:
    _clear_engine_cache()


@pytest.fixture
def xradar_tree() -> xr.DataTree:
    return xr.DataTree.from_dict(
        {
            "/": xr.Dataset({"volume_number": 1}),
            "/sweep_0": xr.Dataset(
                data_vars={
                    "DBZH": (("azimuth", "range"), [[1.0, 2.0]]),
                    "VRADH": (("azimuth", "range"), [[3.0, 4.0]]),
                    "sweep_number": 0,
                },
                coords={
                    "azimuth": ("azimuth", [10.0]),
                    "range": ("range", [0.0, 1000.0]),
                },
            ),
            "/radar_parameters": xr.Dataset(),
            "/georeferencing_correction": xr.Dataset(),
            "/radar_calibration": xr.Dataset(),
        }
    )


@pytest.fixture
def raystack_tree() -> xr.DataTree:
    return xr.DataTree.from_dict(
        {
            "/": xr.Dataset(),
            "/vcps": xr.Dataset({"vcp_number": ("vcp_time", [212])}),
            "/sweeps": xr.Dataset({"sweep_number": ("sweep_time", [0])}),
            "/returns": xr.Dataset(
                data_vars={
                    "azimuth": ("return_time", [10.0]),
                    "DBZH": (("return_time", "range"), [[1.0, 2.0]]),
                    "VRADH": (("return_time", "range"), [[3.0, 4.0]]),
                },
                coords={
                    "return_time": ("return_time", [1]),
                    "range": ("range", [0, 1]),
                },
            ),
            "/activity": xr.Dataset({"volume_valid_count": ("moment", [2])}),
        }
    )


@pytest.fixture
def backend_calls(
    monkeypatch: pytest.MonkeyPatch,
    xradar_tree: xr.DataTree,
    raystack_tree: xr.DataTree,
) -> dict[str, list[tuple[object, dict[str, object]]]]:
    calls: dict[str, list[tuple[object, dict[str, object]]]] = {
        "xradar": [],
        "raystack": [],
    }

    def open_xradar(source: object, **kwargs: object) -> xr.DataTree:
        calls["xradar"].append((source, kwargs))
        return xradar_tree.copy(deep=True)

    def open_raystack(source: object, **kwargs: object) -> xr.DataTree:
        calls["raystack"].append((source, kwargs))
        return raystack_tree.copy(deep=True)

    monkeypatch.setattr(backends.radrs_xradar, "open_datatree", open_xradar)
    monkeypatch.setattr(backends.radrs_raystack, "open_datatree", open_raystack)
    return calls


def test_xarray_lists_radrs_engines() -> None:
    engines = xr.backends.list_engines()

    assert "radrs-xradar" in engines
    assert "radrs-raystack" in engines


@pytest.mark.parametrize(
    ("engine", "expected_group"),
    [
        ("radrs-xradar", "sweep_0"),
        ("radrs-raystack", "returns"),
    ],
)
def test_open_datatree_reads_real_volume(
    test_file_path: str,
    engine: str,
    expected_group: str,
) -> None:
    actual = xr.open_datatree(test_file_path, engine=engine)

    assert expected_group in actual.children


@pytest.mark.parametrize(
    ("engine", "group", "expected_variable"),
    [
        ("radrs-xradar", "sweep_0", "DBZH"),
        ("radrs-raystack", "returns", "DBZH"),
    ],
)
def test_open_datatree_reads_real_group(
    test_file_path: str,
    engine: str,
    group: str,
    expected_variable: str,
) -> None:
    actual = xr.open_datatree(test_file_path, engine=engine, group=group)

    assert not actual.children
    assert expected_variable in actual.dataset


def test_xradar_engine_forwards_options(
    backend_calls: dict[str, list[tuple[object, dict[str, object]]]],
    xradar_tree: xr.DataTree,
) -> None:
    actual = xr.open_datatree(
        "volume.ar2v",
        engine="radrs-xradar",
        sort_by_azimuth=True,
        format=DEFAULT_SOURCE_FORMAT,
    )

    xr.testing.assert_equal(actual, xradar_tree)
    assert backend_calls["xradar"] == [
        (
            "volume.ar2v",
            {"sort_by_azimuth": True, "format": DEFAULT_SOURCE_FORMAT},
        )
    ]


def test_raystack_engine_forwards_options(
    backend_calls: dict[str, list[tuple[object, dict[str, object]]]],
    raystack_tree: xr.DataTree,
) -> None:
    default_actual = xr.open_datatree("volume.ar2v", engine="radrs-raystack")
    actual = xr.open_datatree(
        "volume.ar2v",
        engine="radrs-raystack",
        fold_size=256,
        qc=["qc-step"],
        include_activity=False,
        format=DEFAULT_SOURCE_FORMAT,
    )

    xr.testing.assert_equal(default_actual, raystack_tree)
    xr.testing.assert_equal(actual, raystack_tree)
    assert backend_calls["raystack"] == [
        (
            "volume.ar2v",
            {
                "fold_size": None,
                "qc": None,
                "include_activity": True,
                "format": DEFAULT_SOURCE_FORMAT,
            },
        ),
        (
            "volume.ar2v",
            {
                "fold_size": 256,
                "qc": ["qc-step"],
                "include_activity": False,
                "format": DEFAULT_SOURCE_FORMAT,
            },
        )
    ]


def test_open_dataset_extracts_group(
    backend_calls: dict[str, list[tuple[object, dict[str, object]]]],
    xradar_tree: xr.DataTree,
    raystack_tree: xr.DataTree,
) -> None:
    xradar_ds = xr.open_dataset(
        "volume.ar2v",
        engine="radrs-xradar",
        group="sweep_0",
    )
    raystack_ds = xr.open_dataset(
        "volume.ar2v",
        engine="radrs-raystack",
        group="returns",
    )

    _assert_dataset_content_equal(xradar_ds, xradar_tree["sweep_0"].dataset)
    _assert_dataset_content_equal(raystack_ds, raystack_tree["returns"].dataset)


def test_open_datatree_extracts_group_as_root(
    backend_calls: dict[str, list[tuple[object, dict[str, object]]]],
    xradar_tree: xr.DataTree,
    raystack_tree: xr.DataTree,
) -> None:
    xradar_dt = xr.open_datatree(
        "volume.ar2v",
        engine="radrs-xradar",
        group="sweep_0",
    )
    raystack_dt = xr.open_datatree(
        "volume.ar2v",
        engine="radrs-raystack",
        group="returns",
    )

    assert not xradar_dt.children
    assert not raystack_dt.children
    _assert_dataset_content_equal(xradar_dt.dataset, xradar_tree["sweep_0"].dataset)
    _assert_dataset_content_equal(raystack_dt.dataset, raystack_tree["returns"].dataset)


def test_open_groups_returns_all_groups(
    backend_calls: dict[str, list[tuple[object, dict[str, object]]]],
) -> None:
    xradar_groups = xr.open_groups("volume.ar2v", engine="radrs-xradar")
    raystack_groups = xr.open_groups("volume.ar2v", engine="radrs-raystack")

    assert set(xradar_groups) == {
        "/",
        "/sweep_0",
        "/radar_parameters",
        "/georeferencing_correction",
        "/radar_calibration",
    }
    assert set(raystack_groups) == {"/", "/vcps", "/sweeps", "/returns", "/activity"}


def test_open_groups_extracts_relative_group(
    backend_calls: dict[str, list[tuple[object, dict[str, object]]]],
    xradar_tree: xr.DataTree,
    raystack_tree: xr.DataTree,
) -> None:
    xradar_groups = xr.open_groups(
        "volume.ar2v",
        engine="radrs-xradar",
        group="sweep_0",
    )
    raystack_groups = xr.open_groups(
        "volume.ar2v",
        engine="radrs-raystack",
        group="returns",
    )

    assert set(xradar_groups) == {"."}
    assert set(raystack_groups) == {"."}
    _assert_dataset_content_equal(xradar_groups["."], xradar_tree["sweep_0"].dataset)
    _assert_dataset_content_equal(raystack_groups["."], raystack_tree["returns"].dataset)


def test_drop_variables_applies_to_each_schema(
    backend_calls: dict[str, list[tuple[object, dict[str, object]]]],
) -> None:
    xradar_dt = xr.open_datatree(
        "volume.ar2v",
        engine="radrs-xradar",
        drop_variables="DBZH",
    )
    raystack_dt = xr.open_datatree(
        "volume.ar2v",
        engine="radrs-raystack",
        drop_variables="DBZH",
    )

    assert "DBZH" not in xradar_dt["sweep_0"].dataset
    assert "DBZH" not in raystack_dt["returns"].dataset


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (b"data", b"data"),
        (memoryview(b"data"), b"data"),
        (Path("volume.ar2v"), "volume.ar2v"),
        (BytesIO(b"data"), b"data"),
    ],
)
def test_source_normalization(
    backend_calls: dict[str, list[tuple[object, dict[str, object]]]],
    source: object,
    expected: object,
) -> None:
    backends.RadrsXradarBackend().open_datatree(source)

    assert backend_calls["xradar"][-1][0] == expected


def test_file_like_inputs_must_be_binary() -> None:
    with pytest.raises(TypeError, match="binary mode"):
        backends.RadrsXradarBackend().open_datatree(BytesIOText("text"))


@pytest.mark.parametrize("engine", ["radrs-xradar", "radrs-raystack"])
def test_open_dataset_requires_group(
    backend_calls: dict[str, list[tuple[object, dict[str, object]]]],
    engine: str,
) -> None:
    with pytest.raises(ValueError, match="Use xr.open_datatree"):
        xr.open_dataset("volume.ar2v", engine=engine)


@pytest.mark.parametrize(
    ("engine", "group"),
    [
        ("radrs-xradar", "returns"),
        ("radrs-raystack", "sweep_0"),
    ],
)
def test_invalid_group_error_lists_valid_groups(
    backend_calls: dict[str, list[tuple[object, dict[str, object]]]],
    engine: str,
    group: str,
) -> None:
    with pytest.raises(ValueError, match="Valid groups"):
        xr.open_dataset("volume.ar2v", engine=engine, group=group)


@pytest.mark.parametrize(
    ("engine", "group"),
    [
        ("radrs-xradar", "returns"),
        ("radrs-raystack", "sweep_0"),
    ],
)
def test_open_datatree_invalid_group_error_lists_valid_groups(
    backend_calls: dict[str, list[tuple[object, dict[str, object]]]],
    engine: str,
    group: str,
) -> None:
    with pytest.raises(ValueError, match="Valid groups"):
        xr.open_datatree("volume.ar2v", engine=engine, group=group)


def test_decoder_kwargs_are_rejected(
    backend_calls: dict[str, list[tuple[object, dict[str, object]]]],
) -> None:
    with pytest.raises(ValueError, match="already return decoded"):
        xr.open_datatree(
            "volume.ar2v",
            engine="radrs-xradar",
            mask_and_scale=False,
        )


@pytest.mark.parametrize("engine", ["radrs-xradar", "radrs-raystack"])
def test_invalid_format_is_rejected(
    backend_calls: dict[str, list[tuple[object, dict[str, object]]]],
    engine: str,
) -> None:
    with pytest.raises(ValueError, match="Supported formats: nexrad-level2"):
        xr.open_datatree("volume.ar2v", engine=engine, format="odim")


class BytesIOText:
    def __init__(self, value: str) -> None:
        self._value = value

    def read(self) -> str:
        return self._value
