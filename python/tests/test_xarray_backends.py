"""Tests for radrs xarray backend entry points."""

from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path

import pytest
import xarray as xr
from radrs import backends
from radrs import raystack as radrs_raystack
from radrs import xradar as radrs_xradar
from radrs._source_format import DEFAULT_SOURCE_FORMAT


def _clear_engine_cache() -> None:
    cache_clear = getattr(xr.backends.list_engines, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()


def _assert_dataset_equal(actual: xr.Dataset, expected: xr.Dataset) -> None:
    xr.testing.assert_equal(actual.drop_encoding(), expected.drop_encoding())


def _open_direct(engine: str, source: str | bytes) -> xr.DataTree:
    if engine == "radrs-xradar":
        return radrs_xradar.open_datatree(source)
    if engine == "radrs-raystack":
        return radrs_raystack.open_datatree(source)
    raise AssertionError(f"Unexpected test engine: {engine}")


@pytest.fixture(autouse=True)
def clear_xarray_engine_cache() -> None:
    _clear_engine_cache()


def test_xarray_lists_radrs_engines() -> None:
    engines = xr.backends.list_engines()

    assert isinstance(engines["radrs-xradar"], backends.RadrsXradarBackend)
    assert isinstance(engines["radrs-raystack"], backends.RadrsRaystackBackend)


@pytest.mark.parametrize(
    ("engine", "expected_group"),
    [
        ("radrs-xradar", "sweep_0"),
        ("radrs-raystack", "returns"),
    ],
)
def test_open_datatree_matches_direct_reader(
    test_file_path: str,
    engine: str,
    expected_group: str,
) -> None:
    actual = xr.open_datatree(test_file_path, engine=engine)
    expected = _open_direct(engine, test_file_path)

    xr.testing.assert_equal(actual, expected)
    assert expected_group in actual.children


@pytest.mark.parametrize(
    ("engine", "group", "expected_variable"),
    [
        ("radrs-xradar", "sweep_0", "DBZH"),
        ("radrs-raystack", "returns", "DBZH"),
    ],
)
def test_group_selection_matches_direct_reader(
    test_file_path: str,
    engine: str,
    group: str,
    expected_variable: str,
) -> None:
    expected = _open_direct(engine, test_file_path)[group].dataset

    for group_alias in (group, f"/{group}"):
        actual_dataset = xr.open_dataset(
            test_file_path,
            engine=engine,
            group=group_alias,
        )
        actual_tree = xr.open_datatree(
            test_file_path,
            engine=engine,
            group=group_alias,
        )

        _assert_dataset_equal(actual_dataset, expected)
        _assert_dataset_equal(actual_tree.dataset, expected)
        assert not actual_tree.children
        assert expected_variable in actual_dataset


@pytest.mark.parametrize("engine", ["radrs-xradar", "radrs-raystack"])
def test_open_dataset_defaults_to_root_group(
    test_file_path: str,
    engine: str,
) -> None:
    expected = _open_direct(engine, test_file_path).dataset

    for group in (None, "", ".", "/"):
        actual = xr.open_dataset(test_file_path, engine=engine, group=group)
        _assert_dataset_equal(actual, expected)


def test_xradar_engine_options_match_direct_reader(test_file_path: str) -> None:
    actual = xr.open_datatree(
        test_file_path,
        engine="radrs-xradar",
        sort_by_azimuth=True,
        storage_options={},
        format=DEFAULT_SOURCE_FORMAT,
    )
    expected = radrs_xradar.open_datatree(
        test_file_path,
        sort_by_azimuth=True,
        storage_options={},
        format=DEFAULT_SOURCE_FORMAT,
    )

    xr.testing.assert_equal(actual, expected)


def test_raystack_engine_options_match_direct_reader(test_file_path: str) -> None:
    actual = xr.open_datatree(
        test_file_path,
        engine="radrs-raystack",
        fold_size=128,
        include_activity=False,
        storage_options={},
        format=DEFAULT_SOURCE_FORMAT,
    )
    expected = radrs_raystack.open_datatree(
        test_file_path,
        fold_size=128,
        include_activity=False,
        storage_options={},
        format=DEFAULT_SOURCE_FORMAT,
    )

    xr.testing.assert_equal(actual, expected)
    assert "activity" not in actual.children
    assert actual["returns"].sizes["range"] == 128


@pytest.mark.parametrize(
    ("engine", "selected_group"),
    [
        ("radrs-xradar", "sweep_0"),
        ("radrs-raystack", "returns"),
    ],
)
def test_open_groups_matches_direct_reader(
    test_file_path: str,
    engine: str,
    selected_group: str,
) -> None:
    expected_tree = _open_direct(engine, test_file_path)
    actual_groups = xr.open_groups(test_file_path, engine=engine)
    expected_groups = {
        "/" if key == "." else f"/{key}": node.dataset
        for key, node in expected_tree.subtree_with_keys
    }

    assert set(actual_groups) == set(expected_groups)
    for group, expected in expected_groups.items():
        _assert_dataset_equal(actual_groups[group], expected)

    selected = xr.open_groups(
        test_file_path,
        engine=engine,
        group=selected_group,
    )
    assert set(selected) == {"."}
    _assert_dataset_equal(selected["."], expected_tree[selected_group].dataset)


@pytest.mark.parametrize(
    ("engine", "group"),
    [
        ("radrs-xradar", "sweep_0"),
        ("radrs-raystack", "returns"),
    ],
)
def test_drop_variables_applies_to_selected_group(
    test_file_path: str,
    engine: str,
    group: str,
) -> None:
    actual = xr.open_dataset(
        test_file_path,
        engine=engine,
        group=group,
        drop_variables=["DBZH", "VRADH"],
    )

    assert "DBZH" not in actual
    assert "VRADH" not in actual


def test_supported_source_types_match(test_file_path: str, test_file_bytes: bytes) -> None:
    expected = radrs_xradar.open_datatree(test_file_path)
    sources = (
        Path(test_file_path),
        test_file_bytes,
        memoryview(test_file_bytes),
        BytesIO(test_file_bytes),
    )

    for source in sources:
        actual = xr.open_datatree(source, engine="radrs-xradar")
        xr.testing.assert_equal(actual, expected)


def test_file_like_inputs_must_be_binary() -> None:
    with pytest.raises(TypeError, match="binary mode"):
        backends.RadrsXradarBackend().open_datatree(StringIO("text"))


def test_unsupported_source_type_is_rejected() -> None:
    with pytest.raises(TypeError, match="accept str"):
        backends.RadrsXradarBackend().open_datatree(object())


@pytest.mark.parametrize(
    ("engine", "group"),
    [
        ("radrs-xradar", "returns"),
        ("radrs-raystack", "sweep_0"),
    ],
)
def test_invalid_group_error_lists_valid_groups(
    test_file_path: str,
    engine: str,
    group: str,
) -> None:
    with pytest.raises(ValueError, match="Valid groups"):
        xr.open_dataset(test_file_path, engine=engine, group=group)

    with pytest.raises(ValueError, match="Valid groups"):
        xr.open_datatree(test_file_path, engine=engine, group=group)


@pytest.mark.parametrize("engine", ["radrs-xradar", "radrs-raystack"])
def test_decoder_kwargs_are_rejected(engine: str) -> None:
    with pytest.raises(ValueError, match="already return decoded"):
        xr.open_datatree(b"", engine=engine, mask_and_scale=False)


@pytest.mark.parametrize("engine", ["radrs-xradar", "radrs-raystack"])
def test_unknown_backend_kwargs_are_rejected(engine: str) -> None:
    with pytest.raises(TypeError, match="Unsupported radrs-"):
        xr.open_datatree(b"", engine=engine, unsupported_option=True)


@pytest.mark.parametrize("engine", ["radrs-xradar", "radrs-raystack"])
def test_invalid_format_is_rejected(engine: str) -> None:
    with pytest.raises(ValueError, match="Supported formats: nexrad-level2"):
        xr.open_datatree(b"", engine=engine, format="odim")


def test_nested_group_helpers_reroot_subtrees() -> None:
    tree = xr.DataTree.from_dict(
        {
            "/": xr.Dataset({"root": 1}),
            "/parent": xr.Dataset({"parent": 2}),
            "/parent/child": xr.Dataset({"child": 3}),
        }
    )

    subtree = backends._subtree_for_group(tree, "/parent", "test-engine")
    groups = backends._groups_as_dict(subtree, relative=True)

    assert set(groups) == {".", "child"}
    _assert_dataset_equal(subtree.dataset, tree["parent"].dataset)
    _assert_dataset_equal(subtree["child"].dataset, tree["parent/child"].dataset)
    _assert_dataset_equal(
        backends._dataset_for_group(tree, None, "test-engine"),
        tree.dataset,
    )


def test_nested_group_helpers_report_valid_groups() -> None:
    tree = xr.DataTree.from_dict({"/": xr.Dataset(), "/known": xr.Dataset()})

    with pytest.raises(ValueError, match="Valid groups: /, known"):
        backends._subtree_for_group(tree, "missing", "test-engine")

    with pytest.raises(ValueError, match="Valid groups: /, known"):
        backends._dataset_for_group(tree, "missing", "test-engine")
