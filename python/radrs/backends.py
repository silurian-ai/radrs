"""xarray backend entry points for radrs."""

from __future__ import annotations

import os
from collections.abc import Iterable

import xarray as xr
from xarray.backends import BackendEntrypoint

from radrs import raystack as radrs_raystack
from radrs import xradar as radrs_xradar


_DECODER_KWARGS = {
    "mask_and_scale",
    "decode_times",
    "decode_timedelta",
    "concat_characters",
    "use_cftime",
    "decode_coords",
}


def _normalize_source(filename_or_obj: object) -> str | bytes:
    if isinstance(filename_or_obj, bytes):
        return filename_or_obj
    if isinstance(filename_or_obj, memoryview):
        return filename_or_obj.tobytes()
    if isinstance(filename_or_obj, str):
        return filename_or_obj
    if isinstance(filename_or_obj, os.PathLike):
        return os.fspath(filename_or_obj)
    if hasattr(filename_or_obj, "read"):
        data = filename_or_obj.read()
        if isinstance(data, bytes):
            return data
        if isinstance(data, memoryview):
            return data.tobytes()
        raise TypeError(
            "radrs xarray backends require file-like objects opened in binary mode"
        )
    raise TypeError(
        "radrs xarray backends accept str, os.PathLike, bytes, memoryview, "
        "or binary file-like objects"
    )


def _unsupported_decoder_kwargs(kwargs: dict[str, object]) -> None:
    for name in _DECODER_KWARGS:
        if kwargs.get(name) is None:
            kwargs.pop(name, None)

    unsupported = sorted(name for name in _DECODER_KWARGS if name in kwargs)
    if unsupported:
        joined = ", ".join(unsupported)
        raise ValueError(
            "radrs xarray engines already return decoded in-memory xarray objects; "
            f"unsupported decoder keyword(s): {joined}"
        )


def _drop_variables(
    tree: xr.DataTree,
    drop_variables: str | Iterable[str] | None,
) -> xr.DataTree:
    if drop_variables is None:
        return tree

    variables = (
        [drop_variables] if isinstance(drop_variables, str) else list(drop_variables)
    )
    if not variables:
        return tree

    return tree.map_over_datasets(lambda ds: ds.drop_vars(variables, errors="ignore"))


def _valid_groups(tree: xr.DataTree) -> list[str]:
    return ["/" if key == "." else key for key, _ in tree.subtree_with_keys]


def _dataset_for_group(tree: xr.DataTree, group: str | None, engine_name: str) -> xr.Dataset:
    if group is None:
        raise ValueError(
            f"{engine_name} opens a multi-group radar volume. Use xr.open_datatree(...) "
            "or pass group=... to xr.open_dataset(...)."
        )

    try:
        return tree[group].dataset.copy()
    except KeyError as exc:
        groups = ", ".join(_valid_groups(tree))
        raise ValueError(
            f"Unknown group {group!r} for {engine_name}. Valid groups: {groups}"
        ) from exc


def _groups_as_dict(tree: xr.DataTree) -> dict[str, xr.Dataset]:
    groups: dict[str, xr.Dataset] = {}
    for key, node in tree.subtree_with_keys:
        group = "/" if key == "." else f"/{key}"
        groups[group] = node.dataset.copy()
    return groups


class _RadrsBackendBase(BackendEntrypoint):
    supports_groups = True
    open_dataset_parameters = (
        "filename_or_obj",
        "drop_variables",
        "mask_and_scale",
        "decode_times",
        "decode_timedelta",
        "concat_characters",
        "use_cftime",
        "decode_coords",
        "group",
    )

    def guess_can_open(self, filename_or_obj: object) -> bool:
        return False


class RadrsXradarBackend(_RadrsBackendBase):
    """xarray backend for the radrs xradar-compatible DataTree schema."""

    open_dataset_parameters = _RadrsBackendBase.open_dataset_parameters + (
        "sort_by_azimuth",
    )
    description = "Fast radrs NEXRAD Level II reader returning xradar-style DataTrees"

    def open_datatree(
        self,
        filename_or_obj: object,
        *,
        drop_variables: str | Iterable[str] | None = None,
        sort_by_azimuth: bool = False,
        **kwargs: object,
    ) -> xr.DataTree:
        _unsupported_decoder_kwargs(kwargs)
        if kwargs:
            joined = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported radrs-xradar keyword argument(s): {joined}")

        tree = radrs_xradar.open_datatree(
            _normalize_source(filename_or_obj),
            sort_by_azimuth=sort_by_azimuth,
        )
        return _drop_variables(tree, drop_variables)

    def open_dataset(
        self,
        filename_or_obj: object,
        *,
        drop_variables: str | Iterable[str] | None = None,
        group: str | None = None,
        sort_by_azimuth: bool = False,
        **kwargs: object,
    ) -> xr.Dataset:
        tree = self.open_datatree(
            filename_or_obj,
            drop_variables=drop_variables,
            sort_by_azimuth=sort_by_azimuth,
            **kwargs,
        )
        return _dataset_for_group(tree, group, "radrs-xradar")

    def open_groups_as_dict(
        self,
        filename_or_obj: object,
        *,
        drop_variables: str | Iterable[str] | None = None,
        sort_by_azimuth: bool = False,
        **kwargs: object,
    ) -> dict[str, xr.Dataset]:
        tree = self.open_datatree(
            filename_or_obj,
            drop_variables=drop_variables,
            sort_by_azimuth=sort_by_azimuth,
            **kwargs,
        )
        return _groups_as_dict(tree)


class RadrsRaystackBackend(_RadrsBackendBase):
    """xarray backend for the radrs flattened raystack DataTree schema."""

    open_dataset_parameters = _RadrsBackendBase.open_dataset_parameters + (
        "fold_size",
        "qc",
        "include_activity",
    )
    description = "Fast radrs NEXRAD Level II reader returning raystack DataTrees"

    def open_datatree(
        self,
        filename_or_obj: object,
        *,
        drop_variables: str | Iterable[str] | None = None,
        fold_size: int | None = None,
        qc: object = None,
        include_activity: bool = True,
        **kwargs: object,
    ) -> xr.DataTree:
        _unsupported_decoder_kwargs(kwargs)
        if kwargs:
            joined = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported radrs-raystack keyword argument(s): {joined}")

        tree = radrs_raystack.open_datatree(
            _normalize_source(filename_or_obj),
            fold_size=fold_size,
            qc=qc,
            include_activity=include_activity,
        )
        return _drop_variables(tree, drop_variables)

    def open_dataset(
        self,
        filename_or_obj: object,
        *,
        drop_variables: str | Iterable[str] | None = None,
        group: str | None = None,
        fold_size: int | None = None,
        qc: object = None,
        include_activity: bool = True,
        **kwargs: object,
    ) -> xr.Dataset:
        tree = self.open_datatree(
            filename_or_obj,
            drop_variables=drop_variables,
            fold_size=fold_size,
            qc=qc,
            include_activity=include_activity,
            **kwargs,
        )
        return _dataset_for_group(tree, group, "radrs-raystack")

    def open_groups_as_dict(
        self,
        filename_or_obj: object,
        *,
        drop_variables: str | Iterable[str] | None = None,
        fold_size: int | None = None,
        qc: object = None,
        include_activity: bool = True,
        **kwargs: object,
    ) -> dict[str, xr.Dataset]:
        tree = self.open_datatree(
            filename_or_obj,
            drop_variables=drop_variables,
            fold_size=fold_size,
            qc=qc,
            include_activity=include_activity,
            **kwargs,
        )
        return _groups_as_dict(tree)


__all__ = ["RadrsRaystackBackend", "RadrsXradarBackend"]
