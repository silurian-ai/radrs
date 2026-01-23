import pytest


icechunk = pytest.importorskip("icechunk")
xr = pytest.importorskip("xarray")


VCP_URIS = [
    "gs://silurian-dev-storage/tensorlake/syn_nexrad_v0.zarr.ic.by-vcp/VCP-112.zarr.ic/",
    "gs://silurian-dev-storage/tensorlake/syn_nexrad_v0.zarr.ic.by-vcp/VCP-12.zarr.ic/",
    "gs://silurian-dev-storage/tensorlake/syn_nexrad_v0.zarr.ic.by-vcp/VCP-212.zarr.ic/",
    "gs://silurian-dev-storage/tensorlake/syn_nexrad_v0.zarr.ic.by-vcp/VCP-215.zarr.ic/",
    "gs://silurian-dev-storage/tensorlake/syn_nexrad_v0.zarr.ic.by-vcp/VCP-31.zarr.ic/",
    "gs://silurian-dev-storage/tensorlake/syn_nexrad_v0.zarr.ic.by-vcp/VCP-34.zarr.ic/",
    "gs://silurian-dev-storage/tensorlake/syn_nexrad_v0.zarr.ic.by-vcp/VCP-35.zarr.ic/",
    "gs://silurian-dev-storage/tensorlake/syn_nexrad_v0.zarr.ic.by-vcp/VCP-80.zarr.ic/",
    "gs://silurian-dev-storage/tensorlake/syn_nexrad_v0.zarr.ic.by-vcp/VCP-90.zarr.ic/",
]


def _parse_gs_uri(uri: str) -> tuple[str, str, str]:
    uri = uri.rstrip("/")
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {uri}")
    _, rest = uri.split("gs://", 1)
    bucket, prefix = rest.split("/", 1)
    vcp_name = uri.split("/")[-1].split(".", 1)[0]
    return bucket, prefix, vcp_name


def _open_datatree(uri: str):
    # Prefer the icechunk xarray backend if available (silurian-core registers it).
    if "icechunk" in xr.backends.list_engines():
        return xr.open_datatree(uri, engine="icechunk")

    # Fallback: open via icechunk directly and pass the store to xarray.
    bucket, prefix, _ = _parse_gs_uri(uri)
    storage = icechunk.gcs_storage(bucket=bucket, prefix=prefix, from_env=True)
    repository = icechunk.Repository.open(storage)
    session = repository.readonly_session(branch="main")
    return xr.open_datatree(
        session.store,
        engine="zarr",
        consolidated=False,
    )


def _first_sweep_with_data(vcp_dt):
    sweep_keys = [k for k in vcp_dt.keys() if k.startswith("sweep_")]
    for sweep_key in sweep_keys:
        sweep_dt = vcp_dt[sweep_key]
        inner_keys = list(sweep_dt.keys())
        if inner_keys:
            return sweep_dt[inner_keys[0]]
    return None


@pytest.mark.network
@pytest.mark.parametrize("uri", VCP_URIS)
def test_icechunk_vcp_datatree_structure(uri):
    dt = _open_datatree(uri)
    site_keys = list(dt.keys())
    assert site_keys, "No sites found in Icechunk DataTree"

    site = dt[site_keys[0]]
    _, _, vcp_name = _parse_gs_uri(uri)
    assert vcp_name in site, f"Missing {vcp_name} in site {site_keys[0]}"

    vcp_dt = site[vcp_name]
    sweep_dt = _first_sweep_with_data(vcp_dt)
    assert sweep_dt is not None, f"No sweeps with data found for {vcp_name}"

    # Metadata variables should exist without forcing full array loads
    for var in [
        "latitude",
        "longitude",
        "altitude",
        "instrument_type",
        "platform_type",
        "source_url",
        "source_fs_size",
    ]:
        assert var in vcp_dt, f"Missing {var} in {vcp_name}"

    # Sweep-level content sanity
    assert {"azimuth", "range"} <= set(sweep_dt.coords), "Missing sweep coords"
    moments = {"DBZH", "VRADH", "WRADH", "RHOHV", "ZDR", "PHIDP", "CCORH"}
    assert moments.intersection(set(sweep_dt.data_vars)), "No expected moments found"
