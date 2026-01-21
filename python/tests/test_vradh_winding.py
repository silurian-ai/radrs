"""Tests for VRADH winding number port (Py-ART parity)."""

import numpy as np
import pytest
import xarray as xr

import radrs.qc as qc
import radrs.raystack as rrs
import radrs.xradar as rxr

try:
    import pyart
except Exception:  # pragma: no cover - optional dependency
    pyart = None


def _build_single_sweep_pyart_radar(dt: xr.DataTree, sweep_ds: xr.Dataset):
    if pyart is None:
        pytest.skip("pyart not installed")

    sweep_ds = sweep_ds.copy()
    sweep_ds["sweep_number"] = 0
    root = dt.dataset.copy()
    for coord in ("latitude", "longitude", "altitude"):
        if coord not in root:
            root = root.assign({coord: 0.0})
    single_sweep_dt = xr.DataTree.from_dict(
        {
            ".": root,
            "sweep_0": sweep_ds,
        }
    )
    return single_sweep_dt.pyart.to_radar()


def _clean_vradh(ds: xr.Dataset) -> xr.DataArray:
    return ds["VRADH"].where(~np.isclose(ds["VRADH"], -64.5, atol=1.0), np.nan)


def _select_sweep_with_vradh(dt: xr.DataTree) -> tuple[str, xr.Dataset]:
    for name in dt.children:
        if not name.startswith("sweep_"):
            continue
        sweep_ds = dt[name].dataset
        if "VRADH" in sweep_ds and "DBZH" in sweep_ds:
            return name, sweep_ds
    raise pytest.skip("No sweep with VRADH/DBZH found")


def _compute_pyart_winding(radar, nyq, vel_texture_threshold, reflectivity_threshold):
    if pyart is None:
        pytest.skip("pyart not installed")

    vel_texture = pyart.retrieve.calculate_velocity_texture(
        radar, vel_field="VRADH", nyq=nyq
    )
    radar.add_field("velocity_texture", vel_texture, replace_existing=True)

    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_above("velocity_texture", vel_texture_threshold)
    gatefilter.exclude_below("DBZH", reflectivity_threshold)

    velocity_dealiased = pyart.correct.dealias_region_based(
        radar,
        vel_field="VRADH",
        nyquist_vel=nyq,
        centered=True,
        gatefilter=gatefilter,
    )

    original = np.ma.filled(radar["sweep_0"]["VRADH"].values, np.nan)
    dealiased = np.ma.filled(velocity_dealiased["data"], np.nan)
    return (dealiased - original) / (2 * nyq), original


def _compute_radrs_winding(original, dbzh, nyq, vel_texture_threshold, reflectivity_threshold):
    return qc.vradh_winding_number(
        original,
        dbzh,
        nyquist=nyq,
        wind_size=3,
        velocity_texture_threshold=vel_texture_threshold,
        reflectivity_threshold=reflectivity_threshold,
        interval_splits=3,
        skip_between_rays=100,
        skip_along_ray=100,
        centered=True,
        rays_wrap_around=True,
        fill_value=-64.5,
        fill_tolerance=1.0,
    )


def test_vradh_winding_number_matches_pyart(available_test_files):
    if pyart is None:
        pytest.skip("pyart not installed")

    for test_file_path in available_test_files:
        dt = rxr.open_datatree(test_file_path)
        sweep_name, sweep_ds = _select_sweep_with_vradh(dt)
        sweep_ds = sweep_ds.assign(VRADH=_clean_vradh(sweep_ds))

        radar = _build_single_sweep_pyart_radar(dt, sweep_ds)
        nyq = np.nanmax(np.abs(sweep_ds["VRADH"].values))
        if not np.isfinite(nyq) or nyq <= 0:
            pytest.skip("Invalid Nyquist velocity")

        expected, original = _compute_pyart_winding(radar, nyq, 4.0, 0.0)
        actual = _compute_radrs_winding(original, sweep_ds["DBZH"].values, nyq, 4.0, 0.0)

        diff = np.nan_to_num(np.abs(actual - expected), nan=0.0)
        mismatch = np.count_nonzero(diff > 0)
        mismatch_ratio = mismatch / expected.size

        assert diff.max() <= 1.0
        assert mismatch_ratio < 0.001


@pytest.mark.slow
def test_vradh_winding_number_matches_pyart_all_sweeps(available_test_files):
    if pyart is None:
        pytest.skip("pyart not installed")

    for test_file_path in available_test_files:
        dt = rxr.open_datatree(test_file_path)
        total = 0
        mismatches = 0
        max_diff = 0.0

        for name in dt.children:
            if not name.startswith("sweep_"):
                continue
            sweep_ds = dt[name].dataset
            if "VRADH" not in sweep_ds or "DBZH" not in sweep_ds:
                continue

            sweep_ds = sweep_ds.assign(VRADH=_clean_vradh(sweep_ds))
            nyq = np.nanmax(np.abs(sweep_ds["VRADH"].values))
            if not np.isfinite(nyq) or nyq <= 0:
                continue

            radar = _build_single_sweep_pyart_radar(dt, sweep_ds)
            expected, original = _compute_pyart_winding(radar, nyq, 4.0, 0.0)
            actual = _compute_radrs_winding(original, sweep_ds["DBZH"].values, nyq, 4.0, 0.0)

            diff = np.nan_to_num(np.abs(actual - expected), nan=0.0)
            mismatches += np.count_nonzero(diff > 0)
            total += diff.size
            max_diff = max(max_diff, float(diff.max()))

        if total == 0:
            pytest.skip("No sweeps with VRADH/DBZH found")

        mismatch_ratio = mismatches / total
        # All-sweep parity is slightly looser; rare folds can differ by 2.
        assert max_diff <= 2.0
        assert mismatch_ratio < 0.001


def test_raystack_parse_adds_vradh_winding_number(test_file_bytes):
    rs = rrs.parse(test_file_bytes, qc=[qc.VradhWindingNumber()])
    returns = rs["returns"]
    if "VRADH" not in returns:
        pytest.skip("VRADH not present in test volume")

    assert "vradh_winding_number" in returns
    assert returns["vradh_winding_number"].shape == returns["VRADH"].shape


def test_vradh_winding_number_synthetic_masks():
    vradh = np.linspace(-5, 5, 25, dtype=np.float32).reshape(5, 5)
    dbzh = np.full_like(vradh, -1.0)
    out = qc.vradh_winding_number(
        vradh,
        dbzh,
        nyquist=10.0,
        reflectivity_threshold=0.0,
    )
    assert np.isnan(out).all()


def test_vradh_winding_number_fill_values():
    vradh = np.linspace(-4, 4, 16, dtype=np.float32).reshape(4, 4)
    vradh[0, 0] = -64.5
    vradh[0, 1] = -63.3
    out = qc.vradh_winding_number(
        vradh,
        None,
        nyquist=10.0,
        fill_value=-64.5,
        fill_tolerance=1.0,
    )
    assert np.isnan(out[0, 0])
    assert np.isfinite(out[0, 1])
    assert np.nanmax(np.abs(out)) <= 1.0


def test_vradh_winding_number_auto_nyquist():
    vradh = np.linspace(-8, 8, 49, dtype=np.float32).reshape(7, 7)
    out = qc.vradh_winding_number(vradh, None)
    assert np.isfinite(np.nanmax(out))


def test_vradh_winding_number_explicit_nyquist_matches_auto():
    vradh = np.linspace(-12, 12, 81, dtype=np.float32).reshape(9, 9)
    auto = qc.vradh_winding_number(vradh, None)
    explicit = qc.vradh_winding_number(vradh, None, nyquist=12.0)
    assert np.allclose(auto, explicit, equal_nan=True)
