"""Tests for radrs.viz adapter utilities."""

import numpy as np
import pytest
import xarray as xr

import radrs.viz as viz


def _fixtures() -> tuple[xr.Dataset, xr.Dataset]:
    return_times = np.array(
        ["2024-01-01T00:00:00", "2024-01-01T00:00:01", "2024-01-01T00:00:02"],
        dtype="datetime64[ms]",
    ).astype("datetime64[ns]")
    sweep_times = np.array(
        ["2024-01-01T00:00:00", "2024-01-01T00:00:02"],
        dtype="datetime64[ms]",
    ).astype("datetime64[ns]")

    returns = xr.Dataset(
        data_vars={
            "azimuth": (("return_time",), np.array([0.0, 90.0, 180.0], dtype=np.float32)),
            "elevation": (
                ("return_time",),
                np.array([0.5, 0.5, 1.5], dtype=np.float32),
            ),
            "base_range": (
                ("return_time",),
                np.array([100.0, 200.0, 300.0], dtype=np.float32),
            ),
            "range_step": (
                ("return_time",),
                np.array([50.0, 100.0, 25.0], dtype=np.float32),
            ),
            "sweep_number": (("return_time",), np.array([0, 0, 1], dtype=np.uint32)),
            "DBZH": (
                ("return_time", "range"),
                np.array(
                    [
                        [1.0, np.nan, 3.0, 4.0],
                        [5.0, 6.0, np.nan, 8.0],
                        [9.0, 10.0, 11.0, 12.0],
                    ],
                    dtype=np.float32,
                ),
            ),
            "qc.demo_mask": (
                ("return_time", "range"),
                np.array(
                    [
                        [0, 1, 0, 1],
                        [1, 1, 0, 0],
                        [0, 0, 0, 0],
                    ],
                    dtype=np.float32,
                ),
            ),
        },
        coords={
            "return_time": return_times,
            "range": np.arange(4, dtype=np.float32),
        },
    )

    sweeps = xr.Dataset(
        data_vars={
            "sweep_number": (("sweep_time",), np.array([0, 1], dtype=np.uint32)),
            "elevation_angle": (("sweep_time",), np.array([0.5, 1.5], dtype=np.float32)),
            "num_returns": (("sweep_time",), np.array([2, 1], dtype=np.uint32)),
        },
        coords={"sweep_time": sweep_times},
    )

    return returns, sweeps


def _datatree_fixture() -> xr.DataTree:
    returns, sweeps = _fixtures()
    return xr.DataTree.from_dict({"returns": returns, "sweeps": sweeps})


def _legacy_sweeps_fixture() -> xr.Dataset:
    sweep_times = np.array(
        ["2024-01-01T00:00:00", "2024-01-01T00:00:02"],
        dtype="datetime64[ms]",
    ).astype("datetime64[ns]")
    return xr.Dataset(
        data_vars={
            "sweep_number": (("sweep_time",), np.array([0, 1], dtype=np.uint32)),
            "sweep_fixed_angle": (("sweep_time",), np.array([0.5, 1.5], dtype=np.float32)),
            "n_radials": (("sweep_time",), np.array([2, 1], dtype=np.uint32)),
            "start_index": (("sweep_time",), np.array([0, 2], dtype=np.uint32)),
        },
        coords={"sweep_time": sweep_times},
    )


def test_available_moments_includes_core_and_qc() -> None:
    returns, _ = _fixtures()

    moments = viz.available_moments(returns, include_qc=True)

    assert "DBZH" in moments
    assert "qc.demo_mask" in moments


def test_sweep_offsets_and_infos() -> None:
    _, sweeps = _fixtures()

    offsets = viz.sweep_offsets(sweeps)
    infos = viz.sweep_infos(sweeps)

    np.testing.assert_array_equal(offsets, np.array([0, 2, 3], dtype=np.int64))
    assert len(infos) == 2
    assert infos[0].index == 0
    assert infos[0].sweep_number == 0
    assert infos[0].num_returns == 2
    assert "elev 0.50 deg" in infos[0].label


def test_prepare_polar_payload_builds_expected_gate_geometry() -> None:
    returns, sweeps = _fixtures()

    payload = viz.prepare_polar_payload(
        returns=returns,
        sweeps=sweeps,
        sweep_index=0,
        moment="DBZH",
        max_points=None,
    )

    assert payload.point_count == 6
    np.testing.assert_allclose(payload.values, np.array([1, 3, 4, 5, 6, 8], dtype=np.float32))
    np.testing.assert_allclose(
        payload.range_m,
        np.array([100, 200, 250, 200, 300, 500], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        payload.gate_index,
        np.array([0, 2, 3, 0, 1, 3], dtype=np.uint16),
    )
    np.testing.assert_array_equal(
        payload.return_index,
        np.array([0, 0, 0, 1, 1, 1], dtype=np.uint32),
    )
    np.testing.assert_allclose(
        payload.azimuth_deg,
        np.array([0, 0, 0, 90, 90, 90], dtype=np.float32),
    )
    assert payload.max_range_m == 500.0
    assert payload.vmax > payload.vmin


def test_prepare_polar_payload_applies_sampling_cap() -> None:
    returns, sweeps = _fixtures()

    payload = viz.prepare_polar_payload(
        returns=returns,
        sweeps=sweeps,
        sweep_index=0,
        moment="DBZH",
        max_points=3,
    )

    assert payload.point_count <= 3
    np.testing.assert_allclose(payload.values, np.array([1, 4, 6], dtype=np.float32))


def test_prepare_polar_payload_rejects_unknown_moment() -> None:
    returns, sweeps = _fixtures()

    with pytest.raises(KeyError):
        viz.prepare_polar_payload(
            returns=returns,
            sweeps=sweeps,
            sweep_index=0,
            moment="NOT_A_MOMENT",
        )


def test_prepare_payload_supports_legacy_sweeps_schema() -> None:
    returns, _ = _fixtures()
    sweeps = _legacy_sweeps_fixture()

    offsets = viz.sweep_offsets(sweeps)
    infos = viz.sweep_infos(sweeps)
    payload = viz.prepare_polar_payload(
        returns=returns,
        sweeps=sweeps,
        sweep_index=0,
        moment="DBZH",
    )

    np.testing.assert_array_equal(offsets, np.array([0, 2, 3], dtype=np.int64))
    assert infos[0].num_returns == 2
    assert infos[0].elevation_deg == pytest.approx(0.5)
    assert payload.point_count == 6


def test_prepare_volume_payload_cartesian_geometry() -> None:
    returns, _ = _fixtures()

    payload = viz.prepare_volume_payload(
        returns=returns,
        moment="DBZH",
        max_points=None,
    )

    assert payload.point_count == 10
    # First finite gate belongs to return 0, gate 0, az=0deg, el=0.5deg, range=100m
    assert payload.return_index[0] == 0
    assert payload.gate_index[0] == 0
    np.testing.assert_allclose(payload.x_m[0], 0.0, atol=1e-5)
    np.testing.assert_allclose(payload.y_m[0], 99.99619, rtol=1e-5)
    np.testing.assert_allclose(payload.z_m[0], 0.87265, rtol=1e-5)
    assert payload.vmax > payload.vmin
    assert payload.max_abs_m > 0.0
    assert payload.render_mode == "points"

    state = payload.to_widget_state()
    assert "x_bytes" in state
    assert "y_bytes" in state
    assert "z_bytes" in state
    assert "value_bytes" in state
    assert "range_bytes" not in state
    assert "azimuth_bytes" not in state
    assert "elevation_bytes" not in state
    assert "return_time_ms_bytes" not in state
    assert state["meta"]["render_mode"] == "points"


def test_prepare_ray_payload_uses_returns_only() -> None:
    returns, _ = _fixtures()

    payload = viz.prepare_ray_payload(
        returns=returns,
        moment="DBZH",
        max_points=None,
    )

    assert payload.render_mode == "rays"
    assert payload.point_count == 3
    np.testing.assert_array_equal(payload.return_index, np.array([0, 1, 2], dtype=np.uint32))
    np.testing.assert_array_equal(payload.gate_index, np.array([3, 3, 3], dtype=np.uint16))
    np.testing.assert_allclose(payload.values, np.array([4.0, 8.0, 12.0], dtype=np.float32))
    np.testing.assert_allclose(payload.x_m[0], 0.0, atol=1e-5)
    np.testing.assert_allclose(payload.y_m[0], 249.99048, rtol=1e-5)
    np.testing.assert_allclose(payload.z_m[0], 2.18163, rtol=1e-5)

    state = payload.to_widget_state()
    assert state["meta"]["render_mode"] == "rays"


def test_prepare_cappi_payload_basic() -> None:
    returns, sweeps = _fixtures()

    payload = viz.prepare_cappi_payload(
        returns=returns,
        sweeps=sweeps,
        moment="DBZH",
        altitude_m=100.0,
        tolerance_m=5000.0,
        grid_size=50,
    )

    assert payload.grid.shape == (50, 50)
    assert payload.grid_mode == "cappi"
    assert payload.x_label == "East (km)"
    assert payload.y_label == "North (km)"
    assert payload.moment == "DBZH"
    assert np.any(np.isfinite(payload.grid))
    assert payload.vmin < payload.vmax
    assert payload.cappi_altitude_m == 100.0
    assert payload.cappi_tolerance_m == 5000.0
    assert payload.sweep_elevations_deg is not None
    assert len(payload.sweep_elevations_deg) == 2
    assert payload.sweep_num_returns is not None
    assert len(payload.sweep_num_returns) == 2

    # Roundtrip through widget state
    state = payload.to_widget_state()
    assert "grid_bytes" in state
    assert state["meta"]["grid_mode"] == "cappi"
    assert state["meta"]["n_rows"] == 50
    assert state["meta"]["n_cols"] == 50
    assert isinstance(state["meta"]["sweep_elevations_deg"], list)
    assert isinstance(state["meta"]["sweep_num_returns"], list)


def test_prepare_cappi_payload_empty_band() -> None:
    returns, sweeps = _fixtures()

    # Very high altitude — no gates should fall in this band
    payload = viz.prepare_cappi_payload(
        returns=returns,
        sweeps=sweeps,
        moment="DBZH",
        altitude_m=100_000.0,
        tolerance_m=100.0,
        grid_size=20,
    )

    assert payload.grid.shape == (20, 20)
    assert not np.any(np.isfinite(payload.grid))
    assert payload.vmin == 0.0
    assert payload.vmax == 1.0


def test_prepare_cappi_payload_rejects_unknown_moment() -> None:
    returns, sweeps = _fixtures()

    with pytest.raises(KeyError):
        viz.prepare_cappi_payload(
            returns=returns,
            sweeps=sweeps,
            moment="NOT_A_MOMENT",
            altitude_m=1000.0,
        )


def test_prepare_xsec_payload_basic() -> None:
    returns, sweeps = _fixtures()

    payload = viz.prepare_xsec_payload(
        returns=returns,
        sweeps=sweeps,
        moment="DBZH",
        azimuth_deg=0.0,
        azimuth_tolerance_deg=5.0,
        grid_size=50,
    )

    assert payload.grid.shape == (50, 50)
    assert payload.grid_mode == "xsec"
    assert payload.x_label == "Ground range (km)"
    assert payload.y_label == "Altitude (km)"
    assert payload.moment == "DBZH"
    assert np.any(np.isfinite(payload.grid))
    assert payload.xsec_azimuth_deg == 0.0
    assert payload.xsec_azimuth_tolerance_deg == 5.0
    assert payload.sweep_elevations_deg is not None
    assert payload.sweep_num_returns is not None


def test_prepare_xsec_payload_includes_opposite_azimuth() -> None:
    returns, sweeps = _fixtures()

    # Fixture has returns at az=0, 90, 180. Targeting az=0 should also pick up az=180.
    payload = viz.prepare_xsec_payload(
        returns=returns,
        sweeps=sweeps,
        moment="DBZH",
        azimuth_deg=0.0,
        azimuth_tolerance_deg=5.0,
        grid_size=50,
    )

    # x_min should be negative (backward direction from az=180)
    assert payload.x_min < 0


def test_prepare_xsec_payload_rejects_unknown_moment() -> None:
    returns, sweeps = _fixtures()

    with pytest.raises(KeyError):
        viz.prepare_xsec_payload(
            returns=returns,
            sweeps=sweeps,
            moment="NOT_A_MOMENT",
            azimuth_deg=0.0,
        )


def test_prepare_waterfall_payload_basic() -> None:
    returns, _ = _fixtures()

    payload = viz.prepare_waterfall_payload(
        returns=returns,
        moment="DBZH",
    )

    # 3 returns, 4 gates — no downsampling needed at default limits
    assert payload.grid.shape == (3, 4)
    assert payload.n_returns == 3
    assert payload.n_range == 4
    assert payload.n_returns_orig == 3
    assert payload.n_range_orig == 4
    assert payload.moment == "DBZH"
    assert payload.vmin < payload.vmax

    # Per-return metadata arrays
    assert payload.azimuth_deg.shape == (3,)
    np.testing.assert_allclose(payload.azimuth_deg, [0.0, 90.0, 180.0])
    assert payload.elevation_deg.shape == (3,)
    np.testing.assert_allclose(payload.elevation_deg, [0.5, 0.5, 1.5])
    assert payload.sweep_number.shape == (3,)
    np.testing.assert_array_equal(payload.sweep_number, [0, 0, 1])
    assert payload.return_time_ms.shape == (3,)
    assert payload.return_time_ms.dtype == np.float64
    assert np.all(np.isfinite(payload.return_time_ms))

    # Sweep boundary at index 2 (where sweep changes from 0 to 1)
    assert payload.sweep_boundaries.shape == (1,)
    assert payload.sweep_boundaries[0] == 2

    # Range metadata
    assert payload.range_start_m > 0
    assert payload.range_step_m > 0

    # Widget state roundtrip
    state = payload.to_widget_state()
    assert "grid_bytes" in state
    assert "azimuth_bytes" in state
    assert "elevation_bytes" in state
    assert "return_time_ms_bytes" in state
    assert "sweep_number_bytes" in state
    assert "sweep_boundary_bytes" in state
    assert state["meta"]["n_returns"] == 3
    assert state["meta"]["n_range"] == 4
    assert state["meta"]["moment"] == "DBZH"


def test_prepare_waterfall_payload_downsamples() -> None:
    returns, _ = _fixtures()

    payload = viz.prepare_waterfall_payload(
        returns=returns,
        moment="DBZH",
        max_returns=2,
        max_range=2,
    )

    assert payload.n_returns <= 2
    assert payload.n_range <= 2
    assert payload.n_returns_orig == 3
    assert payload.n_range_orig == 4


def test_prepare_waterfall_payload_rejects_unknown_moment() -> None:
    returns, _ = _fixtures()

    with pytest.raises(KeyError):
        viz.prepare_waterfall_payload(
            returns=returns,
            moment="NOT_A_MOMENT",
        )


def test_prepare_waterfall_payload_fold_size_crops_range() -> None:
    returns, _ = _fixtures()

    payload = viz.prepare_waterfall_payload(
        returns=returns,
        moment="DBZH",
        fold_size=2,
    )

    # Fixture has 4 gates; fold_size=2 crops to first 2
    assert payload.n_range == 2
    assert payload.n_range_orig == 4  # original before crop
    assert payload.grid.shape == (3, 2)
    # Values should be the first 2 gates of each return
    np.testing.assert_allclose(payload.grid[0, 0], 1.0)
    assert np.isnan(payload.grid[0, 1])  # was NaN in original
    np.testing.assert_allclose(payload.grid[1, 0], 5.0)
    np.testing.assert_allclose(payload.grid[1, 1], 6.0)


def test_prepare_waterfall_payload_preserves_nan() -> None:
    returns, _ = _fixtures()

    payload = viz.prepare_waterfall_payload(
        returns=returns,
        moment="DBZH",
    )

    # Original DBZH has NaN at (0,1) and (1,2) — these should survive
    assert np.isnan(payload.grid[0, 1])
    assert np.isnan(payload.grid[1, 2])


def test_volume_widget_from_datatree() -> None:
    dt = _datatree_fixture()

    w = viz.VolumeWidget.from_datatree(dt, "DBZH")

    assert w.meta["point_count"] == 10
    assert w.meta["render_mode"] == "points"
    assert w.meta["moment"] == "DBZH"


def test_volume_widget_from_datatree_rays() -> None:
    dt = _datatree_fixture()

    w = viz.VolumeWidget.from_datatree(dt, "DBZH", render_mode="rays")

    assert w.meta["point_count"] == 3
    assert w.meta["render_mode"] == "rays"


def test_grid_widget_from_cappi() -> None:
    dt = _datatree_fixture()

    w = viz.GridWidget.from_cappi(
        dt, "DBZH", altitude_m=100.0, tolerance_m=5000.0, grid_size=50
    )

    assert w.meta["grid_mode"] == "cappi"
    assert w.meta["n_rows"] == 50
    assert w.meta["n_cols"] == 50
    assert w.meta["moment"] == "DBZH"


def test_grid_widget_from_xsec() -> None:
    dt = _datatree_fixture()

    w = viz.GridWidget.from_xsec(
        dt, "DBZH", azimuth_deg=0.0, azimuth_tolerance_deg=5.0, grid_size=50
    )

    assert w.meta["grid_mode"] == "xsec"
    assert w.meta["n_rows"] == 50
    assert w.meta["n_cols"] == 50
    assert w.meta["moment"] == "DBZH"


def test_waterfall_widget_from_datatree() -> None:
    dt = _datatree_fixture()

    w = viz.WaterfallWidget.from_datatree(dt, "DBZH")

    assert w.meta["n_returns"] == 3
    assert w.meta["n_range"] == 4
    assert w.meta["moment"] == "DBZH"


def test_polar_widget_from_datatree() -> None:
    dt = _datatree_fixture()

    w = viz.PolarWidget.from_datatree(dt, "DBZH", sweep_index=0)

    assert w.meta["point_count"] == 6
    assert w.meta["moment"] == "DBZH"
    assert w.meta["sweep_number"] == 0


def test_volume_payload_optimized() -> None:
    """Optimized prepare_volume_payload produces correct finite values and coordinates."""
    returns, _ = _fixtures()

    payload = viz.prepare_volume_payload(returns=returns, moment="DBZH", max_points=None)

    # 10 finite values in the fixture (12 total, 2 NaN)
    assert payload.point_count == 10
    # All values should be finite
    assert np.all(np.isfinite(payload.values))
    # First point: return 0, gate 0, az=0deg, el=0.5deg, range=100m
    assert payload.return_index[0] == 0
    assert payload.gate_index[0] == 0
    np.testing.assert_allclose(payload.x_m[0], 0.0, atol=1e-5)
    np.testing.assert_allclose(payload.y_m[0], 99.99619, rtol=1e-5)
    np.testing.assert_allclose(payload.z_m[0], 0.87265, rtol=1e-5)


# ---------------------------------------------------------------------------
# to_html() static export tests
# ---------------------------------------------------------------------------


def test_polar_payload_to_html() -> None:
    returns, sweeps = _fixtures()
    payload = viz.prepare_polar_payload(returns, sweeps, 0, "DBZH")
    html = payload.to_html()

    assert isinstance(html, str)
    assert "<!DOCTYPE html>" in html
    assert '<script type="module">' in html
    assert "radrs-viz-root" in html
    assert "DBZH" in html


def test_volume_payload_to_html() -> None:
    returns, _ = _fixtures()
    payload = viz.prepare_volume_payload(returns, moment="DBZH", max_points=None)
    html = payload.to_html()

    assert isinstance(html, str)
    assert "<!DOCTYPE html>" in html
    assert '<script type="module">' in html
    assert "deck.gl" in html
    assert "yaw_deg" in html
    assert "pitch_deg" in html


def test_volume_payload_to_html_preserves_zero_angles() -> None:
    returns, _ = _fixtures()
    payload = viz.prepare_volume_payload(returns, moment="DBZH", max_points=None)
    html = payload.to_html(yaw_deg=0.0, pitch_deg=0.0)

    assert '"yaw_deg": 0' in html
    assert '"pitch_deg": 0' in html
    assert 'Number(model.get("yaw_deg")) ||' not in html
    assert 'Number(model.get("pitch_deg")) ||' not in html


def test_grid_payload_to_html() -> None:
    returns, sweeps = _fixtures()
    payload = viz.prepare_cappi_payload(
        returns, sweeps, "DBZH", altitude_m=100.0, tolerance_m=5000.0, grid_size=50
    )
    html = payload.to_html()

    assert isinstance(html, str)
    assert "<!DOCTYPE html>" in html
    assert '"grid_mode"' in html or "cappi" in html


def test_waterfall_payload_to_html() -> None:
    returns, _ = _fixtures()
    payload = viz.prepare_waterfall_payload(returns, moment="DBZH")
    html = payload.to_html()

    assert isinstance(html, str)
    assert "<!DOCTYPE html>" in html
    assert "n_returns" in html
    assert "n_range" in html


def test_to_html_custom_size() -> None:
    returns, sweeps = _fixtures()
    payload = viz.prepare_polar_payload(returns, sweeps, 0, "DBZH")
    html = payload.to_html(width=1024, height=512)

    assert "1024" in html
    assert "512" in html
