"""Tests for radrs.viz adapter utilities."""

import shutil
import subprocess

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


def _dense_returns(n_returns: int, n_gates: int) -> tuple[xr.Dataset, xr.Dataset]:
    """One fully finite sweep, every gate populated. No NaN anywhere."""
    times = (
        np.datetime64("2024-01-01T00:00:00", "ms") + np.arange(n_returns)
    ).astype("datetime64[ns]")
    returns = xr.Dataset(
        data_vars={
            "azimuth": (
                ("return_time",),
                (np.arange(n_returns) * (360.0 / n_returns)).astype(np.float32),
            ),
            "elevation": (("return_time",), np.full(n_returns, 0.5, dtype=np.float32)),
            "base_range": (("return_time",), np.full(n_returns, 100.0, dtype=np.float32)),
            "range_step": (("return_time",), np.full(n_returns, 250.0, dtype=np.float32)),
            "sweep_number": (("return_time",), np.zeros(n_returns, dtype=np.uint32)),
            "DBZH": (
                ("return_time", "range"),
                np.arange(n_returns * n_gates, dtype=np.float32).reshape(
                    n_returns, n_gates
                ),
            ),
        },
        coords={
            "return_time": times,
            "range": np.arange(n_gates, dtype=np.float32),
        },
    )
    sweeps = xr.Dataset(
        data_vars={
            "sweep_number": (("sweep_time",), np.array([0], dtype=np.uint32)),
            "elevation_angle": (("sweep_time",), np.array([0.5], dtype=np.float32)),
            "num_returns": (("sweep_time",), np.array([n_returns], dtype=np.uint32)),
        },
        coords={"sweep_time": times[:1]},
    )
    return returns, sweeps


def _realistic_volume(
    n_sweeps: int = 14, n_radials: int = 360, n_gates: int = 1832
) -> tuple[xr.Dataset, xr.Dataset]:
    """A volume the size of a real one: 14 sweeps, 360 radials, 1832 gates."""
    rng = np.random.default_rng(7)
    n_returns = n_sweeps * n_radials
    elev_table = np.array(
        [0.5, 0.9, 1.3, 1.8, 2.4, 3.1, 4.0, 5.1, 6.4, 8.0, 10.0, 12.0, 14.0, 19.5],
        dtype=np.float32,
    )[:n_sweeps]
    values = (rng.random((n_returns, n_gates), dtype=np.float32) * 70.0 - 10.0)
    # Three quarters of a volume carries data; the rest is below the noise floor.
    values[rng.random((n_returns, n_gates)) > 0.75] = np.nan

    start = np.datetime64("2024-07-02T00:00:16", "ms")
    return_time = (start + (np.arange(n_returns) * 16).astype("timedelta64[ms]")).astype(
        "datetime64[ns]"
    )
    returns = xr.Dataset(
        data_vars={
            "azimuth": (
                ("return_time",),
                np.tile(
                    np.arange(n_radials, dtype=np.float32) * (360.0 / n_radials), n_sweeps
                ),
            ),
            "elevation": (("return_time",), np.repeat(elev_table, n_radials)),
            "base_range": (
                ("return_time",),
                np.full(n_returns, 2125.0, dtype=np.float32),
            ),
            "range_step": (("return_time",), np.full(n_returns, 250.0, dtype=np.float32)),
            "sweep_number": (
                ("return_time",),
                np.repeat(np.arange(n_sweeps, dtype=np.uint32), n_radials),
            ),
            "vcp_time": (("return_time",), np.repeat(return_time[:1], n_returns)),
            "DBZH": (("return_time", "range"), values),
        },
        coords={
            "return_time": return_time,
            "range": np.arange(n_gates, dtype=np.float32),
        },
    )
    sweeps = xr.Dataset(
        data_vars={
            "sweep_number": (("sweep_time",), np.arange(n_sweeps, dtype=np.uint32)),
            "elevation_angle": (("sweep_time",), elev_table),
            "num_returns": (
                ("sweep_time",),
                np.full(n_sweeps, n_radials, dtype=np.uint32),
            ),
        },
        coords={"sweep_time": return_time[::n_radials]},
    )
    return returns, sweeps


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

    # A dense (return, gate) grid: NaN gates stay in place rather than being
    # compacted away, because the widget derives coordinates from the indices.
    assert payload.values.shape == (2, 4)
    assert payload.n_returns == 2
    assert payload.n_gates == 4
    assert payload.point_count == 6  # finite gates
    assert payload.cell_count == 8
    np.testing.assert_allclose(
        payload.values,
        np.array([[1, np.nan, 3, 4], [5, 6, np.nan, 8]], dtype=np.float32),
    )

    # Per-return geometry, one entry per row, is all the widget needs.
    np.testing.assert_allclose(payload.azimuth_deg, np.array([0, 90], dtype=np.float32))
    np.testing.assert_allclose(payload.elevation_deg, np.array([0.5, 0.5], dtype=np.float32))
    np.testing.assert_allclose(payload.base_range_m, np.array([100, 200], dtype=np.float32))
    np.testing.assert_allclose(payload.range_step_m, np.array([50, 100], dtype=np.float32))
    np.testing.assert_array_equal(payload.return_index, np.array([0, 1], dtype=np.uint32))

    # The derived range grid is what a widget rebuilds from row and column.
    np.testing.assert_allclose(
        payload.range_m,
        np.array([[100, 150, 200, 250], [200, 300, 400, 500]], dtype=np.float32),
    )
    np.testing.assert_array_equal(payload.gate_index, np.array([0, 1, 2, 3], dtype=np.uint16))
    assert payload.max_range_m == 500.0
    assert payload.vmax > payload.vmin


def test_prepare_polar_payload_applies_sampling_cap() -> None:
    """``max_points`` caps shipped cells by decimating rows and gates."""
    returns, sweeps = _fixtures()

    payload = viz.prepare_polar_payload(
        returns=returns,
        sweeps=sweeps,
        sweep_index=0,
        moment="DBZH",
        max_points=4,
    )

    assert payload.cell_count <= 4
    assert payload.gate_stride > 1
    # Decimating the gate axis keeps every other gate, so the range axis is
    # still sampled end to end rather than collapsing onto one ring.
    np.testing.assert_allclose(
        payload.values, np.array([[1, 3], [5, np.nan]], dtype=np.float32)
    )
    np.testing.assert_array_equal(payload.gate_index, np.array([0, 2], dtype=np.uint16))


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
    np.testing.assert_allclose(payload.y_m[0], 99.99618, rtol=1e-5)
    # 4/3 effective-earth height, slightly above the flat-earth 0.87265 m
    np.testing.assert_allclose(payload.z_m[0], 0.87324, rtol=1e-5)
    assert payload.vmax > payload.vmin
    assert payload.max_abs_m > 0.0
    assert payload.render_mode == "points"

    state = payload.to_widget_state()
    # Positions are rebuilt in JS, so no x/y/z buffer crosses the wire.
    assert "x_bytes" not in state
    assert "y_bytes" not in state
    assert "z_bytes" not in state
    assert "value_bytes" in state
    # Per-point: a slot into the per-return arrays plus a gate index.
    assert "return_slot_bytes" in state
    assert "gate_index_bytes" in state
    # Per-return: the polar geometry the widget places each gate with, so the
    # tooltip never has to invert the curved-earth mapping.
    assert "azimuth_bytes" in state
    assert "elevation_bytes" in state
    assert "base_range_bytes" in state
    assert "range_step_bytes" in state
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
    np.testing.assert_allclose(payload.y_m[0], 249.99042, rtol=1e-5)
    # 4/3 effective-earth height, slightly above the flat-earth 2.18163 m
    np.testing.assert_allclose(payload.z_m[0], 2.18531, rtol=1e-5)

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

    # Per-return range ramp, plus the median axis the ticks are drawn on
    np.testing.assert_allclose(payload.base_range_m, [100.0, 200.0, 300.0])
    np.testing.assert_allclose(payload.range_step_m, [50.0, 100.0, 25.0])
    assert payload.axis_start_m > 0
    assert payload.axis_step_m > 0

    # Widget state roundtrip
    state = payload.to_widget_state()
    assert "grid_bytes" in state
    assert "azimuth_bytes" in state
    assert "elevation_bytes" in state
    assert "return_time_ms_bytes" in state
    assert "sweep_number_bytes" in state
    assert "base_range_bytes" in state
    assert "range_step_bytes" in state
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


def test_polar_payload_empty_for_all_nan_sweep() -> None:
    """A sweep whose moment is entirely NaN yields an empty, still-renderable payload."""
    returns, sweeps = _fixtures()
    returns = returns.copy()
    returns["DBZH"] = returns["DBZH"].where(False)

    payload = viz.prepare_polar_payload(returns, sweeps, 0, "DBZH")

    assert payload.point_count == 0
    assert payload.max_range_m == 0.0
    state = payload.to_widget_state()
    assert state["azimuth_bytes"] == b""
    assert state["meta"]["point_count"] == 0
    assert state["meta"]["sweep_number"] == 0


def test_polar_esm_drops_pick_buffer_on_empty_frame() -> None:
    """The empty-frame branch must reset `pick`, or hover indexes freed arrays."""
    esm = viz.assemble_esm("polar.js")
    start = esm.index("if (nReturns === 0 || nGates === 0) {")
    empty_branch = esm[start : esm.index("}", start)]

    assert "pick = new Int32Array(0);" in empty_branch
    assert "lastHover = -1;" in empty_branch


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
    np.testing.assert_allclose(payload.y_m[0], 99.99618, rtol=1e-5)
    np.testing.assert_allclose(payload.z_m[0], 0.87324, rtol=1e-5)


# ---------------------------------------------------------------------------
# beam_geometry (4/3 effective earth radius)
# ---------------------------------------------------------------------------


def _expected_beam_geometry(range_m: float, elevation_deg: float) -> tuple[float, float]:
    """Independent reference implementation of the 4/3 earth model."""
    r_e = 4.0 / 3.0 * 6_371_000.0
    el = np.deg2rad(elevation_deg)
    height = np.sqrt(range_m**2 + r_e**2 + 2.0 * range_m * r_e * np.sin(el)) - r_e
    ground = r_e * np.arcsin(range_m * np.cos(el) / (r_e + height))
    return float(ground), float(height)


def test_beam_geometry_zero_range_is_at_the_radar() -> None:
    ground, height = viz.beam_geometry(0.0, 0.5)

    assert float(ground) == pytest.approx(0.0, abs=1e-9)
    assert float(height) == pytest.approx(0.0, abs=1e-9)


def test_beam_geometry_monotonic_in_range() -> None:
    ranges = np.linspace(0.0, 300_000.0, 200)
    ground, height = viz.beam_geometry(ranges, 0.5)

    assert np.all(np.diff(ground) > 0)
    assert np.all(np.diff(height) > 0)


def test_beam_geometry_matches_four_thirds_model() -> None:
    expected_ground, expected_height = _expected_beam_geometry(200_000.0, 0.5)
    ground, height = viz.beam_geometry(200_000.0, 0.5)

    assert float(height) == pytest.approx(expected_height, rel=1e-12)
    assert float(ground) == pytest.approx(expected_ground, rel=1e-12)

    # ~4.1 km, well above the ~1.75 km a flat-earth model would report.
    flat_height = 200_000.0 * np.sin(np.deg2rad(0.5))
    assert float(height) == pytest.approx(4100.0, abs=100.0)
    assert float(height) - flat_height > 2000.0


def test_beam_geometry_ground_range_below_slant_range() -> None:
    slant = np.array([1_000.0, 50_000.0, 150_000.0, 300_000.0])
    ground, _ = viz.beam_geometry(slant, 0.5)

    assert np.all(ground < slant)
    assert np.all(ground > 0.99 * slant)


def test_beam_geometry_accepts_scalars_and_arrays() -> None:
    scalar_ground, scalar_height = viz.beam_geometry(150_000.0, 1.5)
    assert np.ndim(scalar_ground) == 0
    assert np.ndim(scalar_height) == 0

    ranges = np.array([100.0, 150_000.0], dtype=np.float32)
    elevations = np.array([0.5, 1.5], dtype=np.float32)
    ground, height = viz.beam_geometry(ranges, elevations)

    assert ground.shape == (2,)
    assert height.shape == (2,)
    assert float(height[1]) == pytest.approx(float(scalar_height), rel=1e-6)

    # A scalar elevation broadcasts across a range array.
    broadcast_ground, broadcast_height = viz.beam_geometry(ranges, 1.5)
    assert broadcast_ground.shape == (2,)
    assert float(broadcast_height[1]) == pytest.approx(float(scalar_height), rel=1e-6)


def test_cappi_selects_gates_by_curved_height() -> None:
    """A gate at 150 km sits ~2.63 km up, not the ~1.31 km flat earth predicts."""
    return_times = np.array(["2024-01-01T00:00:00"], dtype="datetime64[ms]").astype(
        "datetime64[ns]"
    )
    sweep_times = return_times.copy()

    # One radial, one gate, 150 km slant range at 0.5 deg elevation.
    returns = xr.Dataset(
        data_vars={
            "azimuth": (("return_time",), np.array([0.0], dtype=np.float32)),
            "elevation": (("return_time",), np.array([0.5], dtype=np.float32)),
            "base_range": (("return_time",), np.array([150_000.0], dtype=np.float32)),
            "range_step": (("return_time",), np.array([250.0], dtype=np.float32)),
            "sweep_number": (("return_time",), np.array([0], dtype=np.uint32)),
            "DBZH": (("return_time", "range"), np.array([[42.0]], dtype=np.float32)),
        },
        coords={
            "return_time": return_times,
            "range": np.arange(1, dtype=np.float32),
        },
    )
    sweeps = xr.Dataset(
        data_vars={
            "sweep_number": (("sweep_time",), np.array([0], dtype=np.uint32)),
            "elevation_angle": (("sweep_time",), np.array([0.5], dtype=np.float32)),
            "num_returns": (("sweep_time",), np.array([1], dtype=np.uint32)),
        },
        coords={"sweep_time": sweep_times},
    )

    _, height = viz.beam_geometry(150_000.0, 0.5)
    curved_altitude = float(height)
    flat_altitude = 150_000.0 * float(np.sin(np.deg2rad(0.5)))
    assert curved_altitude == pytest.approx(2632.9, abs=1.0)

    def _has_data(altitude_m: float) -> bool:
        payload = viz.prepare_cappi_payload(
            returns=returns,
            sweeps=sweeps,
            moment="DBZH",
            altitude_m=altitude_m,
            tolerance_m=500.0,
            grid_size=20,
        )
        return bool(np.any(np.isfinite(payload.grid)))

    # The gate lands in the band around its curved height...
    assert _has_data(curved_altitude)
    # ...and not in the band the flat-earth height would have put it in.
    assert not _has_data(flat_altitude)


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


# ---------------------------------------------------------------------------
# Color scale tests
# ---------------------------------------------------------------------------


def _nyquist_sweeps_fixture(nyquist: float) -> xr.Dataset:
    """Sweeps dataset carrying a Nyquist velocity, as xradar-sourced trees do."""
    _, sweeps = _fixtures()
    return sweeps.assign(
        nyquist_velocity=(
            ("sweep_time",),
            np.full(sweeps.sizes["sweep_time"], nyquist, dtype=np.float32),
        )
    )


def _returns_with_vcp_time() -> xr.Dataset:
    """Returns dataset carrying the ``vcp_time`` the folded view reads."""
    returns, _ = _fixtures()
    return returns.assign(
        vcp_time=(
            ("return_time",),
            np.repeat(returns["return_time"].values[:1], returns.sizes["return_time"]),
        )
    )


def test_color_scale_table_covers_every_moment() -> None:
    for moment in viz.MOMENT_NAMES:
        scale = viz.COLOR_SCALES[moment]
        assert scale.vmin < scale.vmax
        assert scale.name


def test_color_scale_for_known_moment_is_fixed() -> None:
    dbzh = viz.color_scale_for("DBZH")

    assert dbzh is not None
    assert dbzh.name == "nws_reflectivity"
    assert (dbzh.vmin, dbzh.vmax) == (-30.0, 75.0)
    assert dbzh.units == "dBZ"

    # PHIDP follows the parser's 0..360 convention on a cyclic palette.
    phidp = viz.color_scale_for("PHIDP")
    assert phidp is not None
    assert phidp.name == "cyclic"
    assert (phidp.vmin, phidp.vmax) == (0.0, 360.0)


def test_color_scale_for_unknown_moment_is_none() -> None:
    assert viz.color_scale_for("qc.demo_mask") is None
    assert viz.color_scale_for("NOT_A_MOMENT") is None


def test_color_scale_for_velocity_defaults_to_fixed_bounds() -> None:
    returns, sweeps = _fixtures()

    scale = viz.color_scale_for("VRADH", returns=returns, sweeps=sweeps)

    assert scale is not None
    assert (scale.vmin, scale.vmax) == (-32.0, 32.0)


def test_color_scale_for_velocity_uses_nyquist_when_present() -> None:
    returns, _ = _fixtures()
    sweeps = _nyquist_sweeps_fixture(21.5)

    scale = viz.color_scale_for("VRADH", returns=returns, sweeps=sweeps)

    assert scale is not None
    assert (scale.vmin, scale.vmax) == (-21.5, 21.5)
    assert scale.name == "nws_velocity"


def test_known_moment_payload_uses_table_bounds() -> None:
    returns, sweeps = _fixtures()

    payload = viz.prepare_polar_payload(returns, sweeps, 0, "DBZH")

    # Fixture DBZH spans 1..8, but the scale is absolute, not stretched.
    assert (payload.vmin, payload.vmax) == (-30.0, 75.0)
    assert payload.colormap == "nws_reflectivity"
    assert payload.units == "dBZ"


def test_unknown_moment_payload_falls_back_to_percentile_stretch() -> None:
    returns, sweeps = _fixtures()

    payload = viz.prepare_polar_payload(returns, sweeps, 0, "qc.demo_mask")

    assert payload.colormap == viz.FALLBACK_COLORMAP == "viridis"
    assert payload.units == ""
    expected = viz._value_bounds(payload.values)
    assert (payload.vmin, payload.vmax) == expected
    assert payload.vmin >= 0.0 and payload.vmax <= 1.0


def test_color_scale_override_is_honored() -> None:
    returns, sweeps = _fixtures()
    override = viz.ColorScale("magma", -5.0, 12.0, "custom")

    payload = viz.prepare_polar_payload(
        returns, sweeps, 0, "DBZH", color_scale=override
    )

    assert (payload.vmin, payload.vmax) == (-5.0, 12.0)
    assert payload.colormap == "magma"
    assert payload.units == "custom"


def test_color_scale_override_applies_to_every_prepare_function() -> None:
    returns, sweeps = _fixtures()
    override = viz.ColorScale("cyclic", -1.0, 2.0, "u")
    payloads = [
        viz.prepare_polar_payload(returns, sweeps, 0, "DBZH", color_scale=override),
        viz.prepare_volume_payload(returns, "DBZH", color_scale=override),
        viz.prepare_ray_payload(returns, "DBZH", color_scale=override),
        viz.prepare_cappi_payload(
            returns, sweeps, "DBZH", altitude_m=100.0, tolerance_m=5000.0,
            grid_size=20, color_scale=override,
        ),
        viz.prepare_xsec_payload(
            returns, sweeps, "DBZH", azimuth_deg=0.0, azimuth_tolerance_deg=5.0,
            grid_size=20, color_scale=override,
        ),
        viz.prepare_waterfall_payload(returns, "DBZH", color_scale=override),
        viz.prepare_folded_waterfall_payload(
            _returns_with_vcp_time(), "DBZH", color_scale=override
        ),
    ]

    for payload in payloads:
        assert (payload.vmin, payload.vmax) == (-1.0, 2.0), type(payload).__name__
        assert payload.colormap == "cyclic", type(payload).__name__
        assert payload.units == "u", type(payload).__name__


def test_payloads_default_to_the_table_scale() -> None:
    returns, sweeps = _fixtures()
    payloads = [
        viz.prepare_volume_payload(returns, "DBZH"),
        viz.prepare_ray_payload(returns, "DBZH"),
        viz.prepare_waterfall_payload(returns, "DBZH"),
        viz.prepare_folded_waterfall_payload(_returns_with_vcp_time(), "DBZH"),
    ]

    for payload in payloads:
        assert (payload.vmin, payload.vmax) == (-30.0, 75.0), type(payload).__name__
        assert payload.colormap == "nws_reflectivity", type(payload).__name__


def test_payload_without_finite_data_keeps_neutral_bounds() -> None:
    returns, sweeps = _fixtures()

    # No gates fall in this altitude band, so there is nothing to scale.
    payload = viz.prepare_cappi_payload(
        returns, sweeps, "DBZH", altitude_m=100_000.0, tolerance_m=100.0, grid_size=20
    )

    assert (payload.vmin, payload.vmax) == (0.0, 1.0)
    # The palette still names the moment even with no data drawn.
    assert payload.colormap == "nws_reflectivity"
    assert payload.units == "dBZ"


def test_widget_state_meta_carries_colormap_and_units() -> None:
    returns, sweeps = _fixtures()
    payloads = [
        viz.prepare_polar_payload(returns, sweeps, 0, "DBZH"),
        viz.prepare_volume_payload(returns, "DBZH"),
        viz.prepare_cappi_payload(
            returns, sweeps, "DBZH", altitude_m=100.0, tolerance_m=5000.0, grid_size=20
        ),
        viz.prepare_waterfall_payload(returns, "DBZH"),
        viz.prepare_folded_waterfall_payload(_returns_with_vcp_time(), "DBZH"),
    ]

    for payload in payloads:
        meta = payload.to_widget_state()["meta"]
        assert meta["colormap"] == "nws_reflectivity", type(payload).__name__
        assert meta["units"] == "dBZ", type(payload).__name__
        assert meta["vmin"] == -30.0 and meta["vmax"] == 75.0


def test_every_widget_html_carries_a_colorbar() -> None:
    returns, sweeps = _fixtures()
    htmls = [
        viz.prepare_polar_payload(returns, sweeps, 0, "DBZH").to_html(),
        viz.prepare_volume_payload(returns, "DBZH").to_html(),
        viz.prepare_cappi_payload(
            returns, sweeps, "DBZH", altitude_m=100.0, tolerance_m=5000.0, grid_size=20
        ).to_html(),
        viz.prepare_waterfall_payload(returns, "DBZH").to_html(),
        viz.prepare_folded_waterfall_payload(
            _returns_with_vcp_time(), "DBZH"
        ).to_html(),
    ]

    for html in htmls:
        assert "radrs-viz-root" in html
        assert "drawColorbar" in html
        assert "nws_reflectivity" in html


# ---------------------------------------------------------------------------
# Front-end assembly: static/*.js concatenated into one ES module per widget
# ---------------------------------------------------------------------------

WIDGET_JS_FILES = (
    "polar.js",
    "volume.js",
    "grid.js",
    "waterfall.js",
    "folded_waterfall.js",
)

DECK_GL_IMPORT = 'import { COORDINATE_SYSTEM, Deck, LineLayer, OrbitView, PointCloudLayer } from "https://esm.sh/deck.gl@9.2.2?bundle";'

requires_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not on PATH"
)


def _node(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["node", *args], capture_output=True, text=True, timeout=60, check=False
    )


@pytest.mark.parametrize("widget_js", WIDGET_JS_FILES)
def test_assembled_module_embeds_shared_exactly_once(widget_js: str) -> None:
    shared = viz.assets.read_static("shared.js")
    assembled = viz.assemble_esm(widget_js)

    assert assembled.count(shared) == 1
    # Nothing that shared.js owns may be redeclared by a widget file.
    for helper in ("function decodeArray(", "const PALETTES = {", "function buildLut("):
        assert assembled.count(helper) == 1, helper


def test_volume_module_keeps_the_deck_gl_import_first() -> None:
    """ES modules require imports before any other statement."""
    assembled = viz.assemble_esm("volume.js")

    assert assembled.startswith(DECK_GL_IMPORT)
    assert assembled.index(DECK_GL_IMPORT) < assembled.index("function decodeArray(")


def test_only_the_volume_module_imports_anything() -> None:
    for widget_js in WIDGET_JS_FILES:
        assembled = viz.assemble_esm(widget_js)
        expected = 1 if widget_js == "volume.js" else 0
        assert assembled.count("\nimport ") + assembled.startswith("import ") == expected


def test_palette_table_names_every_color_scale() -> None:
    shared = viz.assets.read_static("shared.js")

    for scale in viz.COLOR_SCALES.values():
        assert f"  {scale.name}: [" in shared
    assert f"  {viz.FALLBACK_COLORMAP}: [" in shared


@requires_node
@pytest.mark.parametrize("widget_js", WIDGET_JS_FILES)
def test_assembled_module_is_valid_javascript(widget_js: str, tmp_path) -> None:
    module_path = tmp_path / f"{widget_js.removesuffix('.js')}.mjs"
    module_path.write_text(viz.assemble_esm(widget_js), encoding="utf-8")

    result = _node("--check", str(module_path))

    assert result.returncode == 0, result.stderr


@requires_node
def test_shared_js_helpers_behave(tmp_path) -> None:
    """Run shared.js under node and exercise the helpers it defines."""
    script = tmp_path / "shared_check.mjs"
    script.write_text(
        viz.assets.read_static("shared.js")
        + """
function check(label, ok) {
  if (!ok) {
    console.error(`FAIL ${label}`);
    process.exit(1);
  }
}

const lut = buildLut("nws_reflectivity");
check("lut length", lut.length === 768);
check("lut in range", lut.every((v) => v >= 0 && v <= 255));
check("lut low stop", lut[0] === 40 && lut[1] === 40 && lut[2] === 48);
check("lut high stop", lut[765] === 255 && lut[766] === 255 && lut[767] === 255);

// Out-of-range normalized values clamp to the ends rather than reading past them.
check("clamp low", String(lutColor(lut, -5)) === String(lutColor(lut, 0)));
check("clamp high", String(lutColor(lut, 5)) === String(lutColor(lut, 1)));
check("nan colour defined", lutColor(lut, 0.5).every(Number.isFinite));

// An unknown palette name falls back rather than throwing.
check("fallback palette", buildLut("not_a_palette").length === 768);

// decodeArray drops a partial trailing element instead of throwing.
check("trim partial", decodeArray(new Uint8Array(5), Float32Array).length === 1);
check("exact fit", decodeArray(new Uint8Array(8), Float32Array).length === 2);
check("empty", decodeArray(null, Float32Array).length === 0);
check("undersized", decodeArray(new Uint8Array(3), Float32Array).length === 0);

// beamHeightM mirrors radrs.viz.beam_geometry: 200 km at 0.5 deg is ~4099 m,
// well above the ~1745 m of a flat earth.
const h = beamHeightM(200000, 0.5);
check("curved height", Math.abs(h - 4099) < 1.0);
check("above flat earth", h > 200000 * Math.sin((0.5 * Math.PI) / 180) + 2000);
check("zero range", Math.abs(beamHeightM(0, 0.5)) < 1e-6);

// beamGroundRangeM is the other half of the same model; the gate cloud needs
// both to rebuild a position from azimuth, elevation and range.
const g = beamGroundRangeM(200000, 0.5);
check("ground below slant", g < 200000 && g > 0.99 * 200000);
check("ground zero range", Math.abs(beamGroundRangeM(0, 0.5)) < 1e-6);

// Quantization mirrors radrs.viz.payloads.quantize.
check("quant nan", Number.isNaN(dequantize(QUANT_NAN, -30, 75)));
check("quant low", Math.abs(dequantize(0, -30, 75) - -30) < 1e-9);
check("quant high", Math.abs(dequantize(QUANT_MAX, -30, 75) - 75) < 1e-9);
check("quant mid", Math.abs(dequantize(QUANT_MAX / 2, -30, 75) - 22.5) < 1e-6);
check("quant fraction ends", quantFraction(0) === 0 && quantFraction(QUANT_MAX) === 1);
// A degenerate scale must not divide by zero.
check("quant flat scale", Number.isFinite(dequantize(100, 3, 3)));

check("nearestPick miss", nearestPick(new Int32Array(16).fill(-1), 4, 4, 1, 1) === -1);
const pick = new Int32Array(16).fill(-1);
pick[2 * 4 + 2] = 7;
check("nearestPick hit", nearestPick(pick, 4, 4, 2, 2) === 7);
check("nearestPick nearby", nearestPick(pick, 4, 4, 1, 1) === 7);

check("ticks", String(niceTicks(0, 10, 5)) === String([0, 2, 4, 6, 8, 10]));
check("ticks empty", niceTicks(5, 5, 5).length === 0);

check("utc", formatTimeUTC(Date.UTC(2024, 0, 1, 3, 4, 5)) === "03:04:05");
check("utc ms", formatTimeUTCms(Date.UTC(2024, 0, 1, 3, 4, 5, 6)) === "03:04:05.006");

console.log("OK");
""",
        encoding="utf-8",
    )

    result = _node(str(script))

    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


@pytest.mark.parametrize(
    ("widget_cls", "widget_js"),
    [
        (viz.PolarWidget, "polar.js"),
        (viz.VolumeWidget, "volume.js"),
        (viz.GridWidget, "grid.js"),
        (viz.WaterfallWidget, "waterfall.js"),
        (viz.FoldedWaterfallWidget, "folded_waterfall.js"),
    ],
)
def test_widget_classes_carry_assembled_sources(widget_cls, widget_js: str) -> None:
    assert widget_cls._esm
    assert widget_cls._esm == viz.assemble_esm(widget_js)
    assert widget_cls._css == viz.widget_css()
    assert "export default {" in widget_cls._esm


def _payload_for(widget_js: str):
    returns, sweeps = _fixtures()
    if widget_js == "polar.js":
        return viz.prepare_polar_payload(returns, sweeps, 0, "DBZH")
    if widget_js == "volume.js":
        return viz.prepare_volume_payload(returns, "DBZH")
    if widget_js == "grid.js":
        return viz.prepare_cappi_payload(
            returns, sweeps, "DBZH", altitude_m=100.0, tolerance_m=5000.0, grid_size=20
        )
    if widget_js == "waterfall.js":
        return viz.prepare_waterfall_payload(returns, "DBZH")
    return viz.prepare_folded_waterfall_payload(_returns_with_vcp_time(), "DBZH")


@pytest.mark.parametrize(
    ("widget_cls", "widget_js"),
    [
        (viz.PolarWidget, "polar.js"),
        (viz.VolumeWidget, "volume.js"),
        (viz.GridWidget, "grid.js"),
        (viz.WaterfallWidget, "waterfall.js"),
        (viz.FoldedWaterfallWidget, "folded_waterfall.js"),
    ],
)
def test_every_payload_buffer_reaches_the_widget(widget_cls, widget_js: str) -> None:
    """A payload buffer with no trait, or no `change:` watch, never reaches the canvas."""
    payload = _payload_for(widget_js)
    widget = widget_cls()
    widget.set_payload(payload)
    esm = viz.assemble_esm(widget_js)

    buffers = {k: v for k, v in payload.to_widget_state().items() if k != "meta"}
    assert buffers
    for key, value in buffers.items():
        assert widget.has_trait(key), f"{widget_cls.__name__} has no trait '{key}'"
        assert getattr(widget, key) == value, key
        assert f'"{key}"' in esm, f"{widget_js} never reads '{key}'"


def test_dev_switch_rereads_sources(monkeypatch, tmp_path) -> None:
    """`RADRS_VIZ_DEV=1` bypasses the read cache so edits show up."""
    assert viz.dev_mode() is False
    monkeypatch.setenv(viz.DEV_ENV_VAR, "1")
    assert viz.dev_mode() is True

    marker = "// dev-reload marker\n"
    shared_path = viz.STATIC_DIR / "shared.js"
    original = shared_path.read_text(encoding="utf-8")
    try:
        shared_path.write_text(original + marker, encoding="utf-8")
        assert marker in viz.assemble_esm("polar.js")
    finally:
        shared_path.write_text(original, encoding="utf-8")

    monkeypatch.delenv(viz.DEV_ENV_VAR)
    assert marker not in viz.assemble_esm("polar.js")


# ---------------------------------------------------------------------------
# Curved-earth fields the widgets read instead of redoing the geometry in JS
# ---------------------------------------------------------------------------


def _long_range_returns() -> xr.Dataset:
    """One gate at 200 km slant range, 0.5 deg elevation, due north."""
    return_times = np.array(["2024-01-01T00:00:00"], dtype="datetime64[ms]").astype(
        "datetime64[ns]"
    )
    return xr.Dataset(
        data_vars={
            "azimuth": (("return_time",), np.array([0.0], dtype=np.float32)),
            "elevation": (("return_time",), np.array([0.5], dtype=np.float32)),
            "base_range": (("return_time",), np.array([200_000.0], dtype=np.float32)),
            "range_step": (("return_time",), np.array([250.0], dtype=np.float32)),
            "sweep_number": (("return_time",), np.array([0], dtype=np.uint32)),
            "DBZH": (("return_time", "range"), np.array([[42.0]], dtype=np.float32)),
        },
        coords={
            "return_time": return_times,
            "range": np.arange(1, dtype=np.float32),
        },
    )


def test_volume_payload_carries_native_polar_coordinates() -> None:
    """The tooltip reads these; recovering them from x/y/z would need a flat earth."""
    returns = _long_range_returns()

    payload = viz.prepare_volume_payload(returns, "DBZH", max_points=None)

    assert payload.point_count == 1
    assert float(payload.azimuth_deg[0]) == pytest.approx(0.0)
    assert float(payload.elevation_deg[0]) == pytest.approx(0.5)
    assert float(payload.range_m[0]) == pytest.approx(200_000.0)

    # Inverting the Cartesian triple with flat trigonometry, which is what the
    # widget used to do, lands on a very different elevation and range.
    x, y, z = float(payload.x_m[0]), float(payload.y_m[0]), float(payload.z_m[0])
    flat_elevation = np.degrees(np.arctan2(z, np.hypot(x, y)))
    flat_range = np.sqrt(x * x + y * y + z * z)
    assert flat_elevation > 1.0
    assert abs(flat_range - 200_000.0) > 20.0


def test_ray_payload_carries_native_polar_coordinates() -> None:
    returns = _long_range_returns()

    payload = viz.prepare_ray_payload(returns, "DBZH", max_points=None)

    assert payload.render_mode == "rays"
    assert float(payload.elevation_deg[0]) == pytest.approx(0.5)
    assert float(payload.range_m[0]) == pytest.approx(200_000.0)


def test_volume_payload_polar_buffers_match_point_count() -> None:
    returns, _ = _fixtures()

    for payload in (
        viz.prepare_volume_payload(returns, "DBZH"),
        viz.prepare_ray_payload(returns, "DBZH"),
        viz.prepare_volume_payload(returns, "DBZH", max_points=4),
        viz.prepare_ray_payload(returns, "DBZH", max_points=2),
    ):
        n = payload.point_count
        used = payload.return_count
        assert used <= n, payload.render_mode

        # Per-return geometry is one entry per referenced return...
        assert payload.azimuth_deg.shape == (used,), payload.render_mode
        assert payload.elevation_deg.shape == (used,), payload.render_mode
        assert payload.base_range_m.shape == (used,), payload.render_mode
        assert payload.range_step_m.shape == (used,), payload.render_mode

        # ...and every point resolves against it.
        assert payload.return_slot.shape == (n,), payload.render_mode
        assert int(payload.return_slot.max(initial=0)) < max(1, used)
        assert payload.range_m.shape == (n,), payload.render_mode
        assert payload.point_azimuth_deg.shape == (n,), payload.render_mode

        state = payload.to_widget_state()
        assert len(state["azimuth_bytes"]) == 4 * used
        assert len(state["elevation_bytes"]) == 4 * used
        assert len(state["base_range_bytes"]) == 4 * used
        assert len(state["return_slot_bytes"]) == 4 * n
        assert len(state["value_bytes"]) == 2 * n


def test_grid_payload_ships_curved_sweep_altitudes() -> None:
    """The elevation inset needs the height each sweep reaches, not r*sin(el)."""
    returns, sweeps = _fixtures()

    payload = viz.prepare_cappi_payload(
        returns, sweeps, "DBZH", altitude_m=100.0, tolerance_m=5000.0, grid_size=20
    )

    assert payload.sweep_max_altitude_m is not None
    assert payload.sweep_elevations_deg is not None
    assert len(payload.sweep_max_altitude_m) == len(payload.sweep_elevations_deg)

    _, expected = viz.beam_geometry(
        abs(payload.x_max), payload.sweep_elevations_deg.astype(np.float64)
    )
    np.testing.assert_allclose(payload.sweep_max_altitude_m, expected, rtol=1e-5)

    meta = payload.to_widget_state()["meta"]
    assert meta["sweep_max_altitude_m"] == [
        float(a) for a in payload.sweep_max_altitude_m
    ]


def test_grid_sweep_altitudes_exceed_the_flat_earth_value() -> None:
    """At long range the curved height is what decides whether a sweep is active."""
    returns = _long_range_returns()
    sweep_times = np.array(["2024-01-01T00:00:00"], dtype="datetime64[ms]").astype(
        "datetime64[ns]"
    )
    sweeps = xr.Dataset(
        data_vars={
            "sweep_number": (("sweep_time",), np.array([0], dtype=np.uint32)),
            "elevation_angle": (("sweep_time",), np.array([0.5], dtype=np.float32)),
            "num_returns": (("sweep_time",), np.array([1], dtype=np.uint32)),
        },
        coords={"sweep_time": sweep_times},
    )

    payload = viz.prepare_cappi_payload(
        returns, sweeps, "DBZH", altitude_m=2600.0, tolerance_m=500.0, grid_size=20
    )

    assert payload.sweep_max_altitude_m is not None
    curved = float(payload.sweep_max_altitude_m[0])
    flat = abs(payload.x_max) * float(np.sin(np.deg2rad(0.5)))
    assert curved > flat + 1000.0


def test_xsec_payload_also_ships_sweep_altitudes() -> None:
    returns, sweeps = _fixtures()

    payload = viz.prepare_xsec_payload(
        returns, sweeps, "DBZH", azimuth_deg=0.0, azimuth_tolerance_deg=5.0, grid_size=20
    )

    assert payload.sweep_max_altitude_m is not None
    assert np.all(np.isfinite(payload.sweep_max_altitude_m))
    assert "sweep_max_altitude_m" in payload.to_widget_state()["meta"]


def test_grid_js_reads_the_shipped_altitudes() -> None:
    """The inset must not fall back to flat-earth trigonometry."""
    grid_js = viz.assets.read_static("grid.js")

    assert "sweep_max_altitude_m" in grid_js
    # The fallback path, for payloads built before the field existed, is curved too.
    assert "beamHeightM(rangeM, el)" in grid_js
    # No altitude is derived from `x_max * sin(elevation)` any more. (Plain
    # `Math.sin(rad)` survives: the inset still draws each beam as a ray at its
    # true angle, which is a schematic, not a height.)
    assert "xMaxM" not in grid_js
    assert "* Math.sin(Math.max(" not in grid_js


# ---------------------------------------------------------------------------
# Value quantization: uint16 codes over [vmin, vmax], one sentinel for missing
# ---------------------------------------------------------------------------


def test_quantize_roundtrips_far_below_instrument_resolution() -> None:
    """DBZH's step is ~0.0016 dB, so hover stays exact enough for QC work."""
    vmin, vmax = -30.0, 75.0
    values = np.linspace(vmin, vmax, 5000, dtype=np.float32)

    codes = viz.quantize(values, vmin, vmax)
    recovered = viz.dequantize(codes, vmin, vmax)

    assert codes.dtype == np.uint16
    step = (vmax - vmin) / viz.QUANT_MAX
    assert step < 0.002
    np.testing.assert_allclose(recovered, values, atol=step)


def test_quantize_clamps_values_outside_the_scale() -> None:
    """Documented behaviour: out-of-range values pin to the ends, not wrap."""
    codes = viz.quantize(np.array([-100.0, -30.0, 75.0, 500.0], np.float32), -30.0, 75.0)

    np.testing.assert_array_equal(codes, [0, 0, viz.QUANT_MAX, viz.QUANT_MAX])


def test_quantize_marks_missing_data_with_the_sentinel() -> None:
    codes = viz.quantize(np.array([1.0, np.nan, np.inf, -np.inf], np.float32), 0.0, 10.0)

    assert codes[0] != viz.QUANT_NAN
    assert list(codes[1:]) == [viz.QUANT_NAN] * 3
    assert np.isnan(viz.dequantize(codes, 0.0, 10.0)[1:]).all()
    # The sentinel is outside the finite code range, so no real value collides.
    assert viz.QUANT_NAN > viz.QUANT_MAX


def test_quantize_survives_a_degenerate_scale() -> None:
    """An empty payload's neutral 0..1 bounds must not divide by zero."""
    codes = viz.quantize(np.array([5.0], np.float32), 3.0, 3.0)

    assert codes.dtype == np.uint16
    assert np.isfinite(viz.dequantize(codes, 3.0, 3.0)).all()


@pytest.mark.parametrize("widget_js", WIDGET_JS_FILES)
def test_value_buffers_ship_as_uint16(widget_js: str) -> None:
    """Every payload quantizes, including the grids."""
    payload = _payload_for(widget_js)
    state = payload.to_widget_state()
    key = "value_bytes" if "value_bytes" in state else "grid_bytes"

    if widget_js == "polar.js":
        expected = payload.cell_count
    elif widget_js == "volume.js":
        expected = payload.point_count
    else:
        expected = payload.grid.size

    assert len(state[key]) == 2 * expected, widget_js


def test_shared_js_quantization_constants_match_python() -> None:
    """A drift here would silently recolour or hide every gate."""
    shared = viz.assets.read_static("shared.js")

    assert f"const QUANT_MAX = {viz.QUANT_MAX};" in shared
    assert f"const QUANT_NAN = {viz.QUANT_NAN};" in shared


# ---------------------------------------------------------------------------
# Byte budget: the library picks its own reduction rather than the caller
# ---------------------------------------------------------------------------


def _default_budget_payloads(returns: xr.Dataset, sweeps: xr.Dataset) -> dict:
    return {
        "polar": viz.prepare_polar_payload(returns, sweeps, 0, "DBZH"),
        "volume": viz.prepare_volume_payload(returns, "DBZH"),
        "ray": viz.prepare_ray_payload(returns, "DBZH"),
        "cappi": viz.prepare_cappi_payload(
            returns, sweeps, "DBZH", altitude_m=2000.0, tolerance_m=500.0
        ),
        "xsec": viz.prepare_xsec_payload(returns, sweeps, "DBZH", azimuth_deg=0.0),
        "waterfall": viz.prepare_waterfall_payload(returns, "DBZH"),
        "folded": viz.prepare_folded_waterfall_payload(returns, "DBZH"),
    }


def test_default_budget_fits_every_payload_from_a_real_sized_volume() -> None:
    """14 sweeps x 360 radials x 1832 gates, the shape a NEXRAD volume actually has."""
    returns, sweeps = _realistic_volume()

    for name, payload in _default_budget_payloads(returns, sweeps).items():
        assert payload.nbytes <= viz.DEFAULT_BYTE_BUDGET, (
            f"{name} is {payload.nbytes:,} bytes, over the "
            f"{viz.DEFAULT_BYTE_BUDGET:,} byte budget"
        )
        # A budget that reduced everything to nothing would also "fit".
        assert payload.nbytes > 0, name


def test_nbytes_is_the_size_of_the_widget_state() -> None:
    returns, sweeps = _fixtures()

    for widget_js in WIDGET_JS_FILES:
        payload = _payload_for(widget_js)
        assert payload.nbytes == viz.widget_state_nbytes(payload.to_widget_state())


def test_full_ppi_sweep_ships_whole_inside_the_default_budget() -> None:
    """A 360x1832 sweep is ~1.3 MB quantized, so nothing has to be thrown away."""
    returns, sweeps = _dense_returns(360, 1832)

    payload = viz.prepare_polar_payload(returns, sweeps, 0, "DBZH")

    assert payload.return_stride == 1
    assert payload.gate_stride == 1
    assert payload.values.shape == (360, 1832)
    assert payload.nbytes < viz.DEFAULT_BYTE_BUDGET


def test_tighter_budget_buys_a_coarser_picture() -> None:
    returns, sweeps = _dense_returns(360, 1832)

    generous = viz.prepare_polar_payload(returns, sweeps, 0, "DBZH")
    tight = viz.prepare_polar_payload(returns, sweeps, 0, "DBZH", byte_budget=200_000)

    assert tight.nbytes <= 200_000
    assert tight.cell_count < generous.cell_count
    # Both axes give ground, rather than one collapsing.
    assert tight.return_stride > 1
    assert tight.gate_stride > 1


def test_folded_waterfall_row_window_follows_the_budget() -> None:
    """A wide fold buys fewer rows; the caller no longer does that arithmetic."""
    returns, _ = _fixtures()
    wide = _returns_with_vcp_time()

    narrow_rows = viz.prepare_folded_waterfall_payload(
        wide.isel(range=slice(0, 1)), "DBZH", byte_budget=10_000
    )
    wide_rows = viz.prepare_folded_waterfall_payload(wide, "DBZH", byte_budget=10_000)

    assert narrow_rows.nbytes <= 10_000
    assert wide_rows.nbytes <= 10_000
    # Both windows are clamped by the fixture's 3 rows, so compare the budgets
    # the payloads were sized against on a dataset long enough to show it.
    long_returns, _ = _dense_returns(4000, 128)
    long_returns = long_returns.assign(
        vcp_time=(
            ("return_time",),
            np.repeat(long_returns["return_time"].values[:1], 4000),
        )
    )
    narrow = viz.prepare_folded_waterfall_payload(long_returns, "DBZH", byte_budget=100_000)
    assert narrow.n_rows < 4000
    assert narrow.nbytes <= 100_000


def test_byte_budget_none_ships_the_payload_whole() -> None:
    returns, sweeps = _dense_returns(4000, 1024)

    unbounded = viz.prepare_polar_payload(returns, sweeps, 0, "DBZH", byte_budget=None)
    budgeted = viz.prepare_polar_payload(returns, sweeps, 0, "DBZH")

    assert unbounded.return_stride == 1
    assert unbounded.gate_stride == 1
    assert unbounded.cell_count == 4000 * 1024
    assert unbounded.nbytes > viz.DEFAULT_BYTE_BUDGET
    assert budgeted.nbytes <= viz.DEFAULT_BYTE_BUDGET


def test_max_points_still_overrides_the_budget() -> None:
    returns, sweeps = _dense_returns(360, 1832)

    capped = viz.prepare_volume_payload(returns, "DBZH", max_points=5_000)

    assert capped.point_count <= 5_000
    assert capped.nbytes < viz.DEFAULT_BYTE_BUDGET


# ---------------------------------------------------------------------------
# Downsampling: a stride over a return-major array only lands on a few gates
# ---------------------------------------------------------------------------


def test_gate_cloud_sampling_spreads_across_the_range_axis() -> None:
    """Regression: a constant stride put all 100 samples on gate 0."""
    returns, _ = _dense_returns(400, 4)

    payload = viz.prepare_volume_payload(returns, "DBZH", max_points=100)

    histogram = np.bincount(np.asarray(payload.gate_index), minlength=4)
    assert payload.point_count == 100
    assert int(histogram.min()) > 0, histogram.tolist()
    # Every gate takes a fair share, give or take sampling noise; the old
    # stride gave [100, 0, 0, 0].
    assert int(histogram.max()) < 60, histogram.tolist()


def test_polar_gate_decimation_covers_the_whole_range_axis() -> None:
    """The PPI's other collapse mode: rings, not a full sweep."""
    returns, sweeps = _dense_returns(400, 64)

    payload = viz.prepare_polar_payload(returns, sweeps, 0, "DBZH", max_points=400)

    gates = np.asarray(payload.gate_index)
    assert payload.cell_count <= 400
    # Gates are evenly spaced from the first to near the last, so the drawn
    # rings span the sweep instead of clustering at one radius.
    assert gates[0] == 0
    assert gates[-1] >= 64 - payload.gate_stride
    assert len(np.unique(np.diff(gates))) <= 1


def test_sampling_is_reproducible() -> None:
    """Same inputs, same picture: a fixed seed, not a fixed stride."""
    returns, _ = _dense_returns(400, 4)

    first = viz.prepare_volume_payload(returns, "DBZH", max_points=100)
    second = viz.prepare_volume_payload(returns, "DBZH", max_points=100)

    np.testing.assert_array_equal(first.gate_index, second.gate_index)
    np.testing.assert_array_equal(first.return_slot, second.return_slot)
    np.testing.assert_allclose(first.values, second.values)


def test_sampled_points_keep_their_own_values() -> None:
    """Sampling must not shuffle a value onto the wrong gate."""
    returns, _ = _dense_returns(50, 8)
    source = np.asarray(returns["DBZH"].values)

    payload = viz.prepare_volume_payload(returns, "DBZH", max_points=40)

    rows = np.asarray(payload.point_return_index, dtype=np.int64)
    cols = np.asarray(payload.gate_index, dtype=np.int64)
    np.testing.assert_allclose(payload.values, source[rows, cols])


def test_sample_indices_are_sorted_and_unique() -> None:
    keep = viz.payloads._sample_indices(1000, 100)

    assert keep.size == 100
    assert np.all(np.diff(keep) > 0)
    assert keep.min() >= 0 and keep.max() < 1000


# ---------------------------------------------------------------------------
# The widgets rebuild coordinates the payloads no longer ship. Run that JS and
# check it lands where Python says the gates are.
# ---------------------------------------------------------------------------


def _b64(raw: bytes) -> str:
    import base64

    return base64.b64encode(raw).decode("ascii")


@requires_node
def test_volume_js_rebuilds_the_positions_python_dropped(tmp_path) -> None:
    returns, _ = _fixtures()
    payload = viz.prepare_volume_payload(returns, "DBZH")
    state = payload.to_widget_state()

    expected = np.stack([payload.x_m, payload.y_m, payload.z_m], axis=1)

    script = tmp_path / "volume_positions.mjs"
    script.write_text(
        viz.assets.read_static("shared.js")
        + f"""
const S = {{
  codes: "{_b64(state["value_bytes"])}",
  slots: "{_b64(state["return_slot_bytes"])}",
  gates: "{_b64(state["gate_index_bytes"])}",
  az: "{_b64(state["azimuth_bytes"])}",
  el: "{_b64(state["elevation_bytes"])}",
  base: "{_b64(state["base_range_bytes"])}",
  step: "{_b64(state["range_step_bytes"])}",
}};
function buf(b64) {{
  return Uint8Array.from(Buffer.from(b64, "base64")).buffer;
}}
const codes = decodeArray(buf(S.codes), Uint16Array);
const slots = decodeArray(buf(S.slots), Uint32Array);
const gates = decodeArray(buf(S.gates), Uint16Array);
const az = decodeArray(buf(S.az), Float32Array);
const el = decodeArray(buf(S.el), Float32Array);
const base = decodeArray(buf(S.base), Float32Array);
const step = decodeArray(buf(S.step), Float32Array);

const out = [];
for (let i = 0; i < codes.length; i += 1) {{
  const slot = slots[i];
  const r = base[slot] + gates[i] * step[slot];
  const ground = beamGroundRangeM(r, el[slot]);
  const azRad = (az[slot] * Math.PI) / 180.0;
  out.push([
    ground * Math.sin(azRad),
    ground * Math.cos(azRad),
    beamHeightM(r, el[slot]),
    dequantize(codes[i], {payload.vmin}, {payload.vmax}),
    r,
  ]);
}}
console.log(JSON.stringify(out));
""",
        encoding="utf-8",
    )

    result = _node(str(script))
    assert result.returncode == 0, result.stderr

    import json

    rebuilt = np.array(json.loads(result.stdout), dtype=np.float64)
    assert rebuilt.shape == (payload.point_count, 5)
    np.testing.assert_allclose(rebuilt[:, :3], expected, rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(rebuilt[:, 4], payload.range_m, rtol=1e-6)
    # Quantized values come back within one step of the source.
    step_size = (payload.vmax - payload.vmin) / viz.QUANT_MAX
    np.testing.assert_allclose(rebuilt[:, 3], payload.values, atol=step_size)


@requires_node
def test_polar_js_rebuilds_the_range_grid_python_dropped(tmp_path) -> None:
    returns, sweeps = _fixtures()
    payload = viz.prepare_polar_payload(returns, sweeps, 0, "DBZH")
    state = payload.to_widget_state()

    script = tmp_path / "polar_ranges.mjs"
    script.write_text(
        viz.assets.read_static("shared.js")
        + f"""
function buf(b64) {{
  return Uint8Array.from(Buffer.from(b64, "base64")).buffer;
}}
const base = decodeArray(buf("{_b64(state["base_range_bytes"])}"), Float32Array);
const step = decodeArray(buf("{_b64(state["range_step_bytes"])}"), Float32Array);
const codes = decodeArray(buf("{_b64(state["value_bytes"])}"), Uint16Array);
const nGates = {payload.n_gates};
const gateStride = {payload.gate_stride};

const ranges = [];
const values = [];
for (let row = 0; row < base.length; row += 1) {{
  for (let col = 0; col < nGates; col += 1) {{
    ranges.push(base[row] + col * gateStride * step[row]);
    values.push(dequantize(codes[row * nGates + col], {payload.vmin}, {payload.vmax}));
  }}
}}
console.log(JSON.stringify({{ ranges, values }}));
""",
        encoding="utf-8",
    )

    result = _node(str(script))
    assert result.returncode == 0, result.stderr

    import json

    got = json.loads(result.stdout)
    np.testing.assert_allclose(
        np.array(got["ranges"], dtype=np.float64).reshape(payload.values.shape),
        payload.range_m,
        rtol=1e-6,
    )
    # JSON has no NaN literal, so missing gates arrive as null.
    values = np.array(
        [np.nan if v is None else v for v in got["values"]], dtype=np.float64
    ).reshape(payload.values.shape)
    step_size = (payload.vmax - payload.vmin) / viz.QUANT_MAX
    np.testing.assert_allclose(values, payload.values, atol=step_size)
