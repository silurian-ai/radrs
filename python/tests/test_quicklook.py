"""Tests for radrs.quicklook adapter utilities."""

import numpy as np
import pytest
import xarray as xr

import radrs.quicklook as ql


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

    moments = ql.available_moments(returns, include_qc=True)

    assert "DBZH" in moments
    assert "qc.demo_mask" in moments


def test_sweep_offsets_and_infos() -> None:
    _, sweeps = _fixtures()

    offsets = ql.sweep_offsets(sweeps)
    infos = ql.sweep_infos(sweeps)

    np.testing.assert_array_equal(offsets, np.array([0, 2, 3], dtype=np.int64))
    assert len(infos) == 2
    assert infos[0].index == 0
    assert infos[0].sweep_number == 0
    assert infos[0].num_returns == 2
    assert "elev 0.50 deg" in infos[0].label


def test_prepare_polar_payload_builds_expected_gate_geometry() -> None:
    returns, sweeps = _fixtures()

    payload = ql.prepare_polar_payload(
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

    payload = ql.prepare_polar_payload(
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
        ql.prepare_polar_payload(
            returns=returns,
            sweeps=sweeps,
            sweep_index=0,
            moment="NOT_A_MOMENT",
        )


def test_prepare_payload_supports_legacy_sweeps_schema() -> None:
    returns, _ = _fixtures()
    sweeps = _legacy_sweeps_fixture()

    offsets = ql.sweep_offsets(sweeps)
    infos = ql.sweep_infos(sweeps)
    payload = ql.prepare_polar_payload(
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

    payload = ql.prepare_volume_payload(
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

    payload = ql.prepare_ray_payload(
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
