"""Tests for radrs.viz raystack plots."""

import numpy as np
import pytest
import xarray as xr

import radrs.viz as viz

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def _raystack(fold_size: int = 4, n_folds: int = 2, n_radials: int = 3) -> xr.DataTree:
    """Two sweeps of a single VCP, folded so each radial spans several rows."""

    n_rows = n_radials * n_folds * 2
    return_times = np.arange(n_rows, dtype="int64") // n_folds
    return_times = return_times.astype("datetime64[s]").astype("datetime64[ns]")

    # Row r covers gates [fold * fold_size, (fold + 1) * fold_size).
    fold = np.tile(np.arange(n_folds), n_radials * 2)
    moment = np.arange(n_rows * fold_size, dtype=np.float32).reshape(n_rows, fold_size)
    moment[0, 1] = np.nan

    returns = xr.Dataset(
        data_vars={
            "vcp_time": (("return_time",), np.full(n_rows, np.datetime64("2024-01-01", "ns"))),
            "sweep_number": (
                ("return_time",),
                np.repeat([0, 1], n_rows // 2).astype(np.uint32),
            ),
            "azimuth": (
                ("return_time",),
                np.repeat(np.tile([0.0, 120.0, 240.0], 2), n_folds).astype(np.float32),
            ),
            "elevation": (("return_time",), np.full(n_rows, 0.5, dtype=np.float32)),
            "base_range": (
                ("return_time",),
                (125.0 + fold * fold_size * 250.0).astype(np.float32),
            ),
            "range_step": (("return_time",), np.full(n_rows, 250.0, dtype=np.float32)),
            "DBZH": (("return_time", "range"), moment),
            "qc.demo_mask": (("return_time", "range"), np.zeros_like(moment)),
        },
        coords={"return_time": return_times, "range": np.arange(fold_size, dtype=np.uint32)},
    )
    sweeps = xr.Dataset(
        data_vars={
            "vcp_time": (("sweep_time",), np.full(2, np.datetime64("2024-01-01", "ns"))),
            "sweep_number": (("sweep_time",), np.array([0, 1], dtype=np.uint32)),
            "elevation_angle": (("sweep_time",), np.array([0.5, 1.5], dtype=np.float32)),
            "num_returns": (("sweep_time",), np.full(2, n_rows // 2, dtype=np.uint32)),
        },
        coords={
            "sweep_time": np.array(
                ["2024-01-01T00:00:00", "2024-01-01T00:00:10"], dtype="datetime64[ns]"
            )
        },
    )
    vcps = xr.Dataset(
        data_vars={
            "vcp_number": (("vcp_time",), np.array([212], dtype=np.uint16)),
            "instrument_name": (("vcp_time",), np.array(["KTLX"])),
            "latitude": (("vcp_time",), np.array([35.3333], dtype=np.float32)),
            "longitude": (("vcp_time",), np.array([-97.2778], dtype=np.float32)),
            "altitude": (("vcp_time",), np.array([389.0], dtype=np.float32)),
        },
        coords={"vcp_time": np.array(["2024-01-01T00:00:00"], dtype="datetime64[ns]")},
    )
    return xr.DataTree.from_dict({"vcps": vcps, "sweeps": sweeps, "returns": returns})


def test_available_moments_includes_core_and_qc() -> None:
    moments = viz.available_moments(_raystack(), include_qc=True)

    assert "DBZH" in moments
    assert "qc.demo_mask" in moments
    assert "qc.demo_mask" not in viz.available_moments(_raystack(), include_qc=False)


def test_vcp_and_sweep_infos_describe_the_tree() -> None:
    rs_dt = _raystack()

    vcps = viz.vcp_infos(rs_dt)
    sweeps = viz.sweep_infos(rs_dt)

    assert len(vcps) == 1
    assert vcps[0].vcp_number == 212
    assert vcps[0].instrument_name == "KTLX"
    assert vcps[0].num_sweeps == 2
    assert [info.sweep_number for info in sweeps] == [0, 1]
    assert "elev  0.50 deg" in sweeps[0].label


def test_value_bounds_survives_all_nan_and_constant_input() -> None:
    assert viz.value_bounds(np.full(4, np.nan, dtype=np.float32)) == (0.0, 1.0)
    assert viz.value_bounds(np.full(4, 7.0, dtype=np.float32)) == (7.0, 8.0)


def test_plot_sweep_titles_with_vcp_sweep_moment_and_site() -> None:
    fig, ax = viz.plot_sweep(_raystack(), 0)

    title = ax.get_title()
    assert "VCP 212" in title
    assert "sweep 0" in title
    assert "DBZH" in title
    assert "KTLX" in title
    assert "97.2778 deg W" in title
    assert ax.get_xlabel() == "X distance from radar (km)"
    matplotlib.pyplot.close(fig)


def test_plot_sweeps_draws_one_mesh_per_sweep() -> None:
    fig, ax = viz.plot_sweeps(_raystack(), [0, 1])

    assert len(ax.collections) == 2
    # Later sweeps are drawn semi-transparent over the first.
    assert ax.collections[0].get_alpha() == 1.0
    assert ax.collections[1].get_alpha() == pytest.approx(0.6)
    matplotlib.pyplot.close(fig)


def test_plot_sweeps_rejects_empty_and_unknown_selections() -> None:
    rs_dt = _raystack()

    with pytest.raises(ValueError):
        viz.plot_sweeps(rs_dt, [])
    with pytest.raises(KeyError):
        viz.plot_sweeps(rs_dt, [99])
    with pytest.raises(KeyError):
        viz.plot_sweep(rs_dt, 0, moment_name="NOT_A_MOMENT")
    matplotlib.pyplot.close("all")


def test_plot_waterfall_unfolds_to_the_full_range_axis() -> None:
    # Folds of 4 gates at 250 m, two folds deep, so range runs out to 2 km.
    fig, ax = viz.plot_waterfall(_raystack())

    x_min, x_max = ax.get_xlim()
    assert x_min == pytest.approx(0.0)
    assert x_max == pytest.approx(2.0)
    assert ax.get_xlabel() == "range from radar (km)"
    matplotlib.pyplot.close(fig)


def test_plot_waterfall_windows_rows() -> None:
    fig, ax = viz.plot_waterfall(_raystack(), row_offset=2, row_count=4)

    assert "rows 2-5 of 12" in fig.axes[0].get_ylabel()
    matplotlib.pyplot.close(fig)


def test_plot_waterfall_rejects_unknown_reducer() -> None:
    with pytest.raises(ValueError):
        viz.plot_waterfall(_raystack(), reduce="median")


def test_gate_positions_geolocates_around_the_site() -> None:
    points = viz.gate_positions(_raystack(), "DBZH")

    site = viz.vcp_infos(_raystack())[0]
    # Azimuth 0 is north, so the first radial's gates sit due north of the site.
    assert points["latitude"][0] > site.latitude
    assert points["longitude"][0] == pytest.approx(site.longitude, abs=1e-9)
    # Altitude is above sea level, so it starts at the tower and climbs.
    assert points["altitude"].min() >= site.altitude
    assert np.all(np.isfinite(points["value"]))


def test_gate_positions_filters_and_caps() -> None:
    rs_dt = _raystack()

    assert viz.gate_positions(rs_dt, "DBZH", min_value=20.0)["value"].min() >= 20.0
    assert viz.gate_positions(rs_dt, "DBZH", max_points=5)["value"].size <= 5


def test_plot_geo_builds_a_deck_with_a_point_cloud() -> None:
    pytest.importorskip("pydeck")

    deck = viz.plot_geo(_raystack(), max_points=100)

    layers = [layer.type for layer in deck.layers]
    assert layers == ["PointCloudLayer", "ScatterplotLayer"]
    # Compact serialization is what keeps the notebook output small.
    assert '": ' not in deck.to_json()


def test_plot_geo_rejects_a_fully_filtered_volume() -> None:
    pytest.importorskip("pydeck")

    with pytest.raises(ValueError):
        viz.plot_geo(_raystack(), min_value=1e9)
