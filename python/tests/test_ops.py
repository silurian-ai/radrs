import numpy as np
import pytest
import xarray as xr

import radrs.ops as ops

try:
    import pyart
except ImportError:  # pragma: no cover - dependency failure is reported by integration tests
    pyart = None


def test_align_azimuth_plan_apply_2d():
    src_az = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    dst_az = np.array([3.0, 1.0, 4.0], dtype=np.float32)

    plan = ops.align_azimuth(src_az, dst_az, tolerance_deg=0.4)
    indices = plan.indices
    assert indices.tolist() == [3, 1, -1]

    data = np.arange(8, dtype=np.float32).reshape(4, 2)
    aligned = plan.apply_2d(data)
    assert aligned.shape == (3, 2)
    assert np.allclose(aligned[0], data[3])
    assert np.allclose(aligned[1], data[1])
    assert np.isnan(aligned[2]).all()


def test_align_azimuth_wrap():
    src_az = np.array([359.0, 0.0, 1.0], dtype=np.float32)
    dst_az = np.array([359.8], dtype=np.float32)
    plan = ops.align_azimuth(src_az, dst_az, tolerance_deg=0.5, wrap=True)
    assert plan.indices.tolist() == [1]


def test_align_azimuth_nan_and_no_wrap():
    src_az = np.array([0.0, 10.0, np.nan], dtype=np.float32)
    dst_az = np.array([10.0, np.nan, 350.0], dtype=np.float32)
    plan = ops.align_azimuth(src_az, dst_az, tolerance_deg=0.5, wrap=False)
    assert plan.indices.tolist() == [1, -1, -1]


def test_align_azimuth_tolerance_boundary():
    src_az = np.array([0.0, 1.0], dtype=np.float32)
    dst_az = np.array([0.4], dtype=np.float32)
    plan = ops.align_azimuth(src_az, dst_az, tolerance_deg=0.4, wrap=True)
    assert plan.indices.tolist() == [0]


def test_align_azimuth_duplicates():
    src_az = np.array([1.0, 1.0, 2.0], dtype=np.float32)
    dst_az = np.array([1.0], dtype=np.float32)
    plan = ops.align_azimuth(src_az, dst_az, tolerance_deg=0.1)
    # Stable tie-break: first matching index.
    assert plan.indices.tolist() == [0]


def test_align_range_plan_apply_2d():
    plan = ops.align_range(
        src_start_m=0.0,
        src_step_m=1000.0,
        src_len=3,
        dst_start_m=1000.0,
        dst_step_m=1000.0,
        dst_len=3,
        tolerance_m=1e-3,
    )
    assert plan.indices.tolist() == [1, 2, -1]

    data = np.arange(12, dtype=np.float32).reshape(4, 3)
    aligned = plan.apply_2d(data)
    assert aligned.shape == (4, 3)
    assert np.allclose(aligned[:, 0], data[:, 1])
    assert np.allclose(aligned[:, 1], data[:, 2])
    assert np.isnan(aligned[:, 2]).all()


def test_align_range_negative_step():
    plan = ops.align_range(
        src_start_m=0.0,
        src_step_m=-1000.0,
        src_len=3,
        dst_start_m=0.0,
        dst_step_m=-1000.0,
        dst_len=3,
        tolerance_m=1e-3,
    )
    assert plan.indices.tolist() == [0, 1, 2]


def test_align_range_rounding_boundary():
    plan = ops.align_range(
        src_start_m=0.0,
        src_step_m=1000.0,
        src_len=3,
        dst_start_m=500.0,
        dst_step_m=1000.0,
        dst_len=1,
        tolerance_m=0.1,
    )
    assert plan.indices.tolist() == [-1]


def test_apply_1d_fill_value():
    src_az = np.array([0.0, 1.0], dtype=np.float32)
    dst_az = np.array([1.0, 2.0], dtype=np.float32)
    plan = ops.align_azimuth(src_az, dst_az, tolerance_deg=0.1)
    data = np.array([5.0, 6.0], dtype=np.float32)
    out = plan.apply_1d(data, fill=-999.0)
    assert np.allclose(out[0], 6.0)
    assert np.isclose(out[1], -999.0)


def test_apply_2d_non_contiguous_errors():
    src_az = np.array([0.0, 1.0], dtype=np.float32)
    dst_az = np.array([1.0], dtype=np.float32)
    plan = ops.align_azimuth(src_az, dst_az, tolerance_deg=0.1)
    data = np.arange(12, dtype=np.float32).reshape(2, 6)[:, ::2]
    with pytest.raises(Exception):
        plan.apply_2d(data)


def test_velocity_texture_constant():
    vel = np.zeros((5, 5), dtype=np.float32)
    tex = ops.velocity_texture(vel, nyquist=10.0, wind_size=3)
    assert tex.shape == vel.shape
    assert np.allclose(tex, 0.0)


def _build_single_sweep_pyart_radar(dt: xr.DataTree, sweep_ds: xr.Dataset):
    if pyart is None:
        pytest.fail("The declared arm-pyart dev dependency is not installed")

    sweep_ds = sweep_ds.copy()
    sweep_ds["sweep_number"] = 0
    root = dt.dataset.copy()
    for coord in ("latitude", "longitude", "altitude"):
        if coord not in root:
            root = root.assign({coord: 0.0})
    single_sweep_dt = xr.DataTree.from_dict({".": root, "sweep_0": sweep_ds})
    return single_sweep_dt.pyart.to_radar()


def _clean_vradh(ds: xr.Dataset) -> xr.DataArray:
    return ds["VRADH"].where(~np.isclose(ds["VRADH"], -64.5, atol=1.0), np.nan)


def _select_sweep_with_vradh(dt: xr.DataTree) -> xr.Dataset:
    for name in dt.children:
        if not name.startswith("sweep_"):
            continue
        sweep_ds = dt[name].dataset
        if "VRADH" in sweep_ds and "DBZH" in sweep_ds:
            return sweep_ds
    pytest.fail("Full velocity fixture has no sweep with VRADH/DBZH")


def _symmetric_index(idx: int, length: int) -> int:
    if length == 0:
        return 0
    while idx < 0 or idx >= length:
        if idx < 0:
            idx = -idx - 1
        else:
            idx = 2 * length - idx - 1
    return idx


def _reflect_index(idx: int, length: int) -> int:
    if length == 0:
        return 0
    while idx < 0 or idx >= length:
        if idx < 0:
            idx = -idx
        else:
            idx = 2 * length - idx - 2
    return idx


def _velocity_texture_reference(vel: np.ndarray, nyq: float, wind_size: int) -> np.ndarray:
    n_rows, n_cols = vel.shape
    inv = np.pi / nyq
    x = np.full_like(vel, np.nan, dtype=np.float64)
    y = np.full_like(vel, np.nan, dtype=np.float64)

    mask = np.isfinite(vel)
    x[mask] = np.cos(vel[mask] * inv)
    y[mask] = np.sin(vel[mask] * inv)

    def convolve(data):
        out = np.empty_like(data, dtype=np.float64)
        half = wind_size // 2
        for r in range(n_rows):
            for c in range(n_cols):
                acc = 0.0
                for dr in range(wind_size):
                    rr = _symmetric_index(r + dr - half, n_rows)
                    for dc in range(wind_size):
                        cc = _symmetric_index(c + dc - half, n_cols)
                        acc += data[rr, cc]
                out[r, c] = acc
        return out

    xs = convolve(x)
    ys = convolve(y)

    ns = float(wind_size * wind_size)
    nyq_pi = nyq / np.pi
    std_dev = np.full_like(xs, np.nan, dtype=np.float64)
    for r in range(n_rows):
        for c in range(n_cols):
            xmean = xs[r, c] / ns
            ymean = ys[r, c] / ns
            norm = np.sqrt(xmean * xmean + ymean * ymean)
            if np.isnan(norm) or norm <= 0.0:
                std_dev[r, c] = np.nan
            else:
                std_dev[r, c] = np.sqrt(-2.0 * np.log(norm)) * nyq_pi

    half = wind_size // 2
    out = np.empty_like(std_dev, dtype=np.float64)
    for r in range(n_rows):
        for c in range(n_cols):
            buf = []
            for dr in range(wind_size):
                rr = _reflect_index(r + dr - half, n_rows)
                for dc in range(wind_size):
                    cc = _reflect_index(c + dc - half, n_cols)
                    buf.append(std_dev[rr, cc])
            buf.sort(key=lambda v: (np.isnan(v), v))
            out[r, c] = buf[len(buf) // 2]

    return out.astype(np.float32)


def test_velocity_texture_reference_small():
    vel = np.array(
        [
            [0.0, 1.0, np.nan, 2.0],
            [1.0, 0.5, 0.2, 1.5],
            [0.0, -0.5, -1.0, 0.0],
            [np.nan, 0.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    nyq = 10.0
    ref = _velocity_texture_reference(vel, nyq, wind_size=3)
    actual = ops.velocity_texture(vel, nyquist=nyq, wind_size=3)
    assert np.allclose(actual, ref, atol=1e-5, equal_nan=True)


@pytest.mark.slow
def test_velocity_texture_matches_pyart(full_radrs_datatree):
    if pyart is None:
        pytest.fail("The declared arm-pyart dev dependency is not installed")

    sweep_ds = _select_sweep_with_vradh(full_radrs_datatree)
    sweep_ds = sweep_ds.assign(VRADH=_clean_vradh(sweep_ds))

    radar = _build_single_sweep_pyart_radar(full_radrs_datatree, sweep_ds)
    nyq = np.nanmax(np.abs(sweep_ds["VRADH"].values))
    assert np.isfinite(nyq) and nyq > 0, "Full velocity fixture has invalid Nyquist velocity"

    expected_field = pyart.retrieve.calculate_velocity_texture(
        radar, vel_field="VRADH", nyq=nyq
    )
    expected = np.ma.filled(expected_field["data"], np.nan)
    actual = ops.velocity_texture(sweep_ds["VRADH"].values, nyquist=nyq, wind_size=3)

    mask = np.isfinite(expected) & np.isfinite(actual)
    assert np.any(mask), "Full velocity fixture produced no finite values to compare"

    diff = np.abs(expected[mask] - actual[mask])
    mean_abs = float(np.mean(diff))
    p99 = float(np.quantile(diff, 0.99))
    # Py-ART uses slightly different edge/median handling; allow modest tail differences.
    assert mean_abs < 0.3
    assert p99 < 4.0
