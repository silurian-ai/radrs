"""Py-ART compatibility tests for radrs raystack parsing."""

import numpy as np
import radrs.raystack as rrs
import pytest


def _pyart_ngates(radar):
    ngates = getattr(radar, "ngates", None)
    if ngates is not None:
        return int(ngates)
    return int(np.asarray(radar.range["data"]).shape[0])


def _pyart_field_values(radar, field_name: str) -> np.ndarray | None:
    if field_name not in radar.fields:
        return None
    data = radar.fields[field_name]["data"]
    if np.ma.isMaskedArray(data):
        return data.filled(np.nan).astype(np.float32)
    return np.asarray(data, dtype=np.float32)


def _sweep_offsets(sweeps):
    """Compute (start_index, num_returns) per sweep from sweeps dict-of-arrays."""
    num_returns = np.asarray(sweeps["num_returns"])
    starts = np.concatenate([[0], np.cumsum(num_returns[:-1])]) if len(num_returns) > 0 else np.array([], dtype=np.int64)
    return starts, num_returns


def _n_sweeps(sweeps):
    return len(sweeps["elevation_angle"])


def _match_sweeps_by_elevation(sweeps_dict, pyart_fixed_angles, tol=0.2):
    pairs = []
    unused = set(range(len(pyart_fixed_angles)))
    elev = np.asarray(sweeps_dict["elevation_angle"])
    for rs_idx in range(len(elev)):
        target = float(elev[rs_idx])
        best_idx = None
        best_delta = None
        for pa_idx in unused:
            delta = abs(float(pyart_fixed_angles[pa_idx]) - target)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_idx = pa_idx
        if best_idx is not None and best_delta is not None and best_delta <= tol:
            pairs.append((rs_idx, best_idx))
            unused.remove(best_idx)
    return pairs


def _pair_sweeps(sweeps_dict, radar):
    """Pair sweeps by index when counts match; otherwise use elevation matching."""
    n = _n_sweeps(sweeps_dict)
    if n == int(radar.nsweeps):
        return list(zip(range(n), range(int(radar.nsweeps)), strict=False))

    pyart_fixed = np.asarray(radar.fixed_angle["data"])
    pairs = _match_sweeps_by_elevation(sweeps_dict, pyart_fixed)
    if not pairs:
        pairs = list(zip(range(n), range(int(radar.nsweeps)), strict=False))
    return pairs


def _align_by_azimuth(
    rs_vals: np.ndarray,
    rs_az: np.ndarray,
    rs_time: np.ndarray | None,
    pa_vals: np.ndarray,
    pa_az: np.ndarray,
    pa_time: np.ndarray | None,
    *,
    decimals: int = 2,
    time_tol_s: float = 2.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    # Prefer azimuth + time alignment when time is available.
    if rs_time is not None and pa_time is not None:
        rs_time = np.asarray(rs_time)
        pa_time = np.asarray(pa_time)
        # Convert radrs epoch ms to seconds relative to sweep
        rs_t = (rs_time - rs_time[0]) / 1000.0
        # Py-ART time is already seconds since volume start; use relative per sweep
        pa_t = pa_time - pa_time[0]

        az_tol = 10 ** (-decimals)
        matched_rs = []
        matched_pa = []
        for i, az in enumerate(rs_az):
            az_diff = np.abs(pa_az - az)
            cand = np.where(az_diff <= az_tol)[0]
            if cand.size == 0:
                continue
            # choose closest time among candidates
            dt = np.abs(pa_t[cand] - rs_t[i])
            best = int(np.argmin(dt))
            if dt[best] > time_tol_s:
                continue
            j = int(cand[best])
            matched_rs.append(rs_vals[i])
            matched_pa.append(pa_vals[j])

        if matched_rs:
            return np.asarray(matched_rs), np.asarray(matched_pa)

    # Fallback: azimuth-only alignment
    scale = 10**decimals
    rs_az_r = np.round(rs_az * scale).astype(np.int32)
    pa_az_r = np.round(pa_az * scale).astype(np.int32)

    common = np.intersect1d(rs_az_r, pa_az_r)
    if common.size == 0:
        return None

    rs_mask = np.isin(rs_az_r, common)
    pa_mask = np.isin(pa_az_r, common)

    rs_vals = rs_vals[rs_mask]
    pa_vals = pa_vals[pa_mask]
    rs_az_r = rs_az_r[rs_mask]
    pa_az_r = pa_az_r[pa_mask]

    rs_order = np.argsort(rs_az_r)
    pa_order = np.argsort(pa_az_r)

    rs_vals = rs_vals[rs_order]
    pa_vals = pa_vals[pa_order]

    n = min(rs_vals.shape[0], pa_vals.shape[0])
    if n == 0:
        return None
    return rs_vals[:n], pa_vals[:n]


@pytest.mark.slow
def test_parse_vs_pyart_sweep_and_radial_counts(full_volume_bytes, full_pyart_radar):
    """Compare sweep/radial counts against Py-ART."""
    radar = full_pyart_radar

    fold_size = _pyart_ngates(radar)
    rs = rrs.parse(full_volume_bytes, fold_size=fold_size)

    sweeps = rs["sweeps"]
    pairs = _pair_sweeps(sweeps, radar)
    _, num_returns = _sweep_offsets(sweeps)

    assert len(pairs) > 0, "No sweep pairs matched between radrs and Py-ART"

    for rs_idx, pa_idx in pairs:
        rs_n = int(num_returns[rs_idx])

        start = int(radar.sweep_start_ray_index["data"][pa_idx])
        end = int(radar.sweep_end_ray_index["data"][pa_idx])
        pa_n = end - start + 1

        assert rs_n == pa_n, \
            f"sweep {rs_idx}: radrs={rs_n}, pyart={pa_n}"


@pytest.mark.slow
def test_parse_vs_pyart_azimuth_alignment(full_volume_bytes, full_pyart_radar):
    """Ensure radrs azimuths align with Py-ART within tolerance."""
    radar = full_pyart_radar

    fold_size = _pyart_ngates(radar)
    rs = rrs.parse(full_volume_bytes, fold_size=fold_size)

    returns = rs["returns"]
    sweeps = rs["sweeps"]
    pairs = _pair_sweeps(sweeps, radar)
    starts, num_ret = _sweep_offsets(sweeps)

    az_all = np.asarray(radar.azimuth["data"])

    for rs_idx, pa_idx in pairs:
        start = int(starts[rs_idx])
        n = int(num_ret[rs_idx])

        rs_az = np.asarray(returns["azimuth"][start : start + n])
        rs_time = np.asarray(returns["return_time"][start : start + n])

        pa_start = int(radar.sweep_start_ray_index["data"][pa_idx])
        pa_end = int(radar.sweep_end_ray_index["data"][pa_idx])
        pa_az = np.asarray(az_all[pa_start : pa_end + 1])
        pa_time = np.asarray(radar.time["data"][pa_start : pa_end + 1])

        max_min_diff = 0.0
        for az in rs_az:
            min_diff = float(np.min(np.abs(pa_az - az)))
            max_min_diff = max(max_min_diff, min_diff)

        expected_spacing = 360.0 / len(rs_az)
        tolerance = expected_spacing * 1.1
        assert max_min_diff < tolerance, \
            f"sweep {rs_idx}: worst azimuth match {max_min_diff:.2f}° >= {tolerance:.2f}°"


@pytest.mark.slow
def test_parse_vs_pyart_moment_values(full_volume_bytes, full_pyart_radar):
    """Compare radrs moment values to Py-ART for matched azimuths."""
    radar = full_pyart_radar

    fold_size = _pyart_ngates(radar)
    rs = rrs.parse(full_volume_bytes, fold_size=fold_size)

    returns = rs["returns"]
    sweeps = rs["sweeps"]
    pairs = _pair_sweeps(sweeps, radar)
    starts, num_ret = _sweep_offsets(sweeps)

    fold_size = len(returns["range"])

    # Map radrs -> pyart field names (exclude known dual-pol decoding issues)
    field_map = {
        "DBZH": "reflectivity",
        "VRADH": "velocity",
        "WRADH": "spectrum_width",
        "RHOHV": "cross_correlation_ratio",
    }

    az_all = np.asarray(radar.azimuth["data"])

    for rs_idx, pa_idx in pairs:
        start = int(starts[rs_idx])
        n = int(num_ret[rs_idx])

        rs_az = np.asarray(returns["azimuth"][start : start + n])
        rs_time = np.asarray(returns["return_time"][start : start + n])

        pa_start = int(radar.sweep_start_ray_index["data"][pa_idx])
        pa_end = int(radar.sweep_end_ray_index["data"][pa_idx])
        pa_az = np.asarray(az_all[pa_start : pa_end + 1])
        pa_time = np.asarray(radar.time["data"][pa_start : pa_end + 1])

        for rs_field, pa_field in field_map.items():
            if rs_field not in returns:
                continue
            pa_vals = _pyart_field_values(radar, pa_field)
            if pa_vals is None:
                continue

            # Reshape flat moment array to (n_returns, fold_size) and slice this sweep
            rs_moment_flat = np.asarray(returns[rs_field])
            rs_moment_2d = rs_moment_flat.reshape(-1, fold_size)
            rs_vals = rs_moment_2d[start : start + n]
            pa_vals = np.asarray(pa_vals[pa_start : pa_end + 1])

            aligned = _align_by_azimuth(rs_vals, rs_az, rs_time, pa_vals, pa_az, pa_time)
            if aligned is None:
                continue
            r_aligned, p_aligned = aligned
            mask = np.isfinite(r_aligned) & np.isfinite(p_aligned)
            if not np.any(mask):
                continue
            diff = np.abs(r_aligned[mask] - p_aligned[mask])

            mean_abs = float(np.mean(diff))
            p99 = float(np.quantile(diff, 0.99))
            assert mean_abs < 0.6 and p99 < 5.0, \
                f"sweep {rs_idx}/{rs_field}: mean {mean_abs:.2f}, p99 {p99:.2f} exceeds tolerance"


@pytest.mark.slow
def test_parse_vs_pyart_dualpol_moment_values(full_volume_bytes, full_pyart_radar):
    """Compare radrs dual-pol moments to Py-ART."""
    radar = full_pyart_radar

    fold_size = _pyart_ngates(radar)
    rs = rrs.parse(full_volume_bytes, fold_size=fold_size)

    returns = rs["returns"]
    sweeps = rs["sweeps"]
    pairs = _pair_sweeps(sweeps, radar)
    starts, num_ret = _sweep_offsets(sweeps)

    fold_size = len(returns["range"])

    field_map = {
        "ZDR": "differential_reflectivity",
        "PHIDP": "differential_phase",
    }

    az_all = np.asarray(radar.azimuth["data"])

    for rs_idx, pa_idx in pairs:
        start = int(starts[rs_idx])
        n = int(num_ret[rs_idx])

        rs_az = np.asarray(returns["azimuth"][start : start + n])
        rs_time = np.asarray(returns["return_time"][start : start + n])

        pa_start = int(radar.sweep_start_ray_index["data"][pa_idx])
        pa_end = int(radar.sweep_end_ray_index["data"][pa_idx])
        pa_az = np.asarray(az_all[pa_start : pa_end + 1])
        pa_time = np.asarray(radar.time["data"][pa_start : pa_end + 1])

        for rs_field, pa_field in field_map.items():
            if rs_field not in returns:
                continue
            pa_vals = _pyart_field_values(radar, pa_field)
            if pa_vals is None:
                continue

            rs_moment_flat = np.asarray(returns[rs_field])
            rs_moment_2d = rs_moment_flat.reshape(-1, fold_size)
            rs_vals = rs_moment_2d[start : start + n]
            pa_vals = np.asarray(pa_vals[pa_start : pa_end + 1])

            aligned = _align_by_azimuth(rs_vals, rs_az, rs_time, pa_vals, pa_az, pa_time)
            if aligned is None:
                continue
            r_aligned, p_aligned = aligned
            mask = np.isfinite(r_aligned) & np.isfinite(p_aligned)
            if not np.any(mask):
                continue
            diff = np.abs(r_aligned[mask] - p_aligned[mask])

            mean_abs = float(np.mean(diff))
            p99 = float(np.quantile(diff, 0.99))
            assert mean_abs < 0.6 and p99 < 5.0, \
                f"sweep {rs_idx}/{rs_field}: mean {mean_abs:.2f}, p99 {p99:.2f} exceeds tolerance"
