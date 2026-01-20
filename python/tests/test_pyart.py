"""Py-ART compatibility tests for radrs raystack parsing."""

import numpy as np
import radrs.raystack as rrs
import pytest


def _load_pyart_radar(test_file_path):
    pyart = pytest.importorskip("pyart")
    read_fn = getattr(pyart.io, "read_nexrad_archive", None)
    if read_fn is None:
        pytest.skip("pyart.io.read_nexrad_archive not available")
    return pyart, read_fn(test_file_path)


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


def _match_sweeps_by_elevation(radrs_sweeps, pyart_fixed_angles, tol=0.2):
    pairs = []
    unused = set(range(len(pyart_fixed_angles)))
    for rs_idx, sweep in enumerate(radrs_sweeps):
        target = float(sweep["elevation_angle"])
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


def _align_by_azimuth(
    rs_vals: np.ndarray,
    rs_az: np.ndarray,
    pa_vals: np.ndarray,
    pa_az: np.ndarray,
    *,
    decimals: int = 2,
) -> tuple[np.ndarray, np.ndarray] | None:
    rs_az_r = np.round(rs_az, decimals)
    pa_az_r = np.round(pa_az, decimals)

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
def test_parse_vs_pyart_sweep_and_radial_counts(test_file_path, test_file_bytes):
    """Compare sweep/radial counts against Py-ART."""
    pyart, radar = _load_pyart_radar(test_file_path)

    fold_size = _pyart_ngates(radar)
    rs = rrs.parse(test_file_bytes, fold_size=fold_size)

    radrs_sweeps = rs["sweeps"]
    pyart_fixed = np.asarray(radar.fixed_angle["data"])
    pairs = _match_sweeps_by_elevation(radrs_sweeps, pyart_fixed)

    # If elevation matching fails, fall back to index-order comparison
    if not pairs:
        pairs = list(
            zip(range(len(radrs_sweeps)), range(int(radar.nsweeps)), strict=False)
        )

    assert len(pairs) > 0, "No sweep pairs matched between radrs and Py-ART"

    for rs_idx, pa_idx in pairs:
        rs_sweep = radrs_sweeps[rs_idx]
        rs_n = int(rs_sweep["n_radials"])

        start = int(radar.sweep_start_ray_index["data"][pa_idx])
        end = int(radar.sweep_end_ray_index["data"][pa_idx])
        pa_n = end - start + 1

        assert rs_n == pa_n, \
            f"sweep {rs_idx}: radrs={rs_n}, pyart={pa_n}"


@pytest.mark.slow
def test_parse_vs_pyart_azimuth_alignment(test_file_path, test_file_bytes):
    """Ensure radrs azimuths align with Py-ART within tolerance."""
    pyart, radar = _load_pyart_radar(test_file_path)

    fold_size = _pyart_ngates(radar)
    rs = rrs.parse(test_file_bytes, fold_size=fold_size)

    returns = rs["returns"]
    radrs_sweeps = rs["sweeps"]
    pyart_fixed = np.asarray(radar.fixed_angle["data"])
    pairs = _match_sweeps_by_elevation(radrs_sweeps, pyart_fixed)
    if not pairs:
        pairs = list(
            zip(range(len(radrs_sweeps)), range(int(radar.nsweeps)), strict=False)
        )

    az_all = np.asarray(radar.azimuth["data"])

    for rs_idx, pa_idx in pairs:
        rs_sweep = radrs_sweeps[rs_idx]
        start = int(rs_sweep["start_index"])
        n_radials = int(rs_sweep["n_radials"])

        rs_az = np.asarray(returns["azimuth"][start : start + n_radials])

        pa_start = int(radar.sweep_start_ray_index["data"][pa_idx])
        pa_end = int(radar.sweep_end_ray_index["data"][pa_idx])
        pa_az = np.asarray(az_all[pa_start : pa_end + 1])

        max_min_diff = 0.0
        for az in rs_az:
            min_diff = float(np.min(np.abs(pa_az - az)))
            max_min_diff = max(max_min_diff, min_diff)

        expected_spacing = 360.0 / len(rs_az)
        tolerance = expected_spacing * 1.1
        assert max_min_diff < tolerance, \
            f"sweep {rs_idx}: worst azimuth match {max_min_diff:.2f}° >= {tolerance:.2f}°"


@pytest.mark.slow
def test_parse_vs_pyart_moment_values(test_file_path, test_file_bytes):
    """Compare radrs moment values to Py-ART for matched azimuths."""
    pyart, radar = _load_pyart_radar(test_file_path)

    fold_size = _pyart_ngates(radar)
    rs = rrs.parse(test_file_bytes, fold_size=fold_size)

    returns = rs["returns"]
    radrs_sweeps = rs["sweeps"]
    pyart_fixed = np.asarray(radar.fixed_angle["data"])
    pairs = _match_sweeps_by_elevation(radrs_sweeps, pyart_fixed)
    if not pairs:
        pairs = list(
            zip(range(len(radrs_sweeps)), range(int(radar.nsweeps)), strict=False)
        )

    # Map radrs -> pyart field names (exclude known dual-pol decoding issues)
    field_map = {
        "DBZH": "reflectivity",
        "VRADH": "velocity",
        "WRADH": "spectrum_width",
        "RHOHV": "cross_correlation_ratio",
    }

    az_all = np.asarray(radar.azimuth["data"])

    for rs_idx, pa_idx in pairs:
        rs_sweep = radrs_sweeps[rs_idx]
        start = int(rs_sweep["start_index"])
        n_radials = int(rs_sweep["n_radials"])

        rs_az = np.asarray(returns["azimuth"][start : start + n_radials])

        pa_start = int(radar.sweep_start_ray_index["data"][pa_idx])
        pa_end = int(radar.sweep_end_ray_index["data"][pa_idx])
        pa_az = np.asarray(az_all[pa_start : pa_end + 1])

        for rs_field, pa_field in field_map.items():
            if rs_field not in returns:
                continue
            pa_vals = _pyart_field_values(radar, pa_field)
            if pa_vals is None:
                continue

            rs_vals = np.asarray(returns[rs_field][start : start + n_radials])
            pa_vals = np.asarray(pa_vals[pa_start : pa_end + 1])

            aligned = _align_by_azimuth(rs_vals, rs_az, pa_vals, pa_az)
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
def test_parse_vs_pyart_dualpol_moment_values(test_file_path, test_file_bytes):
    """Compare radrs dual-pol moments to Py-ART."""
    pyart, radar = _load_pyart_radar(test_file_path)

    fold_size = _pyart_ngates(radar)
    rs = rrs.parse(test_file_bytes, fold_size=fold_size)

    returns = rs["returns"]
    radrs_sweeps = rs["sweeps"]
    pyart_fixed = np.asarray(radar.fixed_angle["data"])
    pairs = _match_sweeps_by_elevation(radrs_sweeps, pyart_fixed)
    if not pairs:
        pairs = list(
            zip(range(len(radrs_sweeps)), range(int(radar.nsweeps)), strict=False)
        )

    field_map = {
        "ZDR": "differential_reflectivity",
        "PHIDP": "differential_phase",
        "KDP": "specific_differential_phase",
    }

    az_all = np.asarray(radar.azimuth["data"])

    for rs_idx, pa_idx in pairs:
        rs_sweep = radrs_sweeps[rs_idx]
        start = int(rs_sweep["start_index"])
        n_radials = int(rs_sweep["n_radials"])

        rs_az = np.asarray(returns["azimuth"][start : start + n_radials])

        pa_start = int(radar.sweep_start_ray_index["data"][pa_idx])
        pa_end = int(radar.sweep_end_ray_index["data"][pa_idx])
        pa_az = np.asarray(az_all[pa_start : pa_end + 1])

        for rs_field, pa_field in field_map.items():
            if rs_field not in returns:
                continue
            pa_vals = _pyart_field_values(radar, pa_field)
            if pa_vals is None:
                continue

            rs_vals = np.asarray(returns[rs_field][start : start + n_radials])
            pa_vals = np.asarray(pa_vals[pa_start : pa_end + 1])

            aligned = _align_by_azimuth(rs_vals, rs_az, pa_vals, pa_az)
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
