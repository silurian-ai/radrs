"""Tests for QC integration in raystack parsing."""

import numpy as np
import pytest
import radrs.qc as qc
import radrs.raystack as rrs


def _rhohv_threshold_mask(rhohv: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(
        np.isnan(rhohv),
        -1,
        np.where(rhohv >= threshold, 1, 0),
    ).astype(np.int8)


def _vectorized_correlation(masked_array_2d: np.ma.MaskedArray, reference_1d: np.ndarray) -> np.ndarray:
    reference_2d = np.broadcast_to(reference_1d, masked_array_2d.shape)
    reference_masked = np.ma.array(reference_2d, mask=masked_array_2d.mask)

    mean_data = np.ma.mean(masked_array_2d, axis=1, keepdims=True)
    mean_ref = np.ma.mean(reference_masked, axis=1, keepdims=True)

    dev_data = masked_array_2d - mean_data
    dev_ref = reference_masked - mean_ref

    numerator = np.ma.sum(dev_data * dev_ref, axis=1)
    denominator = np.ma.sqrt(np.ma.sum(dev_data**2, axis=1) * np.ma.sum(dev_ref**2, axis=1))

    correlations = np.ma.where(denominator != 0, numerator / denominator, np.nan)
    valid_counts = np.ma.count(masked_array_2d, axis=1)
    correlations = np.ma.where(valid_counts >= 3, correlations, np.nan)

    return correlations.filled(np.nan)


def _sun_spike_mask(
    dbzh: np.ndarray,
    dbzh_threshold: float = 0.0,
    fill_threshold: float = 0.9,
    correlation_threshold: float = 0.8,
) -> np.ndarray:
    dbzh_masked = np.ma.masked_invalid(dbzh)
    mask_data = np.where(np.isnan(dbzh), -1, 1).astype(np.int8)

    filled_gates = dbzh_masked >= dbzh_threshold
    valid_count = np.ma.count(dbzh_masked, axis=1)
    filled_count = np.ma.sum(filled_gates, axis=1)

    fill_fraction = np.zeros(len(valid_count), dtype=np.float32)
    valid_mask = valid_count > 0
    fill_fraction[valid_mask] = filled_count[valid_mask] / valid_count[valid_mask]

    candidate_mask = (fill_fraction >= fill_threshold) & (valid_count >= 3)
    candidate_rays = np.where(candidate_mask)[0]
    if len(candidate_rays) == 0:
        return mask_data

    range_indices = np.arange(dbzh.shape[1], dtype=np.float32)
    correlations = _vectorized_correlation(dbzh_masked[candidate_rays], range_indices)

    sun_spike_mask = correlations >= correlation_threshold
    sun_spike_rays = candidate_rays[sun_spike_mask]

    for ray_idx in sun_spike_rays:
        nan_locations = np.isnan(dbzh[ray_idx, :])
        mask_data[ray_idx, :] = np.where(nan_locations, -1, 0)

    return mask_data


@pytest.mark.slow
def test_parse_with_rhohv_qc_mask(full_volume_bytes):

    rs = rrs.parse(full_volume_bytes, qc=[qc.RhohvThreshold()])
    returns = rs["returns"]

    assert "RHOHV" in returns, "Full dual-pol fixture must contain RHOHV"
    assert "qc.rhohv_threshold_mask" in returns
    mask = np.asarray(returns["qc.rhohv_threshold_mask"])
    assert mask.dtype == np.int8

    rhohv = np.asarray(returns["RHOHV"])
    assert mask.shape == rhohv.shape
    assert set(np.unique(mask)).issubset({-1, 0, 1})

    assert np.all(mask[np.isnan(rhohv)] == -1)


def test_open_datatree_with_sun_spike_mask(test_file_path):

    dt = rrs.open_datatree(test_file_path, qc=[qc.SunSpike()])
    returns_ds = dt["returns"].dataset

    assert "qc.sun_spike_mask" in returns_ds
    mask = returns_ds["qc.sun_spike_mask"].values.astype(np.int8)

    dbzh = returns_ds["DBZH"].values
    assert mask.shape == dbzh.shape
    assert set(np.unique(mask)).issubset({-1, 0, 1})
    assert np.all(mask[np.isnan(dbzh)] == -1)


def test_rhohv_mask_matches_reference():

    rhohv = np.array(
        [
            [0.6, 0.9, 0.8, np.nan],
            [0.95, 0.2, 0.85, 0.1],
        ],
        dtype=np.float32,
    )

    expected = _rhohv_threshold_mask(rhohv, threshold=0.8)
    actual = qc.rhohv_threshold(rhohv, threshold=0.8)
    assert np.array_equal(actual, expected)


def test_sun_spike_mask_matches_reference():

    dbzh = np.array(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],  # linear spike
            [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],  # constant
            [np.nan, np.nan, 0.0, 0.0, 0.0, 0.0],  # NaNs preserved
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],  # below threshold
        ],
        dtype=np.float32,
    )

    expected = _sun_spike_mask(
        dbzh, dbzh_threshold=0.0, fill_threshold=0.9, correlation_threshold=0.8
    )
    actual = qc.sun_spike(
        dbzh, dbzh_threshold=0.0, fill_threshold=0.9, corr_threshold=0.8
    )
    assert np.array_equal(actual, expected)
