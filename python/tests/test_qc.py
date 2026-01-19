"""Tests for radrs.qc module."""

import pytest
import numpy as np


class TestRhohvThreshold:
    """Tests for rhohv_threshold function."""

    def test_rhohv_threshold_basic(self):
        """Test basic RHOHV threshold functionality."""
        import radrs.qc as qc

        rhohv = np.array([
            [0.5, 0.8, 0.9, 0.7],
            [0.95, 0.6, 0.85, 0.3],
        ], dtype=np.float32)

        mask = qc.rhohv_threshold(rhohv, threshold=0.8)

        assert mask.shape == rhohv.shape
        assert mask.dtype == np.int8

        # Check expected values
        expected = np.array([
            [0, 1, 1, 0],  # 0.5<0.8, 0.8>=0.8, 0.9>=0.8, 0.7<0.8
            [1, 0, 1, 0],  # 0.95>=0.8, 0.6<0.8, 0.85>=0.8, 0.3<0.8
        ], dtype=np.int8)

        np.testing.assert_array_equal(mask, expected)

    def test_rhohv_threshold_with_nan(self):
        """Test RHOHV threshold with NaN values."""
        import radrs.qc as qc

        rhohv = np.array([
            [0.9, np.nan, 0.7],
            [np.nan, 0.85, np.nan],
        ], dtype=np.float32)

        mask = qc.rhohv_threshold(rhohv, threshold=0.8)

        # NaN should become -1
        assert mask[0, 1] == -1
        assert mask[1, 0] == -1
        assert mask[1, 2] == -1

        # Valid values should be properly classified
        assert mask[0, 0] == 1  # 0.9 >= 0.8
        assert mask[0, 2] == 0  # 0.7 < 0.8
        assert mask[1, 1] == 1  # 0.85 >= 0.8

    def test_rhohv_threshold_default(self):
        """Test RHOHV threshold with default value."""
        import radrs.qc as qc

        rhohv = np.array([[0.79, 0.80, 0.81]], dtype=np.float32)

        mask = qc.rhohv_threshold(rhohv)  # Default threshold 0.8

        assert mask[0, 0] == 0  # 0.79 < 0.8
        assert mask[0, 1] == 1  # 0.80 >= 0.8
        assert mask[0, 2] == 1  # 0.81 >= 0.8


class TestSunSpike:
    """Tests for sun_spike function."""

    def test_sun_spike_constant_values(self):
        """Test sun spike detection with constant values (sun spike pattern)."""
        import radrs.qc as qc

        # Create a sun spike pattern: constant high values across range
        dbzh = np.full((1, 100), 30.0, dtype=np.float32)

        mask = qc.sun_spike(dbzh, fill_threshold=0.9, corr_threshold=0.8)

        assert mask.shape == dbzh.shape
        # Constant signal should have high autocorrelation -> detected as sun spike
        assert np.all(mask == 0)  # All marked as sun spike

    def test_sun_spike_with_nan(self):
        """Test sun spike detection with NaN values."""
        import radrs.qc as qc

        # Sparse data with many NaN values - low fill rate
        dbzh = np.full((1, 100), np.nan, dtype=np.float32)
        dbzh[0, :10] = 30.0  # Only 10% filled

        mask = qc.sun_spike(dbzh, fill_threshold=0.9, corr_threshold=0.8)

        # Low fill rate -> NaN values should be -1, valid values should be 1
        assert mask[0, 0] == 1  # Valid data
        assert mask[0, 50] == -1  # NaN -> missing

    def test_sun_spike_random_data(self):
        """Test sun spike detection with random data (not a sun spike)."""
        import radrs.qc as qc

        np.random.seed(42)
        dbzh = np.random.randn(1, 100).astype(np.float32) * 10 + 20

        mask = qc.sun_spike(dbzh, fill_threshold=0.9, corr_threshold=0.8)

        # Random data should have low autocorrelation -> not detected as sun spike
        # Most values should be valid (1)
        assert np.sum(mask == 1) > 50  # Most should be valid
