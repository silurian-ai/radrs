"""Tests for radrs.xradar module."""

import pytest
import numpy as np
import radrs.xradar as rxr


class TestOpenDatatree:
    """Tests for open_datatree function."""

    def test_open_datatree_from_path(self, test_file_path):
        """Test opening a NEXRAD file from path."""

        dt = rxr.open_datatree(test_file_path)

        # Check that we got a DataTree
        assert hasattr(dt, "children")

        # Check that we have sweeps
        sweep_keys = [k for k in dt.children.keys() if k.startswith("sweep_")]
        assert len(sweep_keys) > 0, "Should have at least one sweep"

    def test_open_datatree_from_bytes(self, test_file_bytes):
        """Test opening a NEXRAD file from bytes."""

        dt = rxr.open_datatree(test_file_bytes)

        # Check structure
        assert hasattr(dt, "children")
        sweep_keys = [k for k in dt.children.keys() if k.startswith("sweep_")]
        assert len(sweep_keys) > 0

    def test_datatree_has_moments(self, test_file_path):
        """Test that DataTree contains expected moment variables."""

        dt = rxr.open_datatree(test_file_path)

        # Check first sweep for moment data
        sweep_0 = dt["sweep_0"]
        ds = sweep_0.dataset

        # Should have at least reflectivity
        moment_vars = ["DBZH", "VRADH", "RHOHV", "ZDR"]
        found_vars = [v for v in moment_vars if v in ds]
        assert len(found_vars) > 0, f"Should have at least one moment variable, got: {list(ds.keys())}"

    def test_datatree_has_coordinates(self, test_file_path):
        """Test that DataTree has expected coordinates."""

        dt = rxr.open_datatree(test_file_path)

        sweep_0 = dt["sweep_0"]
        ds = sweep_0.dataset

        # Should have azimuth, elevation, time, and range
        expected_coords = ["azimuth", "elevation", "time", "range"]
        for coord in expected_coords:
            assert coord in ds.coords or coord in ds, f"Missing coordinate: {coord}"

        # Azimuth should be the primary dimension
        if "azimuth" in ds.coords:
            assert ds["azimuth"].dims == ("azimuth",)
        if "time" in ds.coords:
            assert ds["time"].dims == ("azimuth",)

    def test_datatree_sweep_structure(self, test_file_path):
        """Test that each sweep has correct structure."""

        dt = rxr.open_datatree(test_file_path)

        for key in dt.children:
            if not key.startswith("sweep_"):
                continue

            sweep = dt[key]
            ds = sweep.dataset

            # Should have azimuth and elevation
            assert "azimuth" in ds or "azimuth" in ds.coords
            assert "elevation" in ds or "elevation" in ds.coords

            # Azimuth should be 1D with reasonable count
            if "azimuth" in ds:
                assert len(ds["azimuth"].dims) == 1
                assert len(ds["azimuth"]) > 0

    def test_datatree_root_attributes(self, test_file_path):
        """Test that root has expected attributes."""

        dt = rxr.open_datatree(test_file_path)

        # Root should have instrument_type or similar metadata
        root_attrs = dt.attrs
        assert isinstance(root_attrs, dict)

    def test_datatree_root_metadata(self, test_file_path):
        """Root dataset should include key metadata variables."""

        dt = rxr.open_datatree(test_file_path)
        root_ds = dt.dataset
        for var in [
            "volume_number",
            "platform_type",
            "instrument_type",
            "latitude",
            "longitude",
            "altitude",
            "time_coverage_start",
            "time_coverage_end",
        ]:
            assert var in root_ds, f"Missing root metadata variable: {var}"

    def test_datatree_sweep_metadata(self, test_file_path):
        """Sweep datasets should include sweep/prt/follow mode metadata."""

        dt = rxr.open_datatree(test_file_path)
        sweep_0 = dt["sweep_0"]
        ds = sweep_0.dataset
        for var in ["sweep_mode", "prt_mode", "follow_mode"]:
            assert var in ds, f"Missing sweep metadata variable: {var}"

    def test_datatree_azimuth_values(self, test_file_path):
        """Test that azimuth values are in valid range."""

        dt = rxr.open_datatree(test_file_path)

        for key in dt.children:
            if not key.startswith("sweep_"):
                continue

            sweep = dt[key]
            if "azimuth" in sweep.dataset:
                azimuth = sweep["azimuth"].values
                assert np.nanmin(azimuth) >= 0.0, f"{key}: azimuth min < 0"
                assert np.nanmax(azimuth) <= 360.0, f"{key}: azimuth max > 360"

    def test_datatree_elevation_values(self, test_file_path):
        """Test that elevation values are in valid range."""

        dt = rxr.open_datatree(test_file_path)

        for key in dt.children:
            if not key.startswith("sweep_"):
                continue

            sweep = dt[key]
            if "elevation" in sweep.dataset:
                elevation = sweep["elevation"].values
                assert np.nanmin(elevation) >= -5.0, f"{key}: elevation min too low"
                assert np.nanmax(elevation) <= 90.0, f"{key}: elevation max too high"


class TestXradarCompatibility:
    """Tests comparing radrs.xradar output with xradar.

    Key differences between radrs and xradar:

    1. **Radial ordering**: xradar sorts radials by azimuth (ascending 0→360),
       while radrs preserves the original file order. After sorting by azimuth,
       the values should match.

    2. **Below-threshold handling**: radrs marks below-threshold and range-folded
       values as NaN (semantically: no valid measurement), while xradar preserves
       the raw encoded values (e.g., -33 dBZ for reflectivity threshold). Both
       interpretations are valid - radrs is cleaner for analysis, xradar preserves
       more raw information.

    3. **Radial count**: Small differences (1-2 radials) may occur due to different
       handling of duplicate or incomplete radials between implementations.
    """

    @pytest.fixture
    def xradar_datatree(self, test_file_path):
        """Get xradar DataTree for comparison."""
        try:
            import xradar as xd
            return xd.io.open_nexradlevel2_datatree(test_file_path)
        except ImportError:
            pytest.skip("xradar not installed")

    def test_sweep_count_matches(self, test_file_path, xradar_datatree):
        """Test that sweep count matches xradar."""

        rust_dt = rxr.open_datatree(test_file_path)

        rust_sweeps = [k for k in rust_dt.children.keys() if k.startswith("sweep_")]
        xrad_sweeps = [k for k in xradar_datatree.children.keys() if k.startswith("sweep_")]

        assert len(rust_sweeps) == len(xrad_sweeps), \
            f"Sweep count mismatch: radrs={len(rust_sweeps)}, xradar={len(xrad_sweeps)}"

    def test_radial_count_close(self, test_file_path, xradar_datatree):
        """Test that radial count per sweep is close to xradar.

        Note: Small differences (1-2 radials) may occur due to different handling
        of duplicate or incomplete radials between implementations.
        """

        rust_dt = rxr.open_datatree(test_file_path)

        for key in rust_dt.children:
            if not key.startswith("sweep_"):
                continue

            if key not in xradar_datatree.children:
                continue

            rust_n = len(rust_dt[key]["azimuth"])
            xrad_n = len(xradar_datatree[key]["azimuth"])

            # Allow small difference (up to 1% or 3 radials, whichever is larger)
            max_diff = max(3, int(rust_n * 0.01))
            assert abs(rust_n - xrad_n) <= max_diff, \
                f"{key}: radial count diff too large: radrs={rust_n}, xradar={xrad_n}"

    def test_azimuth_values_match(self, test_file_path, xradar_datatree):
        """Test that azimuth values match xradar.

        Both sorted by azimuth, so we match by finding closest azimuths
        (radial counts may differ slightly between implementations).
        """
        rust_dt = rxr.open_datatree(test_file_path, sort_by_azimuth=True)

        for key in rust_dt.children:
            if not key.startswith("sweep_"):
                continue

            rust_az = rust_dt[key]["azimuth"].values
            xrad_az = xradar_datatree[key]["azimuth"].values

            # Each azimuth should have a close match
            max_diff = max(np.min(np.abs(xrad_az - az)) for az in rust_az)
            tolerance = 360.0 / len(rust_az) * 1.1  # ~1 azimuth spacing
            assert max_diff < tolerance, f"{key}: worst azimuth match = {max_diff:.2f}°"

    def test_moment_values_match(self, test_file_path, xradar_datatree):
        """Test that moment values match xradar when aligned by azimuth.

        Note: radrs marks below-threshold/range-folded as NaN, while xradar
        preserves raw values. We only compare where both have finite values.
        """
        rust_dt = rxr.open_datatree(test_file_path, sort_by_azimuth=True)

        for key in rust_dt.children:
            if not key.startswith("sweep_"):
                continue
            if key not in xradar_datatree.children:
                continue

            rust_az = rust_dt[key]["azimuth"].values
            xrad_az = xradar_datatree[key]["azimuth"].values
            az_spacing = 360.0 / len(xrad_az)

            for moment in ["DBZH", "VRADH", "RHOHV"]:
                if moment not in rust_dt[key].dataset or moment not in xradar_datatree[key].dataset:
                    continue

                rust_vals = rust_dt[key][moment].values
                xrad_vals = xradar_datatree[key][moment].values

                # Match rows by closest azimuth
                matched_diffs = []
                for i, az in enumerate(rust_az):
                    j = np.argmin(np.abs(xrad_az - az))
                    if np.abs(xrad_az[j] - az) > az_spacing * 0.55:
                        continue

                    rust_row = rust_vals[i]
                    xrad_row = xrad_vals[j]
                    xrad_row = np.where(np.isclose(xrad_row, -33.0, atol=0.01), np.nan, xrad_row)

                    mask = np.isfinite(rust_row) & np.isfinite(xrad_row)
                    if np.any(mask):
                        matched_diffs.extend(np.abs(rust_row[mask] - xrad_row[mask]).tolist())

                if matched_diffs:
                    assert max(matched_diffs) < 0.01, f"{key}/{moment}: max diff = {max(matched_diffs)}"


class TestPerformance:
    """Performance-related tests for xradar module."""

    @pytest.mark.benchmark
    def test_open_datatree_benchmark(self, benchmark, test_file_bytes):
        """Benchmark open_datatree performance."""

        result = benchmark(lambda: rxr.open_datatree(test_file_bytes))
        assert hasattr(result, "children")
