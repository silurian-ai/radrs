"""Tests for radrs.raystack module."""

import pytest
import numpy as np


class TestParse:
    """Tests for raystack.parse function."""

    def test_parse_returns_dict(self, test_file_bytes):
        """Test that parse returns a dict with expected keys."""
        import radrs.raystack as rrs

        rs = rrs.parse(test_file_bytes)

        assert isinstance(rs, dict)
        assert "vcps" in rs
        assert "sweeps" in rs
        assert "returns" in rs

    def test_parse_with_fold_size(self, test_file_bytes):
        """Test parse with custom fold size."""
        import radrs.raystack as rrs

        for fold_size in [32, 64, 128, 256]:
            rs = rrs.parse(test_file_bytes, fold_size=fold_size)

            # Check that moment arrays have correct range dimension
            returns = rs["returns"]
            if "DBZH" in returns:
                assert returns["DBZH"].shape[1] == fold_size, \
                    f"Expected fold_size={fold_size}, got {returns['DBZH'].shape[1]}"

    def test_parse_has_coordinates(self, test_file_bytes):
        """Test that parsed data has coordinate arrays."""
        import radrs.raystack as rrs

        rs = rrs.parse(test_file_bytes)
        returns = rs["returns"]

        assert "azimuth" in returns
        assert "elevation" in returns
        assert "time" in returns
        assert "sweep_idx" in returns

        # All should be 1D arrays with same length
        n_returns = len(returns["azimuth"])
        assert len(returns["elevation"]) == n_returns
        assert len(returns["time"]) == n_returns
        assert len(returns["sweep_idx"]) == n_returns

    def test_parse_has_moments(self, test_file_bytes):
        """Test that parsed data has moment arrays."""
        import radrs.raystack as rrs

        rs = rrs.parse(test_file_bytes)
        returns = rs["returns"]

        # Should have at least reflectivity
        moment_names = ["DBZH", "VRADH", "WRADH", "ZDR", "PHIDP", "RHOHV", "KDP"]
        found = [m for m in moment_names if m in returns]
        assert len(found) > 0, "Should have at least one moment variable"

    def test_parse_sweep_idx_alignment(self, test_file_bytes):
        """Test that sweep_idx values align with sweeps metadata."""
        import radrs.raystack as rrs

        rs = rrs.parse(test_file_bytes)

        sweeps = rs["sweeps"]
        sweep_idx = rs["returns"]["sweep_idx"]

        # sweep_idx should have values from 0 to len(sweeps)-1
        unique_idx = np.unique(sweep_idx)
        assert unique_idx.min() == 0, "sweep_idx should start at 0"
        assert unique_idx.max() == len(sweeps) - 1, \
            f"sweep_idx max ({unique_idx.max()}) should equal len(sweeps)-1 ({len(sweeps)-1})"

        # Verify radial counts match
        for i, sweep in enumerate(sweeps):
            radials_in_sweep = np.sum(sweep_idx == i)
            assert radials_in_sweep == sweep["n_radials"], \
                f"Sweep {i}: expected {sweep['n_radials']} radials, got {radials_in_sweep}"

    def test_parse_sweeps_metadata(self, test_file_bytes):
        """Test that sweeps metadata is correctly extracted."""
        import radrs.raystack as rrs

        rs = rrs.parse(test_file_bytes)

        for i, sweep in enumerate(rs["sweeps"]):
            assert "elevation_number" in sweep
            assert "elevation_angle" in sweep
            assert "n_radials" in sweep
            assert "start_index" in sweep

            # Basic sanity checks
            assert sweep["n_radials"] > 0
            assert sweep["elevation_angle"] >= -1.0  # Some sweeps can be slightly negative
            assert sweep["elevation_angle"] <= 90.0

    def test_parse_azimuth_range(self, test_file_bytes):
        """Test that azimuth values are in valid range."""
        import radrs.raystack as rrs

        rs = rrs.parse(test_file_bytes)
        azimuth = rs["returns"]["azimuth"]

        # Azimuth should be 0-360 degrees
        assert np.nanmin(azimuth) >= 0.0
        assert np.nanmax(azimuth) <= 360.0

    def test_parse_time_monotonic_per_sweep(self, test_file_bytes):
        """Test that time is roughly monotonic within each sweep."""
        import radrs.raystack as rrs

        rs = rrs.parse(test_file_bytes)
        time = rs["returns"]["time"]
        sweep_idx = rs["returns"]["sweep_idx"]

        for i in range(len(rs["sweeps"])):
            sweep_time = time[sweep_idx == i]
            if len(sweep_time) > 1:
                # Allow some time jitter, but overall should be increasing
                # (not strictly monotonic due to radar timing)
                assert sweep_time[-1] >= sweep_time[0], \
                    f"Sweep {i}: end time should be >= start time"


class TestFromDatatree:
    """Tests for from_datatree function."""

    def test_from_datatree_returns_dict(self, test_file_path):
        """Test that from_datatree returns a dict."""
        import radrs.xradar as rxr
        import radrs.raystack as rrs

        dt = rxr.open_datatree(test_file_path)
        rs = rrs.from_datatree(dt)

        assert isinstance(rs, dict)
        assert "vcps" in rs
        assert "sweeps" in rs
        assert "returns" in rs

    def test_from_datatree_sweep_idx_alignment(self, test_file_path):
        """Test that sweep_idx aligns with sweeps metadata from DataTree."""
        import radrs.xradar as rxr
        import radrs.raystack as rrs

        dt = rxr.open_datatree(test_file_path)
        rs = rrs.from_datatree(dt)

        sweeps = rs["sweeps"]
        sweep_idx = rs["returns"]["sweep_idx"]

        # sweep_idx should have values from 0 to len(sweeps)-1
        unique_idx = np.unique(sweep_idx)
        assert unique_idx.max() == len(sweeps) - 1, \
            f"sweep_idx max ({unique_idx.max()}) should equal len(sweeps)-1 ({len(sweeps)-1})"

    def test_from_datatree_with_xradar_datatree(self, test_file_path):
        """Test from_datatree handles xradar DataTree with non-sweep nodes."""
        try:
            import xradar as xd
        except ImportError:
            pytest.skip("xradar not installed")

        import radrs.raystack as rrs

        # xradar DataTree has extra nodes like radar_parameters, georeferencing_correction
        xrad_dt = xd.io.open_nexradlevel2_datatree(test_file_path)

        # Check that non-sweep nodes exist
        non_sweep = [k for k in xrad_dt.children.keys() if not k.startswith("sweep_")]
        # xradar typically adds: radar_parameters, georeferencing_correction, radar_calibration
        assert len(non_sweep) > 0, "Expected xradar to have non-sweep nodes"

        # Convert to raystack
        rs = rrs.from_datatree(xrad_dt)

        # sweep_idx should still align correctly
        sweeps = rs["sweeps"]
        sweep_idx = rs["returns"]["sweep_idx"]

        # Should have correct number of unique sweep indices
        unique_idx = np.unique(sweep_idx)
        assert len(unique_idx) == len(sweeps), \
            f"Expected {len(sweeps)} unique sweep indices, got {len(unique_idx)}"
        assert unique_idx.max() == len(sweeps) - 1, \
            "sweep_idx max should match sweeps count (not include non-sweep nodes)"


class TestToDatatree:
    """Tests for to_datatree function."""

    def test_to_datatree_returns_datatree(self, test_file_bytes):
        """Test that to_datatree returns a DataTree."""
        import radrs.raystack as rrs

        rs = rrs.parse(test_file_bytes)
        dt = rrs.to_datatree(rs)

        assert hasattr(dt, "children")
        sweep_keys = [k for k in dt.children.keys() if k.startswith("sweep_")]
        assert len(sweep_keys) > 0

    def test_to_datatree_preserves_moments(self, test_file_bytes):
        """Test that to_datatree preserves moment data."""
        import radrs.raystack as rrs

        rs = rrs.parse(test_file_bytes)
        dt = rrs.to_datatree(rs)

        sweep_0 = dt["sweep_0"]
        ds = sweep_0.dataset

        # Should have moment data
        moment_vars = ["DBZH", "VRADH", "RHOHV", "ZDR"]
        found_vars = [v for v in moment_vars if v in ds]
        assert len(found_vars) > 0


class TestRoundtrip:
    """Tests for DataTree <-> raystack roundtrip with numeric validation."""

    def test_roundtrip_preserves_sweeps(self, test_file_path):
        """Test that roundtrip preserves sweep count."""
        import radrs.xradar as rxr
        import radrs.raystack as rrs

        dt1 = rxr.open_datatree(test_file_path)
        rs = rrs.from_datatree(dt1)
        dt2 = rrs.to_datatree(rs)

        sweeps1 = [k for k in dt1.children.keys() if k.startswith("sweep_")]
        sweeps2 = [k for k in dt2.children.keys() if k.startswith("sweep_")]

        assert len(sweeps1) == len(sweeps2)

    def test_roundtrip_preserves_radial_count(self, test_file_path):
        """Test that roundtrip preserves radial count per sweep."""
        import radrs.xradar as rxr
        import radrs.raystack as rrs

        dt1 = rxr.open_datatree(test_file_path)
        rs = rrs.from_datatree(dt1)
        dt2 = rrs.to_datatree(rs)

        for key in dt1.children:
            if not key.startswith("sweep_"):
                continue

            if key not in dt2.children:
                continue

            n1 = len(dt1[key]["azimuth"])
            n2 = len(dt2[key]["azimuth"])
            assert n1 == n2, f"{key}: radial count mismatch ({n1} vs {n2})"

    def test_roundtrip_preserves_coordinates(self, test_file_path):
        """Test that roundtrip preserves azimuth/elevation values exactly."""
        import radrs.xradar as rxr
        import radrs.raystack as rrs

        dt1 = rxr.open_datatree(test_file_path)
        rs = rrs.from_datatree(dt1)
        dt2 = rrs.to_datatree(rs)

        for key in dt1.children:
            if not key.startswith("sweep_"):
                continue
            if key not in dt2.children:
                continue

            # Azimuth should match exactly
            az1 = dt1[key]["azimuth"].values
            az2 = dt2[key]["azimuth"].values
            np.testing.assert_allclose(az1, az2, rtol=1e-5,
                err_msg=f"{key}: azimuth values differ")

            # Elevation should match exactly
            el1 = dt1[key]["elevation"].values
            el2 = dt2[key]["elevation"].values
            np.testing.assert_allclose(el1, el2, rtol=1e-5,
                err_msg=f"{key}: elevation values differ")

    def test_roundtrip_preserves_moment_values(self, test_file_path):
        """Test that roundtrip preserves moment data values (not just shapes)."""
        import radrs.xradar as rxr
        import radrs.raystack as rrs

        dt1 = rxr.open_datatree(test_file_path)
        # Use no folding to preserve exact values
        rs = rrs.from_datatree(dt1, fold_size=2048)
        dt2 = rrs.to_datatree(rs)

        moments_to_check = ["DBZH", "RHOHV", "ZDR"]

        for key in dt1.children:
            if not key.startswith("sweep_"):
                continue
            if key not in dt2.children:
                continue

            for moment in moments_to_check:
                if moment not in dt1[key].dataset:
                    continue
                if moment not in dt2[key].dataset:
                    continue

                v1 = dt1[key][moment].values
                v2 = dt2[key][moment].values

                # Shapes must match
                assert v1.shape == v2.shape, \
                    f"{key}/{moment}: shape mismatch {v1.shape} vs {v2.shape}"

                # Compare finite values
                mask = np.isfinite(v1) & np.isfinite(v2)
                if np.any(mask):
                    np.testing.assert_allclose(v1[mask], v2[mask], rtol=1e-5,
                        err_msg=f"{key}/{moment}: values differ")

                # NaN positions should match (same gates marked invalid)
                nan_match = np.isnan(v1) == np.isnan(v2)
                nan_mismatch_pct = 100 * (1 - nan_match.mean())
                assert nan_mismatch_pct < 1.0, \
                    f"{key}/{moment}: NaN positions differ by {nan_mismatch_pct:.1f}%"

    def test_parse_vs_from_datatree_consistency(self, test_file_path, test_file_bytes):
        """Test that parse and from_datatree produce consistent results."""
        import radrs.xradar as rxr
        import radrs.raystack as rrs

        # Direct parse
        rs1 = rrs.parse(test_file_bytes, fold_size=128)

        # Via DataTree
        dt = rxr.open_datatree(test_file_path)
        rs2 = rrs.from_datatree(dt, fold_size=128)

        # Should have same structure
        assert len(rs1["sweeps"]) == len(rs2["sweeps"])
        assert rs1["returns"]["DBZH"].shape == rs2["returns"]["DBZH"].shape

        # Coordinates should match
        np.testing.assert_allclose(
            rs1["returns"]["azimuth"],
            rs2["returns"]["azimuth"],
            rtol=1e-5,
            err_msg="azimuth differs between parse and from_datatree"
        )

        # Moment values should match where both are finite
        for moment in ["DBZH", "RHOHV"]:
            if moment not in rs1["returns"] or moment not in rs2["returns"]:
                continue
            v1 = rs1["returns"][moment]
            v2 = rs2["returns"][moment]
            mask = np.isfinite(v1) & np.isfinite(v2)
            if np.any(mask):
                max_diff = np.max(np.abs(v1[mask] - v2[mask]))
                assert max_diff < 0.01, \
                    f"{moment}: max diff = {max_diff}, expected < 0.01"


class TestPerformance:
    """Performance-related tests."""

    @pytest.mark.benchmark
    def test_parse_benchmark(self, benchmark, test_file_bytes):
        """Benchmark raystack.parse performance."""
        import radrs.raystack as rrs

        rs = benchmark(lambda: rrs.parse(test_file_bytes))
        assert isinstance(rs, dict)
