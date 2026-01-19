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

    def test_parse_sweep_start_index_consistency(self, test_file_bytes):
        """Test that sweep start_index values are contiguous and aligned."""
        import radrs.raystack as rrs

        rs = rrs.parse(test_file_bytes)
        sweeps = rs["sweeps"]

        if not sweeps:
            pytest.skip("No sweeps found in test file")

        # start_index should be cumulative sum of prior n_radials
        expected_start = 0
        for sweep in sweeps:
            assert sweep["start_index"] == expected_start, \
                f"Expected start_index {expected_start}, got {sweep['start_index']}"
            expected_start += sweep["n_radials"]

        # Final index should match total radial count
        n_radials = len(rs["returns"]["azimuth"])
        assert expected_start == n_radials, \
            f"Expected total radials {expected_start}, got {n_radials}"

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

    def test_parse_dualpol_even_odd_nan_balance(self, test_file_bytes):
        """Ensure dual-pol moments don't show alternating NaN patterns."""
        import radrs.raystack as rrs

        rs = rrs.parse(test_file_bytes, fold_size=2048)
        returns = rs["returns"]

        if not rs["sweeps"]:
            pytest.skip("No sweeps found in test file")

        sweep0 = rs["sweeps"][0]
        start = int(sweep0["start_index"])
        n = int(sweep0["n_radials"])

        for field in ("ZDR", "PHIDP"):
            if field not in returns:
                pytest.skip(f"{field} not present in test file")

            vals = np.asarray(returns[field][start : start + n])
            if vals.size == 0:
                pytest.skip(f"{field} has no data")

            finite = np.isfinite(vals)
            if finite.sum() < 50:
                pytest.skip(f"{field} has insufficient finite data for check")

            even = vals[:, 0::2]
            odd = vals[:, 1::2]
            even_nan = np.isnan(even).mean()
            odd_nan = np.isnan(odd).mean()
            diff = abs(even_nan - odd_nan)

            assert diff < 0.05, \
                f"{field}: even/odd NaN imbalance too large ({diff:.3f})"

    def test_parse_pattern_number_matches_datatree(self, test_file_path, test_file_bytes):
        """Test that pattern_number matches DataTree metadata."""
        import radrs.xradar as rxr
        import radrs.raystack as rrs

        dt = rxr.open_datatree(test_file_path)
        rs = rrs.parse(test_file_bytes)

        pattern = dt.attrs.get("volume_coverage_pattern", 0)
        assert rs["vcps"]["pattern_number"] == pattern

    def test_parse_time_matches_datatree(self, test_file_path, test_file_bytes):
        """Test that parse time array matches DataTree time ordering."""
        import radrs.xradar as rxr
        import radrs.raystack as rrs

        dt = rxr.open_datatree(test_file_path)
        rs = rrs.parse(test_file_bytes)

        sweep_keys = [k for k in dt.children.keys() if k.startswith("sweep_")]
        sweep_keys.sort(key=lambda k: int(k.split("_", 1)[1]))

        times = []
        for key in sweep_keys:
            if "time" not in dt[key].dataset:
                continue
            sweep_times = dt[key]["time"].values.astype("datetime64[ms]").astype("int64")
            if sweep_times.size:
                times.append(sweep_times)

        if not times:
            pytest.skip("No sweep times found in DataTree")

        dt_time = np.concatenate(times)
        rs_time = rs["returns"]["time"]
        assert len(dt_time) == len(rs_time)
        np.testing.assert_array_equal(rs_time, dt_time)


def test_parse_accepts_gzip_bytes(test_file_bytes):
    """Test that parse handles gzipped NEXRAD input bytes."""
    import gzip
    import radrs.raystack as rrs

    raw_rs = rrs.parse(test_file_bytes)
    gz_bytes = gzip.compress(test_file_bytes)
    gz_rs = rrs.parse(gz_bytes)

    assert raw_rs["vcps"]["pattern_number"] == gz_rs["vcps"]["pattern_number"]
    assert len(raw_rs["sweeps"]) == len(gz_rs["sweeps"])

    # Spot-check coordinate and moment data for equality
    np.testing.assert_allclose(
        raw_rs["returns"]["azimuth"][:50],
        gz_rs["returns"]["azimuth"][:50],
        rtol=1e-5,
    )
    if "DBZH" in raw_rs["returns"] and "DBZH" in gz_rs["returns"]:
        np.testing.assert_allclose(
            raw_rs["returns"]["DBZH"][:10],
            gz_rs["returns"]["DBZH"][:10],
            rtol=1e-5,
            equal_nan=True,
        )


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


class TestRaystackDatatree:
    """Tests for raystack-style DataTree conversion."""

    def test_to_raystack_datatree_has_nodes(self, test_file_bytes):
        """Test that to_raystack_datatree yields vcps/sweeps/returns nodes."""
        import radrs.raystack as rrs

        rs = rrs.parse(test_file_bytes)
        dt = rrs.to_raystack_datatree(rs)

        assert hasattr(dt, "children")
        assert "vcps" in dt.children
        assert "sweeps" in dt.children
        assert "returns" in dt.children

    def test_open_datatree_returns_raystack(self, test_file_path):
        """Test that open_datatree returns raystack DataTree."""
        import radrs.raystack as rrs

        dt = rrs.open_datatree(test_file_path)
        assert "returns" in dt.children
        assert "vcps" in dt.children
        assert "sweeps" in dt.children

    @pytest.mark.asyncio
    async def test_open_datatree_async_returns_raystack(self, test_file_path):
        """Test that open_datatree_async returns raystack DataTree."""
        import radrs.raystack as rrs

        dt = await rrs.open_datatree_async(test_file_path)
        assert "returns" in dt.children
        assert "vcps" in dt.children
        assert "sweeps" in dt.children


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
        """Test that roundtrip preserves moment data values.

        Note: raystack output always has shape (n_radials, fold_size), so we compare
        values at the overlapping range indices, not expect exact shapes.
        """
        import radrs.xradar as rxr
        import radrs.raystack as rrs

        dt1 = rxr.open_datatree(test_file_path)
        # Use large fold_size to minimize folding (preserve more exact values)
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

                # Compare values at overlapping range indices
                # v1 shape: (n_radials, original_n_gates)
                # v2 shape: (n_radials, fold_size)
                n_compare = min(v1.shape[1], v2.shape[1])
                v1_slice = v1[:, :n_compare]
                v2_slice = v2[:, :n_compare]

                # Compare finite values
                mask = np.isfinite(v1_slice) & np.isfinite(v2_slice)
                if np.any(mask):
                    np.testing.assert_allclose(v1_slice[mask], v2_slice[mask], rtol=1e-5,
                        err_msg=f"{key}/{moment}: values differ")

                # NaN positions should match at overlapping range
                nan_match = np.isnan(v1_slice) == np.isnan(v2_slice)
                nan_mismatch_pct = 100 * (1 - nan_match.mean())
                assert nan_mismatch_pct < 1.0, \
                    f"{key}/{moment}: NaN positions differ by {nan_mismatch_pct:.1f}%"

    def test_parse_vs_from_datatree_consistency(self, test_file_path, test_file_bytes):
        """Test that parse and from_datatree produce consistent structure/coords."""
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


class TestPerformance:
    """Performance-related tests."""

    @pytest.mark.benchmark
    def test_parse_benchmark(self, benchmark, test_file_bytes):
        """Benchmark raystack.parse performance."""
        import radrs.raystack as rrs

        rs = benchmark(lambda: rrs.parse(test_file_bytes))
        assert isinstance(rs, dict)
