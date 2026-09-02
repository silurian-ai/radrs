import gzip
import os
from pathlib import Path

import numpy as np
import pytest

import radrs.raystack as rrs
import radrs.xradar as rxr


def _align_by_azimuth_time(
    rust_vals: np.ndarray,
    rust_az: np.ndarray,
    rust_time: np.ndarray | None,
    xrad_vals: np.ndarray,
    xrad_az: np.ndarray,
    xrad_time: np.ndarray | None,
    *,
    az_tol: float,
    time_tol_s: float = 2.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Align rows by azimuth, then disambiguate using closest time."""
    use_time = rust_time is not None and xrad_time is not None
    if use_time:
        rust_time = np.asarray(rust_time)
        xrad_time = np.asarray(xrad_time)

    matched_r = []
    matched_x = []

    for i, az in enumerate(rust_az):
        az_diff = np.abs(xrad_az - az)
        cand = np.where(az_diff <= az_tol)[0]
        if cand.size == 0:
            continue

        j = None
        if use_time and rust_time is not None and xrad_time is not None:
            t = rust_time[i]
            if not np.isnat(t):
                valid = ~np.isnat(xrad_time[cand])
                cand_t = cand[valid]
                if cand_t.size > 0:
                    dt = np.abs(xrad_time[cand_t] - t) / np.timedelta64(1, "s")
                    best = int(np.argmin(dt))
                    if dt[best] <= time_tol_s:
                        j = int(cand_t[best])

        if j is None:
            j = int(cand[np.argmin(az_diff[cand])])

        matched_r.append(rust_vals[i])
        matched_x.append(xrad_vals[j])

    if not matched_r:
        return None

    return np.asarray(matched_r), np.asarray(matched_x)


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

    def test_datatree_has_moments(self, compact_radrs_datatree):
        """Test that DataTree contains expected moment variables."""

        dt = compact_radrs_datatree

        # Check first sweep for moment data
        sweep_0 = dt["sweep_0"]
        ds = sweep_0.dataset

        # Should have at least reflectivity
        moment_vars = ["DBZH", "VRADH", "RHOHV", "ZDR"]
        found_vars = [v for v in moment_vars if v in ds]
        assert len(found_vars) > 0, f"Should have at least one moment variable, got: {list(ds.keys())}"

    def test_datatree_has_coordinates(self, compact_radrs_datatree):
        """Test that DataTree has expected coordinates."""

        dt = compact_radrs_datatree

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

    def test_datatree_sweep_structure(self, compact_radrs_datatree):
        """Test that each sweep has correct structure."""

        dt = compact_radrs_datatree

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

    def test_datatree_root_attributes(self, compact_radrs_datatree):
        """Test that root has expected attributes."""

        dt = compact_radrs_datatree

        # Root should have instrument_type or similar metadata
        root_attrs = dt.attrs
        assert isinstance(root_attrs, dict)

    def test_datatree_root_metadata(self, compact_radrs_datatree):
        """Root dataset should include key metadata variables."""

        dt = compact_radrs_datatree
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

    def test_datatree_sweep_metadata(self, compact_radrs_datatree):
        """Sweep datasets should include sweep/prt/follow mode metadata."""

        dt = compact_radrs_datatree
        sweep_0 = dt["sweep_0"]
        ds = sweep_0.dataset
        for var in ["sweep_mode", "prt_mode", "follow_mode"]:
            assert var in ds, f"Missing sweep metadata variable: {var}"

    def test_datatree_azimuth_values(self, compact_radrs_datatree):
        """Test that azimuth values are in valid range."""

        dt = compact_radrs_datatree

        for key in dt.children:
            if not key.startswith("sweep_"):
                continue

            sweep = dt[key]
            if "azimuth" in sweep.dataset:
                azimuth = sweep["azimuth"].values
                assert np.nanmin(azimuth) >= 0.0, f"{key}: azimuth min < 0"
                assert np.nanmax(azimuth) <= 360.0, f"{key}: azimuth max > 360"

    def test_datatree_elevation_values(self, compact_radrs_datatree):
        """Test that elevation values are in valid range."""

        dt = compact_radrs_datatree

        for key in dt.children:
            if not key.startswith("sweep_"):
                continue

            sweep = dt[key]
            if "elevation" in sweep.dataset:
                elevation = sweep["elevation"].values
                assert np.nanmin(elevation) >= -5.0, f"{key}: elevation min too low"
                assert np.nanmax(elevation) <= 90.0, f"{key}: elevation max too high"

    def test_mixed_moment_geometry_uses_physical_gate_centers(self):
        """Exercise the legacy volume with 1 km REF and 250 m Doppler gates.

        The fixture is kept outside the repository because it is a historical
        11 MB volume. Set RADRS_MIXED_GEOMETRY_TEST_FILE to run this regression.
        """
        path = os.environ.get("RADRS_MIXED_GEOMETRY_TEST_FILE")
        if not path or not Path(path).exists():
            pytest.skip(
                "set RADRS_MIXED_GEOMETRY_TEST_FILE to the historical mixed-grid volume"
            )

        with gzip.open(path, "rb") as source:
            dt = rxr.open_datatree(source.read())

        for sweep_number in (4, 5, 6):
            ds = dt[f"sweep_{sweep_number}"].dataset
            ranges = ds["range"].values
            assert ranges[0] == pytest.approx(2125.0)
            assert np.diff(ranges).min() == pytest.approx(125.0)
            assert ranges[-1] == pytest.approx(335500.0)

            dbzh_ranges = ranges[np.isfinite(ds["DBZH"].values).any(axis=0)]
            vradh_ranges = ranges[np.isfinite(ds["VRADH"].values).any(axis=0)]
            assert np.allclose((dbzh_ranges - 2500.0) % 1000.0, 0.0)
            assert np.allclose((vradh_ranges - 2125.0) % 250.0, 0.0)

        with gzip.open(path, "rb") as source:
            raystack = rrs.parse(source.read(), fold_size=128)
        sweeps = raystack["sweeps"]
        assert np.allclose(sweeps["range_start"][4:7], 2125.0)
        assert np.allclose(sweeps["range_step"][4:7], 125.0)
        assert np.all(sweeps["max_gates"][4:7] == 2668)


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

    @pytest.mark.slow
    def test_sweep_count_matches(self, full_radrs_datatree, full_xradar_datatree):
        """Test that sweep count matches xradar."""

        rust_dt = full_radrs_datatree

        rust_sweeps = [k for k in rust_dt.children.keys() if k.startswith("sweep_")]
        xrad_sweeps = [
            k for k in full_xradar_datatree.children.keys() if k.startswith("sweep_")
        ]

        assert len(rust_sweeps) == len(xrad_sweeps), \
            f"Sweep count mismatch: radrs={len(rust_sweeps)}, xradar={len(xrad_sweeps)}"

    @pytest.mark.slow
    def test_radial_count_close(self, full_radrs_datatree, full_xradar_datatree):
        """Test that radial count per sweep is close to xradar.

        Note: Small differences (1-2 radials) may occur due to different handling
        of duplicate or incomplete radials between implementations.
        """

        rust_dt = full_radrs_datatree

        for key in rust_dt.children:
            if not key.startswith("sweep_"):
                continue

            if key not in full_xradar_datatree.children:
                continue

            rust_n = len(rust_dt[key]["azimuth"])
            xrad_n = len(full_xradar_datatree[key]["azimuth"])

            # Allow small difference (up to 1% or 3 radials, whichever is larger)
            max_diff = max(3, int(rust_n * 0.01))
            assert abs(rust_n - xrad_n) <= max_diff, \
                f"{key}: radial count diff too large: radrs={rust_n}, xradar={xrad_n}"

    @pytest.mark.slow
    def test_azimuth_values_match(self, full_volume_path, full_xradar_datatree):
        """Test that azimuth values match xradar.

        Both sorted by azimuth, so we match by finding closest azimuths
        (radial counts may differ slightly between implementations).
        """
        rust_dt = rxr.open_datatree(full_volume_path, sort_by_azimuth=True)

        for key in rust_dt.children:
            if not key.startswith("sweep_"):
                continue

            rust_az = rust_dt[key]["azimuth"].values
            xrad_az = full_xradar_datatree[key]["azimuth"].values

            # Each azimuth should have a close match
            max_diff = max(np.min(np.abs(xrad_az - az)) for az in rust_az)
            tolerance = 360.0 / len(rust_az) * 1.1  # ~1 azimuth spacing
            assert max_diff < tolerance, f"{key}: worst azimuth match = {max_diff:.2f}°"

    @pytest.mark.slow
    def test_moment_values_match(self, full_volume_path, full_xradar_datatree):
        """Test that moment values match xradar when aligned by azimuth.

        Note: radrs marks below-threshold/range-folded as NaN, while xradar
        preserves raw values. We only compare where both have finite values.
        """
        rust_dt = rxr.open_datatree(full_volume_path, sort_by_azimuth=True)

        for key in rust_dt.children:
            if not key.startswith("sweep_"):
                continue
            if key not in full_xradar_datatree.children:
                continue

            rust_az = rust_dt[key]["azimuth"].values
            xrad_az = full_xradar_datatree[key]["azimuth"].values
            rust_time = rust_dt[key]["time"].values if "time" in rust_dt[key].dataset else None
            xrad_time = (
                full_xradar_datatree[key]["time"].values
                if "time" in full_xradar_datatree[key].dataset
                else None
            )
            az_spacing = 360.0 / len(xrad_az)
            az_tol = az_spacing * 0.55

            for moment in ["DBZH", "VRADH", "RHOHV"]:
                if (
                    moment not in rust_dt[key].dataset
                    or moment not in full_xradar_datatree[key].dataset
                ):
                    continue

                rust_vals = rust_dt[key][moment].values
                xrad_vals = full_xradar_datatree[key][moment].values

                # Treat xradar below-threshold sentinel as NaN
                xrad_vals = np.where(np.isclose(xrad_vals, -33.0, atol=0.01), np.nan, xrad_vals)

                aligned = _align_by_azimuth_time(
                    rust_vals,
                    rust_az,
                    rust_time,
                    xrad_vals,
                    xrad_az,
                    xrad_time,
                    az_tol=az_tol,
                )
                if aligned is None:
                    continue
                r_aligned, x_aligned = aligned
                mask = np.isfinite(r_aligned) & np.isfinite(x_aligned)
                if not np.any(mask):
                    continue
                matched_diffs = np.abs(r_aligned[mask] - x_aligned[mask])

                if matched_diffs.size > 0:
                    assert float(np.max(matched_diffs)) < 0.01, \
                        f"{key}/{moment}: max diff = {float(np.max(matched_diffs))}"

    @pytest.mark.slow
    def test_ccorh_values_match(self, full_volume_path, full_xradar_datatree):
        """Test that CCORH values match xradar when aligned by azimuth.

        CCORH can include CFP status codes which radrs treats as NaN. We compare
        only finite values and allow a very small number of outliers due to
        azimuth alignment edge cases.
        """
        rust_dt = rxr.open_datatree(full_volume_path, sort_by_azimuth=True)

        for key in rust_dt.children:
            if not key.startswith("sweep_"):
                continue
            if key not in full_xradar_datatree.children:
                continue
            if (
                "CCORH" not in rust_dt[key].dataset
                or "CCORH" not in full_xradar_datatree[key].dataset
            ):
                continue

            rust_az = rust_dt[key]["azimuth"].values
            xrad_az = full_xradar_datatree[key]["azimuth"].values
            rust_time = rust_dt[key]["time"].values if "time" in rust_dt[key].dataset else None
            xrad_time = (
                full_xradar_datatree[key]["time"].values
                if "time" in full_xradar_datatree[key].dataset
                else None
            )
            az_spacing = 360.0 / len(xrad_az)
            az_tol = az_spacing * 0.55

            rust_vals = rust_dt[key]["CCORH"].values
            xrad_vals = full_xradar_datatree[key]["CCORH"].values

            aligned = _align_by_azimuth_time(
                rust_vals,
                rust_az,
                rust_time,
                xrad_vals,
                xrad_az,
                xrad_time,
                az_tol=az_tol,
            )
            if aligned is None:
                continue
            r_aligned, x_aligned = aligned
            mask = np.isfinite(r_aligned) & np.isfinite(x_aligned)
            if not np.any(mask):
                continue
            diffs = np.abs(r_aligned[mask] - x_aligned[mask])
            mean = float(np.mean(diffs))
            p99 = float(np.quantile(diffs, 0.99))
            outliers = int(np.count_nonzero(diffs > 1.0))

            assert mean < 0.05 and p99 < 0.5, \
                f"{key}/CCORH: mean {mean:.3f}, p99 {p99:.3f} too large"
            assert outliers <= 1, \
                f"{key}/CCORH: too many outliers > 1.0 ({outliers})"


class TestPerformance:
    """Performance-related tests for xradar module."""

    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_open_datatree_benchmark(self, benchmark, full_volume_bytes):
        """Benchmark open_datatree performance."""

        result = benchmark(lambda: rxr.open_datatree(full_volume_bytes))
        assert hasattr(result, "children")


class TestMultiCloudRouting:
    """Verify open_datatree routes non-s3 URIs through fetch_bytes_from_url
    rather than the old s3-only / local-fs branch.
    """

    def test_local_path_missing_raises(self):
        """A bogus local path should surface as a fetch/IO error, not a
        scheme-not-supported error.
        """
        with pytest.raises(Exception) as excinfo:
            rxr.open_datatree("/nonexistent/definitely/not/a/file")
        msg = str(excinfo.value).lower()
        assert "unsupported uri scheme" not in msg

    @pytest.mark.network
    def test_gs_uri_routes_to_object_store(self):
        """A gs:// URI must reach the object_store fetch path. The public GCS
        NEXRAD mirror stores tar archives rather than single-volume files, so
        parse will fail with a "truncated record" error — that's expected and
        is exactly the proof that bytes were fetched from GCS rather than
        rejected at the URI-routing layer. Single-volume gs:// URIs (if you
        have them) work end-to-end; the public bucket simply isn't one.
        """
        url = (
            "gs://gcp-public-data-nexrad-l2/2025/01/01/KABR/"
            "NWS_NEXRAD_NXL2DPBL_KABR_20250101000000_20250101005959.tar"
        )
        with pytest.raises(Exception) as excinfo:
            rxr.open_datatree(url, storage_options={"skip_signature": "true"})
        msg = str(excinfo.value).lower()
        assert "unsupported uri scheme" not in msg
        assert "truncated record" in msg or "nexrad data error" in msg

    def test_storage_options_kwarg_accepted(self, test_file_path):
        """storage_options should be a no-op for local paths but accepted as a
        kwarg without error.
        """
        dt = rxr.open_datatree(test_file_path, storage_options=None)
        assert hasattr(dt, "children")

    def test_relative_local_path(self, test_file_path, tmp_path, monkeypatch):
        """Relative local paths must keep working — historical fs::read
        behavior. Regression test for the URI router rejecting paths without
        a leading '/' or 'file://'.
        """
        import os
        import shutil

        # Copy the volume into a tmp dir so we can pass it by relative name.
        copied = tmp_path / os.path.basename(test_file_path)
        shutil.copy(test_file_path, copied)
        monkeypatch.chdir(tmp_path)

        dt = rxr.open_datatree(copied.name)
        assert hasattr(dt, "children")

    def test_parent_relative_local_path(self, test_file_path, tmp_path, monkeypatch):
        """Parent-relative paths (../data/file) must resolve correctly.

        Regression test: object_store rejects URLs containing `..` segments,
        so the path must be normalized before URL construction.
        """
        import os
        import shutil

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        copied = data_dir / os.path.basename(test_file_path)
        shutil.copy(test_file_path, copied)
        monkeypatch.chdir(work_dir)

        dt = rxr.open_datatree(f"../data/{copied.name}")
        assert hasattr(dt, "children")

    def test_no_silent_substitution_via_parent_dir_collapse(
        self, test_file_path, tmp_path, monkeypatch
    ):
        """When `link/../target` doesn't resolve via OS semantics, we must
        NOT lexically collapse to a cwd-relative `target` that happens to
        exist (data-correctness regression flagged in PR #31 review).

        Layout:
            tmp/work/  (cwd)
            tmp/work/decoy_V06   ← exists in cwd, would be picked up by lexical collapse
            tmp/work/link → /nonexistent_dir   ← broken symlink
            (no tmp/target_V06; OS-resolved path doesn't exist)

        Calling open_datatree("link/../decoy_V06") under the buggy lexical
        fallback would resolve to "decoy_V06" (cwd) and silently read the
        decoy. Correct behavior: raise an error.
        """
        import os
        import shutil

        work_dir = tmp_path / "work"
        work_dir.mkdir()

        # The decoy file the bug would silently substitute to.
        decoy = work_dir / "decoy_V06"
        shutil.copy(test_file_path, decoy)

        # Broken symlink so canonicalize fails.
        link = work_dir / "link"
        os.symlink(tmp_path / "nonexistent_dir", link)
        monkeypatch.chdir(work_dir)

        # `link/../decoy_V06`: under OS semantics, resolves through link
        # (broken) → fails. Under lexical collapse, becomes `decoy_V06`
        # (exists, but wrong file). We require an error.
        with pytest.raises(Exception) as excinfo:
            rxr.open_datatree("link/../decoy_V06")
        # The error should NOT be a successful parse — i.e., we did not
        # silently substitute the decoy.
        msg = str(excinfo.value).lower()
        assert "unsupported uri scheme" not in msg

    def test_symlinked_parent_dir_resolves_with_os_semantics(
        self, test_file_path, tmp_path, monkeypatch
    ):
        """`link/../file` where `link` is a symlink must follow OS semantics.

        Regression test for the symlink-aware-resolution P2: if we collapse
        `..` syntactically, `link/../file` becomes just `file` (relative to
        cwd), which reads the wrong file. The OS resolves through the
        symlink first, then applies `..` from the link target.
        """
        import os
        import shutil

        # Layout:
        #   tmp/real_dir/<volume>          ← actual file
        #   tmp/sibling/decoy              ← unrelated file (would shadow on syntactic collapse)
        #   tmp/work/link → tmp/real_dir   ← symlink
        # Resolving `link/../<basename>` syntactically:  work/<basename>     (no such file)
        # Resolving with symlink awareness:              real_dir/../<basename> = tmp/<basename>
        # Use a unique basename that exists only via the symlink path.
        real_dir = tmp_path / "real_dir"
        real_dir.mkdir()
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        # Place the volume in tmp so OS resolution finds it via real_dir/..
        volume = tmp_path / "via_symlink_V06"
        shutil.copy(test_file_path, volume)

        link = work_dir / "link"
        os.symlink(real_dir, link)
        monkeypatch.chdir(work_dir)

        # `link/../via_symlink_V06`:
        #   - Symlink-aware (OS):  real_dir/../via_symlink_V06 → tmp/via_symlink_V06 (exists)
        #   - Syntactic collapse: work/via_symlink_V06 (does not exist)
        dt = rxr.open_datatree(f"link/../{volume.name}")
        assert hasattr(dt, "children")
