"""Tests for radrs.raystack module."""

import pytest
import numpy as np
import radrs.raystack as rrs
import radrs.xradar as rxr

S3_TEST_FILE = "s3://unidata-nexrad-level2/2024/07/02/KABR/KABR20240702_000016_V06"
MOMENT_NAMES = ["DBZH", "VRADH", "WRADH", "ZDR", "PHIDP", "RHOHV", "CCORH"]


def _moment_matrix(returns, moment):
    n_returns = len(returns["return_time"])
    fold_size = len(returns["range"])
    data = np.asarray(returns[moment])
    assert data.ndim == 1
    assert data.size == n_returns * fold_size
    return data.reshape(n_returns, fold_size)


def _compute_activity_python(rs, moments=None):
    returns = rs["returns"]
    sweeps = rs["sweeps"]
    vcps = rs["vcps"]

    if moments is None:
        moments = [m for m in MOMENT_NAMES if m in returns]

    n_returns = len(returns["return_time"])
    fold_size = len(returns["range"])
    n_sweeps = len(sweeps["sweep_time"])
    n_vcps = len(vcps["vcp_time"])

    ray_valid_count = np.zeros((len(moments), n_returns), dtype=np.uint32)
    ray_valid_fraction = np.full((len(moments), n_returns), np.nan, dtype=np.float32)

    if n_returns > 0 and fold_size > 0:
        for m_idx, moment in enumerate(moments):
            vals = _moment_matrix(returns, moment)
            counts = np.isfinite(vals).sum(axis=1).astype(np.uint32)
            ray_valid_count[m_idx] = counts
            ray_valid_fraction[m_idx] = counts.astype(np.float32) / float(fold_size)

    sweep_valid_count = np.zeros((len(moments), n_sweeps), dtype=np.uint32)
    sweep_valid_fraction = np.full((len(moments), n_sweeps), np.nan, dtype=np.float32)

    sweep_num_returns = np.asarray(sweeps["num_returns"], dtype=np.int64)
    sweep_offsets = np.zeros(n_sweeps + 1, dtype=np.int64)
    sweep_offsets[1:] = np.cumsum(sweep_num_returns)

    for s_idx in range(n_sweeps):
        start = int(min(sweep_offsets[s_idx], n_returns))
        end = int(min(sweep_offsets[s_idx + 1], n_returns))
        denom = max(0, end - start) * fold_size
        counts = ray_valid_count[:, start:end].sum(axis=1).astype(np.uint32)
        sweep_valid_count[:, s_idx] = counts
        if denom > 0:
            sweep_valid_fraction[:, s_idx] = counts.astype(np.float32) / float(denom)

    volume_valid_count = np.zeros((len(moments), n_vcps), dtype=np.uint32)
    volume_valid_fraction = np.full((len(moments), n_vcps), np.nan, dtype=np.float32)

    vcp_num_sweeps = np.asarray(vcps["num_sweeps"], dtype=np.int64)
    sweep_start = 0
    for v_idx, n_sweeps_in_vcp in enumerate(vcp_num_sweeps):
        sweep_end = min(sweep_start + int(n_sweeps_in_vcp), n_sweeps)
        start = int(min(sweep_offsets[sweep_start], n_returns))
        end = int(min(sweep_offsets[sweep_end], n_returns))
        denom = max(0, end - start) * fold_size

        counts = ray_valid_count[:, start:end].sum(axis=1).astype(np.uint32)
        volume_valid_count[:, v_idx] = counts
        if denom > 0:
            volume_valid_fraction[:, v_idx] = counts.astype(np.float32) / float(denom)

        sweep_start = sweep_end

    return {
        "moment": moments,
        "ray_valid_count": ray_valid_count,
        "ray_valid_fraction": ray_valid_fraction,
        "sweep_valid_count": sweep_valid_count,
        "sweep_valid_fraction": sweep_valid_fraction,
        "volume_valid_count": volume_valid_count,
        "volume_valid_fraction": volume_valid_fraction,
    }


class TestParse:
    """Tests for raystack.parse function."""

    def test_parse_returns_dict(self, test_file_bytes):
        rs = rrs.parse(test_file_bytes)
        assert isinstance(rs, dict)
        assert "vcps" in rs
        assert "sweeps" in rs
        assert "returns" in rs

    def test_parse_with_fold_size(self, test_file_bytes):
        for fold_size in [32, 64, 128, 256]:
            rs = rrs.parse(test_file_bytes, fold_size=fold_size)
            returns = rs["returns"]
            assert len(returns["range"]) == fold_size

            if "DBZH" in returns:
                dbzh = np.asarray(returns["DBZH"])
                assert dbzh.ndim == 1
                assert dbzh.size % fold_size == 0

    def test_parse_has_coordinates(self, test_file_bytes):
        rs = rrs.parse(test_file_bytes)
        returns = rs["returns"]

        assert "azimuth" in returns
        assert "elevation" in returns
        assert "return_time" in returns
        assert "sweep_number" in returns
        assert "sweep_time" in returns

        n_returns = len(returns["return_time"])
        assert len(returns["azimuth"]) == n_returns
        assert len(returns["elevation"]) == n_returns
        assert len(returns["sweep_number"]) == n_returns
        assert len(returns["sweep_time"]) == n_returns

    def test_parse_has_moments(self, test_file_bytes):
        rs = rrs.parse(test_file_bytes)
        returns = rs["returns"]
        found = [m for m in MOMENT_NAMES if m in returns]
        assert len(found) > 0

    def test_parse_has_activity(self, test_file_bytes):
        rs = rrs.parse(test_file_bytes)
        assert "activity" in rs

        activity = rs["activity"]
        moments = list(activity["moment"])
        n_moments = len(moments)
        n_returns = len(rs["returns"]["return_time"])
        n_sweeps = len(rs["sweeps"]["sweep_time"])
        n_vcps = len(rs["vcps"]["vcp_time"])

        assert activity["ray_valid_count"].shape == (n_moments, n_returns)
        assert activity["ray_valid_fraction"].shape == (n_moments, n_returns)
        assert activity["sweep_valid_count"].shape == (n_moments, n_sweeps)
        assert activity["sweep_valid_fraction"].shape == (n_moments, n_sweeps)
        assert activity["volume_valid_count"].shape == (n_moments, n_vcps)
        assert activity["volume_valid_fraction"].shape == (n_moments, n_vcps)

        if "DBZH" in rs["returns"] and "DBZH" in moments:
            idx = moments.index("DBZH")
            expected = np.isfinite(_moment_matrix(rs["returns"], "DBZH")).sum(axis=1)
            actual = activity["ray_valid_count"][idx]
            assert np.array_equal(actual, expected)

    def test_parse_activity_can_be_disabled(self, test_file_bytes):
        rs = rrs.parse(test_file_bytes, include_activity=False)
        assert "activity" not in rs

    def test_parse_sweep_number_alignment(self, test_file_bytes):
        rs = rrs.parse(test_file_bytes)

        sweeps = rs["sweeps"]
        sweep_numbers = np.asarray(sweeps["sweep_number"], dtype=np.uint32)
        sweep_num_returns = np.asarray(sweeps["num_returns"], dtype=np.int64)

        return_sweep_numbers = np.asarray(rs["returns"]["sweep_number"], dtype=np.uint32)

        unique_return_sweeps = np.unique(return_sweep_numbers)
        assert set(unique_return_sweeps).issubset(set(sweep_numbers))

        for idx, sweep_number in enumerate(sweep_numbers):
            actual = int(np.sum(return_sweep_numbers == sweep_number))
            expected = int(sweep_num_returns[idx])
            assert actual == expected

    def test_parse_sweeps_metadata(self, test_file_bytes):
        rs = rrs.parse(test_file_bytes)
        sweeps = rs["sweeps"]

        keys = [
            "sweep_time",
            "sweep_number",
            "elevation_number",
            "elevation_angle",
            "max_gates",
            "range_start",
            "range_step",
            "num_returns",
        ]
        for key in keys:
            assert key in sweeps

        n_sweeps = len(sweeps["sweep_time"])
        for key in keys:
            assert len(sweeps[key]) == n_sweeps

        assert np.all(np.asarray(sweeps["elevation_angle"]) <= 90.0)
        assert np.all(np.asarray(sweeps["elevation_angle"]) >= -1.0)

    def test_parse_num_returns_consistency(self, test_file_bytes):
        rs = rrs.parse(test_file_bytes)
        expected = int(np.asarray(rs["sweeps"]["num_returns"], dtype=np.int64).sum())
        actual = len(rs["returns"]["return_time"])
        assert expected == actual

    def test_parse_azimuth_range(self, test_file_bytes):
        rs = rrs.parse(test_file_bytes)
        azimuth = np.asarray(rs["returns"]["azimuth"])
        assert np.nanmin(azimuth) >= 0.0
        assert np.nanmax(azimuth) <= 360.0

    def test_parse_time_monotonic_per_sweep(self, test_file_bytes):
        rs = rrs.parse(test_file_bytes)
        time = np.asarray(rs["returns"]["return_time"], dtype=np.int64)
        sweep_number = np.asarray(rs["returns"]["sweep_number"], dtype=np.uint32)

        for sweep in np.unique(sweep_number):
            sweep_time = time[sweep_number == sweep]
            if sweep_time.size > 1:
                assert sweep_time[-1] >= sweep_time[0]

    def test_parse_returns_are_chunked(self, test_file_bytes):
        rs = rrs.parse(test_file_bytes, fold_size=128)
        sweeps = rs["sweeps"]

        max_gates = int(np.nanmax(np.asarray(sweeps["max_gates"], dtype=np.int64)))
        if max_gates <= 128:
            pytest.skip("Test file does not exceed fold size")

        # Chunking means number of returns exceeds number of physical radials.
        n_returns = len(rs["returns"]["return_time"])
        keys = np.column_stack(
            [
                np.asarray(rs["returns"]["return_time"], dtype=np.int64),
                np.asarray(rs["returns"]["sweep_number"], dtype=np.int64),
                np.round(np.asarray(rs["returns"]["azimuth"], dtype=np.float64) * 100.0).astype(
                    np.int64
                ),
            ]
        )
        n_radials = np.unique(keys, axis=0).shape[0]
        assert n_returns > n_radials

    def test_parse_pattern_number_matches_datatree(self, test_file_path, test_file_bytes):
        dt = rxr.open_datatree(test_file_path)
        rs = rrs.parse(test_file_bytes)

        pattern = dt.attrs.get("volume_coverage_pattern", 0)
        assert rs["vcps"]["vcp_number"][0] == pattern

    def test_parse_time_covers_datatree(self, test_file_path, test_file_bytes):
        dt = rxr.open_datatree(test_file_path)
        rs = rrs.parse(test_file_bytes)

        sweep_keys = [k for k in dt.children.keys() if k.startswith("sweep_")]
        sweep_keys.sort(key=lambda k: int(k.split("_", 1)[1]))

        dt_times = []
        for key in sweep_keys:
            if "time" not in dt[key].dataset:
                continue
            sweep_time = dt[key]["time"].values.astype("datetime64[ms]").astype("int64")
            if sweep_time.size:
                dt_times.append(sweep_time)

        if not dt_times:
            pytest.skip("No sweep times found in DataTree")

        dt_time = np.concatenate(dt_times)
        rs_time = np.asarray(rs["returns"]["return_time"], dtype=np.int64)

        assert rs_time.min() == dt_time.min()
        assert rs_time.max() == dt_time.max()
        assert np.isin(dt_time, rs_time).all()


def test_parse_accepts_gzip_bytes(test_file_bytes):
    import gzip

    raw_rs = rrs.parse(test_file_bytes)
    gz_rs = rrs.parse(gzip.compress(test_file_bytes))

    assert raw_rs["vcps"]["vcp_number"][0] == gz_rs["vcps"]["vcp_number"][0]
    assert len(raw_rs["sweeps"]["sweep_time"]) == len(gz_rs["sweeps"]["sweep_time"])

    np.testing.assert_allclose(
        raw_rs["returns"]["azimuth"][:50],
        gz_rs["returns"]["azimuth"][:50],
        rtol=1e-5,
    )
    if "DBZH" in raw_rs["returns"] and "DBZH" in gz_rs["returns"]:
        np.testing.assert_allclose(
            raw_rs["returns"]["DBZH"][:200],
            gz_rs["returns"]["DBZH"][:200],
            rtol=1e-5,
            equal_nan=True,
        )


class TestFromXradarDatatree:
    def test_from_xradar_datatree_returns_dict(self, test_file_path):
        dt = rxr.open_datatree(test_file_path)
        rs = rrs.from_xradar_datatree(dt)

        assert isinstance(rs, dict)
        assert "vcps" in rs
        assert "sweeps" in rs
        assert "returns" in rs

    def test_from_xradar_datatree_sweep_number_alignment(self, test_file_path):
        dt = rxr.open_datatree(test_file_path)
        rs = rrs.from_xradar_datatree(dt)

        sweeps = rs["sweeps"]
        return_sweep_number = np.asarray(rs["returns"]["sweep_number"], dtype=np.uint32)

        unique_return_sweeps = np.unique(return_sweep_number)
        unique_sweeps = np.unique(np.asarray(sweeps["sweep_number"], dtype=np.uint32))

        assert set(unique_return_sweeps).issubset(set(unique_sweeps))

    def test_from_xradar_datatree_with_xradar_datatree(self, test_file_path):
        try:
            import xradar as xd
        except ImportError:
            pytest.skip("xradar not installed")

        xrad_dt = xd.io.open_nexradlevel2_datatree(test_file_path)
        non_sweep = [k for k in xrad_dt.children.keys() if not k.startswith("sweep_")]
        assert len(non_sweep) > 0

        rs = rrs.from_xradar_datatree(xrad_dt)

        n_sweeps = len(rs["sweeps"]["sweep_time"])
        unique_return_sweeps = np.unique(np.asarray(rs["returns"]["sweep_number"]))
        assert len(unique_return_sweeps) == n_sweeps


class TestToXradarDatatree:
    def test_to_xradar_datatree_returns_datatree(self, test_file_bytes):
        rs = rrs.parse(test_file_bytes)
        dt = rrs.to_xradar_datatree(rs)

        assert hasattr(dt, "children")
        sweep_keys = [k for k in dt.children.keys() if k.startswith("sweep_")]
        assert len(sweep_keys) > 0

    def test_to_xradar_datatree_preserves_moments(self, test_file_bytes):
        rs = rrs.parse(test_file_bytes)
        dt = rrs.to_xradar_datatree(rs)

        sweep_0 = dt["sweep_0"]
        ds = sweep_0.dataset
        found = [v for v in ["DBZH", "VRADH", "RHOHV", "ZDR"] if v in ds]
        assert len(found) > 0


class TestRaystackDatatree:
    def test_to_raystack_datatree_has_nodes(self, test_file_bytes):
        rs = rrs.parse(test_file_bytes)
        dt = rrs.to_raystack_datatree(rs)

        assert hasattr(dt, "children")
        assert "vcps" in dt.children
        assert "sweeps" in dt.children
        assert "returns" in dt.children

    def test_open_datatree_returns_raystack(self, test_file_path):
        dt = rrs.open_datatree(test_file_path)
        assert "returns" in dt.children
        assert "vcps" in dt.children
        assert "sweeps" in dt.children

    @pytest.mark.asyncio
    async def test_open_datatree_async_returns_raystack(self, test_file_path):
        dt = await rrs.open_datatree_async(test_file_path)
        assert "returns" in dt.children
        assert "vcps" in dt.children
        assert "sweeps" in dt.children


class TestRoundtrip:
    def test_roundtrip_preserves_sweeps(self, test_file_path):
        dt1 = rxr.open_datatree(test_file_path)
        rs = rrs.from_xradar_datatree(dt1)
        dt2 = rrs.to_xradar_datatree(rs)

        sweeps1 = [k for k in dt1.children.keys() if k.startswith("sweep_")]
        sweeps2 = [k for k in dt2.children.keys() if k.startswith("sweep_")]
        assert len(sweeps1) == len(sweeps2)

    def test_roundtrip_preserves_radial_count(self, test_file_path):
        dt1 = rxr.open_datatree(test_file_path)
        rs = rrs.from_xradar_datatree(dt1)
        dt2 = rrs.to_xradar_datatree(rs)

        for key in dt1.children:
            if not key.startswith("sweep_"):
                continue
            if key not in dt2.children:
                continue
            n1 = len(dt1[key]["azimuth"])
            n2 = len(dt2[key]["azimuth"])
            assert n1 == n2, f"{key}: radial count mismatch ({n1} vs {n2})"

    def test_roundtrip_preserves_coordinates(self, test_file_path):
        dt1 = rxr.open_datatree(test_file_path)
        rs = rrs.from_xradar_datatree(dt1)
        dt2 = rrs.to_xradar_datatree(rs)

        for key in dt1.children:
            if not key.startswith("sweep_"):
                continue
            if key not in dt2.children:
                continue

            np.testing.assert_allclose(
                dt1[key]["azimuth"].values,
                dt2[key]["azimuth"].values,
                rtol=1e-5,
                err_msg=f"{key}: azimuth values differ",
            )
            np.testing.assert_allclose(
                dt1[key]["elevation"].values,
                dt2[key]["elevation"].values,
                rtol=1e-5,
                err_msg=f"{key}: elevation values differ",
            )

    def test_roundtrip_preserves_moment_values(self, test_file_path):
        dt1 = rxr.open_datatree(test_file_path)
        rs = rrs.from_xradar_datatree(dt1, fold_size=2048)
        dt2 = rrs.to_xradar_datatree(rs)

        moments_to_check = ["DBZH", "RHOHV", "ZDR"]

        for key in dt1.children:
            if not key.startswith("sweep_"):
                continue
            if key not in dt2.children:
                continue

            for moment in moments_to_check:
                if moment not in dt1[key].dataset or moment not in dt2[key].dataset:
                    continue

                v1 = dt1[key][moment].values
                v2 = dt2[key][moment].values

                n_compare = min(v1.shape[1], v2.shape[1])
                v1_slice = v1[:, :n_compare]
                v2_slice = v2[:, :n_compare]

                mask = np.isfinite(v1_slice) & np.isfinite(v2_slice)
                if np.any(mask):
                    np.testing.assert_allclose(
                        v1_slice[mask],
                        v2_slice[mask],
                        rtol=1e-5,
                        err_msg=f"{key}/{moment}: values differ",
                    )

                nan_match = np.isnan(v1_slice) == np.isnan(v2_slice)
                nan_mismatch_pct = 100 * (1 - nan_match.mean())
                assert nan_mismatch_pct < 1.0, (
                    f"{key}/{moment}: NaN positions differ by {nan_mismatch_pct:.1f}%"
                )

    def test_parse_vs_from_xradar_datatree_consistency(self, test_file_path, test_file_bytes):
        rs1 = rrs.parse(test_file_bytes, fold_size=128)
        dt = rxr.open_datatree(test_file_path)
        rs2 = rrs.from_xradar_datatree(dt, fold_size=128)

        assert len(rs1["sweeps"]["sweep_time"]) == len(rs2["sweeps"]["sweep_time"])
        assert len(rs1["returns"]["return_time"]) == len(rs2["returns"]["return_time"])

        np.testing.assert_allclose(
            rs1["returns"]["azimuth"],
            rs2["returns"]["azimuth"],
            rtol=1e-5,
            err_msg="azimuth differs between parse and from_xradar_datatree",
        )

    def test_from_xradar_activity_can_be_disabled(self, test_file_path):
        dt = rxr.open_datatree(test_file_path)
        rs = rrs.from_xradar_datatree(dt, include_activity=False)
        assert "activity" not in rs


class TestActivityDataTree:
    def test_open_datatree_includes_activity(self, test_file_path):
        dt = rrs.open_datatree(test_file_path)
        assert "activity" in dt

        activity = dt["activity"].dataset
        assert "moment" in activity.coords
        assert "ray_valid_count" in activity
        assert "ray_valid_fraction" in activity

        n_returns = dt["returns"].sizes["return_time"]
        assert activity["ray_valid_count"].shape[1] == n_returns

    @pytest.mark.network
    @pytest.mark.slow
    def test_open_datatree_activity_s3(self):
        dt = rrs.open_datatree(S3_TEST_FILE)
        assert "activity" in dt
        activity = dt["activity"].dataset
        assert "volume_valid_fraction" in activity
        assert activity["volume_valid_fraction"].shape[1] == dt["vcps"].sizes["vcp_time"]

    def test_open_datatree_activity_can_be_disabled(self, test_file_path):
        dt = rrs.open_datatree(test_file_path, include_activity=False)
        assert "activity" not in dt


class TestActivityConsistency:
    def test_activity_python_matches_rust(self, test_file_bytes):
        rs = rrs.parse(test_file_bytes)
        activity = rs["activity"]
        moments = [str(m) for m in activity["moment"]]

        expected = _compute_activity_python(rs, moments)

        np.testing.assert_array_equal(activity["ray_valid_count"], expected["ray_valid_count"])
        np.testing.assert_array_equal(
            activity["sweep_valid_count"], expected["sweep_valid_count"]
        )
        np.testing.assert_array_equal(
            activity["volume_valid_count"], expected["volume_valid_count"]
        )
        np.testing.assert_allclose(
            activity["ray_valid_fraction"],
            expected["ray_valid_fraction"],
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            activity["sweep_valid_fraction"],
            expected["sweep_valid_fraction"],
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            activity["volume_valid_fraction"],
            expected["volume_valid_fraction"],
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
        )

    @pytest.mark.network
    @pytest.mark.slow
    def test_activity_python_matches_rust_s3(self):
        import fsspec

        with fsspec.open(S3_TEST_FILE, "rb", anon=True) as f:
            rs = rrs.parse(f.read())

        activity = rs["activity"]
        moments = [str(m) for m in activity["moment"]]
        expected = _compute_activity_python(rs, moments)

        np.testing.assert_array_equal(activity["ray_valid_count"], expected["ray_valid_count"])
        np.testing.assert_array_equal(
            activity["sweep_valid_count"], expected["sweep_valid_count"]
        )
        np.testing.assert_array_equal(
            activity["volume_valid_count"], expected["volume_valid_count"]
        )


class TestMultiCloudRouting:
    """Verify open_datatree routes non-s3 URIs through fetch_bytes_from_url
    rather than the old s3-only / local-fs branch.
    """

    def test_local_path_missing_raises(self):
        with pytest.raises(Exception) as excinfo:
            rrs.open_datatree("/nonexistent/definitely/not/a/file")
        msg = str(excinfo.value).lower()
        assert "unsupported uri scheme" not in msg

    @pytest.mark.network
    def test_gs_uri_routes_to_object_store(self):
        """A gs:// URI must reach the object_store fetch path. The public GCS
        NEXRAD mirror stores tar archives rather than single-volume files, so
        parse will fail with a "truncated record" error — that's expected and
        proves bytes were fetched from GCS rather than rejected at the
        URI-routing layer.
        """
        url = (
            "gs://gcp-public-data-nexrad-l2/2025/01/01/KABR/"
            "NWS_NEXRAD_NXL2DPBL_KABR_20250101000000_20250101005959.tar"
        )
        with pytest.raises(Exception) as excinfo:
            rrs.open_datatree(url, storage_options={"skip_signature": "true"})
        msg = str(excinfo.value).lower()
        assert "unsupported uri scheme" not in msg
        assert "truncated record" in msg or "nexrad data error" in msg

    def test_storage_options_kwarg_accepted(self, test_file_path):
        dt = rrs.open_datatree(test_file_path, storage_options=None)
        assert "returns" in dt.children

    def test_relative_local_path(self, test_file_path, tmp_path, monkeypatch):
        """Relative local paths must keep working — regression test."""
        import os
        import shutil

        copied = tmp_path / os.path.basename(test_file_path)
        shutil.copy(test_file_path, copied)
        monkeypatch.chdir(tmp_path)

        dt = rrs.open_datatree(copied.name)
        assert "returns" in dt.children

    def test_parent_relative_local_path(self, test_file_path, tmp_path, monkeypatch):
        """Parent-relative paths (../data/file) must resolve correctly."""
        import os
        import shutil

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        copied = data_dir / os.path.basename(test_file_path)
        shutil.copy(test_file_path, copied)
        monkeypatch.chdir(work_dir)

        dt = rrs.open_datatree(f"../data/{copied.name}")
        assert "returns" in dt.children


class TestPerformance:
    @pytest.mark.benchmark
    def test_parse_benchmark(self, benchmark, test_file_bytes):
        rs = benchmark(lambda: rrs.parse(test_file_bytes))
        assert isinstance(rs, dict)
