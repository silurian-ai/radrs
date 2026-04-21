"""Tests for radrs.raystack.BatchedRaystack batch accumulator."""

import pytest
import numpy as np
import xarray as xr
import radrs
import radrs.raystack as rrs
import radrs.qc as rqc
from datetime import UTC, datetime


class TestBatching:
    """Tests for BatchedRaystack batch accumulator."""

    @pytest.mark.slow
    @pytest.mark.network
    @pytest.mark.parametrize("truncate", [True, False])
    def test_batch_from_s3_small_range(self, truncate):
        """Test that BatchedRaystack can accumulate volumes into a batch from S3.

        This ensures we can create a sensible raystack batch from S3 over a small time range.
        """

        # Use the same site and date as test_iter.py
        site = "KABR"
        start_time = datetime(2024, 8, 15, 0, 0, 0, tzinfo=UTC)
        end_time = datetime(2024, 8, 15, 0, 10, 0, tzinfo=UTC)

        # Create batch accumulator with capacity for 3 volumes
        # Estimate: ~14 sweeps per pattern, ~720 returns per sweep
        max_vcps = 4
        max_sweeps = max_vcps * 24  # Conservative estimate
        max_returns = max_sweeps * 540  # Conservative estimate

        batch = rrs.BatchedRaystack(
            max_vcps=max_vcps,
            max_sweeps=max_sweeps,
            max_returns=max_returns,
            fold_size=1832,
            truncate=truncate,
        )

        root_url = "s3://unidata-nexrad-level2"
        storage_options = dict(
            anon="true",
            region="us-east-1",
        )

        num_files = len(
            list(
                radrs.NexradL2ArchiveIter(
                    root_url,
                    start_time=start_time,
                    end_time=end_time,
                    storage_options=storage_options,
                    site_filter=[site],
                )
            )
        )
        assert num_files == 2

        n_added = batch.add_volumes_from_l2(
            radrs.NexradL2ArchiveIter(
                root_url,
                start_time=start_time,
                end_time=end_time,
                storage_options=storage_options,
                site_filter=[site],
            ),
            prefetch=1,
        )

        assert n_added == num_files, (
            f"Should have added {max_vcps} volumes, got {n_added}"
        )
        assert batch.has_capacity()

        progress = batch.progress()
        print(progress)

        # Check progress
        assert progress["patterns_filled"] == num_files
        assert progress["returns_filled"] > 0

        qc_steps = [
            rqc.RhohvThreshold(vname="rhohv_mask"),
            rqc.SunSpike(
                dbzh_threshold=50.0,
                vname="sun_spike",
            ),
            rqc.VradhWindingNumber(
                nyquist=25.0,
                vname="vradh_winding",
            ),
        ]

        batch.add_qc_outputs(qc_steps)

        # Convert to raystack
        raystack = batch.finalize_to_rs_dt()
        with xr.set_options(display_max_rows=99):
            print(raystack)

        # Check truncation behavior
        if truncate:
            # With truncate=True, arrays should be exactly sized to filled data
            assert len(raystack["/vcps"].dataset["vcp_time"].values) == num_files
            assert len(raystack["/sweeps"].dataset["vcp_time"].values) < max_sweeps
            assert len(raystack["/returns"].dataset["vcp_time"].values) < max_returns
        else:
            # With truncate=False, arrays should be padded to max capacity with fill values
            assert len(raystack["/vcps"].dataset["vcp_time"].values) == max_vcps
            assert len(raystack["/sweeps"].dataset["vcp_time"].values) == max_sweeps
            assert len(raystack["/returns"].dataset["vcp_time"].values) == max_returns

        assert raystack["/vcps"].dataset["vcp_time"].values[0].astype(
            np.int64
        ) == pytest.approx(np.datetime64("2024-08-15T00:01:14", "ns").astype(np.int64))
        assert (
            raystack["/sweeps"].dataset["sweep_time"].values[0]
            == raystack["/vcps"].dataset["vcp_time"].values[0]
        )
        assert (
            raystack["/returns"].dataset["return_time"].values[0]
            == raystack["/vcps"].dataset["vcp_time"].values[0]
        )

        assert raystack["/vcps"].dataset["vcp_number"].values[0] == 212
        assert raystack["/sweeps"].dataset["elevation_angle"].values[
            0
        ] == pytest.approx(0.4834, abs=0.01)
        assert raystack["/returns"].dataset["azimuth"].values[0] == pytest.approx(
            60.23, abs=0.01
        )

        if not truncate:
            assert raystack["/vcps"].dataset["vcp_number"].values[-1] == 0
            assert raystack["/vcps"].dataset["instrument_type"].values[-1] == ""
            assert np.isnat(raystack["/sweeps"].dataset["sweep_time"].values[-1])
            assert np.isnat(raystack["/returns"].dataset["return_time"].values[-1])
            assert raystack["/returns"].dataset["qc.sun_spike"].values.ravel()[-1] == 0
            assert np.isnan(
                raystack["/returns"].dataset["qc.vradh_winding"].values.ravel()[-1]
            )

    @pytest.mark.slow
    @pytest.mark.network
    def test_fold_batch_from_s3_small_range(self):
        """Test that BatchedRaystack can accumulate folded-ray volumes into a batch from S3.

        This ensures we can create a sensible raystack batch from S3 over a small time range.
        """

        # Use the same site and date as test_iter.py
        site = "KABR"
        start_time = datetime(2024, 8, 15, 0, 0, 0, tzinfo=UTC)
        end_time = datetime(2024, 8, 15, 0, 10, 0, tzinfo=UTC)

        # Create batch accumulator with capacity for 3 volumes
        # Estimate: ~14 sweeps per pattern, ~720 returns per sweep
        max_vcps = 4
        max_sweeps = max_vcps * 24  # Conservative estimate
        max_rays = max_sweeps * 540  # Conservative estimate

        max_ray_range = 1832
        fold_size = 128
        max_returns = max_rays * int(np.ceil(max_ray_range / fold_size))

        batch = rrs.BatchedRaystack(
            max_vcps=max_vcps,
            max_sweeps=max_sweeps,
            max_returns=max_returns,
            fold_size=fold_size,
            truncate=True,
            drop_empty_returns=True,
        )

        root_url = "s3://unidata-nexrad-level2"
        storage_options = dict(
            anon="true",
            region="us-east-1",
        )

        num_files = len(
            list(
                radrs.NexradL2ArchiveIter(
                    root_url,
                    start_time=start_time,
                    end_time=end_time,
                    storage_options=storage_options,
                    site_filter=[site],
                )
            )
        )
        assert num_files == 2

        n_added = batch.add_volumes_from_l2(
            radrs.NexradL2ArchiveIter(
                root_url,
                start_time=start_time,
                end_time=end_time,
                storage_options=storage_options,
                site_filter=[site],
            ),
            prefetch=1,
        )

        assert n_added == num_files, (
            f"Should have added {max_vcps} volumes, got {n_added}"
        )
        assert batch.has_capacity()

        progress = batch.progress()
        print(progress)

        # Check progress
        assert progress["patterns_filled"] == num_files
        assert progress["returns_filled"] > 0

        # Convert to raystack
        raystack = batch.finalize_to_rs_dt()
        with xr.set_options(display_max_rows=99):
            print(raystack)

        # With truncate=True, arrays should be exactly sized to filled data
        assert len(raystack["/vcps"].dataset["vcp_time"].values) == num_files
        assert len(raystack["/sweeps"].dataset["vcp_time"].values) < max_sweeps
        assert len(raystack["/returns"].dataset["vcp_time"].values) < max_returns

        # Check that the range is actually folded to size 128
        returns_ds = raystack["/returns"].dataset
        assert "range" in returns_ds.dims, (
            "Range dimension should exist in returns dataset"
        )
        assert returns_ds.dims["range"] == fold_size, (
            f"Range should be folded to size {fold_size}, got {returns_ds.dims['range']}"
        )

        # Check that the sum of num_returns in all sweeps equals the number of return times
        sweeps_ds = raystack["/sweeps"].dataset
        total_num_returns = sweeps_ds["num_returns"].sum().values
        num_return_times = len(returns_ds["return_time"].values)
        assert total_num_returns == num_return_times, (
            f"Sum of num_returns ({total_num_returns}) should equal "
            f"number of return times ({num_return_times})"
        )

    @pytest.mark.slow
    @pytest.mark.network
    def test_meta_batch_from_s3_small_range(self):
        """Test that BatchedRaystack can return metadata-only batches from S3."""

        #
        # VCPs only
        #

        # Use the same site and date as test_iter.py
        site = "KABR"
        start_time = datetime(2024, 8, 15, 0, 0, 0, tzinfo=UTC)
        end_time = datetime(2024, 8, 15, 1, 0, 0, tzinfo=UTC)

        # Create batch accumulator with capacity for 3 volumes
        # Estimate: ~14 sweeps per pattern, ~720 returns per sweep
        max_vcps = 10
        max_sweeps = max_vcps * 24  # Conservative estimate

        batch = rrs.BatchedRaystack(
            max_vcps=max_vcps,
            max_sweeps=0,
            max_returns=0,
            fold_size=0,
            truncate=False,
            include_sweeps=False,
        )

        root_url = "s3://unidata-nexrad-level2"
        storage_options = dict(
            anon="true",
            region="us-east-1",
        )

        num_files = len(
            list(
                radrs.NexradL2ArchiveIter(
                    root_url,
                    start_time=start_time,
                    end_time=end_time,
                    storage_options=storage_options,
                    site_filter=[site],
                )
            )
        )
        assert num_files == 9

        n_added = batch.add_volumes_from_l2(
            radrs.NexradL2ArchiveIter(
                root_url,
                start_time=start_time,
                end_time=end_time,
                storage_options=storage_options,
                site_filter=[site],
            ),
            prefetch=10,
        )

        assert n_added == num_files, (
            f"Should have added {max_vcps} volumes, got {n_added}"
        )
        assert batch.has_capacity()

        progress = batch.progress()
        print(progress)

        # Check progress
        assert progress["patterns_filled"] == num_files
        assert progress["sweeps_filled"] == 0
        assert progress["returns_filled"] == 0

        # Convert to raystack
        raystack = batch.finalize_to_rs_dt()
        with xr.set_options(display_max_rows=99):
            print(raystack)

        # Check VCP metadata
        vcps_ds = raystack["/vcps"].dataset

        # Check vcp_time is size 10
        assert len(vcps_ds["vcp_time"].values) == 10, (
            f"vcp_time should have size 10, got {len(vcps_ds['vcp_time'].values)}"
        )

        # Check that 9 times are filled (not NaT)
        filled_times = ~np.isnat(vcps_ds["vcp_time"].values)
        num_filled = np.sum(filled_times)
        assert num_filled == 9, (
            f"Expected 9 filled times, got {num_filled}"
        )

        # Check that the first time is in the specified time bounds
        first_time = vcps_ds["vcp_time"].values[0]
        start_time_np = np.datetime64(start_time)
        end_time_np = np.datetime64(end_time)
        assert start_time_np <= first_time <= end_time_np, (
            f"First time {first_time} should be between {start_time_np} and {end_time_np}"
        )

        # Check that all vcp numbers are filled (not 0)
        vcp_numbers = vcps_ds["vcp_number"].values
        assert np.all(vcp_numbers[:9] != 0), (
            f"First 9 VCP numbers should be filled (non-zero), got {vcp_numbers[:9]}"
        )

        # Check that all latitudes are filled except the last value
        latitudes = vcps_ds["latitude"].values
        assert np.all(~np.isnan(latitudes[:9])), (
            f"First 9 latitudes should be filled (not NaN), got {latitudes[:9]}"
        )
        assert np.isnan(latitudes[9]), (
            f"Last latitude should be NaN, got {latitudes[9]}"
        )

        #
        # VCP and sweep metadata
        #
    
        batch = rrs.BatchedRaystack(
            max_vcps=max_vcps,
            max_sweeps=max_sweeps,
            max_returns=0,
            fold_size=0,
            truncate=True,
            include_returns=False,
        )

        root_url = "s3://unidata-nexrad-level2"
        storage_options = dict(
            anon="true",
            region="us-east-1",
        )

        n_added = batch.add_volumes_from_l2(
            radrs.NexradL2ArchiveIter(
                root_url,
                start_time=start_time,
                end_time=datetime(2024, 8, 15, 0, 10, 0, tzinfo=UTC),
                storage_options=storage_options,
                site_filter=[site],
            ),
            prefetch=1,
        )

        num_files = 2
        assert n_added == num_files, (
            f"Should have added {max_vcps} volumes, got {n_added}"
        )
        assert batch.has_capacity()

        progress = batch.progress()
        print(progress)

        # Check progress
        assert progress["patterns_filled"] == num_files
        assert progress["sweeps_filled"] == 46
        assert progress["returns_filled"] == 0

        # Convert to raystack
        raystack = batch.finalize_to_rs_dt()
        with xr.set_options(display_max_rows=99):
            print(raystack)

        # Check that the sum of num_sweeps in all VCPs equals the number of sweep times
        vcps_ds = raystack["/vcps"].dataset
        sweeps_ds = raystack["/sweeps"].dataset

        total_num_sweeps = vcps_ds["num_sweeps"].sum().values
        num_sweep_times = len(sweeps_ds["sweep_time"].values)

        assert total_num_sweeps == num_sweep_times, (
            f"Sum of num_sweeps ({total_num_sweeps}) should equal "
            f"number of sweep times ({num_sweep_times})"
        )

