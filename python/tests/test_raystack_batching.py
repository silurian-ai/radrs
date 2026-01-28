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
            max_patterns=max_vcps,
            max_sweeps=max_sweeps,
            max_returns=max_returns,
            fold_size=1832,
            truncate=truncate,
        )

        # Create source and use add_volumes helper
        # source = radrs.VolumeSource.nexrad(site, start=date, end=date)

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
        ) == pytest.approx(np.datetime64("2024-08-15T00:01:14", "ms").astype(np.int64))
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
        ] == pytest.approx(0.2884, abs=0.01)
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
