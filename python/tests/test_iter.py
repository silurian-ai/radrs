"""Tests for radrs iterator functions."""

import pytest
import radrs


class TestListVolumes:
    """Tests for list_volumes function."""

    @pytest.mark.slow
    @pytest.mark.network
    def test_list_volumes_fetches_real_data(self):
        """Test that list_volumes actually fetches volume names from S3.

        This test requires network access and verifies actual data is returned.
        Skip in CI with: pytest -m 'not network'
        """

        # Use a known date with data (March 15, 2024 - a date with known KTLX data)
        volumes = radrs.list_volumes("KTLX", "2024-03-15")

        # Must return a non-empty list (this date definitely has data)
        assert isinstance(volumes, list), "list_volumes should return a list"
        assert len(volumes) > 0, \
            "list_volumes returned empty list for date with known data"

        # All entries should be strings starting with site ID
        assert all(isinstance(v, str) for v in volumes), \
            "All volume names should be strings"
        assert all(v.startswith("KTLX") for v in volumes), \
            "All volume names should start with site ID"

        # Should have reasonable number of volumes (a day has ~288 volumes at 5-min intervals)
        assert len(volumes) > 100, \
            f"Expected >100 volumes for a full day, got {len(volumes)}"


class TestIterVolumes:
    """Tests for iter_volumes function."""

    def test_iter_volumes_returns_iterator(self):
        """Test that iter_volumes returns an iterator (interface check only)."""

        source = radrs.VolumeSource.nexrad("KTLX", start="2024-03-15", end="2024-03-15")
        iterator = radrs.iter_volumes(source)

        # Should be iterable
        assert hasattr(iterator, "__iter__"), "iter_volumes should return iterable"
        assert hasattr(iterator, "__next__"), "iter_volumes should return iterator"

    @pytest.mark.slow
    @pytest.mark.network
    def test_iter_volumes_fetches_and_parses_data(self):
        """Test that iter_volumes actually fetches and parses NEXRAD data.

        This test requires network access and verifies end-to-end functionality.
        Skip in CI with: pytest -m 'not network'
        """

        source = radrs.VolumeSource.nexrad("KTLX", start="2024-03-15", end="2024-03-15")
        iterator = radrs.iter_volumes(source)

        # Actually consume one item - this tests fetch AND parse
        dt = next(iterator)

        # Verify we got a valid DataTree
        assert hasattr(dt, "children"), "iter_volumes should yield DataTree objects"

        sweep_keys = [k for k in dt.children.keys() if k.startswith("sweep_")]
        assert len(sweep_keys) > 0, "DataTree should have sweep children"

        # Verify sweep has expected structure
        sweep_0 = dt["sweep_0"]
        assert "azimuth" in sweep_0.dataset or "azimuth" in sweep_0.coords, \
            "Sweep should have azimuth coordinate"
        assert "DBZH" in sweep_0.dataset or len(sweep_0.dataset.data_vars) > 0, \
            "Sweep should have moment data"


class TestStreamArchive:
    """Tests for stream_archive function."""

    def test_stream_archive_returns_async_iterator(self):
        """Test that stream_archive returns an async iterator."""

        stream = radrs.stream_archive("KTLX")

        # Should be async iterable
        assert hasattr(stream, "__aiter__")
        assert hasattr(stream, "__anext__")

    def test_stream_archive_with_poll_interval(self):
        """Test that stream_archive accepts poll_interval parameter."""

        stream = radrs.stream_archive("KTLX", poll_interval=60)

        assert hasattr(stream, "__aiter__")


class TestStreamRealtime:
    """Tests for stream_realtime function."""

    def test_stream_realtime_not_implemented(self):
        """Test that stream_realtime raises NotImplementedError."""

        with pytest.raises(NotImplementedError):
            radrs.stream_realtime("KTLX")
