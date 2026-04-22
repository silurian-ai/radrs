"""Tests for radrs iterator functions."""

import pytest
import radrs


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


class TestNexradL2ArchiveIter:
    """Tests for NexradL2ArchiveIter."""

    @pytest.mark.slow
    @pytest.mark.network
    def test_no_duplicate_files_at_time_boundaries(self):
        """Test that iterating in 10-minute blocks doesn't produce duplicate files.

        This test verifies that when a file's timestamp falls exactly on a 10-minute
        boundary, it only appears once across adjacent time blocks and not in both.

        Tests KATR data from 2024-01-01, which has a known file that splits exactly
        at a 10-minute mark.
        """
        from datetime import datetime, timedelta, timezone

        # Define 10-minute blocks for the first 30 mins to capture the first few files
        start_date = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        block_duration = timedelta(minutes=10)
        num_blocks = 3  # First 30 mins

        all_files = []
        file_to_blocks = {}  # Track which blocks each file appears in

        # Iterate through each 10-minute block
        for i in range(num_blocks):
            block_start = start_date + i * block_duration
            block_end = block_start + block_duration

            # Create iterator for this specific 10-minute block
            iterator = radrs.NexradL2ArchiveIter(
                base_uri="s3://unidata-nexrad-level2",
                start_time=block_start,
                end_time=block_end,
                storage_options={"anon": "true", "region": "us-east-1"},
                site_filter=["KATX"],
            )

            # Collect files from this block
            block_files = []
            for info in iterator:
                block_files.append(info.uri)
                all_files.append(info.uri)

                # Track which blocks this file appears in
                if info.uri not in file_to_blocks:
                    file_to_blocks[info.uri] = []
                file_to_blocks[info.uri].append(i)

            print(f"Block {i} ({block_start} to {block_end}): {len(block_files)} files")

        # Check for duplicates across all blocks
        unique_files = set(all_files)
        print(f"\nTotal files collected: {len(all_files)}")
        print(f"Unique files: {len(unique_files)}")

        # Find any files that appear in multiple blocks
        duplicates = {
            uri: blocks for uri, blocks in file_to_blocks.items() if len(blocks) > 1
        }

        if duplicates:
            print("\nDuplicate files found:")
            for uri, blocks in duplicates.items():
                print(f"  {uri}")
                print(f"    Appears in blocks: {blocks}")

        # Assert no duplicates
        assert len(all_files) == len(unique_files), (
            f"Found {len(all_files) - len(unique_files)} duplicate files across time blocks"
        )

        # Verify we actually got some data
        assert len(unique_files) >= 3, (
            f"Expected at least 3 files, but only got {len(unique_files)}"
        )
