"""Tests for radrs iterator functions."""

import json
import os
from pathlib import Path
import subprocess
import sys

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

    def test_local_archive_uses_utc_contract_outside_utc(self, tmp_path):
        """Naive and aware bounds agree, and the returned time is aware UTC."""
        archive_root = tmp_path / "archive"
        site_dir = archive_root / "2024" / "03" / "15" / "KTLX"
        site_dir.mkdir(parents=True)
        for filename in (
            "KTLX20240315_000000_V06",
            "KTLX20240315_010000_V06",
            "KTLX20240315_020000_V06",
        ):
            (site_dir / filename).touch()

        script = r"""
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
import radrs

time.tzset()
root = Path(os.environ["RADRS_ARCHIVE_ROOT"])

def collect(start, end):
    return [
        {
            "uri": info.uri,
            "vcp_time": info.vcp_time.isoformat(),
            "is_utc": info.vcp_time.tzinfo is timezone.utc,
            "offset": info.vcp_time.utcoffset().total_seconds(),
        }
        for info in radrs.NexradL2ArchiveIter(
            str(root), start, end, site_filter=["KTLX"]
        )
    ]

naive = collect(datetime(2024, 3, 15), datetime(2024, 3, 15, 2))
after_first = collect(
    datetime(2024, 3, 15, 0, 0, 0, 500000), datetime(2024, 3, 15, 2)
)
pdt = timezone(timedelta(hours=-7))
aware = collect(
    datetime(2024, 3, 14, 17, tzinfo=pdt),
    datetime(2024, 3, 14, 19, tzinfo=pdt),
)
assert datetime(2024, 3, 15).astimezone().utcoffset() == timedelta(hours=-7)
print(json.dumps({"naive": naive, "after_first": after_first, "aware": aware}))
"""
        env = os.environ.copy()
        env["TZ"] = "America/Los_Angeles"
        env["RADRS_ARCHIVE_ROOT"] = str(archive_root)
        source_root = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = os.pathsep.join(
            [source_root, env.get("PYTHONPATH", "")]
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        output = json.loads(result.stdout)

        expected_uris = [
            str(site_dir / "KTLX20240315_000000_V06").lstrip(os.sep),
            str(site_dir / "KTLX20240315_010000_V06").lstrip(os.sep),
        ]
        assert [item["uri"] for item in output["naive"]] == expected_uris
        assert [item["uri"] for item in output["after_first"]] == [expected_uris[1]]
        assert output["aware"] == output["naive"]
        assert all(item["is_utc"] and item["offset"] == 0 for item in output["naive"])

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
