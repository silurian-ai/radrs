"""Tests for radrs iterator functions."""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys
import time

import pytest
import radrs


def _write_local_archive(tmp_path):
    """Create a local NEXRAD L2 archive tree and return the site directory."""
    site_dir = tmp_path / "archive" / "2024" / "03" / "15" / "KTLX"
    site_dir.mkdir(parents=True)
    for filename in (
        "KTLX20240315_000000_V06",
        "KTLX20240315_010000_V06",
        "KTLX20240315_020000_V06",
    ):
        (site_dir / filename).touch()
    return site_dir


def _basename(uri):
    """Last path component of a URI, independent of the host separator."""
    return uri.replace("\\", "/").rsplit("/", 1)[-1]


@pytest.mark.parametrize("name", ["stream_archive", "stream_realtime"])
def test_unimplemented_streams_are_not_exported(name):
    assert not hasattr(radrs, name)
    assert not hasattr(radrs._radrs, name)


class TestNexradL2ArchiveIter:
    """Tests for NexradL2ArchiveIter."""

    def test_local_archive_utc_contract(self, tmp_path):
        """Naive and aware bounds agree, and the returned time is aware UTC."""
        site_dir = _write_local_archive(tmp_path)

        def collect(start, end):
            return list(
                radrs.NexradL2ArchiveIter(
                    str(site_dir.parents[3]), start, end, site_filter=["KTLX"]
                )
            )

        naive = collect(datetime(2024, 3, 15), datetime(2024, 3, 15, 2))
        assert [_basename(info.uri) for info in naive] == [
            "KTLX20240315_000000_V06",
            "KTLX20240315_010000_V06",
        ]

        # start_time is inclusive at sub-second granularity, end_time exclusive.
        after_first = collect(
            datetime(2024, 3, 15, 0, 0, 0, 500000), datetime(2024, 3, 15, 2)
        )
        assert [_basename(info.uri) for info in after_first] == [
            "KTLX20240315_010000_V06"
        ]

        # Aware bounds are interpreted by instant, so a -07:00 window covering
        # the same range selects the same volumes as the naive (UTC) one.
        pdt = timezone(timedelta(hours=-7))
        aware = collect(
            datetime(2024, 3, 14, 17, tzinfo=pdt),
            datetime(2024, 3, 14, 19, tzinfo=pdt),
        )
        assert [info.uri for info in aware] == [info.uri for info in naive]

        assert [info.vcp_time for info in naive] == [
            datetime(2024, 3, 15, 0, 0, tzinfo=timezone.utc),
            datetime(2024, 3, 15, 1, 0, tzinfo=timezone.utc),
        ]
        assert all(info.vcp_time.utcoffset() == timedelta(0) for info in naive)

    @pytest.mark.skipif(
        not hasattr(time, "tzset"),
        reason="time.tzset is POSIX-only; the host timezone cannot be overridden",
    )
    def test_local_archive_utc_contract_outside_utc(self, tmp_path):
        """The UTC contract holds when the host timezone is not UTC."""
        site_dir = _write_local_archive(tmp_path)

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
        env["RADRS_ARCHIVE_ROOT"] = str(site_dir.parents[3])
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
        assert [item["vcp_time"] for item in output["naive"]] == [
            "2024-03-15T00:00:00+00:00",
            "2024-03-15T01:00:00+00:00",
        ]
        assert all(item["offset"] == 0 for item in output["naive"])

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
