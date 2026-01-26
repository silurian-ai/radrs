"""Tests for archive analysis utilities."""

from datetime import datetime

import radrs.archive as archive


class _Info:
    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size

    def s3_url(self, site: str, date: str) -> str:
        return f"s3://bucket/{date}/{site}/{self.name}"


class _Meta:
    def __init__(self, vcp: int, dt: datetime):
        self.vcp = vcp
        self.datetime = dt
        self.is_clear_air = vcp in (31, 32, 35)
        self.is_precipitation = vcp in (12, 21, 112, 121, 212, 215, 221)


def test_analyze_archive_filters_and_counts():
    def list_volumes(site: str, date: str):
        if date == "2024-07-02":
            return [_Info("A", 500_000), _Info("B", 2_000_000)]
        if date == "2024-07-03":
            return [_Info("C", 3_000_000)]
        return []

    def iter_meta_candidates(candidates, peek_mode="fast", **_kwargs):
        for url, info in candidates:
            name = info.name
            vcp = 212 if name == "B" else 31
            yield info, _Meta(vcp, datetime(2024, 7, 2))

    rows, summary = archive.analyze_archive(
        "KABR",
        "2024-07-02",
        "2024-07-03",
        min_size_bytes=1_000_000,
        list_volumes_fn=list_volumes,
        iter_meta_candidates_fn=iter_meta_candidates,
    )

    assert summary["candidates"] == 2
    assert summary["size_stats"]["min"] == 2_000_000
    assert summary["size_stats"]["max"] == 3_000_000
    assert summary["peeked"] == 2
    assert summary["vcp_counts"][212] == 1
    assert summary["vcp_counts"][31] == 1
    assert all(row.size >= 1_000_000 for row in rows)
