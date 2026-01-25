"""Tests for peek_volume metadata extraction."""

import radrs
import radrs.xradar as rxr


def test_peek_volume_vcp_matches_datatree(test_file_path):
    """Peek should find VCP and match DataTree metadata."""
    dt = rxr.open_datatree(test_file_path)
    expected = dt.attrs.get("volume_coverage_pattern")
    if expected is None:
        return

    meta = radrs.peek_volume(test_file_path)
    assert meta.vcp is not None
    assert meta.vcp == int(expected)


def test_peek_volume_header_only(test_file_path):
    """Header-only peek should skip deeper fields."""
    meta = radrs.peek_volume(test_file_path, header_only=True)
    assert meta.vcp is None
    assert isinstance(meta.site, str)
    assert len(meta.site) == 4
    assert meta.version.startswith("V")


def test_peek_volume_small_mode(test_file_path):
    """Small mode should still find VCP."""
    meta = radrs.peek_volume(test_file_path, peek_mode="small")
    assert meta.vcp is not None
