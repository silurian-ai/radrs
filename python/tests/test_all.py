"""Basic sanity tests for radrs module."""

import pytest


def test_import_radrs():
    """Test that radrs can be imported."""
    import radrs
    assert hasattr(radrs, "NexradL2ArchiveIter")
    assert hasattr(radrs, "peek_volume")
    assert hasattr(radrs, "stream_realtime")


def test_import_submodules():
    """Test that submodules can be imported."""
    import radrs.xradar
    import radrs.raystack
    import radrs.qc
    import radrs.ops

    assert hasattr(radrs.xradar, "open_datatree")
    assert hasattr(radrs.raystack, "parse")
    assert hasattr(radrs.qc, "rhohv_threshold")
    assert hasattr(radrs.ops, "align_azimuth")
