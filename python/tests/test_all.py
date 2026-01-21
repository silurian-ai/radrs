"""Basic sanity tests for radrs module."""

import pytest


def test_import_radrs():
    """Test that radrs can be imported."""
    import radrs
    assert hasattr(radrs, "list_volumes")
    assert hasattr(radrs, "VolumeSource")
    assert hasattr(radrs, "iter_volumes")
    assert hasattr(radrs, "stream_realtime")


def test_import_submodules():
    """Test that submodules can be imported."""
    import radrs.xradar
    import radrs.raystack
    import radrs.qc

    assert hasattr(radrs.xradar, "open_datatree")
    assert hasattr(radrs.raystack, "parse")
    assert hasattr(radrs.qc, "rhohv_threshold")
