"""Test fixtures for radrs tests."""

import pytest
import os
import tempfile

# Path to local test data files (for development)
# Default: ../nexrad/downloads relative to the radrs repo root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_TEST_DATA_DIR = os.path.join(_REPO_ROOT, "..", "nexrad", "downloads")
TEST_DATA_DIR = os.environ.get("RADRS_TEST_DATA_DIR", _DEFAULT_TEST_DATA_DIR)
TEST_FILE_1 = os.path.join(TEST_DATA_DIR, "KDMX20220305_232324_V06")
TEST_FILE_2 = os.path.join(TEST_DATA_DIR, "KCRP20170826_044114_V06")
TEST_FILES = [TEST_FILE_1, TEST_FILE_2]

# S3 URL for CI (when local files not available)
S3_TEST_FILE = "s3://unidata-nexrad-level2/2024/07/02/KABR/KABR20240702_000016_V06"

# Cache for downloaded test file
_cached_test_bytes = None
_cached_test_path = None


def _download_test_file():
    """Download a test file from S3 (for CI)."""
    global _cached_test_bytes, _cached_test_path
    if _cached_test_bytes is not None:
        return _cached_test_path, _cached_test_bytes

    try:
        import fsspec

        with fsspec.open(S3_TEST_FILE, "rb", anon=True) as f:
            _cached_test_bytes = f.read()

        # Also save to temp file for path-based tests
        fd, _cached_test_path = tempfile.mkstemp(suffix="_V06")
        with os.fdopen(fd, "wb") as f:
            f.write(_cached_test_bytes)

        return _cached_test_path, _cached_test_bytes
    except Exception as e:
        pytest.skip(f"Could not download test file from S3: {e}")


@pytest.fixture
def test_file_path():
    """Path to a test NEXRAD file."""
    # Try local files first
    for path in TEST_FILES:
        if os.path.exists(path):
            return path

    # Fall back to downloading from S3
    path, _ = _download_test_file()
    return path


@pytest.fixture
def test_file_bytes(test_file_path):
    """Contents of a test NEXRAD file."""
    with open(test_file_path, "rb") as f:
        return f.read()


@pytest.fixture(scope="session")
def available_test_files():
    """List of available test files."""
    paths = [path for path in TEST_FILES if os.path.exists(path)]
    if paths:
        return paths

    # Fall back to S3 download
    path, _ = _download_test_file()
    return [path]
