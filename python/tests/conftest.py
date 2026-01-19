"""Test fixtures for radrs tests."""

import pytest
import os

# Path to test data files
TEST_DATA_DIR = "/Users/rejuvyesh/src/silurian/nexrad/downloads"
TEST_FILE_1 = os.path.join(TEST_DATA_DIR, "KDMX20220305_232324_V06")
TEST_FILE_2 = os.path.join(TEST_DATA_DIR, "KCRP20170826_044114_V06")


@pytest.fixture
def test_file_path():
    """Path to a test NEXRAD file."""
    if os.path.exists(TEST_FILE_1):
        return TEST_FILE_1
    elif os.path.exists(TEST_FILE_2):
        return TEST_FILE_2
    else:
        pytest.skip("No test data files available")


@pytest.fixture
def test_file_bytes(test_file_path):
    """Contents of a test NEXRAD file."""
    with open(test_file_path, "rb") as f:
        return f.read()
