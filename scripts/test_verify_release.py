from __future__ import annotations

import unittest
from pathlib import Path

from scripts.verify_release import load_metadata, verify_release_tag


REPOSITORY = Path(__file__).resolve().parent.parent


class VerifyReleaseTagTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = load_metadata(REPOSITORY)

    def test_current_release_tag_matches_all_metadata(self) -> None:
        tag = f"radrs-v{self.metadata.cargo_version}"
        self.assertEqual(
            verify_release_tag(tag, self.metadata), self.metadata.cargo_version
        )

    def test_rejects_non_release_tag(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported release tag"):
            verify_release_tag("v0.3.7", self.metadata)

    def test_rejects_version_mismatch(self) -> None:
        mismatched = self.metadata.__class__(
            cargo_name=self.metadata.cargo_name,
            cargo_version="0.3.6",
            project_name=self.metadata.project_name,
            project_version_is_dynamic=self.metadata.project_version_is_dynamic,
            manifest_version=self.metadata.manifest_version,
        )
        with self.assertRaisesRegex(ValueError, "Cargo.toml=0.3.6"):
            verify_release_tag(f"radrs-v{self.metadata.cargo_version}", mismatched)

    def test_rejects_static_python_version(self) -> None:
        static_version = self.metadata.__class__(
            cargo_name=self.metadata.cargo_name,
            cargo_version=self.metadata.cargo_version,
            project_name=self.metadata.project_name,
            project_version_is_dynamic=False,
            manifest_version=self.metadata.manifest_version,
        )
        with self.assertRaisesRegex(ValueError, "derive its version"):
            verify_release_tag(
                f"radrs-v{self.metadata.cargo_version}", static_version
            )


if __name__ == "__main__":
    unittest.main()
