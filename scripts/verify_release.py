#!/usr/bin/env python3
"""Verify that a release tag agrees with the repository's version metadata."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path


VERSION_PATTERN = r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
RELEASE_TAG = re.compile(rf"^radrs-v(?P<version>{VERSION_PATTERN})$")


@dataclass(frozen=True)
class ReleaseMetadata:
    cargo_name: str
    cargo_version: str
    project_name: str
    project_version_is_dynamic: bool
    manifest_version: str


def load_metadata(repository: Path) -> ReleaseMetadata:
    with (repository / "Cargo.toml").open("rb") as file:
        cargo = tomllib.load(file)
    with (repository / "pyproject.toml").open("rb") as file:
        project = tomllib.load(file)
    with (repository / ".release-please-manifest.json").open() as file:
        manifest = json.load(file)

    package = cargo["package"]
    python_project = project["project"]
    return ReleaseMetadata(
        cargo_name=package["name"],
        cargo_version=package["version"],
        project_name=python_project["name"],
        project_version_is_dynamic="version" in python_project["dynamic"],
        manifest_version=manifest["."],
    )


def verify_release_tag(tag: str, metadata: ReleaseMetadata) -> str:
    match = RELEASE_TAG.fullmatch(tag)
    if match is None:
        raise ValueError(
            f"unsupported release tag {tag!r}; expected radrs-v<MAJOR>.<MINOR>.<PATCH>"
        )

    version = match["version"]
    versions = {
        "Cargo.toml": metadata.cargo_version,
        ".release-please-manifest.json": metadata.manifest_version,
    }
    mismatches = {
        source: value for source, value in versions.items() if value != version
    }
    if mismatches:
        details = ", ".join(f"{source}={value}" for source, value in mismatches.items())
        raise ValueError(f"release tag {tag!r} has version {version}, but {details}")

    if metadata.cargo_name != metadata.project_name:
        raise ValueError(
            "package names differ: "
            f"Cargo.toml={metadata.cargo_name}, pyproject.toml={metadata.project_name}"
        )
    if not metadata.project_version_is_dynamic:
        raise ValueError("pyproject.toml must derive its version from Cargo.toml")

    return version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="tag name being released")
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repository root (defaults to the parent of scripts/)",
    )
    args = parser.parse_args()

    try:
        version = verify_release_tag(args.tag, load_metadata(args.repository))
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"release verification failed: {error}", file=sys.stderr)
        return 1

    print(f"verified release {args.tag} against version metadata ({version})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
