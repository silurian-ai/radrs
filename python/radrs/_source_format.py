"""Supported radar source formats."""

from __future__ import annotations

DEFAULT_SOURCE_FORMAT = "nexrad-level2"


def require_supported_format(source_format: str = DEFAULT_SOURCE_FORMAT) -> str:
    if source_format != DEFAULT_SOURCE_FORMAT:
        raise ValueError(
            f"Unsupported format {source_format!r}. "
            f"Supported formats: {DEFAULT_SOURCE_FORMAT}"
        )
    return source_format
