"""Shared source-format validation for Python-facing APIs."""

from __future__ import annotations

DEFAULT_SOURCE_FORMAT = "nexrad-level2"
SUPPORTED_SOURCE_FORMATS = (DEFAULT_SOURCE_FORMAT,)


def validate_source_format(format: str = DEFAULT_SOURCE_FORMAT) -> str:
    if format not in SUPPORTED_SOURCE_FORMATS:
        supported = ", ".join(SUPPORTED_SOURCE_FORMATS)
        raise ValueError(
            f"Unsupported format {format!r}. Supported formats: {supported}"
        )
    return format
