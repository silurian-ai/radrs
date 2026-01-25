"""Archive analysis utilities (listing + VCP peeks)."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Iterable, Sequence

import radrs


@dataclass(frozen=True)
class ArchiveCandidate:
    """A listed volume candidate with URL + size metadata."""

    date: str
    info: radrs.VolumeInfo
    url: str


@dataclass(frozen=True)
class ArchivePeekRow:
    """Row of metadata extracted via peek."""

    date: str
    name: str
    size: int
    vcp: int | None
    is_clear_air: bool | None
    is_precipitation: bool | None


def _iter_dates(start: str, end: str) -> Iterable[str]:
    start_dt = datetime.strptime(start, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end, "%Y-%m-%d").date()
    current = start_dt
    while current <= end_dt:
        yield current.strftime("%Y-%m-%d")
        current = current + timedelta(days=1)


def _quantile(sorted_values: Sequence[int], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    idx = q * (len(sorted_values) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    if lo == hi:
        return float(sorted_values[lo])
    frac = idx - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def iter_candidates(
    site: str,
    start: str,
    end: str | None = None,
    *,
    min_size_bytes: int = 0,
    limit: int | None = None,
    list_volumes_fn: Callable[[str, str], list] | None = None,
) -> Iterable[ArchiveCandidate]:
    """Yield archive candidates from listings."""
    if end is None:
        end = start

    list_volumes_fn = list_volumes_fn or radrs.list_volumes
    yielded = 0
    for date in _iter_dates(start, end):
        infos = list_volumes_fn(site, date)
        for info in infos:
            if info.size < min_size_bytes:
                continue
            url = info.s3_url(site, date)
            yield ArchiveCandidate(date=date, info=info, url=url)
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def summarize_sizes(candidates: Sequence[ArchiveCandidate]) -> dict:
    sizes = [candidate.info.size for candidate in candidates]
    sizes.sort()
    total = sum(sizes)
    return {
        "count": len(sizes),
        "total_bytes": total,
        "min": sizes[0] if sizes else None,
        "median": _quantile(sizes, 0.5),
        "p90": _quantile(sizes, 0.9),
        "max": sizes[-1] if sizes else None,
    }


def peek_candidates(
    candidates: Sequence[ArchiveCandidate],
    *,
    header_only: bool = False,
    peek_mode: str = "fast",
    prefetch: int = 8,
    iter_meta_candidates_fn: Callable[..., Iterable] | None = None,
) -> list[ArchivePeekRow]:
    if not candidates:
        return []

    iter_meta_candidates_fn = iter_meta_candidates_fn or radrs.iter_meta_candidates
    pairs = [(candidate.url, candidate.info) for candidate in candidates]

    rows = []
    for info, meta in iter_meta_candidates_fn(
        pairs,
        header_only=header_only,
        peek_mode=peek_mode,
        prefetch=prefetch,
    ):
        rows.append(
            ArchivePeekRow(
                date=str(meta.datetime.date()),
                name=info.name,
                size=info.size,
                vcp=meta.vcp,
                is_clear_air=getattr(meta, "is_clear_air", None),
                is_precipitation=getattr(meta, "is_precipitation", None),
            )
        )
    return rows


def summarize_vcps(rows: Sequence[ArchivePeekRow], candidate_count: int) -> dict:
    vcp_counts = Counter(row.vcp for row in rows if row.vcp is not None)
    return {
        "peeked": len(rows),
        "peek_errors": max(candidate_count - len(rows), 0),
        "vcp_counts": dict(vcp_counts),
    }


def analyze_archive(
    site: str,
    start: str,
    end: str | None = None,
    *,
    min_size_bytes: int = 0,
    limit: int | None = None,
    peek_mode: str = "fast",
    prefetch: int = 8,
    include_peek: bool = True,
    list_volumes_fn: Callable[[str, str], list] | None = None,
    iter_meta_candidates_fn: Callable[..., Iterable] | None = None,
    as_dataframe: bool = False,
) -> tuple[list[ArchivePeekRow], dict]:
    """List candidates (size stats) and optionally peek VCPs."""
    candidates = list(
        iter_candidates(
            site,
            start,
            end,
            min_size_bytes=min_size_bytes,
            limit=limit,
            list_volumes_fn=list_volumes_fn,
        )
    )

    size_stats = summarize_sizes(candidates)
    rows: list[ArchivePeekRow] = []
    vcp_stats: dict = {"peeked": 0, "peek_errors": 0, "vcp_counts": {}}

    if include_peek:
        rows = peek_candidates(
            candidates,
            peek_mode=peek_mode,
            prefetch=prefetch,
            iter_meta_candidates_fn=iter_meta_candidates_fn,
        )
        vcp_stats = summarize_vcps(rows, len(candidates))

    summary = {
        "site": site,
        "start": start,
        "end": end or start,
        "candidates": len(candidates),
        "size_stats": size_stats,
        **vcp_stats,
    }

    if as_dataframe:
        try:
            import pandas as pd  # type: ignore

            df = pd.DataFrame([row.__dict__ for row in rows])
            return df, summary
        except Exception:
            pass

    return rows, summary


__all__ = [
    "ArchiveCandidate",
    "ArchivePeekRow",
    "iter_candidates",
    "summarize_sizes",
    "peek_candidates",
    "summarize_vcps",
    "analyze_archive",
]
