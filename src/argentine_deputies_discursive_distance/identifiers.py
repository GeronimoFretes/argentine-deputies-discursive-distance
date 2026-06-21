"""Deterministic identifiers and conservative text normalization."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import date


def normalize_display_text(value: str) -> str:
    """Normalize Unicode and repeated whitespace without removing accents."""
    normalized = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", normalized).strip()


def fold_for_matching(value: str) -> str:
    """Create an accent-insensitive uppercase form for rules and identity."""
    decomposed = unicodedata.normalize("NFKD", value)
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return re.sub(r"[^A-Z0-9]+", " ", without_marks.upper()).strip()


def build_source_record_id(
    *,
    period: int,
    session_date: date,
    meeting_number: int | None,
    title: str,
) -> str:
    """Build a stable source identifier from official entry attributes.

    Meeting-based records use period, date, and meeting number. Records without
    a meeting number use period, date, and normalized title.
    """
    if meeting_number is not None:
        canonical = (
            f"deputies|period={period}|date={session_date.isoformat()}|meeting={meeting_number}"
        )
    else:
        canonical = (
            f"deputies|period={period}|date={session_date.isoformat()}|"
            f"title={fold_for_matching(title)}"
        )

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
