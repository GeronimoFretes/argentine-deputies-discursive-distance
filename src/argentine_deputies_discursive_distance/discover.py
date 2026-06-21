"""Discover sessions from the official Diario de Sesiones index."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import tomllib
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
from bs4 import BeautifulSoup, Tag

from argentine_deputies_discursive_distance.identifiers import (
    build_source_record_id,
    fold_for_matching,
    normalize_display_text,
)
from argentine_deputies_discursive_distance.schemas import (
    DiscoveryStatus,
    EventStatus,
    SessionCategory,
    SessionManifestRecord,
    SessionTerm,
)

DATE_PATTERN = re.compile(r"\((\d{2})/(\d{2})/(\d{4})\)\s*$")
PERIOD_PATTERN = re.compile(r"PER[IÍ]ODO\s+(\d+)", re.IGNORECASE)
MEETING_PATTERN = re.compile(
    r"^\s*(\d+)\s*[°ºª]?\s*Reuni[oó]n\b",
    re.IGNORECASE,
)
SESSION_NUMBER_PATTERN = re.compile(
    r"-\s*(\d+)\s*[°ºª]?\s*Sesi[oó]n\b",
    re.IGNORECASE,
)
VIEWER_PATTERN = re.compile(
    r"""abrirPDF\(\s*['"](?P<url>[^'"]+)['"]\s*,""",
    re.IGNORECASE,
)

USER_AGENT = (
    "argentine-deputies-discursive-distance/0.1 (academic research; public parliamentary records)"
)


class DiscoveryError(RuntimeError):
    """Base exception raised by the discovery pipeline."""


class DiscoveryParseError(DiscoveryError):
    """Raised when the official index cannot be parsed safely."""


@dataclass(frozen=True)
class DiscoveryConfig:
    """Configuration needed by the discovery stage."""

    source_url: str
    candidate_start_date: date


@dataclass(frozen=True)
class SourceSnapshot:
    """Downloaded or locally loaded source HTML and its metadata."""

    content: bytes
    source_url: str
    final_url: str
    retrieved_at: datetime
    http_status: int | None
    content_type: str | None
    encoding: str
    retrieval_mode: str
    input_path: Path | None = None

    @property
    def sha256(self) -> str:
        """Return the SHA-256 digest of the source bytes."""
        return hashlib.sha256(self.content).hexdigest()


def load_discovery_config(config_path: Path) -> DiscoveryConfig:
    """Load discovery settings from the project TOML configuration."""
    raw_config = tomllib.loads(config_path.read_text(encoding="utf-8"))

    try:
        source_section = raw_config["source"]
        project_section = raw_config["project"]

        source_url = str(source_section["session_index_url"])
        candidate_start_date = date.fromisoformat(str(project_section["candidate_start_date"]))
    except (KeyError, TypeError, ValueError) as error:
        raise DiscoveryError(
            f"Invalid discovery configuration in {config_path}: {error}"
        ) from error

    return DiscoveryConfig(
        source_url=source_url,
        candidate_start_date=candidate_start_date,
    )


def fetch_source_snapshot(source_url: str) -> SourceSnapshot:
    """Download the official session index using a normal HTTP request."""
    retrieved_at = datetime.now(UTC)

    with httpx.Client(
        follow_redirects=True,
        timeout=60.0,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        response = client.get(source_url)
        response.raise_for_status()

    encoding = response.encoding or "utf-8"

    return SourceSnapshot(
        content=response.content,
        source_url=source_url,
        final_url=str(response.url),
        retrieved_at=retrieved_at,
        http_status=response.status_code,
        content_type=response.headers.get("content-type"),
        encoding=encoding,
        retrieval_mode="http",
        input_path=None,
    )


def load_local_snapshot(
    *,
    html_path: Path,
    source_url: str,
) -> SourceSnapshot:
    """Load a frozen HTML snapshot and preserve its acquisition metadata."""
    metadata_path = html_path.with_suffix(".metadata.json")

    if not metadata_path.exists():
        raise DiscoveryError(
            f"Frozen HTML snapshots require a sibling metadata file: {metadata_path}"
        )

    content = html_path.read_bytes()

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DiscoveryError(f"Could not read frozen snapshot metadata: {metadata_path}") from error

    actual_sha256 = hashlib.sha256(content).hexdigest()
    expected_sha256 = str(metadata.get("sha256", ""))

    if actual_sha256 != expected_sha256:
        raise DiscoveryError(
            "Frozen HTML SHA-256 does not match its metadata: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    try:
        retrieved_at = datetime.fromisoformat(str(metadata["retrieved_at_utc"]))
    except (KeyError, TypeError, ValueError) as error:
        raise DiscoveryError("Frozen snapshot metadata has no valid retrieved_at_utc.") from error

    if retrieved_at.tzinfo is None:
        raise DiscoveryError("Frozen snapshot retrieved_at_utc must include a timezone.")

    http_status_value = metadata.get("http_status")
    http_status = int(http_status_value) if http_status_value is not None else None

    return SourceSnapshot(
        content=content,
        source_url=str(metadata.get("source_url", source_url)),
        final_url=str(metadata.get("final_url", source_url)),
        retrieved_at=retrieved_at.astimezone(UTC),
        http_status=http_status,
        content_type=(
            str(metadata["content_type"]) if metadata.get("content_type") is not None else None
        ),
        encoding=str(metadata.get("encoding", "utf-8")),
        retrieval_mode="local_html",
        input_path=html_path,
    )


def _tag_text(tag: Tag) -> str:
    """Return normalized visible text from an HTML element."""
    return normalize_display_text(" ".join(tag.stripped_strings))


def _string_attribute(tag: Tag, attribute: str) -> str | None:
    """Return an HTML attribute only when its value is a string."""
    value = tag.get(attribute)
    return value if isinstance(value, str) else None


def _parse_period(period_item: Tag) -> int:
    button = period_item.select_one(".accordion-header button")

    if not isinstance(button, Tag):
        raise DiscoveryParseError("A period block has no accordion header button.")

    period_text = _tag_text(button)
    match = PERIOD_PATTERN.fullmatch(period_text)

    if match is None:
        raise DiscoveryParseError(f"Could not parse period header: {period_text!r}")

    return int(match.group(1))


def _parse_entry_date(entry_text: str) -> date:
    match = DATE_PATTERN.search(entry_text)

    if match is None:
        raise DiscoveryParseError(f"Could not parse final entry date: {entry_text!r}")

    day, month, year = match.groups()

    try:
        return date(int(year), int(month), int(day))
    except ValueError as error:
        raise DiscoveryParseError(f"Invalid date in entry: {entry_text!r}") from error


def _remove_final_date(entry_text: str) -> str:
    """Remove the terminal parenthesized date from an index entry."""
    without_date = DATE_PATTERN.sub("", entry_text)
    return normalize_display_text(without_date).strip(" -")


def _parse_optional_number(
    *,
    pattern: re.Pattern[str],
    title: str,
) -> int | None:
    match = pattern.search(title)
    return int(match.group(1)) if match is not None else None


def _extract_urls(entry: Tag) -> tuple[str | None, str | None]:
    """Extract the viewer and direct PDF URLs from an official entry."""
    link = entry.select_one("a.pdf-link")

    if not isinstance(link, Tag):
        return None, None

    onclick = _string_attribute(link, "onclick")

    if onclick is None:
        raise DiscoveryParseError(f"PDF entry has no onclick attribute: {_tag_text(entry)!r}")

    viewer_match = VIEWER_PATTERN.search(onclick)

    if viewer_match is None:
        raise DiscoveryParseError(f"Could not parse viewer URL from onclick: {onclick!r}")

    viewer_url = viewer_match.group("url")
    query = parse_qs(urlparse(viewer_url).query)
    pdf_url = query.get("file", [None])[0]

    if pdf_url is None:
        raise DiscoveryParseError(f"Viewer URL has no file parameter: {viewer_url!r}")

    return viewer_url, pdf_url


def _classify_category(title: str) -> SessionCategory:
    folded = fold_for_matching(title)

    if "ASAMBLEA LEGISLATIVA" in folded:
        return SessionCategory.ASSEMBLY
    if "PRESENTACION DE PRESUPUESTO" in folded:
        return SessionCategory.BUDGET_PRESENTATION
    if "EXPRESIONES EN MINORIA" in folded:
        return SessionCategory.EXPRESSIONS_IN_MINORITY
    if "PREPARATORIA" in folded:
        return SessionCategory.PREPARATORY
    if "HOMENAJE" in folded:
        return SessionCategory.HOMAGE
    if "INFORMATIVA" in folded:
        return SessionCategory.INFORMATIVE
    if "SESION" in folded or "ORDINARIA" in folded or "EXTRAORDINARIA" in folded:
        return SessionCategory.LEGISLATIVE_DEBATE

    return SessionCategory.OTHER


def _classify_term(
    *,
    title: str,
    category: SessionCategory,
) -> SessionTerm:
    if category in {
        SessionCategory.ASSEMBLY,
        SessionCategory.BUDGET_PRESENTATION,
    }:
        return SessionTerm.UNKNOWN

    folded = fold_for_matching(title)

    if "PRORROGA" in folded:
        return SessionTerm.EXTENSION
    if "EXTRAORDINARIA" in folded:
        return SessionTerm.EXTRAORDINARY
    if "ORDINARIA" in folded:
        return SessionTerm.ORDINARY

    return SessionTerm.UNKNOWN


def _classify_event_status(
    *,
    title: str,
    has_pdf: bool,
) -> EventStatus:
    folded = fold_for_matching(title)

    if "NO EFECTUADA" in folded:
        return EventStatus.NOT_HELD
    if "FRACASADA" in folded:
        return EventStatus.FAILED
    if has_pdf:
        return EventStatus.HELD

    return EventStatus.UNKNOWN


def _parse_entry(
    *,
    entry: Tag,
    period: int,
    source_entry_position: int,
    period_entry_position: int,
    source_page_url: str,
    source_snapshot_at: datetime,
    candidate_start_date: date,
) -> SessionManifestRecord:
    entry_text = _tag_text(entry)
    session_date = _parse_entry_date(entry_text)
    title_raw = _remove_final_date(entry_text)

    meeting_number = _parse_optional_number(
        pattern=MEETING_PATTERN,
        title=title_raw,
    )
    session_number = _parse_optional_number(
        pattern=SESSION_NUMBER_PATTERN,
        title=title_raw,
    )

    viewer_url, pdf_url = _extract_urls(entry)
    has_pdf = pdf_url is not None

    category = _classify_category(title_raw)
    folded_title = fold_for_matching(title_raw)

    return SessionManifestRecord(
        source_record_id=build_source_record_id(
            period=period,
            session_date=session_date,
            meeting_number=meeting_number,
            title=title_raw,
        ),
        source_entry_position=source_entry_position,
        period_entry_position=period_entry_position,
        period=period,
        meeting_number=meeting_number,
        session_number=session_number,
        session_date=session_date,
        entry_text_raw=entry_text,
        title_raw=title_raw,
        title_normalized=normalize_display_text(title_raw),
        session_category=category,
        session_term=_classify_term(
            title=title_raw,
            category=category,
        ),
        event_status=_classify_event_status(
            title=title_raw,
            has_pdf=has_pdf,
        ),
        discovery_status=(
            DiscoveryStatus.PDF_AVAILABLE if has_pdf else DiscoveryStatus.NO_PDF_LINK
        ),
        is_special="ESPECIAL" in folded_title,
        is_remote="REMOTA" in folded_title,
        is_continuation=bool(re.search(r"\bCONT(?:INUACION)?\b", folded_title)),
        is_joint="CONJUNTA" in folded_title,
        source_page_url=source_page_url,
        viewer_url=viewer_url,
        pdf_url=pdf_url,
        source_snapshot_at=source_snapshot_at,
        in_candidate_window=session_date >= candidate_start_date,
    )


def parse_manifest_html(
    *,
    html: str,
    source_page_url: str,
    source_snapshot_at: datetime,
    candidate_start_date: date,
) -> list[SessionManifestRecord]:
    """Parse every listed source entry without discarding missing PDFs."""
    soup = BeautifulSoup(html, "html.parser")
    period_container = soup.select_one("#periodos")

    if not isinstance(period_container, Tag):
        raise DiscoveryParseError("The official index does not contain the #periodos container.")

    period_items = period_container.find_all(
        "div",
        class_="accordion-item",
        recursive=False,
    )

    if not period_items:
        raise DiscoveryParseError("No period accordion items were found.")

    records: list[SessionManifestRecord] = []
    periods_seen: set[int] = set()
    source_entry_position = 0

    for raw_period_item in period_items:
        if not isinstance(raw_period_item, Tag):
            continue

        period = _parse_period(raw_period_item)

        if period in periods_seen:
            raise DiscoveryParseError(f"Duplicate period block detected: {period}")

        periods_seen.add(period)

        entry_list = raw_period_item.select_one(".accordion-body ul")

        if not isinstance(entry_list, Tag):
            raise DiscoveryParseError(f"Period {period} has no session-entry list.")

        entries = entry_list.find_all("li", recursive=False)

        for period_entry_position, raw_entry in enumerate(entries, start=1):
            if not isinstance(raw_entry, Tag):
                continue

            source_entry_position += 1

            records.append(
                _parse_entry(
                    entry=raw_entry,
                    period=period,
                    source_entry_position=source_entry_position,
                    period_entry_position=period_entry_position,
                    source_page_url=source_page_url,
                    source_snapshot_at=source_snapshot_at,
                    candidate_start_date=candidate_start_date,
                )
            )

    _validate_manifest(records)

    return records


def _validate_manifest(records: list[SessionManifestRecord]) -> None:
    if not records:
        raise DiscoveryParseError("The parsed manifest is empty.")

    identifier_counts = Counter(record.source_record_id for record in records)
    duplicate_identifiers = sorted(
        identifier for identifier, count in identifier_counts.items() if count > 1
    )

    if duplicate_identifiers:
        raise DiscoveryParseError(
            f"Duplicate source_record_id values detected: {duplicate_identifiers[:5]}"
        )

    pdf_counts = Counter(record.pdf_url for record in records if record.pdf_url is not None)
    duplicate_pdf_urls = sorted(pdf_url for pdf_url, count in pdf_counts.items() if count > 1)

    if duplicate_pdf_urls:
        raise DiscoveryParseError(f"Duplicate direct PDF URLs detected: {duplicate_pdf_urls[:5]}")


def _write_snapshot(
    *,
    snapshot: SourceSnapshot,
    source_dir: Path,
) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)

    html_path = source_dir / "diary_index.html"
    metadata_path = source_dir / "diary_index.metadata.json"

    if snapshot.input_path is not None and snapshot.input_path.resolve() == html_path.resolve():
        return

    html_path.write_bytes(snapshot.content)

    metadata: dict[str, Any] = {
        "source_url": snapshot.source_url,
        "final_url": snapshot.final_url,
        "retrieved_at_utc": snapshot.retrieved_at.isoformat(),
        "http_status": snapshot.http_status,
        "content_type": snapshot.content_type,
        "encoding": snapshot.encoding,
        "size_bytes": len(snapshot.content),
        "sha256": snapshot.sha256,
        "retrieval_mode": snapshot.retrieval_mode,
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _write_manifest(
    *,
    records: list[SessionManifestRecord],
    raw_dir: Path,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_path = raw_dir / "session_manifest.csv"
    jsonl_path = raw_dir / "session_manifest.jsonl"

    rows = [record.model_dump(mode="json") for record in records]
    fieldnames = list(SessionManifestRecord.model_fields)

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    with jsonl_path.open(
        "w",
        encoding="utf-8",
    ) as jsonl_file:
        for row in rows:
            jsonl_file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            jsonl_file.write("\n")


def build_manifest_summary(
    *,
    records: list[SessionManifestRecord],
    snapshot: SourceSnapshot,
) -> dict[str, Any]:
    """Create a deterministic quality summary of discovery results."""
    candidate_records = [record for record in records if record.in_candidate_window]

    dates = [record.session_date for record in records]
    candidate_dates = [record.session_date for record in candidate_records]

    return {
        "source_sha256": snapshot.sha256,
        "source_retrieved_at_utc": snapshot.retrieved_at.isoformat(),
        "manifest_built_at_utc": datetime.now(UTC).isoformat(),
        "retrieval_mode": snapshot.retrieval_mode,
        "period_count": len({record.period for record in records}),
        "record_count": len(records),
        "pdf_available_count": sum(
            record.discovery_status == DiscoveryStatus.PDF_AVAILABLE for record in records
        ),
        "no_pdf_link_count": sum(
            record.discovery_status == DiscoveryStatus.NO_PDF_LINK for record in records
        ),
        "minimum_date": min(dates).isoformat(),
        "maximum_date": max(dates).isoformat(),
        "candidate_record_count": len(candidate_records),
        "candidate_pdf_available_count": sum(
            record.discovery_status == DiscoveryStatus.PDF_AVAILABLE for record in candidate_records
        ),
        "candidate_no_pdf_link_count": sum(
            record.discovery_status == DiscoveryStatus.NO_PDF_LINK for record in candidate_records
        ),
        "candidate_minimum_date": min(candidate_dates).isoformat(),
        "candidate_maximum_date": max(candidate_dates).isoformat(),
        "event_status_counts": dict(
            sorted(Counter(record.event_status.value for record in records).items())
        ),
        "session_category_counts": dict(
            sorted(Counter(record.session_category.value for record in records).items())
        ),
    }


def discover_sessions(
    *,
    config_path: Path,
    raw_dir: Path,
    qa_dir: Path,
    html_path: Path | None = None,
) -> dict[str, Any]:
    """Run discovery from either the live website or a frozen HTML file."""
    config = load_discovery_config(config_path)

    if html_path is None:
        snapshot = fetch_source_snapshot(config.source_url)
    else:
        snapshot = load_local_snapshot(
            html_path=html_path,
            source_url=config.source_url,
        )

    _write_snapshot(
        snapshot=snapshot,
        source_dir=raw_dir / "source",
    )

    html = snapshot.content.decode(
        snapshot.encoding,
        errors="strict",
    )

    records = parse_manifest_html(
        html=html,
        source_page_url=snapshot.final_url,
        source_snapshot_at=snapshot.retrieved_at,
        candidate_start_date=config.candidate_start_date,
    )

    _write_manifest(
        records=records,
        raw_dir=raw_dir,
    )

    summary = build_manifest_summary(
        records=records,
        snapshot=snapshot,
    )

    qa_dir.mkdir(parents=True, exist_ok=True)
    summary_path = qa_dir / "session_manifest_summary.json"
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return summary
