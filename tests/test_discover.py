import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from argentine_deputies_discursive_distance.discover import (
    DiscoveryError,
    DiscoveryParseError,
    load_local_snapshot,
    parse_manifest_html,
)
from argentine_deputies_discursive_distance.schemas import (
    DiscoveryStatus,
    EventStatus,
    SessionCategory,
    SessionManifestRecord,
    SessionTerm,
)

FIXTURE_PATH = Path("tests/fixtures/html/diary_index_minimal.html")


def parse_fixture() -> list[SessionManifestRecord]:
    html = FIXTURE_PATH.read_text(encoding="utf-8")

    return parse_manifest_html(
        html=html,
        source_page_url="https://example.org/index.html",
        source_snapshot_at=datetime(
            2026,
            6,
            21,
            tzinfo=UTC,
        ),
        candidate_start_date=date(2008, 1, 1),
    )


def test_parse_manifest_preserves_all_entries() -> None:
    records = parse_fixture()

    assert len(records) == 6
    assert {record.period for record in records} == {143, 144}
    assert len({record.source_record_id for record in records}) == 6


def test_parse_linked_legislative_session() -> None:
    records = parse_fixture()
    record = next(item for item in records if item.meeting_number == 3 and item.period == 144)

    assert record.session_number == 3
    assert record.session_date == date(2026, 5, 20)
    assert record.session_category == SessionCategory.LEGISLATIVE_DEBATE
    assert record.session_term == SessionTerm.ORDINARY
    assert record.event_status == EventStatus.HELD
    assert record.discovery_status == DiscoveryStatus.PDF_AVAILABLE
    assert record.is_special is True
    assert record.pdf_url == ("https://example.org/periodo-144/session-3.pdf")


def test_parse_unheld_entry_without_pdf() -> None:
    records = parse_fixture()
    record = next(item for item in records if item.event_status == EventStatus.NOT_HELD)

    assert record.meeting_number is None
    assert record.pdf_url is None
    assert record.discovery_status == DiscoveryStatus.NO_PDF_LINK
    assert record.in_candidate_window is True


def test_parse_informative_session() -> None:
    records = parse_fixture()
    record = next(item for item in records if item.meeting_number == 2 and item.period == 144)

    assert record.session_category == SessionCategory.INFORMATIVE
    assert record.session_term == SessionTerm.ORDINARY


def test_parse_assembly() -> None:
    records = parse_fixture()
    record = next(item for item in records if item.session_category == SessionCategory.ASSEMBLY)

    assert record.meeting_number is None
    assert record.session_term == SessionTerm.UNKNOWN


def test_parse_failed_session_with_pdf() -> None:
    records = parse_fixture()
    record = next(item for item in records if item.event_status == EventStatus.FAILED)

    assert record.discovery_status == DiscoveryStatus.PDF_AVAILABLE
    assert record.session_category == SessionCategory.INFORMATIVE


def test_missing_period_container_fails_loudly() -> None:
    with pytest.raises(
        DiscoveryParseError,
        match="does not contain the #periodos container",
    ):
        parse_manifest_html(
            html="<html><body></body></html>",
            source_page_url="https://example.org",
            source_snapshot_at=datetime.now(UTC),
            candidate_start_date=date(2008, 1, 1),
        )


def test_entry_without_final_date_fails_loudly() -> None:
    html = """
    <div id="periodos">
        <div class="accordion-item">
            <h2 class="accordion-header">
                <button>PERÍODO 144</button>
            </h2>
            <div class="accordion-body">
                <ul>
                    <li>Entry without a date</li>
                </ul>
            </div>
        </div>
    </div>
    """

    with pytest.raises(
        DiscoveryParseError,
        match="Could not parse final entry date",
    ):
        parse_manifest_html(
            html=html,
            source_page_url="https://example.org",
            source_snapshot_at=datetime.now(UTC),
            candidate_start_date=date(2008, 1, 1),
        )


def test_local_snapshot_preserves_retrieval_metadata(
    tmp_path: Path,
) -> None:
    html_path = tmp_path / "diary_index.html"
    html_content = b"<html><body>Frozen snapshot</body></html>"
    html_path.write_bytes(html_content)

    retrieved_at = "2026-06-21T20:28:46.849388+00:00"
    sha256 = hashlib.sha256(html_content).hexdigest()

    metadata_path = html_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "source_url": "https://example.org/index.html",
                "final_url": "https://example.org/index.html",
                "retrieved_at_utc": retrieved_at,
                "http_status": 200,
                "content_type": "text/html;charset=UTF-8",
                "encoding": "utf-8",
                "size_bytes": len(html_content),
                "sha256": sha256,
            }
        ),
        encoding="utf-8",
    )

    snapshot = load_local_snapshot(
        html_path=html_path,
        source_url="https://fallback.example.org",
    )

    assert snapshot.retrieved_at.isoformat() == retrieved_at
    assert snapshot.sha256 == sha256
    assert snapshot.http_status == 200
    assert snapshot.input_path == html_path
    assert snapshot.retrieval_mode == "local_html"


def test_local_snapshot_rejects_hash_mismatch(
    tmp_path: Path,
) -> None:
    html_path = tmp_path / "diary_index.html"
    html_path.write_text(
        "<html>Changed content</html>",
        encoding="utf-8",
    )

    metadata_path = html_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "retrieved_at_utc": ("2026-06-21T20:28:46.849388+00:00"),
                "sha256": "incorrect-sha256",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        DiscoveryError,
        match="SHA-256 does not match",
    ):
        load_local_snapshot(
            html_path=html_path,
            source_url="https://example.org",
        )
