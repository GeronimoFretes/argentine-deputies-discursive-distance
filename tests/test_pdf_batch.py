import csv
from pathlib import Path

import httpx
import pymupdf
import pytest

from argentine_deputies_discursive_distance.pdf_batch import (
    PdfBatchError,
    read_pdf_selections,
    run_pdf_batch,
)


def make_pdf_bytes() -> bytes:
    document = pymupdf.open()

    try:
        page = document.new_page(
            width=500,
            height=700,
        )
        page.insert_text(
            (72, 100),
            "A test parliamentary intervention.",
        )
        return document.tobytes()
    finally:
        document.close()


def write_csv(
    *,
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def test_pdf_batch_downloads_then_reuses_outputs(
    tmp_path: Path,
) -> None:
    source_record_id = "period-144-date-2026-05-20-meeting-3"
    manifest_path = tmp_path / "manifest.csv"
    selection_path = tmp_path / "selection.csv"

    write_csv(
        path=manifest_path,
        fieldnames=[
            "source_record_id",
            "source_url",
            "pdf_url",
            "session_date",
            "period",
            "meeting_number",
            "session_category",
        ],
        rows=[
            {
                "source_record_id": source_record_id,
                "source_url": ("https://example.org/index.html"),
                "pdf_url": ("https://example.org/session.pdf"),
                "session_date": "2026-05-20",
                "period": "144",
                "meeting_number": "3",
                "session_category": ("legislative_debate"),
            }
        ],
    )

    write_csv(
        path=selection_path,
        fieldnames=[
            "label",
            "source_record_id",
        ],
        rows=[
            {
                "label": "latest_session",
                "source_record_id": source_record_id,
            }
        ],
    )

    pdf_bytes = make_pdf_bytes()
    request_count = 0

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal request_count
        request_count += 1

        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "application/pdf"},
            content=pdf_bytes,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        first = run_pdf_batch(
            client=client,
            manifest_path=manifest_path,
            selection_path=selection_path,
            pdf_directory=tmp_path / "pdfs",
            extraction_root=tmp_path / "extracted",
            summary_path=tmp_path / "summary.json",
        )

        second = run_pdf_batch(
            client=client,
            manifest_path=manifest_path,
            selection_path=selection_path,
            pdf_directory=tmp_path / "pdfs",
            extraction_root=tmp_path / "extracted",
            summary_path=tmp_path / "summary.json",
        )

    assert request_count == 1
    assert first["download_reused_count"] == 0
    assert first["extraction_reused_count"] == 0
    assert second["download_reused_count"] == 1
    assert second["extraction_reused_count"] == 1


def test_pdf_selection_rejects_duplicate_record_ids(
    tmp_path: Path,
) -> None:
    selection_path = tmp_path / "selection.csv"

    write_csv(
        path=selection_path,
        fieldnames=[
            "label",
            "source_record_id",
        ],
        rows=[
            {
                "label": "first",
                "source_record_id": "record-1",
            },
            {
                "label": "second",
                "source_record_id": "record-1",
            },
        ],
    )

    with pytest.raises(
        PdfBatchError,
        match="Duplicate selected source_record_id",
    ):
        read_pdf_selections(selection_path)


def test_pdf_selection_accepts_utf8_bom(
    tmp_path: Path,
) -> None:
    selection_path = tmp_path / "selection.csv"
    selection_path.write_text(
        ("label,source_record_id\npilot,record-1\n"),
        encoding="utf-8-sig",
    )

    selections = read_pdf_selections(selection_path)

    assert len(selections) == 1
    assert selections[0].label == "pilot"
    assert selections[0].source_record_id == "record-1"
