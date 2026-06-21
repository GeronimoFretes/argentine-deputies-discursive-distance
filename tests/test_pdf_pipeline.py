import json
from pathlib import Path

import httpx
import pymupdf

from argentine_deputies_discursive_distance.pdf_pipeline import (
    download_pdf,
    extract_document,
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
            "Hello from the Chamber of Deputies.",
        )
        return document.tobytes()
    finally:
        document.close()


def manifest_record() -> dict[str, str]:
    return {
        "source_record_id": "period-144-date-2026-05-20-meeting-3",
        "source_url": "https://example.org/index.html",
        "pdf_url": "https://example.org/session.pdf",
        "session_date": "2026-05-20",
        "period": "144",
        "meeting_number": "3",
        "session_category": "legislative_debate",
    }


def test_download_pdf_and_reuse_valid_cache(
    tmp_path: Path,
) -> None:
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
        first = download_pdf(
            client=client,
            manifest_record=manifest_record(),
            pdf_directory=tmp_path / "pdfs",
        )

    assert first["reused"] is False
    assert request_count == 1
    assert Path(str(first["pdf_path"])).exists()
    assert Path(str(first["metadata_path"])).exists()

    def unexpected_request(
        request: httpx.Request,
    ) -> httpx.Response:
        raise AssertionError(f"Cache reuse made a request to {request.url}")

    with httpx.Client(transport=httpx.MockTransport(unexpected_request)) as client:
        second = download_pdf(
            client=client,
            manifest_record=manifest_record(),
            pdf_directory=tmp_path / "pdfs",
        )

    assert second["reused"] is True
    assert second["sha256"] == first["sha256"]


def test_extract_document_writes_page_and_block_records(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "session.pdf"
    pdf_path.write_bytes(make_pdf_bytes())

    result = extract_document(
        pdf_path=pdf_path,
        manifest_record=manifest_record(),
        output_root=tmp_path / "extracted",
    )

    assert result["reused"] is False
    assert result["pdf"]["page_count"] == 1

    pages_path = Path(result["extraction"]["pages_path"])
    blocks_path = Path(result["extraction"]["blocks_path"])

    assert pages_path.exists()
    assert blocks_path.exists()

    page_lines = pages_path.read_text(encoding="utf-8").splitlines()
    block_lines = blocks_path.read_text(encoding="utf-8").splitlines()

    assert len(page_lines) == 1
    assert len(block_lines) >= 1

    page_record = json.loads(page_lines[0])
    first_block = json.loads(block_lines[0])

    assert page_record["page_number"] == 1
    assert "Hello from the Chamber" in (page_record["text"])
    assert first_block["reading_order"] == 1
    assert "Hello from the Chamber" in (first_block["text"])


def test_extract_document_reuses_matching_outputs(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "session.pdf"
    pdf_path.write_bytes(make_pdf_bytes())

    first = extract_document(
        pdf_path=pdf_path,
        manifest_record=manifest_record(),
        output_root=tmp_path / "extracted",
    )
    second = extract_document(
        pdf_path=pdf_path,
        manifest_record=manifest_record(),
        output_root=tmp_path / "extracted",
    )

    assert first["reused"] is False
    assert second["reused"] is True
    assert second["pdf"]["sha256"] == first["pdf"]["sha256"]


def test_extract_document_rebuilds_corrupted_output(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "session.pdf"
    pdf_path.write_bytes(make_pdf_bytes())

    first = extract_document(
        pdf_path=pdf_path,
        manifest_record=manifest_record(),
        output_root=tmp_path / "extracted",
    )

    pages_path = Path(first["extraction"]["pages_path"])
    pages_path.write_text(
        "corrupted output\n",
        encoding="utf-8",
    )

    second = extract_document(
        pdf_path=pdf_path,
        manifest_record=manifest_record(),
        output_root=tmp_path / "extracted",
    )

    assert second["reused"] is False

    rebuilt_page = json.loads(pages_path.read_text(encoding="utf-8").splitlines()[0])

    assert "Hello from the Chamber" in rebuilt_page["text"]
