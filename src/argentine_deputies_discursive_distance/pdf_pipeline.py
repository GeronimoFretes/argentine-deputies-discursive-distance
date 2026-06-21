"""Resumable download and extraction of official session PDFs."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pymupdf

from .layout import extract_ordered_page

EXTRACTOR_VERSION = "1"
MAX_DOWNLOAD_ATTEMPTS = 3
FILE_HASH_CHUNK_SIZE = 1024 * 1024


class PdfPipelineError(RuntimeError):
    """Raised when PDF acquisition or extraction cannot continue."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""
    hasher = hashlib.sha256()

    with path.open("rb") as input_file:
        while chunk := input_file.read(FILE_HASH_CHUNK_SIZE):
            hasher.update(chunk)

    return hasher.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object and reject other top-level JSON values."""
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PdfPipelineError(f"Could not read JSON metadata: {path}") from error

    if not isinstance(payload, dict):
        raise PdfPipelineError(f"Expected a JSON object in {path}.")

    return {str(key): value for key, value in payload.items()}


def _write_json_atomic(
    *,
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write JSON through a temporary file and atomically replace the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.part")

    temporary_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _validate_source_record_id(source_record_id: str) -> None:
    """Reject identifiers that could escape their intended directory."""
    if not source_record_id:
        raise PdfPipelineError("Manifest record has no source_record_id.")

    if Path(source_record_id).name != source_record_id:
        raise PdfPipelineError(
            f"source_record_id cannot contain path separators: {source_record_id}"
        )


def _cached_pdf_metadata(
    *,
    pdf_path: Path,
    metadata_path: Path,
    expected_pdf_url: str,
) -> dict[str, Any] | None:
    """Return valid cached metadata or None when the cache is unusable."""
    if not pdf_path.exists() or not metadata_path.exists():
        return None

    try:
        metadata = _read_json_object(metadata_path)
    except PdfPipelineError:
        return None

    if str(metadata.get("pdf_url", "")) != expected_pdf_url:
        return None

    try:
        with pdf_path.open("rb") as input_file:
            signature = input_file.read(5)
    except OSError:
        return None

    if signature != b"%PDF-":
        return None

    actual_size = pdf_path.stat().st_size
    recorded_size = metadata.get("size_bytes")

    if recorded_size is not None:
        try:
            if actual_size != int(recorded_size):
                return None
        except (TypeError, ValueError):
            return None

    actual_sha256 = sha256_file(pdf_path)

    if actual_sha256 != str(metadata.get("sha256", "")):
        return None

    return metadata


def download_pdf(
    *,
    client: httpx.Client,
    manifest_record: dict[str, str],
    pdf_directory: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Download one PDF atomically or reuse a validated cached copy."""
    source_record_id = manifest_record.get(
        "source_record_id",
        "",
    ).strip()
    pdf_url = manifest_record.get("pdf_url", "").strip()

    _validate_source_record_id(source_record_id)

    if not pdf_url:
        raise PdfPipelineError(f"Manifest record has no PDF URL: {source_record_id}")

    pdf_directory.mkdir(parents=True, exist_ok=True)

    pdf_path = pdf_directory / f"{source_record_id}.pdf"
    metadata_path = pdf_directory / f"{source_record_id}.metadata.json"

    if not force:
        cached_metadata = _cached_pdf_metadata(
            pdf_path=pdf_path,
            metadata_path=metadata_path,
            expected_pdf_url=pdf_url,
        )

        if cached_metadata is not None:
            result = dict(cached_metadata)
            result.update(
                {
                    "pdf_path": str(pdf_path),
                    "metadata_path": str(metadata_path),
                    "reused": True,
                }
            )
            return result

    temporary_path = pdf_path.with_suffix(".pdf.part")
    last_error: Exception | None = None

    for attempt in range(
        1,
        MAX_DOWNLOAD_ATTEMPTS + 1,
    ):
        temporary_path.unlink(missing_ok=True)

        hasher = hashlib.sha256()
        first_bytes = b""
        size_bytes = 0

        try:
            with client.stream("GET", pdf_url) as response:
                response.raise_for_status()

                with temporary_path.open("wb") as output_file:
                    for chunk in response.iter_bytes():
                        if not chunk:
                            continue

                        if len(first_bytes) < 8:
                            first_bytes = (first_bytes + chunk)[:8]

                        output_file.write(chunk)
                        hasher.update(chunk)
                        size_bytes += len(chunk)

                if not first_bytes.startswith(b"%PDF-"):
                    raise PdfPipelineError(
                        "Downloaded content does not begin with a PDF signature."
                    )

                temporary_path.replace(pdf_path)

                metadata: dict[str, Any] = {
                    "source_record_id": source_record_id,
                    "source_url": manifest_record.get("source_url"),
                    "pdf_url": pdf_url,
                    "final_url": str(response.url),
                    "session_date": manifest_record.get("session_date"),
                    "period": manifest_record.get("period"),
                    "meeting_number": manifest_record.get("meeting_number"),
                    "retrieved_at_utc": datetime.now(UTC).isoformat(),
                    "http_status": response.status_code,
                    "content_type": response.headers.get("content-type"),
                    "size_bytes": size_bytes,
                    "sha256": hasher.hexdigest(),
                    "retrieval_mode": "http",
                    "download_attempt": attempt,
                }

                _write_json_atomic(
                    path=metadata_path,
                    payload=metadata,
                )

                result = dict(metadata)
                result.update(
                    {
                        "pdf_path": str(pdf_path),
                        "metadata_path": str(metadata_path),
                        "reused": False,
                    }
                )
                return result

        except (
            httpx.HTTPError,
            OSError,
            PdfPipelineError,
        ) as error:
            last_error = error
            temporary_path.unlink(missing_ok=True)

            if attempt < MAX_DOWNLOAD_ATTEMPTS:
                time.sleep(2**attempt)

    raise PdfPipelineError(
        f"Could not download PDF after {MAX_DOWNLOAD_ATTEMPTS} attempts: {pdf_url}"
    ) from last_error


def _output_matches_metadata(
    *,
    path: Path,
    expected_sha256: object,
    expected_size_bytes: object,
) -> bool:
    """Return whether an output matches its recorded hash and size."""
    if not path.exists():
        return False

    if not isinstance(expected_sha256, str):
        return False

    if isinstance(expected_size_bytes, bool):
        return False

    if isinstance(expected_size_bytes, int):
        recorded_size = expected_size_bytes
    elif isinstance(expected_size_bytes, str):
        try:
            recorded_size = int(expected_size_bytes)
        except ValueError:
            return False
    else:
        return False

    if path.stat().st_size != recorded_size:
        return False

    return sha256_file(path) == expected_sha256


def _reusable_extraction_summary(
    *,
    summary_path: Path,
    pages_path: Path,
    blocks_path: Path,
    pdf_sha256: str,
) -> dict[str, Any] | None:
    """Return an existing extraction summary when outputs remain valid."""
    if not (summary_path.exists() and pages_path.exists() and blocks_path.exists()):
        return None

    try:
        summary = _read_json_object(summary_path)
    except PdfPipelineError:
        return None

    if str(summary.get("extractor_version", "")) != EXTRACTOR_VERSION:
        return None

    pdf_section = summary.get("pdf")

    if not isinstance(pdf_section, dict):
        return None

    if str(pdf_section.get("sha256", "")) != pdf_sha256:
        return None

    extraction_section = summary.get("extraction")

    if not isinstance(extraction_section, dict):
        return None

    if not _output_matches_metadata(
        path=pages_path,
        expected_sha256=extraction_section.get("pages_sha256"),
        expected_size_bytes=extraction_section.get("pages_size_bytes"),
    ):
        return None

    if not _output_matches_metadata(
        path=blocks_path,
        expected_sha256=extraction_section.get("blocks_sha256"),
        expected_size_bytes=extraction_section.get("blocks_size_bytes"),
    ):
        return None

    return summary


def extract_document(
    *,
    pdf_path: Path,
    manifest_record: dict[str, str],
    output_root: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Extract ordered pages and blocks from one PDF."""
    if not pdf_path.exists():
        raise PdfPipelineError(f"PDF does not exist: {pdf_path}")

    source_record_id = manifest_record.get(
        "source_record_id",
        "",
    ).strip()
    _validate_source_record_id(source_record_id)

    pdf_sha256 = sha256_file(pdf_path)
    output_directory = output_root / source_record_id
    output_directory.mkdir(parents=True, exist_ok=True)

    pages_path = output_directory / "pages.jsonl"
    blocks_path = output_directory / "blocks.jsonl"
    summary_path = output_directory / "document.json"

    if not force:
        cached_summary = _reusable_extraction_summary(
            summary_path=summary_path,
            pages_path=pages_path,
            blocks_path=blocks_path,
            pdf_sha256=pdf_sha256,
        )

        if cached_summary is not None:
            result = dict(cached_summary)
            result["reused"] = True
            return result

    pages_temporary_path = pages_path.with_suffix(".jsonl.part")
    blocks_temporary_path = blocks_path.with_suffix(".jsonl.part")

    pages_temporary_path.unlink(missing_ok=True)
    blocks_temporary_path.unlink(missing_ok=True)

    layout_counts: Counter[str] = Counter()
    empty_page_numbers: list[int] = []
    low_text_page_numbers: list[int] = []

    total_text_blocks = 0
    total_characters = 0
    total_words = 0

    try:
        with pymupdf.open(str(pdf_path)) as document:  # type: ignore[no-untyped-call]
            if not document.is_pdf:
                raise PdfPipelineError(f"PyMuPDF did not identify a PDF: {pdf_path}")

            if document.needs_pass:
                raise PdfPipelineError(f"PDF requires a password: {pdf_path}")

            page_count = document.page_count
            document_metadata = dict(document.metadata or {})

            with (
                pages_temporary_path.open(
                    "w",
                    encoding="utf-8",
                ) as pages_file,
                blocks_temporary_path.open(
                    "w",
                    encoding="utf-8",
                ) as blocks_file,
            ):
                for page_index in range(page_count):
                    page = document.load_page(page_index)
                    page_number = page_index + 1

                    ordered_page = extract_ordered_page(
                        raw_blocks=page.get_text(
                            "blocks",
                            sort=False,
                        ),
                        page_number=page_number,
                        page_width=float(page.rect.width),
                        page_height=float(page.rect.height),
                    )

                    page_text = ordered_page.text
                    character_count = len(page_text)
                    word_count = len(page_text.split())
                    text_block_count = len(ordered_page.blocks)

                    layout_counts[ordered_page.layout.value] += 1

                    if not page_text.strip():
                        empty_page_numbers.append(page_number)

                    if character_count < 50:
                        low_text_page_numbers.append(page_number)

                    total_text_blocks += text_block_count
                    total_characters += character_count
                    total_words += word_count

                    page_record: dict[str, Any] = {
                        "source_record_id": (source_record_id),
                        "page_number": page_number,
                        "width": round(
                            ordered_page.width,
                            3,
                        ),
                        "height": round(
                            ordered_page.height,
                            3,
                        ),
                        "rotation": int(page.rotation),
                        "layout": (ordered_page.layout.value),
                        "image_block_count": (ordered_page.image_block_count),
                        "text_block_count": (text_block_count),
                        "character_count": (character_count),
                        "word_count": word_count,
                        "text": page_text,
                    }

                    pages_file.write(
                        json.dumps(
                            page_record,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                    pages_file.write("\n")

                    for ordered_block in ordered_page.blocks:
                        block = ordered_block.block

                        block_record: dict[
                            str,
                            Any,
                        ] = {
                            "source_record_id": (source_record_id),
                            "page_number": (page_number),
                            "reading_order": (ordered_block.reading_order),
                            "region": (ordered_block.region.value),
                            "raw_block_number": (block.raw_block_number),
                            "x0": round(block.x0, 3),
                            "y0": round(block.y0, 3),
                            "x1": round(block.x1, 3),
                            "y1": round(block.y1, 3),
                            "text": block.text,
                        }

                        blocks_file.write(
                            json.dumps(
                                block_record,
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                        )
                        blocks_file.write("\n")

        pages_temporary_path.replace(pages_path)
        blocks_temporary_path.replace(blocks_path)
        pages_sha256 = sha256_file(pages_path)
        blocks_sha256 = sha256_file(blocks_path)
        pages_size_bytes = pages_path.stat().st_size
        blocks_size_bytes = blocks_path.stat().st_size

    except Exception:
        pages_temporary_path.unlink(missing_ok=True)
        blocks_temporary_path.unlink(missing_ok=True)
        raise

    summary: dict[str, Any] = {
        "extractor_version": EXTRACTOR_VERSION,
        "extracted_at_utc": datetime.now(UTC).isoformat(),
        "source_record_id": source_record_id,
        "manifest_record": {
            "source_record_id": source_record_id,
            "session_date": manifest_record.get("session_date"),
            "period": manifest_record.get("period"),
            "meeting_number": manifest_record.get("meeting_number"),
            "session_category": manifest_record.get("session_category"),
            "pdf_url": manifest_record.get("pdf_url"),
        },
        "pdf": {
            "path": str(pdf_path),
            "sha256": pdf_sha256,
            "size_bytes": pdf_path.stat().st_size,
            "page_count": page_count,
            "metadata": document_metadata,
        },
        "extraction": {
            "pages_path": str(pages_path),
            "pages_sha256": pages_sha256,
            "pages_size_bytes": pages_size_bytes,
            "blocks_sha256": blocks_sha256,
            "blocks_size_bytes": blocks_size_bytes,
            "blocks_path": str(blocks_path),
            "total_text_blocks": total_text_blocks,
            "total_characters": total_characters,
            "total_words": total_words,
            "layout_counts": dict(sorted(layout_counts.items())),
            "empty_page_numbers": (empty_page_numbers),
            "low_text_page_numbers": (low_text_page_numbers),
        },
    }

    _write_json_atomic(
        path=summary_path,
        payload=summary,
    )

    result = dict(summary)
    result["reused"] = False
    return result
