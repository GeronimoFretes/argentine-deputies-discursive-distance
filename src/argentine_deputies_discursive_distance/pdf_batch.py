"""Batch orchestration for selected official session PDFs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from .pdf_pipeline import (
    PdfPipelineError,
    download_pdf,
    extract_document,
    sha256_file,
)

PDF_USER_AGENT = (
    "argentine-deputies-discursive-distance/0.1 (academic research; public parliamentary records)"
)

BATCH_VERSION = "1"


class PdfBatchError(RuntimeError):
    """Raised when a selected PDF batch cannot be resolved or processed."""


@dataclass(frozen=True, slots=True)
class PdfSelection:
    """One named manifest record selected for PDF processing."""

    label: str
    source_record_id: str


def _write_json_atomic(
    *,
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write a JSON object atomically."""
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


def _normalized_csv_row(
    row: dict[str, str | None],
) -> dict[str, str]:
    """Normalize CSV keys and nullable values."""
    return {
        str(key).strip(): (value.strip() if value is not None else "")
        for key, value in row.items()
        if key is not None
    }


def read_manifest_records(
    path: Path,
) -> dict[str, dict[str, str]]:
    """Read manifest records indexed by stable source identifier."""
    if not path.exists():
        raise PdfBatchError(f"Session manifest does not exist: {path}")

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as input_file:
            reader = csv.DictReader(input_file)

            required_columns = {
                "source_record_id",
                "pdf_url",
            }
            available_columns = set(reader.fieldnames or [])

            missing_columns = required_columns - available_columns

            if missing_columns:
                raise PdfBatchError(
                    f"Session manifest is missing required columns: {sorted(missing_columns)}"
                )

            records: dict[str, dict[str, str]] = {}

            for row_number, raw_row in enumerate(
                reader,
                start=2,
            ):
                row = _normalized_csv_row(raw_row)
                source_record_id = row["source_record_id"]

                if not source_record_id:
                    raise PdfBatchError(
                        f"Manifest contains an empty source_record_id at row {row_number}."
                    )

                if source_record_id in records:
                    raise PdfBatchError(
                        f"Manifest contains duplicate source_record_id: {source_record_id}"
                    )

                records[source_record_id] = row

    except OSError as error:
        raise PdfBatchError(f"Could not read session manifest: {path}") from error

    return records


def read_pdf_selections(
    path: Path,
) -> list[PdfSelection]:
    """Read an ordered list of selected source records."""
    if not path.exists():
        raise PdfBatchError(f"PDF selection file does not exist: {path}")

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as input_file:
            reader = csv.DictReader(input_file)

            required_columns = {
                "label",
                "source_record_id",
            }
            available_columns = set(reader.fieldnames or [])
            missing_columns = required_columns - available_columns

            if missing_columns:
                raise PdfBatchError(
                    f"PDF selection file is missing columns: {sorted(missing_columns)}"
                )

            selections: list[PdfSelection] = []
            seen_labels: set[str] = set()
            seen_record_ids: set[str] = set()

            for row_number, raw_row in enumerate(
                reader,
                start=2,
            ):
                row = _normalized_csv_row(raw_row)
                label = row["label"]
                source_record_id = row["source_record_id"]

                if not label or not source_record_id:
                    raise PdfBatchError(
                        "PDF selection contains an empty label "
                        "or source_record_id at "
                        f"row {row_number}."
                    )

                if label in seen_labels:
                    raise PdfBatchError(f"Duplicate PDF selection label: {label}")

                if source_record_id in seen_record_ids:
                    raise PdfBatchError(f"Duplicate selected source_record_id: {source_record_id}")

                seen_labels.add(label)
                seen_record_ids.add(source_record_id)

                selections.append(
                    PdfSelection(
                        label=label,
                        source_record_id=source_record_id,
                    )
                )

    except OSError as error:
        raise PdfBatchError(f"Could not read PDF selection file: {path}") from error

    if not selections:
        raise PdfBatchError(f"PDF selection file is empty: {path}")

    return selections


def resolve_pdf_selections(
    *,
    manifest_records: dict[
        str,
        dict[str, str],
    ],
    selections: list[PdfSelection],
) -> list[tuple[PdfSelection, dict[str, str]]]:
    """Resolve each selection to exactly one PDF-bearing manifest record."""
    resolved = []

    for selection in selections:
        manifest_record = manifest_records.get(selection.source_record_id)

        if manifest_record is None:
            raise PdfBatchError(
                "Selected source_record_id does not exist "
                "in the manifest: "
                f"{selection.source_record_id}"
            )

        if not manifest_record.get("pdf_url", "").strip():
            raise PdfBatchError(
                f"Selected manifest record has no PDF URL: {selection.source_record_id}"
            )

        resolved.append(
            (
                selection,
                manifest_record,
            )
        )

    return resolved


def run_pdf_batch(
    *,
    client: httpx.Client,
    manifest_path: Path,
    selection_path: Path,
    pdf_directory: Path,
    extraction_root: Path,
    summary_path: Path,
    force_download: bool = False,
    force_extract: bool = False,
) -> dict[str, Any]:
    """Download and extract every selected PDF in deterministic order."""
    started_at = datetime.now(UTC)

    manifest_records = read_manifest_records(manifest_path)
    selections = read_pdf_selections(selection_path)
    resolved = resolve_pdf_selections(
        manifest_records=manifest_records,
        selections=selections,
    )

    result_records: list[dict[str, Any]] = []

    for selection, manifest_record in resolved:
        try:
            download_result = download_pdf(
                client=client,
                manifest_record=manifest_record,
                pdf_directory=pdf_directory,
                force=force_download,
            )

            extraction_result = extract_document(
                pdf_path=Path(str(download_result["pdf_path"])),
                manifest_record=manifest_record,
                output_root=extraction_root,
                force=force_extract,
            )

        except PdfPipelineError as error:
            raise PdfBatchError(
                f"PDF processing failed for {selection.label} ({selection.source_record_id})."
            ) from error

        result_records.append(
            {
                "label": selection.label,
                "source_record_id": (selection.source_record_id),
                "session_date": manifest_record.get("session_date"),
                "period": manifest_record.get("period"),
                "meeting_number": manifest_record.get("meeting_number"),
                "session_category": manifest_record.get("session_category"),
                "download_reused": bool(download_result["reused"]),
                "extraction_reused": bool(extraction_result["reused"]),
                "pdf_sha256": str(download_result["sha256"]),
                "pdf_size_bytes": int(download_result["size_bytes"]),
                "page_count": int(extraction_result["pdf"]["page_count"]),
                "total_words": int(extraction_result["extraction"]["total_words"]),
                "layout_counts": extraction_result["extraction"]["layout_counts"],
                "empty_page_numbers": extraction_result["extraction"]["empty_page_numbers"],
                "low_text_page_numbers": extraction_result["extraction"]["low_text_page_numbers"],
                "pdf_path": str(download_result["pdf_path"]),
                "document_path": str(
                    extraction_root / selection.source_record_id / "document.json"
                ),
            }
        )

    finished_at = datetime.now(UTC)

    summary: dict[str, Any] = {
        "batch_version": BATCH_VERSION,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "selection": {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
        },
        "pdf_directory": str(pdf_directory),
        "extraction_root": str(extraction_root),
        "record_count": len(result_records),
        "download_reused_count": sum(bool(record["download_reused"]) for record in result_records),
        "extraction_reused_count": sum(
            bool(record["extraction_reused"]) for record in result_records
        ),
        "records": result_records,
    }

    _write_json_atomic(
        path=summary_path,
        payload=summary,
    )

    return summary
