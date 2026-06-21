"""Batch orchestration for structural transcript segmentation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .pdf_pipeline import sha256_file
from .structure_pipeline import (
    StructurePipelineError,
    segment_document,
)

STRUCTURE_BATCH_VERSION = "1"


class StructureBatchError(RuntimeError):
    """Raised when a structural segmentation batch cannot run."""


def _read_json_object(
    path: Path,
) -> dict[str, Any]:
    """Read a JSON object."""
    if not path.exists():
        raise StructureBatchError(f"PDF extraction batch summary does not exist: {path}")

    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (
        OSError,
        json.JSONDecodeError,
    ) as error:
        raise StructureBatchError(f"Could not read PDF extraction batch summary: {path}") from error

    if not isinstance(payload, dict):
        raise StructureBatchError(f"Expected a JSON object in PDF extraction summary: {path}")

    return {str(key): value for key, value in payload.items()}


def _write_json_atomic(
    *,
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write a JSON object atomically."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary_path = path.with_suffix(f"{path.suffix}.part")

    try:
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
    except OSError as error:
        temporary_path.unlink(missing_ok=True)
        raise StructureBatchError(f"Could not write structural batch summary: {path}") from error


def _read_batch_records(
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate and return ordered PDF batch records."""
    raw_records = summary.get("records")

    if not isinstance(
        raw_records,
        list,
    ):
        raise StructureBatchError("PDF extraction summary has no record list.")

    if not raw_records:
        raise StructureBatchError("PDF extraction summary contains no records.")

    records = []
    seen_labels: set[str] = set()
    seen_record_ids: set[str] = set()

    for record_number, raw_record in enumerate(
        raw_records,
        start=1,
    ):
        if not isinstance(
            raw_record,
            dict,
        ):
            raise StructureBatchError(f"Invalid PDF batch record at position {record_number}.")

        record = {str(key): value for key, value in raw_record.items()}

        label = str(record.get("label", "")).strip()
        source_record_id = str(
            record.get(
                "source_record_id",
                "",
            )
        ).strip()
        document_path = str(
            record.get(
                "document_path",
                "",
            )
        ).strip()

        if not (label and source_record_id and document_path):
            raise StructureBatchError(
                "PDF batch record is missing "
                "label, source_record_id, or "
                "document_path at position "
                f"{record_number}."
            )

        if label in seen_labels:
            raise StructureBatchError(f"Duplicate PDF batch label: {label}")

        if source_record_id in seen_record_ids:
            raise StructureBatchError(f"Duplicate PDF batch source_record_id: {source_record_id}")

        seen_labels.add(label)
        seen_record_ids.add(source_record_id)
        records.append(record)

    return records


def _required_mapping(
    *,
    payload: dict[str, Any],
    key: str,
    source_record_id: str,
) -> dict[str, Any]:
    """Return one required nested mapping."""
    value = payload.get(key)

    if not isinstance(value, dict):
        raise StructureBatchError(
            f"Structural result for {source_record_id} has no valid {key} section."
        )

    return {str(nested_key): nested_value for nested_key, nested_value in value.items()}


def _anchor_reference(
    value: object,
) -> str | None:
    """Read an optional serialized anchor reference."""
    if value is None:
        return None

    if not isinstance(value, dict):
        raise StructureBatchError("Invalid structural anchor.")

    reference = str(value.get("reference", "")).strip()

    if not reference:
        raise StructureBatchError("Structural anchor has no reference.")

    return reference


def run_structure_batch(
    *,
    pdf_summary_path: Path,
    output_root: Path,
    summary_path: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Segment every document in a PDF extraction batch."""
    started_at = datetime.now(UTC)

    pdf_batch_summary = _read_json_object(pdf_summary_path)
    source_records = _read_batch_records(pdf_batch_summary)

    result_records = []

    for source_record in source_records:
        label = str(source_record["label"])
        source_record_id = str(source_record["source_record_id"])
        document_path = Path(str(source_record["document_path"]))

        try:
            structure_result = segment_document(
                document_path=(document_path),
                output_root=(output_root),
                force=force,
            )
        except StructurePipelineError as error:
            raise StructureBatchError(
                f"Structural segmentation failed for {label} ({source_record_id})."
            ) from error

        boundaries = _required_mapping(
            payload=structure_result,
            key="boundaries",
            source_record_id=(source_record_id),
        )
        outputs = _required_mapping(
            payload=structure_result,
            key="outputs",
            source_record_id=(source_record_id),
        )
        statistics = _required_mapping(
            payload=structure_result,
            key="statistics",
            source_record_id=(source_record_id),
        )

        result_records.append(
            {
                "label": label,
                "source_record_id": (source_record_id),
                "session_date": (source_record.get("session_date")),
                "period": (source_record.get("period")),
                "meeting_number": (source_record.get("meeting_number")),
                "session_category": (source_record.get("session_category")),
                "segmentation_reused": bool(structure_result["reused"]),
                "start_anchor": (_anchor_reference(boundaries.get("start_anchor"))),
                "end_anchor": (_anchor_reference(boundaries.get("end_anchor"))),
                "post_start_anchor": (_anchor_reference(boundaries.get("post_start_anchor"))),
                "end_method": str(
                    boundaries.get(
                        "end_method",
                        "",
                    )
                ),
                "page_count": int(statistics["page_count"]),
                "block_count": int(statistics["block_count"]),
                "included_block_count": int(statistics["included_block_count"]),
                "excluded_block_count": int(statistics["excluded_block_count"]),
                "included_word_count": int(statistics["included_word_count"]),
                "structural_zone_counts": (statistics["structural_zone_counts"]),
                "content_role_counts": (statistics["content_role_counts"]),
                "mixed_structural_zone_page_numbers": (
                    statistics["mixed_structural_zone_page_numbers"]
                ),
                "mixed_content_role_page_numbers": (statistics["mixed_content_role_page_numbers"]),
                "minimum_classification_confidence": (
                    statistics["minimum_classification_confidence"]
                ),
                "structural_blocks_path": str(outputs["structural_blocks_path"]),
                "structural_pages_path": str(outputs["structural_pages_path"]),
                "structure_path": str(output_root / source_record_id / "structure.json"),
            }
        )

    finished_at = datetime.now(UTC)

    summary: dict[str, Any] = {
        "batch_version": (STRUCTURE_BATCH_VERSION),
        "started_at_utc": (started_at.isoformat()),
        "finished_at_utc": (finished_at.isoformat()),
        "source_pdf_summary": {
            "path": str(pdf_summary_path),
            "sha256": sha256_file(pdf_summary_path),
        },
        "output_root": str(output_root),
        "record_count": len(result_records),
        "segmentation_reused_count": (
            sum(bool(record["segmentation_reused"]) for record in result_records)
        ),
        "records": result_records,
    }

    _write_json_atomic(
        path=summary_path,
        payload=summary,
    )

    return summary
