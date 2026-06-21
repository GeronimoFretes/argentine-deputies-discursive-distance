"""Persistent and resumable structural segmentation outputs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .pdf_pipeline import sha256_file
from .structure import (
    BlockClassification,
    StructuralAnchor,
    StructuralInputBlock,
    StructuralSegmentationError,
    classify_structural_blocks,
)

SEGMENTER_VERSION = "1"


class StructurePipelineError(RuntimeError):
    """Raised when structural segmentation cannot be persisted safely."""


def _read_json_object(
    path: Path,
) -> dict[str, Any]:
    """Read a JSON object."""
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (
        OSError,
        json.JSONDecodeError,
    ) as error:
        raise StructurePipelineError(f"Could not read JSON object: {path}") from error

    if not isinstance(payload, dict):
        raise StructurePipelineError(f"Expected a JSON object: {path}")

    return {str(key): value for key, value in payload.items()}


def _read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    """Read a JSON Lines file."""
    records = []

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as input_file:
            for line_number, line in enumerate(
                input_file,
                start=1,
            ):
                if not line.strip():
                    continue

                try:
                    payload: object = json.loads(line)
                except json.JSONDecodeError as error:
                    raise StructurePipelineError(f"Invalid JSON at {path}:{line_number}") from error

                if not isinstance(payload, dict):
                    raise StructurePipelineError(f"Expected a JSON object at {path}:{line_number}")

                records.append({str(key): value for key, value in payload.items()})

    except OSError as error:
        raise StructurePipelineError(f"Could not read JSONL file: {path}") from error

    return records


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
        raise StructurePipelineError(f"Could not write JSON object: {path}") from error


def _write_jsonl(
    *,
    path: Path,
    records: Iterable[dict[str, Any]],
) -> None:
    """Write JSONL records to a specified path."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        with path.open(
            "w",
            encoding="utf-8",
        ) as output_file:
            for record in records:
                output_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                output_file.write("\n")

    except OSError as error:
        raise StructurePipelineError(f"Could not write JSONL file: {path}") from error


def _safe_int(
    value: object,
    *,
    field_name: str,
) -> int:
    """Convert a JSON metadata value to an integer."""
    if isinstance(value, bool):
        raise StructurePipelineError(f"Invalid integer for {field_name}: {value!r}")

    if isinstance(value, int):
        return value

    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as error:
            raise StructurePipelineError(f"Invalid integer for {field_name}: {value!r}") from error

    raise StructurePipelineError(f"Invalid integer for {field_name}: {value!r}")


def _validate_source_record_id(
    source_record_id: str,
) -> None:
    """Reject identifiers that could escape the output root."""
    if not source_record_id:
        raise StructurePipelineError("Document metadata has no source_record_id.")

    if Path(source_record_id).name != source_record_id:
        raise StructurePipelineError(
            f"source_record_id cannot contain path separators: {source_record_id}"
        )


def _output_matches_metadata(
    *,
    path: Path,
    expected_sha256: object,
    expected_size_bytes: object,
) -> bool:
    """Check whether a file matches recorded hash metadata."""
    if not path.exists():
        return False

    if not isinstance(
        expected_sha256,
        str,
    ):
        return False

    if isinstance(
        expected_size_bytes,
        bool,
    ):
        return False

    if isinstance(
        expected_size_bytes,
        int,
    ):
        recorded_size = expected_size_bytes
    elif isinstance(
        expected_size_bytes,
        str,
    ):
        try:
            recorded_size = int(expected_size_bytes)
        except ValueError:
            return False
    else:
        return False

    if path.stat().st_size != recorded_size:
        return False

    return sha256_file(path) == expected_sha256


def _anchor_payload(
    anchor: StructuralAnchor | None,
) -> dict[str, Any] | None:
    """Serialize a structural anchor."""
    if anchor is None:
        return None

    return {
        "page_number": anchor.page_number,
        "reading_order": (anchor.reading_order),
        "reference": anchor.reference,
        "method": anchor.method,
        "matched_text": (anchor.matched_text),
    }


def _reusable_structure_summary(
    *,
    summary_path: Path,
    structural_blocks_path: Path,
    structural_pages_path: Path,
    source_record_id: str,
    extractor_version: str,
    pdf_sha256: str,
    pages_sha256: str,
    blocks_sha256: str,
) -> dict[str, Any] | None:
    """Return reusable structural outputs when all hashes match."""
    if not (
        summary_path.exists() and structural_blocks_path.exists() and structural_pages_path.exists()
    ):
        return None

    try:
        summary = _read_json_object(summary_path)
    except StructurePipelineError:
        return None

    if (
        str(
            summary.get(
                "segmenter_version",
                "",
            )
        )
        != SEGMENTER_VERSION
    ):
        return None

    if (
        str(
            summary.get(
                "source_record_id",
                "",
            )
        )
        != source_record_id
    ):
        return None

    source = summary.get("source")

    if not isinstance(source, dict):
        return None

    expected_source = {
        "extractor_version": (extractor_version),
        "pdf_sha256": pdf_sha256,
        "pages_sha256": pages_sha256,
        "blocks_sha256": blocks_sha256,
    }

    for key, expected_value in expected_source.items():
        if str(source.get(key, "")) != expected_value:
            return None

    outputs = summary.get("outputs")

    if not isinstance(outputs, dict):
        return None

    if not _output_matches_metadata(
        path=structural_blocks_path,
        expected_sha256=outputs.get("structural_blocks_sha256"),
        expected_size_bytes=outputs.get("structural_blocks_size_bytes"),
    ):
        return None

    if not _output_matches_metadata(
        path=structural_pages_path,
        expected_sha256=outputs.get("structural_pages_sha256"),
        expected_size_bytes=outputs.get("structural_pages_size_bytes"),
    ):
        return None

    return summary


def _classification_fields(
    classification: BlockClassification,
) -> dict[str, Any]:
    """Serialize one block classification."""
    return {
        "structural_zone": (classification.structural_zone.value),
        "content_role": (classification.content_role.value),
        "include_in_discourse": (classification.include_in_discourse),
        "exclusion_reason": (classification.exclusion_reason),
        "classification_method": (classification.classification_method),
        "classification_confidence": (classification.classification_confidence),
    }


def _validate_record_source_id(
    *,
    record: dict[str, Any],
    expected_source_record_id: str,
    path: Path,
) -> None:
    """Ensure an extracted record belongs to its document."""
    actual = str(
        record.get(
            "source_record_id",
            "",
        )
    )

    if actual != expected_source_record_id:
        raise StructurePipelineError(f"Unexpected source_record_id in {path}: {actual!r}")


def _build_page_records(
    *,
    source_record_id: str,
    pages: list[dict[str, Any]],
    structural_blocks: list[dict[str, Any]],
    start_anchor: StructuralAnchor,
    end_anchor: StructuralAnchor | None,
    post_start_anchor: (StructuralAnchor | None),
) -> list[dict[str, Any]]:
    """Create page-level summaries from block classifications."""
    blocks_by_page: dict[
        int,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for block in structural_blocks:
        page_number = _safe_int(
            block.get("page_number"),
            field_name="page_number",
        )
        blocks_by_page[page_number].append(block)

    page_records = []

    for page in sorted(
        pages,
        key=lambda record: _safe_int(
            record.get("page_number"),
            field_name="page_number",
        ),
    ):
        page_number = _safe_int(
            page.get("page_number"),
            field_name="page_number",
        )
        page_blocks = blocks_by_page.get(
            page_number,
            [],
        )

        zone_counts = Counter(str(block["structural_zone"]) for block in page_blocks)
        role_counts = Counter(str(block["content_role"]) for block in page_blocks)

        included_blocks = [block for block in page_blocks if bool(block["include_in_discourse"])]

        included_character_count = sum(len(str(block.get("text", ""))) for block in included_blocks)
        included_word_count = sum(
            len(str(block.get("text", "")).split()) for block in included_blocks
        )

        page_records.append(
            {
                "source_record_id": (source_record_id),
                "page_number": page_number,
                "layout": str(page.get("layout", "")),
                "source_text_block_count": (
                    _safe_int(
                        page.get(
                            "text_block_count",
                            0,
                        ),
                        field_name=("text_block_count"),
                    )
                ),
                "source_character_count": (
                    _safe_int(
                        page.get(
                            "character_count",
                            0,
                        ),
                        field_name=("character_count"),
                    )
                ),
                "source_word_count": (
                    _safe_int(
                        page.get(
                            "word_count",
                            0,
                        ),
                        field_name="word_count",
                    )
                ),
                "classified_block_count": len(page_blocks),
                "included_block_count": len(included_blocks),
                "excluded_block_count": (len(page_blocks) - len(included_blocks)),
                "included_character_count": (included_character_count),
                "included_word_count": (included_word_count),
                "structural_zone_counts": dict(sorted(zone_counts.items())),
                "content_role_counts": dict(sorted(role_counts.items())),
                "mixed_structural_zone": (len(zone_counts) > 1),
                "mixed_content_role": (len(role_counts) > 1),
                "has_included_discourse": (bool(included_blocks)),
                "start_anchor_on_page": (start_anchor.page_number == page_number),
                "end_anchor_on_page": (
                    end_anchor is not None and end_anchor.page_number == page_number
                ),
                "post_start_anchor_on_page": (
                    post_start_anchor is not None and (post_start_anchor.page_number == page_number)
                ),
            }
        )

    return page_records


def segment_document(
    *,
    document_path: Path,
    output_root: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Segment one extracted parliamentary document."""
    if not document_path.exists():
        raise StructurePipelineError(
            f"Extraction document metadata does not exist: {document_path}"
        )

    document_summary = _read_json_object(document_path)

    source_record_id = str(
        document_summary.get(
            "source_record_id",
            "",
        )
    ).strip()
    _validate_source_record_id(source_record_id)

    extractor_version = str(
        document_summary.get(
            "extractor_version",
            "",
        )
    )

    if not extractor_version:
        raise StructurePipelineError("Extraction metadata has no extractor_version.")

    extraction = document_summary.get("extraction")
    pdf = document_summary.get("pdf")

    if not isinstance(
        extraction,
        dict,
    ):
        raise StructurePipelineError("Extraction metadata has no valid extraction section.")

    if not isinstance(pdf, dict):
        raise StructurePipelineError("Extraction metadata has no valid PDF section.")

    pages_path = Path(str(extraction.get("pages_path", "")))
    blocks_path = Path(str(extraction.get("blocks_path", "")))

    pages_sha256 = str(
        extraction.get(
            "pages_sha256",
            "",
        )
    )
    blocks_sha256 = str(
        extraction.get(
            "blocks_sha256",
            "",
        )
    )
    pdf_sha256 = str(pdf.get("sha256", ""))

    if not _output_matches_metadata(
        path=pages_path,
        expected_sha256=pages_sha256,
        expected_size_bytes=extraction.get("pages_size_bytes"),
    ):
        raise StructurePipelineError(
            f"Extracted pages do not match document metadata: {pages_path}"
        )

    if not _output_matches_metadata(
        path=blocks_path,
        expected_sha256=blocks_sha256,
        expected_size_bytes=extraction.get("blocks_size_bytes"),
    ):
        raise StructurePipelineError(
            f"Extracted blocks do not match document metadata: {blocks_path}"
        )

    output_directory = output_root / source_record_id
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    structural_blocks_path = output_directory / "structural_blocks.jsonl"
    structural_pages_path = output_directory / "structural_pages.jsonl"
    summary_path = output_directory / "structure.json"

    if not force:
        cached_summary = _reusable_structure_summary(
            summary_path=summary_path,
            structural_blocks_path=(structural_blocks_path),
            structural_pages_path=(structural_pages_path),
            source_record_id=(source_record_id),
            extractor_version=(extractor_version),
            pdf_sha256=pdf_sha256,
            pages_sha256=pages_sha256,
            blocks_sha256=blocks_sha256,
        )

        if cached_summary is not None:
            result = dict(cached_summary)
            result["reused"] = True
            return result

    pages = _read_jsonl(pages_path)
    raw_blocks = _read_jsonl(blocks_path)

    if not pages:
        raise StructurePipelineError(f"No page records found: {pages_path}")

    for page in pages:
        _validate_record_source_id(
            record=page,
            expected_source_record_id=(source_record_id),
            path=pages_path,
        )

    for block in raw_blocks:
        _validate_record_source_id(
            record=block,
            expected_source_record_id=(source_record_id),
            path=blocks_path,
        )

    ordered_blocks = sorted(
        raw_blocks,
        key=lambda block: (
            _safe_int(
                block.get("page_number"),
                field_name="page_number",
            ),
            _safe_int(
                block.get("reading_order"),
                field_name="reading_order",
            ),
        ),
    )

    page_heights: dict[int, float] = {}

    for page in pages:
        page_number = _safe_int(
            page.get("page_number"),
            field_name="page_number",
        )

        if page_number in page_heights:
            raise StructurePipelineError(f"Duplicate page record: {page_number}")

        try:
            page_heights[page_number] = float(page["height"])
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise StructurePipelineError(f"Invalid page height for page {page_number}.") from error

    structural_inputs = []

    for block in ordered_blocks:
        try:
            structural_inputs.append(
                StructuralInputBlock(
                    page_number=_safe_int(
                        block.get("page_number"),
                        field_name=("page_number"),
                    ),
                    reading_order=_safe_int(
                        block.get("reading_order"),
                        field_name=("reading_order"),
                    ),
                    region=str(block["region"]),
                    y0=float(block["y0"]),
                    y1=float(block["y1"]),
                    text=str(block["text"]),
                )
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            raise StructurePipelineError("Invalid extracted block record.") from error

    try:
        segmentation = classify_structural_blocks(
            blocks=structural_inputs,
            page_heights=page_heights,
        )
    except StructuralSegmentationError as error:
        raise StructurePipelineError(
            f"Structural segmentation failed for {source_record_id}."
        ) from error

    if len(segmentation.classifications) != len(ordered_blocks):
        raise StructurePipelineError("Classification count does not match extracted block count.")

    structural_block_records = []

    for block, classification in zip(
        ordered_blocks,
        segmentation.classifications,
        strict=True,
    ):
        expected_reference = f"p{block['page_number']}:b{block['reading_order']}"

        if classification.reference != expected_reference:
            raise StructurePipelineError("Classification order does not match extracted blocks.")

        record = dict(block)
        record.update(_classification_fields(classification))
        structural_block_records.append(record)

    structural_page_records = _build_page_records(
        source_record_id=source_record_id,
        pages=pages,
        structural_blocks=(structural_block_records),
        start_anchor=(segmentation.start_anchor),
        end_anchor=(segmentation.end_anchor),
        post_start_anchor=(segmentation.post_start_anchor),
    )

    structural_blocks_temporary_path = structural_blocks_path.with_suffix(".jsonl.part")
    structural_pages_temporary_path = structural_pages_path.with_suffix(".jsonl.part")

    structural_blocks_temporary_path.unlink(missing_ok=True)
    structural_pages_temporary_path.unlink(missing_ok=True)

    try:
        _write_jsonl(
            path=(structural_blocks_temporary_path),
            records=(structural_block_records),
        )
        _write_jsonl(
            path=(structural_pages_temporary_path),
            records=(structural_page_records),
        )

        structural_blocks_temporary_path.replace(structural_blocks_path)
        structural_pages_temporary_path.replace(structural_pages_path)

    except Exception:
        structural_blocks_temporary_path.unlink(missing_ok=True)
        structural_pages_temporary_path.unlink(missing_ok=True)
        raise

    structural_blocks_sha256 = sha256_file(structural_blocks_path)
    structural_pages_sha256 = sha256_file(structural_pages_path)

    zone_counts = Counter(str(record["structural_zone"]) for record in structural_block_records)
    role_counts = Counter(str(record["content_role"]) for record in structural_block_records)

    included_records = [
        record for record in structural_block_records if bool(record["include_in_discourse"])
    ]

    mixed_zone_page_numbers = [
        _safe_int(
            page["page_number"],
            field_name="page_number",
        )
        for page in structural_page_records
        if bool(page["mixed_structural_zone"])
    ]
    mixed_role_page_numbers = [
        _safe_int(
            page["page_number"],
            field_name="page_number",
        )
        for page in structural_page_records
        if bool(page["mixed_content_role"])
    ]

    confidence_values = [
        float(record["classification_confidence"]) for record in structural_block_records
    ]

    summary: dict[str, Any] = {
        "segmenter_version": (SEGMENTER_VERSION),
        "segmented_at_utc": (datetime.now(UTC).isoformat()),
        "source_record_id": (source_record_id),
        "source": {
            "document_path": str(document_path),
            "document_sha256": (sha256_file(document_path)),
            "extractor_version": (extractor_version),
            "pdf_sha256": pdf_sha256,
            "pages_path": str(pages_path),
            "pages_sha256": (pages_sha256),
            "blocks_path": str(blocks_path),
            "blocks_sha256": (blocks_sha256),
        },
        "boundaries": {
            "start_anchor": (_anchor_payload(segmentation.start_anchor)),
            "end_anchor": (_anchor_payload(segmentation.end_anchor)),
            "post_start_anchor": (_anchor_payload(segmentation.post_start_anchor)),
            "end_method": (segmentation.end_method),
        },
        "outputs": {
            "structural_blocks_path": str(structural_blocks_path),
            "structural_blocks_sha256": (structural_blocks_sha256),
            "structural_blocks_size_bytes": (structural_blocks_path.stat().st_size),
            "structural_pages_path": str(structural_pages_path),
            "structural_pages_sha256": (structural_pages_sha256),
            "structural_pages_size_bytes": (structural_pages_path.stat().st_size),
        },
        "statistics": {
            "page_count": len(structural_page_records),
            "block_count": len(structural_block_records),
            "included_block_count": len(included_records),
            "excluded_block_count": (len(structural_block_records) - len(included_records)),
            "included_character_count": sum(
                len(
                    str(
                        record.get(
                            "text",
                            "",
                        )
                    )
                )
                for record in included_records
            ),
            "included_word_count": sum(
                len(
                    str(
                        record.get(
                            "text",
                            "",
                        )
                    ).split()
                )
                for record in included_records
            ),
            "structural_zone_counts": dict(sorted(zone_counts.items())),
            "content_role_counts": dict(sorted(role_counts.items())),
            "mixed_structural_zone_page_numbers": (mixed_zone_page_numbers),
            "mixed_content_role_page_numbers": (mixed_role_page_numbers),
            "pages_with_included_discourse": (
                sum(bool(page["has_included_discourse"]) for page in structural_page_records)
            ),
            "minimum_classification_confidence": (
                min(confidence_values) if confidence_values else None
            ),
        },
    }

    _write_json_atomic(
        path=summary_path,
        payload=summary,
    )

    result = dict(summary)
    result["reused"] = False
    return result
