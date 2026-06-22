"""Persistent and resumable speaker-turn outputs for one document."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .pdf_pipeline import sha256_file
from .speaker import SpeakerMarker
from .speaker_turns import (
    SpeakerTurn,
    SpeakerTurnError,
    SpeakerTurnInputBlock,
    SpeakerTurnParseResult,
    SpeakerTurnSegment,
    parse_speaker_turns,
)
from .turn_content import (
    DocumentaryBoundary,
    TurnContentError,
    TurnContentKind,
    TurnContentResult,
    TurnContentSpan,
    classify_speaker_turn_content,
)

SPEAKER_TURN_PIPELINE_VERSION = "1"

_COUNT_STATISTIC_FIELDS = (
    "turn_count",
    "explicit_marker_count",
    "unattributed_turn_count",
    "unattributed_segment_count",
    "barrier_reset_count",
    "assigned_segment_count",
    "assigned_character_count",
    "content_span_count",
    "speech_span_count",
    "speech_word_count",
    "documentary_span_count",
    "documentary_word_count",
    "stage_direction_span_count",
    "stage_direction_word_count",
    "editorial_note_span_count",
    "editorial_note_word_count",
    "unattributed_content_span_count",
    "unattributed_content_word_count",
    "zero_speech_turn_count",
    "maximum_speech_word_count",
)
_MAPPING_STATISTIC_FIELDS = (
    "speaker_family_counts",
    "content_kind_counts",
    "attribution_method_counts",
)


class SpeakerTurnPipelineError(RuntimeError):
    """Raised when speaker-turn outputs cannot be persisted safely."""


@dataclass(frozen=True, slots=True)
class _ValidatedSource:
    """Validated structural source and parser-ready proceedings blocks."""

    source_record_id: str
    segmenter_version: str
    structure_sha256: str
    structural_blocks_path: Path
    structural_blocks_sha256: str
    structural_blocks_size_bytes: int
    source_blocks: dict[tuple[int, int], str]
    input_blocks: tuple[SpeakerTurnInputBlock, ...]


@dataclass(frozen=True, slots=True)
class _SerializedDocument:
    """Fully reconciled records for one document."""

    turns: tuple[dict[str, Any], ...]
    segments: tuple[dict[str, Any], ...]
    content_spans: tuple[dict[str, Any], ...]
    statistics: dict[str, Any]


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object from a required file."""
    if not path.is_file():
        raise SpeakerTurnPipelineError(f"JSON source does not exist: {path}")

    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SpeakerTurnPipelineError(f"Could not read JSON object: {path}") from error

    if not isinstance(payload, dict):
        raise SpeakerTurnPipelineError(f"Expected a JSON object: {path}")

    return {str(key): value for key, value in payload.items()}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read and validate JSON objects from a JSON Lines file."""
    records = []

    try:
        with path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue

                try:
                    payload: object = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SpeakerTurnPipelineError(
                        f"Invalid JSON at {path}:{line_number}"
                    ) from error

                if not isinstance(payload, dict):
                    raise SpeakerTurnPipelineError(
                        f"Expected a JSON object at {path}:{line_number}"
                    )

                records.append({str(key): value for key, value in payload.items()})
    except OSError as error:
        raise SpeakerTurnPipelineError(f"Could not read JSONL file: {path}") from error

    return records


def _safe_int(value: object, *, field_name: str) -> int:
    """Convert an integer or integer string while rejecting booleans."""
    if isinstance(value, bool):
        raise SpeakerTurnPipelineError(f"Invalid integer for {field_name}: {value!r}")

    if isinstance(value, int):
        return value

    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as error:
            raise SpeakerTurnPipelineError(
                f"Invalid integer for {field_name}: {value!r}"
            ) from error

    raise SpeakerTurnPipelineError(f"Invalid integer for {field_name}: {value!r}")


def _validate_source_record_id(value: object) -> str:
    """Return a safe single-path-component source record identifier."""
    if not isinstance(value, str):
        raise SpeakerTurnPipelineError("Structural summary has no valid source_record_id.")

    source_record_id = value.strip()
    source_component = Path(source_record_id)

    if (
        not source_record_id
        or source_record_id in {".", ".."}
        or source_component.is_absolute()
        or source_component.name != source_record_id
    ):
        raise SpeakerTurnPipelineError(f"Unsafe source_record_id: {source_record_id!r}")

    return source_record_id


def _required_nonempty_string(value: object, *, field_name: str) -> str:
    """Return a required non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise SpeakerTurnPipelineError(f"Missing or invalid {field_name}.")

    return value.strip()


def _required_mapping(payload: dict[str, Any], *, field_name: str) -> dict[str, Any]:
    """Return a required JSON mapping."""
    value = payload.get(field_name)

    if not isinstance(value, dict):
        raise SpeakerTurnPipelineError(f"Missing or invalid {field_name} mapping.")

    return {str(key): nested_value for key, nested_value in value.items()}


def _validate_structural_block(
    *,
    record: dict[str, Any],
    source_record_id: str,
    path: Path,
) -> SpeakerTurnInputBlock:
    """Validate one structural block and return the parser input shape."""
    actual_source_record_id = record.get("source_record_id")

    if actual_source_record_id != source_record_id:
        raise SpeakerTurnPipelineError(
            f"Unexpected source_record_id in {path}: {actual_source_record_id!r}"
        )

    page_number = _safe_int(record.get("page_number"), field_name="page_number")
    reading_order = _safe_int(record.get("reading_order"), field_name="reading_order")

    if page_number < 1 or reading_order < 1:
        raise SpeakerTurnPipelineError(
            "Structural block references must use positive page_number and reading_order."
        )

    include_in_discourse = record.get("include_in_discourse")

    if not isinstance(include_in_discourse, bool):
        raise SpeakerTurnPipelineError(
            f"Invalid include_in_discourse at p{page_number}:b{reading_order}."
        )

    string_fields = {}

    for field_name in ("structural_zone", "content_role", "text"):
        value = record.get(field_name)

        if not isinstance(value, str):
            raise SpeakerTurnPipelineError(
                f"Invalid {field_name} at p{page_number}:b{reading_order}."
            )

        string_fields[field_name] = value

    return SpeakerTurnInputBlock(
        page_number=page_number,
        reading_order=reading_order,
        structural_zone=string_fields["structural_zone"],
        content_role=string_fields["content_role"],
        include_in_discourse=include_in_discourse,
        text=string_fields["text"],
    )


def _validate_source(structure_path: Path) -> _ValidatedSource:
    """Validate the current structural summary and every structural block."""
    structure = _read_json_object(structure_path)
    source_record_id = _validate_source_record_id(structure.get("source_record_id"))
    segmenter_version = _required_nonempty_string(
        structure.get("segmenter_version"),
        field_name="segmenter_version",
    )
    outputs = _required_mapping(structure, field_name="outputs")
    structural_blocks_path = Path(
        _required_nonempty_string(
            outputs.get("structural_blocks_path"),
            field_name="outputs.structural_blocks_path",
        )
    )
    expected_sha256 = _required_nonempty_string(
        outputs.get("structural_blocks_sha256"),
        field_name="outputs.structural_blocks_sha256",
    )
    expected_size = _safe_int(
        outputs.get("structural_blocks_size_bytes"),
        field_name="outputs.structural_blocks_size_bytes",
    )

    if expected_size < 0:
        raise SpeakerTurnPipelineError("Structural-block size cannot be negative.")

    if not structural_blocks_path.is_file():
        raise SpeakerTurnPipelineError(
            f"Structural-block file does not exist for {source_record_id}: {structural_blocks_path}"
        )

    try:
        actual_size = structural_blocks_path.stat().st_size
        actual_sha256 = sha256_file(structural_blocks_path)
        structure_sha256 = sha256_file(structure_path)
    except OSError as error:
        raise SpeakerTurnPipelineError(
            f"Could not validate structural sources for {source_record_id}."
        ) from error

    if actual_size != expected_size:
        raise SpeakerTurnPipelineError(
            f"Structural-block size does not match metadata for {source_record_id}."
        )

    if actual_sha256 != expected_sha256:
        raise SpeakerTurnPipelineError(
            f"Structural-block SHA-256 does not match metadata for {source_record_id}."
        )

    raw_blocks = _read_jsonl(structural_blocks_path)
    references: set[tuple[int, int]] = set()
    source_blocks: dict[tuple[int, int], str] = {}
    proceedings_blocks = []

    for record in raw_blocks:
        block = _validate_structural_block(
            record=record,
            source_record_id=source_record_id,
            path=structural_blocks_path,
        )
        reference = (block.page_number, block.reading_order)

        if reference in references:
            raise SpeakerTurnPipelineError(
                f"Duplicate page and reading-order reference: {block.reference}"
            )

        references.add(reference)
        source_blocks[reference] = block.text

        if block.structural_zone == "proceedings":
            proceedings_blocks.append(block)

    if not proceedings_blocks:
        raise SpeakerTurnPipelineError(f"No proceedings blocks found for {source_record_id}.")

    return _ValidatedSource(
        source_record_id=source_record_id,
        segmenter_version=segmenter_version,
        structure_sha256=structure_sha256,
        structural_blocks_path=structural_blocks_path,
        structural_blocks_sha256=actual_sha256,
        structural_blocks_size_bytes=actual_size,
        source_blocks=source_blocks,
        input_blocks=tuple(
            sorted(
                proceedings_blocks,
                key=lambda block: (block.page_number, block.reading_order),
            )
        ),
    )


def _marker_payload(marker: SpeakerMarker | None) -> dict[str, Any] | None:
    """Serialize one speaker marker without losing block-relative offsets."""
    if marker is None:
        return None

    return {
        "start": marker.start,
        "end": marker.end,
        "raw_marker": marker.raw_marker,
        "raw_title": marker.raw_title,
        "normalized_title": marker.normalized_title,
        "raw_label": marker.raw_label,
        "normalized_label": marker.normalized_label,
        "family": marker.family.value,
        "separator": marker.separator,
        "separator_kind": marker.separator_kind.value,
        "position": marker.position.value,
        "is_multiline": marker.is_multiline,
        "detection_method": marker.detection_method,
        "detection_confidence": marker.detection_confidence,
    }


def _boundary_payload(boundary: DocumentaryBoundary | None) -> dict[str, Any] | None:
    """Serialize one documentary boundary."""
    if boundary is None:
        return None

    return {
        "turn_offset": boundary.turn_offset,
        "page_number": boundary.page_number,
        "reading_order": boundary.reading_order,
        "source_reference": boundary.source_reference,
        "source_offset": boundary.source_offset,
        "cue": boundary.cue.value,
        "matched_text": boundary.matched_text,
        "classification_method": boundary.classification_method,
        "classification_confidence": boundary.classification_confidence,
        "documentary_word_count": boundary.documentary_word_count,
    }


def _validate_marker_source(*, turn: SpeakerTurn, source: _ValidatedSource) -> None:
    """Validate an explicit marker against its source block."""
    marker = turn.marker

    if marker is None:
        if any(
            value is not None
            for value in (
                turn.marker_page_number,
                turn.marker_reading_order,
                turn.marker_block_included,
            )
        ):
            raise SpeakerTurnPipelineError(
                f"Unattributed turn {turn.turn_index} has marker metadata."
            )
        return

    if turn.marker_page_number is None or turn.marker_reading_order is None:
        raise SpeakerTurnPipelineError(
            f"Explicit turn {turn.turn_index} has no marker source reference."
        )

    reference = (turn.marker_page_number, turn.marker_reading_order)
    source_text = source.source_blocks.get(reference)

    if source_text is None:
        raise SpeakerTurnPipelineError(
            f"Marker source block is missing for turn {turn.turn_index}."
        )

    if not (0 <= marker.start < marker.end <= len(source_text)):
        raise SpeakerTurnPipelineError(f"Invalid marker source range for turn {turn.turn_index}.")

    if source_text[marker.start : marker.end] != marker.raw_marker:
        raise SpeakerTurnPipelineError(
            f"Marker source text does not match for turn {turn.turn_index}."
        )

    input_block = next(
        (
            block
            for block in source.input_blocks
            if (block.page_number, block.reading_order) == reference
        ),
        None,
    )

    if input_block is None or turn.marker_block_included != input_block.include_in_discourse:
        raise SpeakerTurnPipelineError(
            f"Marker inclusion metadata does not match for turn {turn.turn_index}."
        )


def _segment_record(
    *,
    source_record_id: str,
    turn_index: int,
    segment_index: int,
    segment: SpeakerTurnSegment,
) -> dict[str, Any]:
    """Serialize one exact turn segment."""
    return {
        "source_record_id": source_record_id,
        "turn_index": turn_index,
        "segment_index": segment_index,
        "page_number": segment.page_number,
        "reading_order": segment.reading_order,
        "block_reference": segment.block_reference,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "attribution_method": segment.attribution_method.value,
        "character_count": len(segment.text),
        "word_count": len(segment.text.split()),
    }


def _containing_segment_index(*, turn: SpeakerTurn, span: TurnContentSpan) -> int:
    """Return the unique one-based segment containing a content span."""
    candidates = [
        segment_index
        for segment_index, segment in enumerate(turn.segments, start=1)
        if (
            segment.page_number == span.page_number
            and segment.reading_order == span.reading_order
            and segment.start <= span.start
            and span.end <= segment.end
        )
    ]

    if len(candidates) != 1:
        raise SpeakerTurnPipelineError(
            f"Content span in turn {turn.turn_index} does not map to exactly one segment."
        )

    return candidates[0]


def _content_span_record(
    *,
    source_record_id: str,
    turn: SpeakerTurn,
    content_span_index: int,
    span: TurnContentSpan,
) -> dict[str, Any]:
    """Serialize one exact classified source span."""
    return {
        "source_record_id": source_record_id,
        "turn_index": turn.turn_index,
        "content_span_index": content_span_index,
        "source_segment_index": _containing_segment_index(turn=turn, span=span),
        "page_number": span.page_number,
        "reading_order": span.reading_order,
        "block_reference": span.block_reference,
        "start": span.start,
        "end": span.end,
        "text": span.text,
        "content_kind": span.content_kind.value,
        "include_in_speech": span.include_in_speech,
        "classification_method": span.classification_method,
        "classification_confidence": span.classification_confidence,
        "attribution_method": span.attribution_method.value,
        "character_count": len(span.text),
        "word_count": span.word_count,
    }


def _validate_segment_and_span_provenance(
    *,
    turn: SpeakerTurn,
    segment_records: list[dict[str, Any]],
    content_span_records: list[dict[str, Any]],
    source: _ValidatedSource,
) -> None:
    """Validate exact source substrings and lossless per-segment coverage."""
    for expected_index, (segment, record) in enumerate(
        zip(turn.segments, segment_records, strict=True),
        start=1,
    ):
        if record["segment_index"] != expected_index:
            raise SpeakerTurnPipelineError(
                f"Non-consecutive segment index in turn {turn.turn_index}."
            )

        source_text = source.source_blocks.get((segment.page_number, segment.reading_order))

        if source_text is None or not (0 <= segment.start <= segment.end <= len(source_text)):
            raise SpeakerTurnPipelineError(
                f"Invalid source segment range in turn {turn.turn_index}."
            )

        if segment.text != source_text[segment.start : segment.end]:
            raise SpeakerTurnPipelineError(
                f"Segment source text does not match in turn {turn.turn_index}."
            )

        linked_spans = [
            span for span in content_span_records if span["source_segment_index"] == expected_index
        ]

        if not linked_spans:
            raise SpeakerTurnPipelineError(
                f"Segment {expected_index} in turn {turn.turn_index} has no content spans."
            )

        if linked_spans[0]["start"] != segment.start:
            raise SpeakerTurnPipelineError(
                f"Content spans do not start at segment {expected_index} in turn {turn.turn_index}."
            )

        cursor = segment.start

        for span_record in linked_spans:
            if span_record["start"] != cursor:
                raise SpeakerTurnPipelineError(
                    f"Content spans have a gap or overlap in turn {turn.turn_index}."
                )

            source_span_text = source_text[span_record["start"] : span_record["end"]]

            if span_record["text"] != source_span_text:
                raise SpeakerTurnPipelineError(
                    f"Content span source text does not match in turn {turn.turn_index}."
                )

            cursor = int(span_record["end"])

        if cursor != segment.end:
            raise SpeakerTurnPipelineError(
                f"Content spans do not end at segment {expected_index} in turn {turn.turn_index}."
            )

        if "".join(str(span["text"]) for span in linked_spans) != segment.text:
            raise SpeakerTurnPipelineError(
                f"Content spans do not reconstruct segment {expected_index} "
                f"in turn {turn.turn_index}."
            )


def _turn_record(
    *,
    source_record_id: str,
    turn: SpeakerTurn,
    content: TurnContentResult,
) -> dict[str, Any]:
    """Serialize one turn and its reconciled content statistics."""
    speech_spans = [span for span in content.spans if span.include_in_speech]
    documentary_spans = [
        span for span in content.spans if span.content_kind == TurnContentKind.DOCUMENTARY_INSERT
    ]
    stage_spans = [
        span for span in content.spans if span.content_kind == TurnContentKind.STAGE_DIRECTION
    ]
    editorial_spans = [
        span for span in content.spans if span.content_kind == TurnContentKind.EDITORIAL_NOTE
    ]

    return {
        "source_record_id": source_record_id,
        "turn_index": turn.turn_index,
        "marker": _marker_payload(turn.marker),
        "marker_page_number": turn.marker_page_number,
        "marker_reading_order": turn.marker_reading_order,
        "marker_block_reference": turn.marker_block_reference,
        "marker_block_included": turn.marker_block_included,
        "normalized_label": turn.normalized_label,
        "speaker_family": turn.speaker_family.value if turn.speaker_family is not None else None,
        "is_unattributed": turn.is_unattributed,
        "segment_count": len(turn.segments),
        "character_count": turn.character_count,
        "word_count": turn.word_count,
        "content_span_count": len(content.spans),
        "speech_span_count": len(speech_spans),
        "speech_word_count": sum(span.word_count for span in speech_spans),
        "documentary_word_count": sum(span.word_count for span in documentary_spans),
        "stage_direction_word_count": sum(span.word_count for span in stage_spans),
        "editorial_note_word_count": sum(span.word_count for span in editorial_spans),
        "first_reference": turn.segments[0].block_reference if turn.segments else None,
        "last_reference": turn.segments[-1].block_reference if turn.segments else None,
        "documentary_boundary": _boundary_payload(content.documentary_boundary),
    }


def _document_statistics(
    *,
    parse_result: SpeakerTurnParseResult,
    turn_records: list[dict[str, Any]],
    segment_records: list[dict[str, Any]],
    content_span_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build all required per-document statistics from persisted records."""
    content_kind_counts = Counter(str(record["content_kind"]) for record in content_span_records)
    attribution_method_counts = Counter(
        str(record["attribution_method"]) for record in segment_records
    )
    speaker_family_counts = Counter(
        str(record["speaker_family"])
        for record in turn_records
        if record["speaker_family"] is not None
    )

    def spans_of_kind(kind: TurnContentKind) -> list[dict[str, Any]]:
        return [record for record in content_span_records if record["content_kind"] == kind.value]

    speech_spans = [record for record in content_span_records if record["include_in_speech"]]
    documentary_spans = spans_of_kind(TurnContentKind.DOCUMENTARY_INSERT)
    stage_spans = spans_of_kind(TurnContentKind.STAGE_DIRECTION)
    editorial_spans = spans_of_kind(TurnContentKind.EDITORIAL_NOTE)
    unattributed_spans = spans_of_kind(TurnContentKind.UNATTRIBUTED_TEXT)

    statistics: dict[str, Any] = {
        "turn_count": len(turn_records),
        "explicit_marker_count": sum(record["marker"] is not None for record in turn_records),
        "unattributed_turn_count": sum(bool(record["is_unattributed"]) for record in turn_records),
        "unattributed_segment_count": sum(
            int(record["segment_count"]) for record in turn_records if record["is_unattributed"]
        ),
        "barrier_reset_count": parse_result.barrier_reset_count,
        "assigned_segment_count": len(segment_records),
        "assigned_character_count": sum(
            int(record["character_count"]) for record in segment_records
        ),
        "content_span_count": len(content_span_records),
        "speech_span_count": len(speech_spans),
        "speech_word_count": sum(int(record["word_count"]) for record in speech_spans),
        "documentary_span_count": len(documentary_spans),
        "documentary_word_count": sum(int(record["word_count"]) for record in documentary_spans),
        "stage_direction_span_count": len(stage_spans),
        "stage_direction_word_count": sum(int(record["word_count"]) for record in stage_spans),
        "editorial_note_span_count": len(editorial_spans),
        "editorial_note_word_count": sum(int(record["word_count"]) for record in editorial_spans),
        "unattributed_content_span_count": len(unattributed_spans),
        "unattributed_content_word_count": sum(
            int(record["word_count"]) for record in unattributed_spans
        ),
        "zero_speech_turn_count": sum(
            int(record["speech_word_count"]) == 0 for record in turn_records
        ),
        "maximum_speech_word_count": max(
            (int(record["speech_word_count"]) for record in turn_records),
            default=0,
        ),
        "speaker_family_counts": dict(sorted(speaker_family_counts.items())),
        "content_kind_counts": dict(sorted(content_kind_counts.items())),
        "attribution_method_counts": dict(sorted(attribution_method_counts.items())),
    }

    expected_parse_statistics = {
        "explicit_marker_count": parse_result.explicit_marker_count,
        "unattributed_turn_count": parse_result.unattributed_turn_count,
        "unattributed_segment_count": parse_result.unattributed_segment_count,
        "barrier_reset_count": parse_result.barrier_reset_count,
        "assigned_segment_count": parse_result.assigned_segment_count,
        "assigned_character_count": parse_result.assigned_character_count,
    }

    for field_name, expected in expected_parse_statistics.items():
        if statistics[field_name] != expected:
            raise SpeakerTurnPipelineError(
                f"Document statistic {field_name} does not reconcile with parser output."
            )

    if sum(int(record["segment_count"]) for record in turn_records) != len(segment_records):
        raise SpeakerTurnPipelineError("Turn segment counts do not reconcile.")

    if sum(int(record["content_span_count"]) for record in turn_records) != len(
        content_span_records
    ):
        raise SpeakerTurnPipelineError("Turn content-span counts do not reconcile.")

    for field_name in (
        "speech_span_count",
        "speech_word_count",
        "documentary_word_count",
        "stage_direction_word_count",
        "editorial_note_word_count",
    ):
        if sum(int(record[field_name]) for record in turn_records) != statistics[field_name]:
            raise SpeakerTurnPipelineError(f"Turn statistic {field_name} does not reconcile.")

    return statistics


def _serialize_document(
    *,
    source: _ValidatedSource,
    parse_result: SpeakerTurnParseResult,
    content_results: tuple[TurnContentResult, ...],
) -> _SerializedDocument:
    """Serialize and reconcile all parsed output records in memory."""
    if len(parse_result.turns) != len(content_results):
        raise SpeakerTurnPipelineError("Turn and content result counts do not match.")

    turn_records = []
    all_segment_records = []
    all_content_span_records = []

    for expected_turn_index, (turn, content) in enumerate(
        zip(parse_result.turns, content_results, strict=True),
        start=1,
    ):
        if turn.turn_index != expected_turn_index or content.turn_index != expected_turn_index:
            raise SpeakerTurnPipelineError("Turn indices are not consecutive from one.")

        _validate_marker_source(turn=turn, source=source)

        segment_records = [
            _segment_record(
                source_record_id=source.source_record_id,
                turn_index=turn.turn_index,
                segment_index=segment_index,
                segment=segment,
            )
            for segment_index, segment in enumerate(turn.segments, start=1)
        ]
        content_span_records = [
            _content_span_record(
                source_record_id=source.source_record_id,
                turn=turn,
                content_span_index=content_span_index,
                span=span,
            )
            for content_span_index, span in enumerate(content.spans, start=1)
        ]

        _validate_segment_and_span_provenance(
            turn=turn,
            segment_records=segment_records,
            content_span_records=content_span_records,
            source=source,
        )

        turn_record = _turn_record(
            source_record_id=source.source_record_id,
            turn=turn,
            content=content,
        )

        if turn_record["content_span_count"] != len(content_span_records):
            raise SpeakerTurnPipelineError(
                f"Content-span count does not reconcile in turn {turn.turn_index}."
            )

        if turn_record["segment_count"] != len(segment_records):
            raise SpeakerTurnPipelineError(
                f"Segment count does not reconcile in turn {turn.turn_index}."
            )

        if turn_record["character_count"] != sum(
            int(record["character_count"]) for record in segment_records
        ):
            raise SpeakerTurnPipelineError(
                f"Character count does not reconcile in turn {turn.turn_index}."
            )

        if turn_record["word_count"] != sum(
            int(record["word_count"]) for record in segment_records
        ):
            raise SpeakerTurnPipelineError(
                f"Word count does not reconcile in turn {turn.turn_index}."
            )

        expected_content_statistics = {
            "speech_span_count": sum(
                bool(record["include_in_speech"]) for record in content_span_records
            ),
            "speech_word_count": content.speech_word_count,
            "documentary_word_count": content.documentary_word_count,
            "stage_direction_word_count": sum(
                int(record["word_count"])
                for record in content_span_records
                if record["content_kind"] == TurnContentKind.STAGE_DIRECTION.value
            ),
            "editorial_note_word_count": sum(
                int(record["word_count"])
                for record in content_span_records
                if record["content_kind"] == TurnContentKind.EDITORIAL_NOTE.value
            ),
        }

        for field_name, expected_value in expected_content_statistics.items():
            if turn_record[field_name] != expected_value:
                raise SpeakerTurnPipelineError(
                    f"{field_name} does not reconcile in turn {turn.turn_index}."
                )

        if (
            content.documentary_boundary is not None
            and content.documentary_boundary.documentary_word_count
            != turn_record["documentary_word_count"]
        ):
            raise SpeakerTurnPipelineError(
                f"Documentary boundary does not reconcile in turn {turn.turn_index}."
            )

        turn_records.append(turn_record)
        all_segment_records.extend(segment_records)
        all_content_span_records.extend(content_span_records)

    statistics = _document_statistics(
        parse_result=parse_result,
        turn_records=turn_records,
        segment_records=all_segment_records,
        content_span_records=all_content_span_records,
    )

    return _SerializedDocument(
        turns=tuple(turn_records),
        segments=tuple(all_segment_records),
        content_spans=tuple(all_content_span_records),
        statistics=statistics,
    )


def _output_matches_metadata(
    *,
    path: Path,
    expected_sha256: object,
    expected_size_bytes: object,
) -> bool:
    """Return whether one generated output matches its cache metadata."""
    if not path.is_file() or not isinstance(expected_sha256, str) or not expected_sha256:
        return False

    try:
        expected_size = _safe_int(expected_size_bytes, field_name="cached output size")
        return path.stat().st_size == expected_size and sha256_file(path) == expected_sha256
    except (OSError, SpeakerTurnPipelineError):
        return False


def _valid_cached_statistics(value: object) -> bool:
    """Return whether a cached manifest has complete statistic metadata."""
    if not isinstance(value, dict):
        return False

    for field_name in _COUNT_STATISTIC_FIELDS:
        count = value.get(field_name)

        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return False

    for field_name in _MAPPING_STATISTIC_FIELDS:
        counts = value.get(field_name)

        if not isinstance(counts, dict) or list(counts) != sorted(counts):
            return False

        if any(
            not isinstance(key, str)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            for key, count in counts.items()
        ):
            return False

    return True


def _reusable_manifest(
    *,
    manifest_path: Path,
    source: _ValidatedSource,
    structure_path: Path,
    turns_path: Path,
    turn_segments_path: Path,
    content_spans_path: Path,
) -> dict[str, Any] | None:
    """Return a valid cache manifest, or none when regeneration is required."""
    if not manifest_path.is_file():
        return None

    try:
        manifest = _read_json_object(manifest_path)
    except SpeakerTurnPipelineError:
        return None

    if manifest.get("pipeline_version") != SPEAKER_TURN_PIPELINE_VERSION:
        return None

    if manifest.get("source_record_id") != source.source_record_id:
        return None

    if "reused" in manifest:
        return None

    processed_at_utc = manifest.get("processed_at_utc")

    if not isinstance(processed_at_utc, str) or not processed_at_utc:
        return None

    if not _valid_cached_statistics(manifest.get("statistics")):
        return None

    source_metadata = manifest.get("source")
    outputs = manifest.get("outputs")

    if not isinstance(source_metadata, dict) or not isinstance(outputs, dict):
        return None

    expected_source_metadata: dict[str, object] = {
        "structure_path": str(structure_path),
        "structure_sha256": source.structure_sha256,
        "segmenter_version": source.segmenter_version,
        "structural_blocks_path": str(source.structural_blocks_path),
        "structural_blocks_sha256": source.structural_blocks_sha256,
        "structural_blocks_size_bytes": source.structural_blocks_size_bytes,
    }

    for key, expected_value in expected_source_metadata.items():
        if source_metadata.get(key) != expected_value:
            return None

    output_checks = (
        (turns_path, "turns_path", "turns_sha256", "turns_size_bytes"),
        (
            turn_segments_path,
            "turn_segments_path",
            "turn_segments_sha256",
            "turn_segments_size_bytes",
        ),
        (
            content_spans_path,
            "content_spans_path",
            "content_spans_sha256",
            "content_spans_size_bytes",
        ),
    )

    for path, path_key, sha_key, size_key in output_checks:
        if outputs.get(path_key) != str(path):
            return None

        if not _output_matches_metadata(
            path=path,
            expected_sha256=outputs.get(sha_key),
            expected_size_bytes=outputs.get(size_key),
        ):
            return None

    return manifest


def _jsonl_text(records: Iterable[dict[str, Any]]) -> str:
    """Return deterministic JSON Lines text."""
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )


def _json_text(payload: dict[str, Any]) -> str:
    """Return deterministic, human-readable JSON text."""
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _part_path(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.part")


def _backup_path(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.bak")


def _replace_path(source: Path, destination: Path) -> None:
    """Replace a path through a narrow patch point for failure tests."""
    source.replace(destination)


def _cleanup_transaction_paths(final_paths: Iterable[Path]) -> None:
    """Remove temporary and backup paths for a completed transaction."""
    for final_path in final_paths:
        _part_path(final_path).unlink(missing_ok=True)
        _backup_path(final_path).unlink(missing_ok=True)


def _promote_transaction(final_paths: tuple[Path, ...]) -> None:
    """Promote staged outputs with rollback of existing final files."""
    backed_up: list[Path] = []
    promoted: list[Path] = []

    try:
        for final_path in final_paths:
            if final_path.exists():
                _replace_path(final_path, _backup_path(final_path))
                backed_up.append(final_path)

        for final_path in final_paths:
            _replace_path(_part_path(final_path), final_path)
            promoted.append(final_path)
    except Exception as error:
        for final_path in reversed(promoted):
            final_path.unlink(missing_ok=True)

        restore_error: Exception | None = None

        for final_path in reversed(backed_up):
            backup_path = _backup_path(final_path)

            if not backup_path.exists():
                continue

            try:
                _replace_path(backup_path, final_path)
            except Exception as error_during_restore:
                restore_error = error_during_restore

        for final_path in final_paths:
            _part_path(final_path).unlink(missing_ok=True)

        if restore_error is not None:
            raise SpeakerTurnPipelineError(
                "Could not restore prior speaker-turn outputs after promotion failure."
            ) from restore_error

        _cleanup_transaction_paths(final_paths)
        raise SpeakerTurnPipelineError("Could not promote speaker-turn outputs safely.") from error

    _cleanup_transaction_paths(final_paths)


def _write_outputs(
    *,
    source: _ValidatedSource,
    structure_path: Path,
    serialized: _SerializedDocument,
    turns_path: Path,
    turn_segments_path: Path,
    content_spans_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Stage, hash, and transactionally promote one document's outputs."""
    final_paths = (
        turns_path,
        turn_segments_path,
        content_spans_path,
        manifest_path,
    )

    try:
        turns_path.parent.mkdir(parents=True, exist_ok=True)
        _cleanup_transaction_paths(final_paths)
        _part_path(turns_path).write_text(
            _jsonl_text(serialized.turns),
            encoding="utf-8",
            newline="\n",
        )
        _part_path(turn_segments_path).write_text(
            _jsonl_text(serialized.segments),
            encoding="utf-8",
            newline="\n",
        )
        _part_path(content_spans_path).write_text(
            _jsonl_text(serialized.content_spans),
            encoding="utf-8",
            newline="\n",
        )

        output_metadata = {
            "turns_path": str(turns_path),
            "turns_sha256": sha256_file(_part_path(turns_path)),
            "turns_size_bytes": _part_path(turns_path).stat().st_size,
            "turn_segments_path": str(turn_segments_path),
            "turn_segments_sha256": sha256_file(_part_path(turn_segments_path)),
            "turn_segments_size_bytes": _part_path(turn_segments_path).stat().st_size,
            "content_spans_path": str(content_spans_path),
            "content_spans_sha256": sha256_file(_part_path(content_spans_path)),
            "content_spans_size_bytes": _part_path(content_spans_path).stat().st_size,
        }
        manifest: dict[str, Any] = {
            "pipeline_version": SPEAKER_TURN_PIPELINE_VERSION,
            "processed_at_utc": datetime.now(UTC).isoformat(),
            "source_record_id": source.source_record_id,
            "source": {
                "structure_path": str(structure_path),
                "structure_sha256": source.structure_sha256,
                "segmenter_version": source.segmenter_version,
                "structural_blocks_path": str(source.structural_blocks_path),
                "structural_blocks_sha256": source.structural_blocks_sha256,
                "structural_blocks_size_bytes": source.structural_blocks_size_bytes,
            },
            "outputs": output_metadata,
            "statistics": serialized.statistics,
        }
        _part_path(manifest_path).write_text(
            _json_text(manifest),
            encoding="utf-8",
            newline="\n",
        )
    except Exception as error:
        _cleanup_transaction_paths(final_paths)

        if isinstance(error, SpeakerTurnPipelineError):
            raise

        raise SpeakerTurnPipelineError(
            f"Could not stage speaker-turn outputs for {source.source_record_id}."
        ) from error

    _promote_transaction(final_paths)
    return manifest


def process_speaker_turn_document(
    *,
    structure_path: Path,
    output_root: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Parse and persist speaker turns for one structurally segmented document."""
    source = _validate_source(structure_path)
    output_directory = output_root / source.source_record_id
    turns_path = output_directory / "turns.jsonl"
    turn_segments_path = output_directory / "turn_segments.jsonl"
    content_spans_path = output_directory / "content_spans.jsonl"
    manifest_path = output_directory / "speaker_turns.json"

    if not force:
        cached_manifest = _reusable_manifest(
            manifest_path=manifest_path,
            source=source,
            structure_path=structure_path,
            turns_path=turns_path,
            turn_segments_path=turn_segments_path,
            content_spans_path=content_spans_path,
        )

        if cached_manifest is not None:
            result = dict(cached_manifest)
            result["reused"] = True
            return result

    try:
        parse_result = parse_speaker_turns(source.input_blocks)
        content_results = tuple(classify_speaker_turn_content(turn) for turn in parse_result.turns)
    except (SpeakerTurnError, TurnContentError) as error:
        raise SpeakerTurnPipelineError(
            f"Speaker-turn parsing failed for {source.source_record_id}."
        ) from error

    serialized = _serialize_document(
        source=source,
        parse_result=parse_result,
        content_results=content_results,
    )
    manifest = _write_outputs(
        source=source,
        structure_path=structure_path,
        serialized=serialized,
        turns_path=turns_path,
        turn_segments_path=turn_segments_path,
        content_spans_path=content_spans_path,
        manifest_path=manifest_path,
    )
    result = dict(manifest)
    result["reused"] = False
    return result
