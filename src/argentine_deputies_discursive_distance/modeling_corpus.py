"""Export a traceable spoken-discourse modelling corpus."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, BinaryIO

from .pdf_pipeline import sha256_file
from .speaker_turn_pipeline import SPEAKER_TURN_PIPELINE_VERSION
from .turn_content import TURN_CONTENT_CLASSIFIER_VERSION

MODELING_CORPUS_EXPORTER_VERSION = "1"
DEFAULT_SPEAKER_TURN_ROOT = Path("data/interim/speaker_turns")
DEFAULT_MODELING_OVERRIDES_PATH = Path("config/modeling_turn_overrides.json")
DEFAULT_MODELING_METADATA_PATH = Path("data/qa/full_corpus_run_summary.json")
DEFAULT_MODELING_OUTPUT_ROOT = Path("data/processed/modeling_corpus")

DOCUMENTS_FILENAME = "documents.jsonl"
SOURCE_TURNS_FILENAME = "source_turns.jsonl"
TURN_DECISIONS_FILENAME = "turn_decisions.jsonl"
EXCLUSION_LEDGER_FILENAME = "exclusion_ledger.jsonl"
MANIFEST_FILENAME = "export_manifest.json"

DEFAULT_MINIMUM_WORDS = 25
DEFAULT_MAXIMUM_CHUNK_WORDS = 300
SPOKEN_SPAN_SEPARATOR = "\n\n"

INCLUDED_SPEAKER_FAMILIES = (
    "executive_official",
    "named_or_role_unspecified",
)

EXCLUDED_SPEAKER_FAMILIES = (
    "chair",
    "chamber_secretary",
    "collective_or_anonymous",
)

SPAN_SEPARATOR_POLICY = "distinct_selected_spoken_content_spans_are_joined_with_two_newlines"
SOURCE_FRAGMENT_KIND = "source_fragment"
SYNTHETIC_SEPARATOR_KIND = "synthetic_separator"

SENTENCE_END_PATTERN = re.compile(r"[.!?¡¿…]+[\"')\]}»]*$")
CLAUSE_END_PATTERN = re.compile(r"[,;:][\"')\]}»]*$")
WORD_PATTERN = re.compile(r"\S+")


class ModelingCorpusError(RuntimeError):
    """Raised when the modelling corpus cannot be exported safely."""


class OverrideAction(StrEnum):
    """Supported exceptional modelling decisions."""

    EXCLUDE_TURN = "exclude_turn"
    RETAIN_FROM_ANCHOR_AND_RELABEL = "retain_from_anchor_and_relabel"


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    """Session-level metadata required by modelling records."""

    source_record_id: str
    session_date: str
    year: int
    temporal_period: str
    session_category: str
    meeting_number: str | None


@dataclass(frozen=True, slots=True)
class ProvenanceFragment:
    """Trace reconstructed text to source spans or documented separators."""

    fragment_kind: str
    source_record_id: str
    turn_index: int
    text_start: int
    text_end: int
    text: str
    content_span_index: int | None = None
    source_segment_index: int | None = None
    page_number: int | None = None
    reading_order: int | None = None
    block_reference: str | None = None
    source_start: int | None = None
    source_end: int | None = None


@dataclass(frozen=True, slots=True)
class SourceTurn:
    """One parsed source turn with deterministic reconstructed speech text."""

    source_record_id: str
    turn_index: int
    metadata: SourceMetadata
    original_speaker_family: str | None
    original_normalized_label: str | None
    original_speaker_label: str | None
    upstream_speech_word_count: int
    selected_span_word_count: int
    modeling_word_count: int
    exact_text: str
    provenance: tuple[ProvenanceFragment, ...]
    source_fragment_count: int
    synthetic_separator_count: int


@dataclass(frozen=True, slots=True)
class OverrideSpec:
    """One strict override declaration."""

    source_record_id: str
    turn_index: int
    session_date: str
    expected_speech_word_count: int
    action: OverrideAction
    reason: str
    start_anchor: str | None = None
    expected_anchor_occurrences: int | None = None
    speaker_family: str | None = None
    normalized_label: str | None = None

    @property
    def key(self) -> tuple[str, int]:
        """Return the turn lookup key."""
        return (self.source_record_id, self.turn_index)


@dataclass(frozen=True, slots=True)
class OverrideManifest:
    """Validated override file."""

    override_manifest_version: str
    required_pipeline_version: str
    required_content_classifier_version: str
    overrides: tuple[OverrideSpec, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class EffectiveTurn:
    """Post-override source-turn state before filtering."""

    source_turn: SourceTurn
    effective_speaker_family: str | None
    effective_normalized_label: str | None
    effective_speaker_label: str | None
    exact_text: str
    provenance: tuple[ProvenanceFragment, ...]
    modeling_word_count: int
    override: OverrideSpec | None
    discarded_prefix_text: str
    discarded_prefix_provenance: tuple[ProvenanceFragment, ...]


@dataclass(frozen=True, slots=True)
class Chunk:
    """One modelling document chunk."""

    chunk_index: int
    chunk_count_for_turn: int
    exact_text: str
    word_count: int
    provenance: tuple[ProvenanceFragment, ...]


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    if not path.is_file():
        raise ModelingCorpusError(f"JSON source does not exist: {path}")

    try:
        payload: object = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelingCorpusError(f"Could not read JSON object: {path}") from error

    if not isinstance(payload, dict):
        raise ModelingCorpusError(f"Expected a JSON object: {path}")

    return {str(key): value for key, value in payload.items()}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSON Lines records from disk."""
    if not path.is_file():
        raise ModelingCorpusError(f"JSONL source does not exist: {path}")

    records: list[dict[str, Any]] = []

    try:
        with path.open("r", encoding="utf-8-sig") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue

                try:
                    payload: object = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ModelingCorpusError(f"Invalid JSON at {path}:{line_number}") from error

                if not isinstance(payload, dict):
                    raise ModelingCorpusError(f"Expected a JSON object at {path}:{line_number}")

                records.append({str(key): value for key, value in payload.items()})
    except OSError as error:
        raise ModelingCorpusError(f"Could not read JSONL source: {path}") from error

    return records


def _json_text(payload: Mapping[str, Any]) -> str:
    """Return deterministic JSON text."""
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _safe_int(value: object, *, field_name: str) -> int:
    """Return an int while rejecting booleans."""
    if isinstance(value, bool):
        raise ModelingCorpusError(f"Invalid integer for {field_name}: {value!r}")

    if isinstance(value, int):
        return value

    raise ModelingCorpusError(f"Invalid integer for {field_name}: {value!r}")


def _required_string(payload: Mapping[str, Any], field_name: str) -> str:
    """Return a required string field."""
    value = payload.get(field_name)

    if not isinstance(value, str) or not value:
        raise ModelingCorpusError(f"Missing or invalid {field_name}.")

    return value


def _optional_string(value: object) -> str | None:
    """Return a string or None from a JSON value."""
    if value is None:
        return None

    if isinstance(value, str):
        return value

    raise ModelingCorpusError(f"Expected a string or null, got {value!r}")


def _content_hash(text: str) -> str:
    """Return a SHA-256 hash for exact text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_modeling_text(text: str) -> str:
    """Return conservative whitespace-normalized text for modelling."""
    return re.sub(r"\s+", " ", text).strip()


def _document_id(source_record_id: str, turn_index: int, chunk_index: int) -> str:
    """Return a stable modelling-document identifier."""
    return f"{source_record_id}__turn_{turn_index:06d}__chunk_{chunk_index:03d}"


def _source_turn_id(source_record_id: str, turn_index: int) -> str:
    """Return a stable source-turn identifier."""
    return f"{source_record_id}__turn_{turn_index:06d}"


def _temporal_period(session_date: str) -> tuple[int, str]:
    """Return the year and analytical period for a session date."""
    try:
        parsed = date.fromisoformat(session_date)
    except ValueError as error:
        raise ModelingCorpusError(f"Invalid session_date: {session_date!r}") from error

    year = parsed.year

    if not 2008 <= year <= 2025:
        raise ModelingCorpusError(f"Session date outside 2008-2025: {session_date}")

    if year <= 2011:
        return year, "2008-2011"
    if year <= 2015:
        return year, "2012-2015"
    if year <= 2019:
        return year, "2016-2019"
    if year <= 2023:
        return year, "2020-2023"

    return year, "2024-2025"


def _load_metadata(metadata_summary_path: Path) -> dict[str, SourceMetadata]:
    """Load source metadata from the full-corpus run summary."""
    payload = _read_json_object(metadata_summary_path)
    raw_records = payload.get("records")

    if not isinstance(raw_records, list):
        raise ModelingCorpusError("Metadata summary has no valid records list.")

    metadata: dict[str, SourceMetadata] = {}

    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            raise ModelingCorpusError("Metadata record is not an object.")

        record = {str(key): value for key, value in raw_record.items()}
        source_record_id = _required_string(record, "source_record_id")
        session_date = _required_string(record, "session_date")
        session_category = _required_string(record, "session_category")
        year, temporal_period = _temporal_period(session_date)

        if source_record_id in metadata:
            raise ModelingCorpusError(f"Duplicate metadata source_record_id: {source_record_id}")

        metadata[source_record_id] = SourceMetadata(
            source_record_id=source_record_id,
            session_date=session_date,
            year=year,
            temporal_period=temporal_period,
            session_category=session_category,
            meeting_number=_optional_string(record.get("meeting_number")),
        )

    return metadata


def _load_override_manifest(overrides_path: Path) -> OverrideManifest:
    """Load and strictly validate modelling overrides."""
    payload = _read_json_object(overrides_path)
    allowed_top_level = {
        "override_manifest_version",
        "required_pipeline_version",
        "required_content_classifier_version",
        "overrides",
    }

    unexpected_top_level = set(payload) - allowed_top_level

    if unexpected_top_level:
        raise ModelingCorpusError(
            f"Unsupported override manifest fields: {sorted(unexpected_top_level)}"
        )

    override_manifest_version = _required_string(payload, "override_manifest_version")
    required_pipeline_version = _required_string(payload, "required_pipeline_version")
    required_content_classifier_version = _required_string(
        payload,
        "required_content_classifier_version",
    )

    if required_pipeline_version != SPEAKER_TURN_PIPELINE_VERSION:
        raise ModelingCorpusError(
            f"Override manifest requires incompatible pipeline version: {required_pipeline_version}"
        )

    if required_content_classifier_version != TURN_CONTENT_CLASSIFIER_VERSION:
        raise ModelingCorpusError(
            "Override manifest requires incompatible content classifier version: "
            f"{required_content_classifier_version}"
        )

    raw_overrides = payload.get("overrides")

    if not isinstance(raw_overrides, list):
        raise ModelingCorpusError("Override manifest has no valid overrides list.")

    overrides: list[OverrideSpec] = []
    seen_keys: set[tuple[str, int]] = set()

    for raw_override in raw_overrides:
        if not isinstance(raw_override, dict):
            raise ModelingCorpusError("Override entry is not an object.")

        record = {str(key): value for key, value in raw_override.items()}
        action_value = _required_string(record, "action")

        try:
            action = OverrideAction(action_value)
        except ValueError as error:
            raise ModelingCorpusError(f"Unsupported override action: {action_value}") from error

        base_fields = {
            "source_record_id",
            "turn_index",
            "session_date",
            "expected_speech_word_count",
            "action",
            "reason",
        }
        retain_fields = {
            "start_anchor",
            "expected_anchor_occurrences",
            "speaker_family",
            "normalized_label",
        }
        allowed_fields = (
            base_fields | retain_fields
            if action == OverrideAction.RETAIN_FROM_ANCHOR_AND_RELABEL
            else base_fields
        )
        unexpected_fields = set(record) - allowed_fields

        if unexpected_fields:
            raise ModelingCorpusError(
                f"Unsupported override fields for {action.value}: {sorted(unexpected_fields)}"
            )

        missing_fields = allowed_fields - set(record)

        if missing_fields:
            raise ModelingCorpusError(
                f"Missing override fields for {action.value}: {sorted(missing_fields)}"
            )

        override = OverrideSpec(
            source_record_id=_required_string(record, "source_record_id"),
            turn_index=_safe_int(record.get("turn_index"), field_name="turn_index"),
            session_date=_required_string(record, "session_date"),
            expected_speech_word_count=_safe_int(
                record.get("expected_speech_word_count"),
                field_name="expected_speech_word_count",
            ),
            action=action,
            reason=_required_string(record, "reason"),
            start_anchor=(
                _required_string(record, "start_anchor")
                if action == OverrideAction.RETAIN_FROM_ANCHOR_AND_RELABEL
                else None
            ),
            expected_anchor_occurrences=(
                _safe_int(
                    record.get("expected_anchor_occurrences"),
                    field_name="expected_anchor_occurrences",
                )
                if action == OverrideAction.RETAIN_FROM_ANCHOR_AND_RELABEL
                else None
            ),
            speaker_family=(
                _required_string(record, "speaker_family")
                if action == OverrideAction.RETAIN_FROM_ANCHOR_AND_RELABEL
                else None
            ),
            normalized_label=(
                _required_string(record, "normalized_label")
                if action == OverrideAction.RETAIN_FROM_ANCHOR_AND_RELABEL
                else None
            ),
        )

        if override.key in seen_keys:
            raise ModelingCorpusError(
                f"Duplicate override for {override.source_record_id} turn {override.turn_index}"
            )

        seen_keys.add(override.key)
        overrides.append(override)

    return OverrideManifest(
        override_manifest_version=override_manifest_version,
        required_pipeline_version=required_pipeline_version,
        required_content_classifier_version=required_content_classifier_version,
        overrides=tuple(
            sorted(
                overrides,
                key=lambda override: (
                    override.source_record_id,
                    override.turn_index,
                ),
            )
        ),
        sha256=sha256_file(overrides_path),
    )


def _manifest_path(source_directory: Path) -> Path:
    return source_directory / "speaker_turns.json"


def _validate_manifest(manifest: Mapping[str, Any], source_directory: Path) -> None:
    """Validate required speaker-turn manifest versions."""
    if manifest.get("pipeline_version") != SPEAKER_TURN_PIPELINE_VERSION:
        raise ModelingCorpusError(
            f"Pipeline version mismatch in {_manifest_path(source_directory)}"
        )

    if manifest.get("content_classifier_version") != TURN_CONTENT_CLASSIFIER_VERSION:
        raise ModelingCorpusError(
            f"Content classifier version mismatch in {_manifest_path(source_directory)}"
        )


def _path_from_manifest(
    *,
    outputs: Mapping[str, Any],
    key: str,
    fallback: Path,
) -> Path:
    """Return a local output path, using manifest paths only as fallback."""
    if fallback.is_file():
        return fallback

    value = outputs.get(key)

    if isinstance(value, str) and value:
        path = Path(value)
        candidates = [path] if path.is_absolute() else [Path.cwd() / path, fallback.parent / path]

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_file():
                return resolved

    raise ModelingCorpusError(f"Missing or invalid manifest output path: {key}")


def _speaker_label(turn_record: Mapping[str, Any]) -> str | None:
    """Return the visible speaker label for a turn."""
    marker = turn_record.get("marker")

    if isinstance(marker, dict):
        raw_label = marker.get("raw_label")

        if isinstance(raw_label, str) and raw_label:
            return raw_label

    return _optional_string(turn_record.get("normalized_label"))


def _fragment_payload(fragment: ProvenanceFragment) -> dict[str, Any]:
    """Serialize one provenance fragment."""
    payload: dict[str, Any] = {
        "character_count": len(fragment.text),
        "fragment_kind": fragment.fragment_kind,
        "fragment_text_sha256": _content_hash(fragment.text),
        "source_record_id": fragment.source_record_id,
        "text_end": fragment.text_end,
        "text_start": fragment.text_start,
        "turn_index": fragment.turn_index,
    }

    if fragment.fragment_kind == SYNTHETIC_SEPARATOR_KIND:
        payload["separator_policy"] = SPAN_SEPARATOR_POLICY
        return payload

    payload.update(
        {
            "block_reference": fragment.block_reference,
            "content_span_index": fragment.content_span_index,
            "page_number": fragment.page_number,
            "reading_order": fragment.reading_order,
            "source_end": fragment.source_end,
            "source_segment_index": fragment.source_segment_index,
            "source_start": fragment.source_start,
        }
    )
    return payload


def _source_fragment_from_span(
    *,
    source_record_id: str,
    turn_index: int,
    span_record: Mapping[str, Any],
    text_start: int,
    text: str,
) -> ProvenanceFragment:
    """Build one source-fragment provenance entry from a spoken content span."""
    return ProvenanceFragment(
        fragment_kind=SOURCE_FRAGMENT_KIND,
        source_record_id=source_record_id,
        turn_index=turn_index,
        content_span_index=_safe_int(
            span_record.get("content_span_index"),
            field_name="content_span_index",
        ),
        source_segment_index=_safe_int(
            span_record.get("source_segment_index"),
            field_name="source_segment_index",
        ),
        page_number=_safe_int(span_record.get("page_number"), field_name="page_number"),
        reading_order=_safe_int(span_record.get("reading_order"), field_name="reading_order"),
        block_reference=_required_string(span_record, "block_reference"),
        source_start=_safe_int(span_record.get("start"), field_name="start"),
        source_end=_safe_int(span_record.get("end"), field_name="end"),
        text_start=text_start,
        text_end=text_start + len(text),
        text=text,
    )


def _synthetic_separator_fragment(
    *,
    source_record_id: str,
    turn_index: int,
    text_start: int,
) -> ProvenanceFragment:
    """Build one synthetic separator provenance entry."""
    return ProvenanceFragment(
        fragment_kind=SYNTHETIC_SEPARATOR_KIND,
        source_record_id=source_record_id,
        turn_index=turn_index,
        text_start=text_start,
        text_end=text_start + len(SPOKEN_SPAN_SEPARATOR),
        text=SPOKEN_SPAN_SEPARATOR,
    )


def _slice_fragments(
    fragments: Sequence[ProvenanceFragment],
    *,
    start: int,
    end: int,
) -> tuple[ProvenanceFragment, ...]:
    """Slice provenance fragments by exact retained-text offsets."""
    sliced: list[ProvenanceFragment] = []

    for fragment in fragments:
        overlap_start = max(start, fragment.text_start)
        overlap_end = min(end, fragment.text_end)

        if overlap_start >= overlap_end:
            continue

        local_start = overlap_start - fragment.text_start
        local_end = overlap_end - fragment.text_start
        source_start = (
            None if fragment.source_start is None else fragment.source_start + local_start
        )
        source_end = None if fragment.source_start is None else fragment.source_start + local_end

        sliced.append(
            ProvenanceFragment(
                fragment_kind=fragment.fragment_kind,
                source_record_id=fragment.source_record_id,
                turn_index=fragment.turn_index,
                content_span_index=fragment.content_span_index,
                source_segment_index=fragment.source_segment_index,
                page_number=fragment.page_number,
                reading_order=fragment.reading_order,
                block_reference=fragment.block_reference,
                source_start=source_start,
                source_end=source_end,
                text_start=overlap_start - start,
                text_end=overlap_end - start,
                text=fragment.text[local_start:local_end],
            )
        )

    return tuple(sliced)


def _source_pages(fragments: Sequence[ProvenanceFragment]) -> list[int]:
    """Return sorted source pages covered by fragments."""
    return sorted(
        {
            fragment.page_number
            for fragment in fragments
            if fragment.fragment_kind == SOURCE_FRAGMENT_KIND and fragment.page_number is not None
        }
    )


def _validate_separator_policy(
    *,
    text: str,
    fragments: Sequence[ProvenanceFragment],
) -> bool:
    """Return whether reconstructed fragments follow the separator policy."""
    source_count = sum(fragment.fragment_kind == SOURCE_FRAGMENT_KIND for fragment in fragments)
    separator_count = sum(
        fragment.fragment_kind == SYNTHETIC_SEPARATOR_KIND for fragment in fragments
    )

    if source_count == 0:
        return separator_count == 0 and text == ""

    if separator_count != source_count - 1:
        return False

    if "".join(fragment.text for fragment in fragments) != text:
        return False

    previous_kind: str | None = None

    for index, fragment in enumerate(fragments):
        if fragment.text != text[fragment.text_start : fragment.text_end]:
            return False

        if fragment.fragment_kind == SYNTHETIC_SEPARATOR_KIND:
            if fragment.text != SPOKEN_SPAN_SEPARATOR:
                return False
            if index == 0 or index == len(fragments) - 1:
                return False
            if previous_kind != SOURCE_FRAGMENT_KIND:
                return False
        elif fragment.fragment_kind != SOURCE_FRAGMENT_KIND:
            return False

        previous_kind = fragment.fragment_kind

    return True


def _source_turn_from_records(
    *,
    source_record_id: str,
    metadata: SourceMetadata,
    turn_record: Mapping[str, Any],
    span_records: Sequence[Mapping[str, Any]],
) -> SourceTurn:
    """Build one source turn from persisted turn and content-span records."""
    turn_index = _safe_int(turn_record.get("turn_index"), field_name="turn_index")
    fragments: list[ProvenanceFragment] = []
    exact_parts: list[str] = []
    selected_span_word_count = 0
    source_fragment_count = 0
    synthetic_separator_count = 0
    cursor = 0

    for span_record in sorted(
        span_records,
        key=lambda record: _safe_int(
            record.get("content_span_index"),
            field_name="content_span_index",
        ),
    ):
        if (
            span_record.get("content_kind") != "spoken_text"
            or span_record.get("include_in_speech") is not True
        ):
            continue

        if exact_parts:
            exact_parts.append(SPOKEN_SPAN_SEPARATOR)
            fragments.append(
                _synthetic_separator_fragment(
                    source_record_id=source_record_id,
                    turn_index=turn_index,
                    text_start=cursor,
                )
            )
            cursor += len(SPOKEN_SPAN_SEPARATOR)
            synthetic_separator_count += 1

        text = _required_string(span_record, "text")
        exact_parts.append(text)
        fragments.append(
            _source_fragment_from_span(
                source_record_id=source_record_id,
                turn_index=turn_index,
                span_record=span_record,
                text_start=cursor,
                text=text,
            )
        )
        cursor += len(text)
        selected_span_word_count += _safe_int(
            span_record.get("word_count"),
            field_name="word_count",
        )
        source_fragment_count += 1

    exact_text = "".join(exact_parts)
    upstream_speech_word_count = _safe_int(
        turn_record.get("speech_word_count"),
        field_name="speech_word_count",
    )
    modeling_word_count = len(exact_text.split())

    if upstream_speech_word_count != selected_span_word_count:
        raise ModelingCorpusError(
            "Turn speech_word_count does not match selected spoken span word counts: "
            f"{source_record_id} turn {turn_index}"
        )

    if modeling_word_count != selected_span_word_count:
        raise ModelingCorpusError(
            "Turn modeling_word_count does not match reconstructed spoken spans: "
            f"{source_record_id} turn {turn_index}"
        )

    if not _validate_separator_policy(text=exact_text, fragments=fragments):
        raise ModelingCorpusError(
            f"Spoken-span separator policy violation: {source_record_id} turn {turn_index}"
        )

    return SourceTurn(
        source_record_id=source_record_id,
        turn_index=turn_index,
        metadata=metadata,
        original_speaker_family=_optional_string(turn_record.get("speaker_family")),
        original_normalized_label=_optional_string(turn_record.get("normalized_label")),
        original_speaker_label=_speaker_label(turn_record),
        upstream_speech_word_count=upstream_speech_word_count,
        selected_span_word_count=selected_span_word_count,
        modeling_word_count=modeling_word_count,
        exact_text=exact_text,
        provenance=tuple(fragments),
        source_fragment_count=source_fragment_count,
        synthetic_separator_count=synthetic_separator_count,
    )


def _iter_source_turns(
    *,
    speaker_turn_root: Path,
    metadata_by_source: Mapping[str, SourceMetadata],
) -> Iterator[SourceTurn]:
    """Yield source turns from successful speaker-turn directories."""
    if not speaker_turn_root.is_dir():
        raise ModelingCorpusError(f"Speaker-turn root does not exist: {speaker_turn_root}")

    source_directories = sorted(
        path
        for path in speaker_turn_root.iterdir()
        if path.is_dir() and _manifest_path(path).is_file()
    )

    for source_directory in source_directories:
        manifest = _read_json_object(_manifest_path(source_directory))
        _validate_manifest(manifest, source_directory)
        source_record_id = _required_string(manifest, "source_record_id")

        metadata = metadata_by_source.get(source_record_id)

        if metadata is None:
            raise ModelingCorpusError(f"Missing metadata for source_record_id {source_record_id}")

        outputs = manifest.get("outputs")

        if not isinstance(outputs, dict):
            raise ModelingCorpusError(f"Missing outputs in {_manifest_path(source_directory)}")

        turns_path = _path_from_manifest(
            outputs=outputs,
            key="turns_path",
            fallback=source_directory / "turns.jsonl",
        )
        content_spans_path = _path_from_manifest(
            outputs=outputs,
            key="content_spans_path",
            fallback=source_directory / "content_spans.jsonl",
        )
        _path_from_manifest(
            outputs=outputs,
            key="turn_segments_path",
            fallback=source_directory / "turn_segments.jsonl",
        )
        turn_records = _read_jsonl(turns_path)
        span_records = _read_jsonl(content_spans_path)
        span_records_by_turn: dict[int, list[dict[str, Any]]] = defaultdict(list)

        for span_record in span_records:
            span_source_record_id = _required_string(span_record, "source_record_id")

            if span_source_record_id != source_record_id:
                raise ModelingCorpusError(
                    f"Unexpected source_record_id in content span: {span_source_record_id}"
                )

            span_records_by_turn[
                _safe_int(span_record.get("turn_index"), field_name="turn_index")
            ].append(span_record)

        seen_turn_indices: set[int] = set()

        for turn_record in sorted(
            turn_records,
            key=lambda record: _safe_int(record.get("turn_index"), field_name="turn_index"),
        ):
            turn_source_record_id = _required_string(turn_record, "source_record_id")

            if turn_source_record_id != source_record_id:
                raise ModelingCorpusError(
                    f"Unexpected source_record_id in turn: {turn_source_record_id}"
                )

            turn_index = _safe_int(turn_record.get("turn_index"), field_name="turn_index")

            if turn_index in seen_turn_indices:
                raise ModelingCorpusError(
                    f"Duplicate turn_index {turn_index} in source {source_record_id}"
                )

            seen_turn_indices.add(turn_index)
            yield _source_turn_from_records(
                source_record_id=source_record_id,
                metadata=metadata,
                turn_record=turn_record,
                span_records=span_records_by_turn.get(turn_index, []),
            )

        extra_span_turns = set(span_records_by_turn) - seen_turn_indices

        if extra_span_turns:
            raise ModelingCorpusError(
                f"Content spans reference missing turns in {source_record_id}: "
                f"{sorted(extra_span_turns)}"
            )


def _apply_override(
    *,
    turn: SourceTurn,
    override: OverrideSpec | None,
) -> EffectiveTurn:
    """Apply one override to a source turn."""
    if override is None:
        return EffectiveTurn(
            source_turn=turn,
            effective_speaker_family=turn.original_speaker_family,
            effective_normalized_label=turn.original_normalized_label,
            effective_speaker_label=turn.original_speaker_label,
            exact_text=turn.exact_text,
            provenance=turn.provenance,
            modeling_word_count=turn.modeling_word_count,
            override=None,
            discarded_prefix_text="",
            discarded_prefix_provenance=(),
        )

    if override.action == OverrideAction.EXCLUDE_TURN:
        return EffectiveTurn(
            source_turn=turn,
            effective_speaker_family=turn.original_speaker_family,
            effective_normalized_label=turn.original_normalized_label,
            effective_speaker_label=turn.original_speaker_label,
            exact_text=turn.exact_text,
            provenance=turn.provenance,
            modeling_word_count=turn.modeling_word_count,
            override=override,
            discarded_prefix_text="",
            discarded_prefix_provenance=(),
        )

    if override.start_anchor is None or override.expected_anchor_occurrences is None:
        raise ModelingCorpusError("Retain override is missing anchor fields.")

    occurrence_count = turn.exact_text.count(override.start_anchor)

    if occurrence_count != override.expected_anchor_occurrences:
        raise ModelingCorpusError(
            "Override anchor occurrence mismatch for "
            f"{override.source_record_id} turn {override.turn_index}: "
            f"expected {override.expected_anchor_occurrences}, got {occurrence_count}"
        )

    anchor_start = turn.exact_text.index(override.start_anchor)
    retained_text = turn.exact_text[anchor_start:]
    discarded_prefix_text = turn.exact_text[:anchor_start]

    return EffectiveTurn(
        source_turn=turn,
        effective_speaker_family=override.speaker_family,
        effective_normalized_label=override.normalized_label,
        effective_speaker_label=override.normalized_label,
        exact_text=retained_text,
        provenance=_slice_fragments(turn.provenance, start=anchor_start, end=len(turn.exact_text)),
        modeling_word_count=len(retained_text.split()),
        override=override,
        discarded_prefix_text=discarded_prefix_text,
        discarded_prefix_provenance=_slice_fragments(
            turn.provenance,
            start=0,
            end=anchor_start,
        ),
    )


def _break_candidates(
    *,
    text: str,
    words: Sequence[re.Match[str]],
    start_word_index: int,
    max_end_word_index: int,
    kind: str,
) -> list[int]:
    """Return candidate end-word indices for one boundary kind."""
    candidates: list[int] = []

    for word_index in range(start_word_index, max_end_word_index + 1):
        word = words[word_index - 1].group(0)
        separator = text[words[word_index - 1].end() : words[word_index].start()]

        if kind == "paragraph" and separator.count("\n") >= 2:
            candidates.append(word_index)
            continue

        if kind == "sentence" and SENTENCE_END_PATTERN.search(word) is not None:
            candidates.append(word_index)
            continue

        if kind == "clause" and CLAUSE_END_PATTERN.search(word) is not None:
            candidates.append(word_index)

    return candidates


def _select_chunk_end_word_index(
    *,
    paragraph_candidates: Sequence[int],
    sentence_candidates: Sequence[int],
    clause_candidates: Sequence[int],
    start_word_index: int,
    max_end_word_index: int,
    maximum_chunk_words: int,
) -> int:
    """Choose a useful boundary without creating pathologically short chunks."""
    candidates: list[tuple[int, int]] = []
    candidates.extend((word_index, 0) for word_index in paragraph_candidates)
    candidates.extend((word_index, 1) for word_index in sentence_candidates)
    candidates.extend((word_index, 2) for word_index in clause_candidates)

    if not candidates:
        return max_end_word_index

    final_quarter_size = max(
        1,
        maximum_chunk_words // 4,
    )
    final_quarter_floor = max(
        start_word_index,
        max_end_word_index - final_quarter_size + 1,
    )
    final_quarter_candidates = [
        (word_index, priority)
        for word_index, priority in candidates
        if word_index >= final_quarter_floor
    ]

    if final_quarter_candidates:
        return min(
            final_quarter_candidates,
            key=lambda item: (
                item[1],
                -item[0],
            ),
        )[0]

    latter_half_size = max(
        1,
        maximum_chunk_words // 2,
    )
    latter_half_floor = max(
        start_word_index,
        max_end_word_index - latter_half_size + 1,
    )
    latter_half_candidates = [
        (word_index, priority)
        for word_index, priority in candidates
        if word_index >= latter_half_floor
    ]

    if latter_half_candidates:
        return min(
            latter_half_candidates,
            key=lambda item: (
                -item[0],
                item[1],
            ),
        )[0]

    return max_end_word_index


def _chunk_ranges(text: str, *, maximum_chunk_words: int) -> tuple[tuple[int, int], ...]:
    """Return contiguous character ranges for deterministic chunks."""
    if maximum_chunk_words < 1:
        raise ModelingCorpusError("maximum_chunk_words must be positive.")

    words = tuple(WORD_PATTERN.finditer(text))

    if not words:
        return ()

    ranges: list[tuple[int, int]] = []
    start_word_index = 1
    start_char = 0
    total_words = len(words)

    while start_word_index <= total_words:
        max_end_word_index = min(
            total_words,
            start_word_index + maximum_chunk_words - 1,
        )

        if max_end_word_index == total_words:
            end_word_index = total_words
        else:
            paragraph_candidates = _break_candidates(
                text=text,
                words=words,
                start_word_index=start_word_index,
                max_end_word_index=max_end_word_index,
                kind="paragraph",
            )
            sentence_candidates = _break_candidates(
                text=text,
                words=words,
                start_word_index=start_word_index,
                max_end_word_index=max_end_word_index,
                kind="sentence",
            )
            clause_candidates = _break_candidates(
                text=text,
                words=words,
                start_word_index=start_word_index,
                max_end_word_index=max_end_word_index,
                kind="clause",
            )

            end_word_index = _select_chunk_end_word_index(
                paragraph_candidates=paragraph_candidates,
                sentence_candidates=sentence_candidates,
                clause_candidates=clause_candidates,
                start_word_index=start_word_index,
                max_end_word_index=max_end_word_index,
                maximum_chunk_words=maximum_chunk_words,
            )

        if end_word_index >= total_words:
            end_char = len(text)
        else:
            end_char = words[end_word_index].start()

        if end_char <= start_char:
            raise ModelingCorpusError("Chunking produced an empty chunk.")

        ranges.append((start_char, end_char))
        start_word_index = end_word_index + 1
        start_char = end_char

    if "".join(text[start:end] for start, end in ranges) != text:
        raise ModelingCorpusError("Chunk ranges do not reconstruct the source text.")

    return tuple(ranges)


def _chunks_for_turn(
    effective_turn: EffectiveTurn,
    *,
    maximum_chunk_words: int,
) -> tuple[Chunk, ...]:
    """Build deterministic chunks for one retained source turn."""
    ranges = _chunk_ranges(
        effective_turn.exact_text,
        maximum_chunk_words=maximum_chunk_words,
    )
    chunks: list[Chunk] = []

    for chunk_index, (start, end) in enumerate(ranges, start=1):
        chunk_text = effective_turn.exact_text[start:end]

        if not chunk_text.strip():
            raise ModelingCorpusError("Chunking produced whitespace-only chunk.")

        chunks.append(
            Chunk(
                chunk_index=chunk_index,
                chunk_count_for_turn=len(ranges),
                exact_text=chunk_text,
                word_count=len(chunk_text.split()),
                provenance=_slice_fragments(
                    effective_turn.provenance,
                    start=start,
                    end=end,
                ),
            )
        )

    if "".join(chunk.exact_text for chunk in chunks) != effective_turn.exact_text:
        raise ModelingCorpusError("Chunks do not reconstruct retained source turn.")

    if sum(chunk.word_count for chunk in chunks) != effective_turn.modeling_word_count:
        raise ModelingCorpusError("Chunk words do not reconcile with retained source turn.")

    return tuple(chunks)


def _exclusion_record(
    *,
    effective_turn: EffectiveTurn,
    exclusion_reason: str,
    exact_text: str,
    provenance: Sequence[ProvenanceFragment],
    word_count: int,
) -> dict[str, Any]:
    """Build one exclusion ledger record."""
    turn = effective_turn.source_turn
    override = effective_turn.override

    return {
        "content_hash": _content_hash(exact_text),
        "effective_normalized_label": effective_turn.effective_normalized_label,
        "effective_speaker_family": effective_turn.effective_speaker_family,
        "exact_excluded_text": exact_text,
        "exclusion_reason": exclusion_reason,
        "modeling_word_count": word_count,
        "original_normalized_label": turn.original_normalized_label,
        "original_speaker_family": turn.original_speaker_family,
        "original_upstream_speech_word_count": turn.upstream_speech_word_count,
        "override_action": override.action.value if override is not None else None,
        "override_applied": override is not None,
        "override_reason": override.reason if override is not None else None,
        "provenance": [_fragment_payload(fragment) for fragment in provenance],
        "session_category": turn.metadata.session_category,
        "session_date": turn.metadata.session_date,
        "source_record_id": turn.source_record_id,
        "source_turn_id": _source_turn_id(turn.source_record_id, turn.turn_index),
        "turn_index": turn.turn_index,
        "word_count": word_count,
    }


def _decision_reason(
    *,
    effective_turn: EffectiveTurn,
    minimum_words: int,
) -> tuple[str, str]:
    """Return the decision and reason for one effective turn."""
    override = effective_turn.override

    if override is not None and override.action == OverrideAction.EXCLUDE_TURN:
        return "excluded_by_override", override.reason

    if effective_turn.modeling_word_count == 0:
        return "excluded_zero_speech", "no_spoken_content"

    if effective_turn.effective_speaker_family not in INCLUDED_SPEAKER_FAMILIES:
        return "excluded_speaker_family", "speaker_family_not_in_modeling_scope"

    if effective_turn.modeling_word_count < minimum_words:
        return "excluded_below_minimum_words", f"fewer_than_{minimum_words}_words"

    return "retained", "meets_modeling_corpus_policy"


def _base_turn_fields(effective_turn: EffectiveTurn) -> dict[str, Any]:
    """Return common source-turn metadata fields."""
    turn = effective_turn.source_turn
    metadata = turn.metadata

    return {
        "meeting_number": metadata.meeting_number,
        "session_category": metadata.session_category,
        "session_date": metadata.session_date,
        "source_record_id": turn.source_record_id,
        "source_turn_id": _source_turn_id(turn.source_record_id, turn.turn_index),
        "temporal_period": metadata.temporal_period,
        "turn_index": turn.turn_index,
        "year": metadata.year,
    }


def _source_turn_record(
    *,
    effective_turn: EffectiveTurn,
    chunks: Sequence[Chunk],
) -> dict[str, Any]:
    """Build one retained source-turn record."""
    turn = effective_turn.source_turn
    override = effective_turn.override
    record = _base_turn_fields(effective_turn)
    record.update(
        {
            "chunk_count": len(chunks),
            "content_hash": _content_hash(effective_turn.exact_text),
            "effective_normalized_label": effective_turn.effective_normalized_label,
            "effective_speaker_family": effective_turn.effective_speaker_family,
            "effective_speaker_label": effective_turn.effective_speaker_label,
            "exact_retained_text": effective_turn.exact_text,
            "exact_retained_text_kind": "deterministic_reconstructed_spoken_text",
            "original_normalized_label": turn.original_normalized_label,
            "original_speaker_family": turn.original_speaker_family,
            "original_speaker_label": turn.original_speaker_label,
            "original_modeling_word_count": turn.modeling_word_count,
            "original_upstream_speech_word_count": turn.upstream_speech_word_count,
            "override_action": override.action.value if override is not None else None,
            "override_applied": override is not None,
            "override_reason": override.reason if override is not None else None,
            "post_override_modeling_word_count": effective_turn.modeling_word_count,
            "provenance": [_fragment_payload(fragment) for fragment in effective_turn.provenance],
            "reconstruction_separator": SPOKEN_SPAN_SEPARATOR,
            "selected_span_word_count": turn.selected_span_word_count,
            "source_pages_covered": _source_pages(effective_turn.provenance),
            "synthetic_separator_count": sum(
                fragment.fragment_kind == SYNTHETIC_SEPARATOR_KIND
                for fragment in effective_turn.provenance
            ),
        }
    )
    return record


def _document_record(
    *,
    effective_turn: EffectiveTurn,
    chunk: Chunk,
) -> dict[str, Any]:
    """Build one modelling document record."""
    turn = effective_turn.source_turn
    metadata = turn.metadata
    override = effective_turn.override
    modeling_text = _normalize_modeling_text(chunk.exact_text)

    if len(modeling_text.split()) != chunk.word_count:
        raise ModelingCorpusError("Modeling-text word count does not match exact text.")

    return {
        "chunk_count_for_turn": chunk.chunk_count_for_turn,
        "chunk_index": chunk.chunk_index,
        "document_id": _document_id(
            turn.source_record_id,
            turn.turn_index,
            chunk.chunk_index,
        ),
        "exact_text": chunk.exact_text,
        "exact_text_kind": "deterministic_reconstructed_spoken_text_chunk",
        "meeting_number": metadata.meeting_number,
        "modeling_text": modeling_text,
        "modeling_word_count": chunk.word_count,
        "normalized_label": effective_turn.effective_normalized_label,
        "override_action": override.action.value if override is not None else None,
        "override_applied": override is not None,
        "provenance": [_fragment_payload(fragment) for fragment in chunk.provenance],
        "session_category": metadata.session_category,
        "session_date": metadata.session_date,
        "source_pages_covered": _source_pages(chunk.provenance),
        "source_record_id": turn.source_record_id,
        "source_turn_modeling_word_count_after_override": effective_turn.modeling_word_count,
        "speaker_family": effective_turn.effective_speaker_family,
        "speaker_label": effective_turn.effective_speaker_label,
        "temporal_period": metadata.temporal_period,
        "turn_index": turn.turn_index,
        "word_count": chunk.word_count,
        "year": metadata.year,
    }


def _turn_decision_record(
    *,
    effective_turn: EffectiveTurn,
    decision: str,
    reason: str,
    chunk_count: int,
) -> dict[str, Any]:
    """Build one turn decision record."""
    turn = effective_turn.source_turn
    override = effective_turn.override
    record = _base_turn_fields(effective_turn)
    record.update(
        {
            "decision": decision,
            "effective_normalized_label": effective_turn.effective_normalized_label,
            "effective_speaker_family": effective_turn.effective_speaker_family,
            "original_normalized_label": turn.original_normalized_label,
            "original_speaker_family": turn.original_speaker_family,
            "original_modeling_word_count": turn.modeling_word_count,
            "original_upstream_speech_word_count": turn.upstream_speech_word_count,
            "override_action": override.action.value if override is not None else None,
            "override_applied": override is not None,
            "override_reason": override.reason if override is not None else None,
            "post_override_modeling_word_count": effective_turn.modeling_word_count,
            "reason": reason,
            "resulting_chunk_count": chunk_count,
        }
    )
    return record


def _empty_group() -> dict[str, int]:
    """Return an empty grouped manifest counter."""
    return {
        "document_count": 0,
        "source_turn_count": 0,
        "word_count": 0,
    }


def _add_group_count(
    target: dict[str, dict[str, int]],
    key: object,
    *,
    source_turn_count: int = 0,
    document_count: int = 0,
    word_count: int = 0,
) -> None:
    """Increment a grouped manifest counter."""
    normalized_key = str(key)

    if normalized_key not in target:
        target[normalized_key] = _empty_group()

    target[normalized_key]["document_count"] += document_count
    target[normalized_key]["source_turn_count"] += source_turn_count
    target[normalized_key]["word_count"] += word_count


def _sorted_groups(groups: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, int]]:
    """Return groups with deterministic key ordering."""
    return {
        key: {
            "document_count": int(value["document_count"]),
            "source_turn_count": int(value["source_turn_count"]),
            "word_count": int(value["word_count"]),
        }
        for key, value in sorted(groups.items())
    }


@dataclass(slots=True)
class ExportStats:
    """Running export statistics and reconciliation state."""

    input_source_ids: set[str] = field(default_factory=set)
    input_turn_keys: set[tuple[str, int]] = field(default_factory=set)
    decision_keys: set[tuple[str, int]] = field(default_factory=set)
    retained_decision_keys: set[tuple[str, int]] = field(default_factory=set)
    retained_source_turn_keys: set[tuple[str, int]] = field(default_factory=set)
    document_source_turn_keys: set[tuple[str, int]] = field(default_factory=set)
    document_ids: set[str] = field(default_factory=set)
    source_turn_declared_chunk_counts: dict[tuple[str, int], int] = field(default_factory=dict)
    document_counts_by_source_turn: Counter[tuple[str, int]] = field(default_factory=Counter)
    applied_override_counts: Counter[tuple[str, int]] = field(default_factory=Counter)
    excluded_counts_by_reason: Counter[str] = field(default_factory=Counter)
    counts_by_session_category: dict[str, dict[str, int]] = field(default_factory=dict)
    counts_by_speaker_family: dict[str, dict[str, int]] = field(default_factory=dict)
    counts_by_temporal_period: dict[str, dict[str, int]] = field(default_factory=dict)
    counts_by_year: dict[str, dict[str, int]] = field(default_factory=dict)
    session_category_rollup: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            "legislative_debate": _empty_group(),
            "other_session_categories": _empty_group(),
        }
    )
    positive_speech_turn_count: int = 0
    excluded_turn_decision_count: int = 0
    exclusion_ledger_count: int = 0
    retained_source_turn_count: int = 0
    modeling_document_count: int = 0
    retained_modeling_word_total: int = 0
    chunk_modeling_word_total: int = 0
    all_dates_inside_2008_2025: bool = True
    no_chunk_exceeds_maximum_words: bool = True
    no_retained_turn_below_minimum_words: bool = True
    no_retained_family_outside_allowed_set: bool = True
    synthetic_separator_policy_ok: bool = True


class _JsonlPartWriter:
    """Incremental deterministic JSONL part-file writer."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: BinaryIO | None = None
        self._sha256 = hashlib.sha256()
        self.size_bytes = 0
        self.record_count = 0

    def __enter__(self) -> _JsonlPartWriter:
        self._handle = _part_path(self.path).open("wb")
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def write(self, record: Mapping[str, Any]) -> None:
        """Write one deterministic JSONL record and update running hash."""
        if self._handle is None:
            raise ModelingCorpusError("JSONL writer is not open.")

        data = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        self._handle.write(data)
        self._sha256.update(data)
        self.size_bytes += len(data)
        self.record_count += 1

    def metadata(self) -> dict[str, Any]:
        """Return final-file metadata for the staged part file."""
        return {
            "path": str(self.path),
            "sha256": self._sha256.hexdigest(),
            "size_bytes": self.size_bytes,
        }


def _turn_key_from_record(record: Mapping[str, Any]) -> tuple[str, int]:
    """Return a source-turn identity from an emitted record."""
    return (
        str(record["source_record_id"]),
        int(record["turn_index"]),
    )


def _rollup_key(session_category: object) -> str:
    """Return legislative-debate vs other-session-category rollup key."""
    return (
        "legislative_debate"
        if str(session_category) == "legislative_debate"
        else "other_session_categories"
    )


def _record_decision_stats(stats: ExportStats, record: Mapping[str, Any]) -> None:
    """Update running stats for one turn-decision record."""
    key = _turn_key_from_record(record)

    if key in stats.decision_keys:
        raise ModelingCorpusError(f"Duplicate turn decision: {key[0]} turn {key[1]}")

    stats.decision_keys.add(key)

    if int(record["original_upstream_speech_word_count"]) > 0:
        stats.positive_speech_turn_count += 1

    if record["decision"] == "retained":
        stats.retained_decision_keys.add(key)
    else:
        stats.excluded_turn_decision_count += 1
        stats.excluded_counts_by_reason[str(record["decision"])] += 1


def _record_source_turn_stats(
    *,
    stats: ExportStats,
    record: Mapping[str, Any],
    minimum_words: int,
) -> None:
    """Update running stats for one retained source-turn record."""
    key = _turn_key_from_record(record)

    if key in stats.retained_source_turn_keys:
        raise ModelingCorpusError(f"Duplicate source-turn output: {key[0]} turn {key[1]}")

    words = int(record["post_override_modeling_word_count"])
    family = str(record["effective_speaker_family"])
    stats.retained_source_turn_keys.add(key)
    stats.source_turn_declared_chunk_counts[key] = int(record["chunk_count"])
    stats.retained_source_turn_count += 1
    stats.retained_modeling_word_total += words
    stats.all_dates_inside_2008_2025 = (
        stats.all_dates_inside_2008_2025 and 2008 <= int(record["year"]) <= 2025
    )
    stats.no_retained_turn_below_minimum_words = (
        stats.no_retained_turn_below_minimum_words and words >= minimum_words
    )
    stats.no_retained_family_outside_allowed_set = (
        stats.no_retained_family_outside_allowed_set and family in INCLUDED_SPEAKER_FAMILIES
    )

    _add_group_count(
        stats.counts_by_session_category,
        record["session_category"],
        source_turn_count=1,
        word_count=words,
    )
    _add_group_count(
        stats.counts_by_speaker_family,
        family,
        source_turn_count=1,
        word_count=words,
    )
    _add_group_count(
        stats.counts_by_temporal_period,
        record["temporal_period"],
        source_turn_count=1,
        word_count=words,
    )
    _add_group_count(
        stats.counts_by_year,
        record["year"],
        source_turn_count=1,
        word_count=words,
    )
    _add_group_count(
        stats.session_category_rollup,
        _rollup_key(record["session_category"]),
        source_turn_count=1,
        word_count=words,
    )


def _record_document_stats(
    *,
    stats: ExportStats,
    record: Mapping[str, Any],
    maximum_chunk_words: int,
) -> None:
    """Update running stats for one modelling-document record."""
    document_id = str(record["document_id"])

    if document_id in stats.document_ids:
        raise ModelingCorpusError(f"Duplicate document_id: {document_id}")

    key = _turn_key_from_record(record)
    words = int(record["word_count"])
    stats.document_ids.add(document_id)
    stats.document_source_turn_keys.add(key)
    stats.document_counts_by_source_turn[key] += 1
    stats.modeling_document_count += 1
    stats.chunk_modeling_word_total += words
    stats.no_chunk_exceeds_maximum_words = (
        stats.no_chunk_exceeds_maximum_words and words <= maximum_chunk_words
    )

    _add_group_count(
        stats.counts_by_session_category,
        record["session_category"],
        document_count=1,
    )
    _add_group_count(
        stats.counts_by_speaker_family,
        record["speaker_family"],
        document_count=1,
    )
    _add_group_count(
        stats.counts_by_temporal_period,
        record["temporal_period"],
        document_count=1,
    )
    _add_group_count(stats.counts_by_year, record["year"], document_count=1)
    _add_group_count(
        stats.session_category_rollup,
        _rollup_key(record["session_category"]),
        document_count=1,
    )


def _record_exclusion_stats(stats: ExportStats) -> None:
    """Update running stats for one exclusion-ledger record."""
    stats.exclusion_ledger_count += 1


def _validate_override_for_turn(*, turn: SourceTurn, override: OverrideSpec) -> None:
    """Validate one override against its loaded source turn."""
    if turn.metadata.session_date != override.session_date:
        raise ModelingCorpusError(
            "Override session_date mismatch for "
            f"{override.source_record_id} turn {override.turn_index}"
        )

    if turn.upstream_speech_word_count != override.expected_speech_word_count:
        raise ModelingCorpusError(
            "Override expected_speech_word_count mismatch for "
            f"{override.source_record_id} turn {override.turn_index}"
        )


def _validate_all_overrides_applied(
    *,
    stats: ExportStats,
    override_manifest: OverrideManifest,
) -> None:
    """Validate that every configured override matched one input turn."""
    for override in override_manifest.overrides:
        if override.source_record_id not in stats.input_source_ids:
            raise ModelingCorpusError(
                f"Override source_record_id does not exist: {override.source_record_id}"
            )

        if override.key not in stats.input_turn_keys:
            raise ModelingCorpusError(
                "Override turn does not exist: "
                f"{override.source_record_id} turn {override.turn_index}"
            )

        if stats.applied_override_counts.get(override.key, 0) != 1:
            raise ModelingCorpusError(
                "Override was not applied exactly once: "
                f"{override.source_record_id} turn {override.turn_index}"
            )


def _override_application_ledger(
    *,
    overrides: Sequence[OverrideSpec],
    applied_counts: Mapping[tuple[str, int], int],
) -> list[dict[str, Any]]:
    """Return manifest override application ledger."""
    return [
        {
            "action": override.action.value,
            "applied_count": int(applied_counts.get(override.key, 0)),
            "expected_application_count": 1,
            "reason": override.reason,
            "source_record_id": override.source_record_id,
            "turn_index": override.turn_index,
        }
        for override in overrides
    ]


def _stream_reconciliation_checks(
    *,
    stats: ExportStats,
    override_manifest: OverrideManifest,
) -> dict[str, bool]:
    """Return required reconciliation checks from running stats."""
    source_turn_document_counts_match = all(
        stats.document_counts_by_source_turn[key] == declared_count
        for key, declared_count in stats.source_turn_declared_chunk_counts.items()
    )
    override_counts = [
        stats.applied_override_counts.get(override.key, 0) == 1
        for override in override_manifest.overrides
    ]

    return {
        "all_configured_overrides_applied_once": all(override_counts),
        "all_dates_inside_2008_2025": stats.all_dates_inside_2008_2025,
        "all_document_keys_map_to_retained_source_turn": (
            stats.document_source_turn_keys <= stats.retained_source_turn_keys
        ),
        "all_input_turns_have_one_decision": stats.input_turn_keys == stats.decision_keys,
        "all_parsed_turns_reconcile": len(stats.input_turn_keys)
        == stats.retained_source_turn_count + stats.excluded_turn_decision_count,
        "all_retained_decisions_have_one_source_turn": (
            stats.retained_decision_keys == stats.retained_source_turn_keys
        ),
        "all_source_turns_have_declared_document_count": (
            source_turn_document_counts_match
            and set(stats.source_turn_declared_chunk_counts) == stats.retained_source_turn_keys
        ),
        "every_source_turn_has_at_least_one_document": all(
            declared_count > 0
            for declared_count in stats.source_turn_declared_chunk_counts.values()
        ),
        "every_synthetic_separator_follows_policy": stats.synthetic_separator_policy_ok,
        "no_chunk_exceeds_maximum_words": stats.no_chunk_exceeds_maximum_words,
        "no_duplicate_document_ids": len(stats.document_ids) == stats.modeling_document_count,
        "no_duplicate_source_turn_decisions": len(stats.decision_keys)
        == len(stats.input_turn_keys),
        "no_retained_family_outside_allowed_set": (stats.no_retained_family_outside_allowed_set),
        "no_retained_turn_below_minimum_words": stats.no_retained_turn_below_minimum_words,
        "retained_reconstructed_word_totals_match_chunks": (
            stats.retained_modeling_word_total == stats.chunk_modeling_word_total
        ),
    }


def _stream_manifest_record(
    *,
    stats: ExportStats,
    override_manifest: OverrideManifest,
    reconciliation_checks: Mapping[str, bool],
    minimum_words: int,
    maximum_chunk_words: int,
) -> dict[str, Any]:
    """Build the export manifest from running stats except output hashes."""
    return {
        "chunking_configuration": {
            "break_selection": ("final_quarter_strength_then_latter_half_latest_else_hard_cap"),
            "maximum_chunk_words": maximum_chunk_words,
            "separator": SPOKEN_SPAN_SEPARATOR,
            "separator_policy": SPAN_SEPARATOR_POLICY,
            "sentence_aware": True,
            "unit_preference_order": [
                "paragraph_boundary",
                "sentence_ending_punctuation",
                "clause_punctuation",
                "whitespace_hard_split",
            ],
        },
        "content_classifier_version": TURN_CONTENT_CLASSIFIER_VERSION,
        "counts_by_session_category_rollup": _sorted_groups(stats.session_category_rollup),
        "counts_and_words_by_session_category": _sorted_groups(stats.counts_by_session_category),
        "counts_and_words_by_speaker_family": _sorted_groups(stats.counts_by_speaker_family),
        "counts_and_words_by_temporal_period": _sorted_groups(stats.counts_by_temporal_period),
        "counts_and_words_by_year": _sorted_groups(stats.counts_by_year),
        "excluded_counts_by_reason": dict(sorted(stats.excluded_counts_by_reason.items())),
        "excluded_family_values": list(EXCLUDED_SPEAKER_FAMILIES),
        "exclusion_ledger_count": stats.exclusion_ledger_count,
        "exporter_version": MODELING_CORPUS_EXPORTER_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "included_family_values": list(INCLUDED_SPEAKER_FAMILIES),
        "input_session_count": len(stats.input_source_ids),
        "input_turn_count": len(stats.input_turn_keys),
        "minimum_word_threshold": minimum_words,
        "modeling_document_count": stats.modeling_document_count,
        "modeling_word_total": stats.chunk_modeling_word_total,
        "output_files": {},
        "override_application_ledger": _override_application_ledger(
            overrides=override_manifest.overrides,
            applied_counts=stats.applied_override_counts,
        ),
        "override_manifest_version": override_manifest.override_manifest_version,
        "overrides_sha256": override_manifest.sha256,
        "pipeline_version": SPEAKER_TURN_PIPELINE_VERSION,
        "positive_speech_turn_count": stats.positive_speech_turn_count,
        "reconstructed_exact_word_total": stats.retained_modeling_word_total,
        "reconciliation_checks": dict(sorted(reconciliation_checks.items())),
        "retained_modeling_word_total": stats.retained_modeling_word_total,
        "retained_source_turn_count": stats.retained_source_turn_count,
        "retained_source_turn_word_total": stats.retained_modeling_word_total,
    }


def _part_path(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.part")


def _backup_path(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.bak")


def _replace_path(source: Path, destination: Path) -> None:
    """Replace a path through a narrow patch point for failure tests."""
    source.replace(destination)


def _cleanup_transaction_paths(final_paths: Iterable[Path]) -> None:
    """Remove temporary transaction files."""
    for final_path in final_paths:
        _part_path(final_path).unlink(missing_ok=True)
        _backup_path(final_path).unlink(missing_ok=True)


def _promote_transaction(final_paths: tuple[Path, ...]) -> None:
    """Promote part files to final paths with rollback."""
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
            raise ModelingCorpusError(
                "Could not restore prior modelling-corpus outputs after promotion failure."
            ) from restore_error

        _cleanup_transaction_paths(final_paths)
        raise ModelingCorpusError("Could not promote modelling-corpus outputs safely.") from error

    _cleanup_transaction_paths(final_paths)


def _final_output_paths(output_root: Path) -> dict[str, Path]:
    """Return canonical final output paths."""
    return {
        "documents": output_root / DOCUMENTS_FILENAME,
        "exclusion_ledger": output_root / EXCLUSION_LEDGER_FILENAME,
        "manifest": output_root / MANIFEST_FILENAME,
        "source_turns": output_root / SOURCE_TURNS_FILENAME,
        "turn_decisions": output_root / TURN_DECISIONS_FILENAME,
    }


def _ensure_output_directory(output_root: Path, *, force: bool) -> None:
    """Enforce output overwrite protection."""
    if output_root.exists() and not output_root.is_dir():
        raise ModelingCorpusError(f"Output root is not a directory: {output_root}")

    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise ModelingCorpusError(
            f"Output directory is nonempty; use --force to overwrite: {output_root}"
        )

    output_root.mkdir(parents=True, exist_ok=True)


def _write_outputs(
    *,
    speaker_turn_root: Path,
    metadata_by_source: Mapping[str, SourceMetadata],
    override_manifest: OverrideManifest,
    output_root: Path,
    minimum_words: int,
    maximum_chunk_words: int,
    force: bool,
) -> dict[str, Any]:
    """Stream all export outputs to staged part files transactionally."""
    if minimum_words < 1:
        raise ModelingCorpusError("minimum_words must be positive.")

    if maximum_chunk_words < 1:
        raise ModelingCorpusError("maximum_chunk_words must be positive.")

    _ensure_output_directory(output_root, force=force)
    paths = _final_output_paths(output_root)
    final_paths = (
        paths["documents"],
        paths["source_turns"],
        paths["turn_decisions"],
        paths["exclusion_ledger"],
        paths["manifest"],
    )
    _cleanup_transaction_paths(final_paths)
    stats = ExportStats()
    overrides_by_key = {override.key: override for override in override_manifest.overrides}
    manifest: dict[str, Any] | None = None

    try:
        with (
            _JsonlPartWriter(paths["documents"]) as documents_writer,
            _JsonlPartWriter(paths["source_turns"]) as source_turns_writer,
            _JsonlPartWriter(paths["turn_decisions"]) as turn_decisions_writer,
            _JsonlPartWriter(paths["exclusion_ledger"]) as exclusion_ledger_writer,
        ):
            for turn in _iter_source_turns(
                speaker_turn_root=speaker_turn_root,
                metadata_by_source=metadata_by_source,
            ):
                key = (turn.source_record_id, turn.turn_index)

                if key in stats.input_turn_keys:
                    raise ModelingCorpusError(
                        f"Duplicate source turn identity: {key[0]} turn {key[1]}"
                    )

                stats.input_turn_keys.add(key)
                stats.input_source_ids.add(turn.source_record_id)
                stats.synthetic_separator_policy_ok = (
                    stats.synthetic_separator_policy_ok
                    and _validate_separator_policy(
                        text=turn.exact_text,
                        fragments=turn.provenance,
                    )
                )

                override = overrides_by_key.get(key)

                if override is not None:
                    _validate_override_for_turn(turn=turn, override=override)

                effective_turn = _apply_override(turn=turn, override=override)

                if override is not None:
                    stats.applied_override_counts[override.key] += 1

                if effective_turn.discarded_prefix_text:
                    exclusion_record = _exclusion_record(
                        effective_turn=effective_turn,
                        exclusion_reason="discarded_prefix_by_override",
                        exact_text=effective_turn.discarded_prefix_text,
                        provenance=effective_turn.discarded_prefix_provenance,
                        word_count=len(effective_turn.discarded_prefix_text.split()),
                    )
                    exclusion_ledger_writer.write(exclusion_record)
                    _record_exclusion_stats(stats)

                decision, reason = _decision_reason(
                    effective_turn=effective_turn,
                    minimum_words=minimum_words,
                )
                chunks: tuple[Chunk, ...] = ()

                if decision == "retained":
                    chunks = _chunks_for_turn(
                        effective_turn,
                        maximum_chunk_words=maximum_chunk_words,
                    )
                    source_turn_record = _source_turn_record(
                        effective_turn=effective_turn,
                        chunks=chunks,
                    )
                    source_turns_writer.write(source_turn_record)
                    _record_source_turn_stats(
                        stats=stats,
                        record=source_turn_record,
                        minimum_words=minimum_words,
                    )

                    for chunk in chunks:
                        document_record = _document_record(
                            effective_turn=effective_turn,
                            chunk=chunk,
                        )
                        documents_writer.write(document_record)
                        _record_document_stats(
                            stats=stats,
                            record=document_record,
                            maximum_chunk_words=maximum_chunk_words,
                        )
                else:
                    exclusion_record = _exclusion_record(
                        effective_turn=effective_turn,
                        exclusion_reason=decision,
                        exact_text=effective_turn.exact_text,
                        provenance=effective_turn.provenance,
                        word_count=effective_turn.modeling_word_count,
                    )
                    exclusion_ledger_writer.write(exclusion_record)
                    _record_exclusion_stats(stats)

                decision_record = _turn_decision_record(
                    effective_turn=effective_turn,
                    decision=decision,
                    reason=reason,
                    chunk_count=len(chunks),
                )
                turn_decisions_writer.write(decision_record)
                _record_decision_stats(stats, decision_record)

            _validate_all_overrides_applied(
                stats=stats,
                override_manifest=override_manifest,
            )
            checks = _stream_reconciliation_checks(
                stats=stats,
                override_manifest=override_manifest,
            )

            if not all(checks.values()):
                failed = sorted(key for key, passed in checks.items() if not passed)
                raise ModelingCorpusError(f"Export reconciliation failed: {failed}")

            manifest = _stream_manifest_record(
                stats=stats,
                override_manifest=override_manifest,
                reconciliation_checks=checks,
                minimum_words=minimum_words,
                maximum_chunk_words=maximum_chunk_words,
            )
            manifest["output_files"] = {
                "documents": documents_writer.metadata(),
                "exclusion_ledger": exclusion_ledger_writer.metadata(),
                "source_turns": source_turns_writer.metadata(),
                "turn_decisions": turn_decisions_writer.metadata(),
            }

        _part_path(paths["manifest"]).write_bytes(_json_text(manifest).encode("utf-8"))
    except Exception as error:
        _cleanup_transaction_paths(final_paths)

        if isinstance(error, ModelingCorpusError):
            raise

        raise ModelingCorpusError("Could not stage modelling-corpus outputs.") from error

    _promote_transaction(final_paths)
    if manifest is None:
        raise ModelingCorpusError("Modelling-corpus manifest was not staged.")

    return manifest


def export_modeling_corpus(
    *,
    speaker_turn_root: Path = DEFAULT_SPEAKER_TURN_ROOT,
    overrides_path: Path = DEFAULT_MODELING_OVERRIDES_PATH,
    metadata_summary_path: Path = DEFAULT_MODELING_METADATA_PATH,
    output_root: Path = DEFAULT_MODELING_OUTPUT_ROOT,
    minimum_words: int = DEFAULT_MINIMUM_WORDS,
    maximum_chunk_words: int = DEFAULT_MAXIMUM_CHUNK_WORDS,
    force: bool = False,
) -> dict[str, Any]:
    """Export the final spoken-discourse modelling corpus."""
    metadata_by_source = _load_metadata(metadata_summary_path)
    override_manifest = _load_override_manifest(overrides_path)
    return _write_outputs(
        speaker_turn_root=speaker_turn_root,
        metadata_by_source=metadata_by_source,
        override_manifest=override_manifest,
        output_root=output_root,
        minimum_words=minimum_words,
        maximum_chunk_words=maximum_chunk_words,
        force=force,
    )
