import json
from pathlib import Path
from typing import Any

import pytest

from argentine_deputies_discursive_distance import speaker_turn_pipeline
from argentine_deputies_discursive_distance.pdf_pipeline import sha256_file
from argentine_deputies_discursive_distance.speaker_turn_pipeline import (
    SpeakerTurnPipelineError,
    process_speaker_turn_document,
)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
        ),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def primary_blocks(source_record_id: str = "record-1") -> list[dict[str, Any]]:
    return [
        {
            "source_record_id": source_record_id,
            "page_number": 1,
            "reading_order": 1,
            "structural_zone": "proceedings",
            "content_role": "transcript",
            "include_in_discourse": True,
            "text": "Sr. Pérez.- Hola.\n(Aplausos.)",
        },
        {
            "source_record_id": source_record_id,
            "page_number": 1,
            "reading_order": 2,
            "structural_zone": "proceedings",
            "content_role": "transcript",
            "include_in_discourse": True,
            "text": "Continúa.",
        },
        {
            "source_record_id": source_record_id,
            "page_number": 1,
            "reading_order": 3,
            "structural_zone": "proceedings",
            "content_role": "procedural",
            "include_in_discourse": False,
            "text": "VOTACIÓN",
        },
        {
            "source_record_id": source_record_id,
            "page_number": 1,
            "reading_order": 4,
            "structural_zone": "proceedings",
            "content_role": "transcript",
            "include_in_discourse": True,
            "text": "Texto sin atribución.",
        },
    ]


def build_structure_bundle(
    tmp_path: Path,
    *,
    blocks: list[dict[str, Any]] | None = None,
    source_record_id: str = "record-1",
) -> tuple[Path, Path]:
    source_directory = tmp_path / "structure" / source_record_id
    source_directory.mkdir(parents=True, exist_ok=True)
    structural_blocks_path = source_directory / "structural_blocks.jsonl"
    structure_path = source_directory / "structure.json"
    actual_blocks = blocks if blocks is not None else primary_blocks(source_record_id)

    write_jsonl(structural_blocks_path, actual_blocks)
    structure = {
        "segmenter_version": "1",
        "source_record_id": source_record_id,
        "outputs": {
            "structural_blocks_path": str(structural_blocks_path),
            "structural_blocks_sha256": sha256_file(structural_blocks_path),
            "structural_blocks_size_bytes": structural_blocks_path.stat().st_size,
        },
    }
    structure_path.write_text(
        json.dumps(structure, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return structure_path, structural_blocks_path


def output_paths(output_root: Path, source_record_id: str = "record-1") -> tuple[Path, ...]:
    output_directory = output_root / source_record_id
    return (
        output_directory / "turns.jsonl",
        output_directory / "turn_segments.jsonl",
        output_directory / "content_spans.jsonl",
        output_directory / "speaker_turns.json",
    )


def test_writes_all_outputs_and_reconciled_statistics(tmp_path: Path) -> None:
    structure_path, _ = build_structure_bundle(tmp_path)
    output_root = tmp_path / "speaker_turns"

    result = process_speaker_turn_document(
        structure_path=structure_path,
        output_root=output_root,
    )

    paths = output_paths(output_root)
    assert result["reused"] is False
    assert all(path.is_file() for path in paths)

    turns = read_jsonl(paths[0])
    segments = read_jsonl(paths[1])
    spans = read_jsonl(paths[2])
    persisted_manifest = json.loads(paths[3].read_text(encoding="utf-8"))
    returned_manifest = dict(result)
    returned_manifest.pop("reused")

    assert persisted_manifest == returned_manifest
    assert len(turns) == 2
    assert len(segments) == 3
    assert len(spans) == 4
    assert persisted_manifest["statistics"] == {
        "assigned_character_count": sum(len(segment["text"]) for segment in segments),
        "assigned_segment_count": 3,
        "attribution_method_counts": {
            "carried_forward": 1,
            "explicit_marker": 1,
            "unattributed": 1,
        },
        "barrier_reset_count": 1,
        "content_kind_counts": {
            "spoken_text": 2,
            "stage_direction": 1,
            "unattributed_text": 1,
        },
        "content_span_count": 4,
        "documentary_span_count": 0,
        "documentary_word_count": 0,
        "editorial_note_span_count": 0,
        "editorial_note_word_count": 0,
        "explicit_marker_count": 1,
        "maximum_speech_word_count": 2,
        "speaker_family_counts": {"named_or_role_unspecified": 1},
        "speech_span_count": 2,
        "speech_word_count": 2,
        "stage_direction_span_count": 1,
        "stage_direction_word_count": 1,
        "turn_count": 2,
        "unattributed_content_span_count": 1,
        "unattributed_content_word_count": 3,
        "unattributed_segment_count": 1,
        "unattributed_turn_count": 1,
        "zero_speech_turn_count": 1,
    }

    for path, prefix in zip(
        paths[:3],
        ("turns", "turn_segments", "content_spans"),
        strict=True,
    ):
        outputs = persisted_manifest["outputs"]
        assert outputs[f"{prefix}_path"] == str(path)
        assert outputs[f"{prefix}_sha256"] == sha256_file(path)
        assert outputs[f"{prefix}_size_bytes"] == path.stat().st_size


def test_preserves_exact_segment_and_content_span_provenance(tmp_path: Path) -> None:
    structure_path, structural_blocks_path = build_structure_bundle(tmp_path)
    output_root = tmp_path / "speaker_turns"
    process_speaker_turn_document(structure_path=structure_path, output_root=output_root)

    turns_path, segments_path, spans_path, _ = output_paths(output_root)
    turns = read_jsonl(turns_path)
    segments = read_jsonl(segments_path)
    spans = read_jsonl(spans_path)
    source_blocks = {
        (record["page_number"], record["reading_order"]): record["text"]
        for record in read_jsonl(structural_blocks_path)
    }

    assert "text" not in turns[0]
    assert turns[0]["marker"] == {
        "detection_confidence": 1.0,
        "detection_method": "explicit_honorific_dot_dash",
        "end": len("Sr. Pérez.-"),
        "family": "named_or_role_unspecified",
        "is_multiline": False,
        "normalized_label": "PEREZ",
        "normalized_title": "SR.",
        "position": "block_start",
        "raw_label": "Pérez",
        "raw_marker": "Sr. Pérez.-",
        "raw_title": "Sr.",
        "separator": ".-",
        "separator_kind": "dot_dash",
        "start": 0,
    }

    for segment in segments:
        source_text = source_blocks[(segment["page_number"], segment["reading_order"])]
        assert segment["text"] == source_text[segment["start"] : segment["end"]]

    for span in spans:
        source_text = source_blocks[(span["page_number"], span["reading_order"])]
        assert span["text"] == source_text[span["start"] : span["end"]]

    for segment in segments:
        linked = [
            span
            for span in spans
            if (
                span["turn_index"] == segment["turn_index"]
                and span["source_segment_index"] == segment["segment_index"]
            )
        ]
        assert linked[0]["start"] == segment["start"]
        assert linked[-1]["end"] == segment["end"]
        assert "".join(span["text"] for span in linked) == segment["text"]


def test_reuses_valid_outputs_without_changing_bytes_or_modification_times(
    tmp_path: Path,
) -> None:
    structure_path, _ = build_structure_bundle(tmp_path)
    output_root = tmp_path / "speaker_turns"
    first = process_speaker_turn_document(
        structure_path=structure_path,
        output_root=output_root,
    )
    paths = output_paths(output_root)
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}

    second = process_speaker_turn_document(
        structure_path=structure_path,
        output_root=output_root,
    )

    assert first["reused"] is False
    assert second["reused"] is True
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
    assert "reused" not in json.loads(paths[-1].read_text(encoding="utf-8"))


def test_rebuilds_a_corrupted_generated_output(tmp_path: Path) -> None:
    structure_path, _ = build_structure_bundle(tmp_path)
    output_root = tmp_path / "speaker_turns"
    process_speaker_turn_document(structure_path=structure_path, output_root=output_root)
    turns_path = output_paths(output_root)[0]
    expected_turns = turns_path.read_bytes()
    turns_path.write_text("corrupted\n", encoding="utf-8")

    result = process_speaker_turn_document(
        structure_path=structure_path,
        output_root=output_root,
    )

    assert result["reused"] is False
    assert turns_path.read_bytes() == expected_turns
    assert len(read_jsonl(turns_path)) == 2


def test_rejects_changed_upstream_blocks_without_updated_metadata(tmp_path: Path) -> None:
    structure_path, structural_blocks_path = build_structure_bundle(tmp_path)
    output_root = tmp_path / "speaker_turns"
    process_speaker_turn_document(structure_path=structure_path, output_root=output_root)
    paths = output_paths(output_root)
    before = {path: path.read_bytes() for path in paths}
    source_text = structural_blocks_path.read_text(encoding="utf-8")
    structural_blocks_path.write_text(source_text.replace("Hola", "Chau"), encoding="utf-8")

    with pytest.raises(SpeakerTurnPipelineError, match="SHA-256"):
        process_speaker_turn_document(
            structure_path=structure_path,
            output_root=output_root,
        )

    assert before == {path: path.read_bytes() for path in paths}


def test_serializes_an_empty_explicit_turn_safely(tmp_path: Path) -> None:
    blocks = [
        {
            "source_record_id": "record-1",
            "page_number": 1,
            "reading_order": 1,
            "structural_zone": "proceedings",
            "content_role": "transcript",
            "include_in_discourse": True,
            "text": "Sr. Pérez.-",
        }
    ]
    structure_path, _ = build_structure_bundle(tmp_path, blocks=blocks)
    output_root = tmp_path / "speaker_turns"

    result = process_speaker_turn_document(
        structure_path=structure_path,
        output_root=output_root,
    )

    turns_path, segments_path, spans_path, _ = output_paths(output_root)
    turns = read_jsonl(turns_path)
    assert len(turns) == 1
    assert turns[0]["segment_count"] == 0
    assert turns[0]["content_span_count"] == 0
    assert turns[0]["first_reference"] is None
    assert turns[0]["last_reference"] is None
    assert segments_path.read_bytes() == b""
    assert spans_path.read_bytes() == b""
    assert result["statistics"]["explicit_marker_count"] == 1
    assert result["statistics"]["zero_speech_turn_count"] == 1
    assert result["statistics"]["maximum_speech_word_count"] == 0


def test_forced_rebuild_preserves_all_non_timestamp_output_content(tmp_path: Path) -> None:
    structure_path, _ = build_structure_bundle(tmp_path)
    output_root = tmp_path / "speaker_turns"
    first = process_speaker_turn_document(
        structure_path=structure_path,
        output_root=output_root,
    )
    paths = output_paths(output_root)
    before_data = {path: path.read_bytes() for path in paths[:3]}
    first_without_timestamp = dict(first)
    first_without_timestamp.pop("processed_at_utc")
    first_without_timestamp.pop("reused")

    second = process_speaker_turn_document(
        structure_path=structure_path,
        output_root=output_root,
        force=True,
    )
    second_without_timestamp = dict(second)
    second_without_timestamp.pop("processed_at_utc")
    second_without_timestamp.pop("reused")

    assert second["reused"] is False
    assert before_data == {path: path.read_bytes() for path in paths[:3]}
    assert second_without_timestamp == first_without_timestamp


def test_promotion_failure_restores_prior_outputs_and_cleans_transaction_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structure_path, _ = build_structure_bundle(tmp_path)
    output_root = tmp_path / "speaker_turns"
    process_speaker_turn_document(structure_path=structure_path, output_root=output_root)
    paths = output_paths(output_root)
    before = {path: path.read_bytes() for path in paths}
    original_replace = speaker_turn_pipeline._replace_path
    failure_raised = False

    def fail_during_second_promotion(source: Path, destination: Path) -> None:
        nonlocal failure_raised

        if source.name == "turn_segments.jsonl.part" and not failure_raised:
            failure_raised = True
            raise OSError("Synthetic promotion failure.")

        original_replace(source, destination)

    monkeypatch.setattr(
        speaker_turn_pipeline,
        "_replace_path",
        fail_during_second_promotion,
    )

    with pytest.raises(SpeakerTurnPipelineError, match="promote"):
        process_speaker_turn_document(
            structure_path=structure_path,
            output_root=output_root,
            force=True,
        )

    assert failure_raised is True
    assert before == {path: path.read_bytes() for path in paths}
    assert not list((output_root / "record-1").glob("*.part"))
    assert not list((output_root / "record-1").glob("*.bak"))
