import json
from pathlib import Path
from typing import Any

import pytest

from argentine_deputies_discursive_distance import modeling_corpus
from argentine_deputies_discursive_distance.modeling_corpus import (
    MANIFEST_FILENAME,
    ModelingCorpusError,
    _temporal_period,
    export_modeling_corpus,
)
from argentine_deputies_discursive_distance.pdf_pipeline import sha256_file


def words(count: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{index}" for index in range(1, count + 1))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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


def override_payload(overrides: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "override_manifest_version": "1",
        "overrides": [] if overrides is None else overrides,
        "required_content_classifier_version": "2",
        "required_pipeline_version": "2",
    }


def write_overrides(
    tmp_path: Path,
    overrides: list[dict[str, Any]] | None = None,
) -> Path:
    path = tmp_path / "config" / "modeling_turn_overrides.json"
    write_json(path, override_payload(overrides))
    return path


def write_metadata(
    tmp_path: Path,
    records: list[dict[str, Any]],
) -> Path:
    path = tmp_path / "qa" / "full_corpus_run_summary.json"
    write_json(path, {"records": records})
    return path


def metadata_record(
    source_record_id: str,
    session_date: str = "2020-01-01",
    session_category: str = "legislative_debate",
) -> dict[str, Any]:
    return {
        "meeting_number": "1",
        "session_category": session_category,
        "session_date": session_date,
        "source_record_id": source_record_id,
    }


def turn_record(
    *,
    source_record_id: str,
    turn_index: int,
    text: str,
    speaker_family: str | None = "named_or_role_unspecified",
    normalized_label: str | None = "ALPHA",
    raw_label: str | None = "Alpha",
    speech_word_count: int | None = None,
) -> dict[str, Any]:
    marker = None if raw_label is None else {"raw_label": raw_label}

    return {
        "marker": marker,
        "normalized_label": normalized_label,
        "source_record_id": source_record_id,
        "speaker_family": speaker_family,
        "speech_word_count": (
            len(text.split()) if speech_word_count is None else speech_word_count
        ),
        "turn_index": turn_index,
    }


def span_record(
    *,
    source_record_id: str,
    turn_index: int,
    content_span_index: int,
    text: str,
    page_number: int = 1,
    start: int = 0,
    content_kind: str = "spoken_text",
    include_in_speech: bool = True,
    source_segment_index: int | None = None,
) -> dict[str, Any]:
    return {
        "block_reference": f"p{page_number}:b{content_span_index}",
        "content_kind": content_kind,
        "content_span_index": content_span_index,
        "end": start + len(text),
        "include_in_speech": include_in_speech,
        "page_number": page_number,
        "reading_order": content_span_index,
        "source_record_id": source_record_id,
        "source_segment_index": (
            content_span_index if source_segment_index is None else source_segment_index
        ),
        "start": start,
        "text": text,
        "turn_index": turn_index,
        "word_count": len(text.split()),
    }


def make_speaker_turn_document(
    root: Path,
    *,
    source_record_id: str = "source-a",
    turns: list[dict[str, Any]],
    spans: list[dict[str, Any]],
    segments: list[dict[str, Any]] | None = None,
    pipeline_version: str = "2",
    classifier_version: str = "2",
) -> None:
    source_dir = root / source_record_id
    turns_path = source_dir / "turns.jsonl"
    spans_path = source_dir / "content_spans.jsonl"
    segments_path = source_dir / "turn_segments.jsonl"

    write_jsonl(turns_path, turns)
    write_jsonl(spans_path, spans)
    write_jsonl(segments_path, [] if segments is None else segments)
    write_json(
        source_dir / "speaker_turns.json",
        {
            "content_classifier_version": classifier_version,
            "outputs": {
                "content_spans_path": str(spans_path),
                "turn_segments_path": str(segments_path),
                "turns_path": str(turns_path),
            },
            "pipeline_version": pipeline_version,
            "source_record_id": source_record_id,
        },
    )


def single_turn_fixture(
    tmp_path: Path,
    *,
    text: str = "",
    source_record_id: str = "source-a",
    speaker_family: str | None = "named_or_role_unspecified",
    normalized_label: str | None = "ALPHA",
    raw_label: str | None = "Alpha",
    pipeline_version: str = "2",
    classifier_version: str = "2",
    session_date: str = "2020-01-01",
    session_category: str = "legislative_debate",
) -> tuple[Path, Path, Path]:
    root = tmp_path / "speaker_turns"
    turns = [
        turn_record(
            source_record_id=source_record_id,
            turn_index=1,
            text=text,
            speaker_family=speaker_family,
            normalized_label=normalized_label,
            raw_label=raw_label,
        )
    ]
    spans = (
        []
        if not text
        else [
            span_record(
                source_record_id=source_record_id,
                turn_index=1,
                content_span_index=1,
                text=text,
            )
        ]
    )
    make_speaker_turn_document(
        root,
        source_record_id=source_record_id,
        turns=turns,
        spans=spans,
        pipeline_version=pipeline_version,
        classifier_version=classifier_version,
    )
    metadata_path = write_metadata(
        tmp_path,
        [metadata_record(source_record_id, session_date, session_category)],
    )
    overrides_path = write_overrides(tmp_path)
    return root, metadata_path, overrides_path


def run_export(
    tmp_path: Path,
    *,
    root: Path,
    metadata_path: Path,
    overrides_path: Path,
    minimum_words: int = 1,
    maximum_chunk_words: int = 300,
    output_name: str = "modeling",
    force: bool = False,
) -> dict[str, Any]:
    return export_modeling_corpus(
        speaker_turn_root=root,
        overrides_path=overrides_path,
        metadata_summary_path=metadata_path,
        output_root=tmp_path / output_name,
        minimum_words=minimum_words,
        maximum_chunk_words=maximum_chunk_words,
        force=force,
    )


def test_override_manifest_accepts_utf8_bom(tmp_path: Path) -> None:
    root, metadata_path, overrides_path = single_turn_fixture(tmp_path, text=words(25))
    overrides_path.write_text(
        json.dumps(override_payload(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8-sig",
    )

    run_export(tmp_path, root=root, metadata_path=metadata_path, overrides_path=overrides_path)

    assert (tmp_path / "modeling" / MANIFEST_FILENAME).is_file()


def test_metadata_summary_accepts_utf8_bom(tmp_path: Path) -> None:
    root, metadata_path, overrides_path = single_turn_fixture(tmp_path, text=words(25))
    metadata_path.write_text(
        json.dumps({"records": [metadata_record("source-a")]}, ensure_ascii=False) + "\n",
        encoding="utf-8-sig",
    )

    run_export(tmp_path, root=root, metadata_path=metadata_path, overrides_path=overrides_path)

    assert (tmp_path / "modeling" / MANIFEST_FILENAME).is_file()


def test_manifest_paths_do_not_override_local_copied_files(tmp_path: Path) -> None:
    stale_root = tmp_path / "production" / "speaker_turns"
    local_root = tmp_path / "smoke" / "speaker_turns"
    make_speaker_turn_document(
        stale_root,
        source_record_id="source-a",
        turns=[
            turn_record(
                source_record_id="source-a",
                turn_index=1,
                text=words(25, "stale"),
            )
        ],
        spans=[
            span_record(
                source_record_id="source-a",
                turn_index=1,
                content_span_index=1,
                text=words(25, "stale"),
            )
        ],
    )
    make_speaker_turn_document(
        local_root,
        source_record_id="source-a",
        turns=[
            turn_record(
                source_record_id="source-a",
                turn_index=1,
                text=words(25, "local"),
            )
        ],
        spans=[
            span_record(
                source_record_id="source-a",
                turn_index=1,
                content_span_index=1,
                text=words(25, "local"),
            )
        ],
    )
    local_manifest = local_root / "source-a" / "speaker_turns.json"
    stale_source_dir = stale_root / "source-a"
    payload = json.loads(local_manifest.read_text(encoding="utf-8"))
    payload["outputs"] = {
        "content_spans_path": str(stale_source_dir / "content_spans.jsonl"),
        "turn_segments_path": str(stale_source_dir / "turn_segments.jsonl"),
        "turns_path": str(stale_source_dir / "turns.jsonl"),
    }
    write_json(local_manifest, payload)
    metadata_path = write_metadata(tmp_path, [metadata_record("source-a")])
    overrides_path = write_overrides(tmp_path)

    run_export(
        tmp_path,
        root=local_root,
        metadata_path=metadata_path,
        overrides_path=overrides_path,
    )
    source_turns = read_jsonl(tmp_path / "modeling" / "source_turns.jsonl")

    assert source_turns[0]["exact_retained_text"] == words(25, "local")


def test_two_spoken_spans_without_boundary_whitespace_are_token_separated(
    tmp_path: Path,
) -> None:
    root = tmp_path / "speaker_turns"
    spans = [
        span_record(
            source_record_id="source-a",
            turn_index=1,
            content_span_index=1,
            text="primera palabra",
        ),
        span_record(
            source_record_id="source-a",
            turn_index=1,
            content_span_index=2,
            text="segunda parte",
        ),
    ]
    make_speaker_turn_document(
        root,
        turns=[
            turn_record(
                source_record_id="source-a",
                turn_index=1,
                text="",
                speech_word_count=4,
            )
        ],
        spans=spans,
    )
    metadata_path = write_metadata(tmp_path, [metadata_record("source-a")])
    overrides_path = write_overrides(tmp_path)

    run_export(tmp_path, root=root, metadata_path=metadata_path, overrides_path=overrides_path)
    source_turns = read_jsonl(tmp_path / "modeling" / "source_turns.jsonl")

    assert source_turns[0]["exact_retained_text"] == "primera palabra\n\nsegunda parte"
    assert "palabrasegunda" not in source_turns[0]["exact_retained_text"]
    assert source_turns[0]["original_upstream_speech_word_count"] == 4
    assert source_turns[0]["post_override_modeling_word_count"] == 4
    assert source_turns[0]["synthetic_separator_count"] == 1


def test_three_spoken_spans_across_multiple_source_segments(tmp_path: Path) -> None:
    root = tmp_path / "speaker_turns"
    spans = [
        span_record(
            source_record_id="source-a",
            turn_index=1,
            content_span_index=1,
            source_segment_index=1,
            text="uno dos",
        ),
        span_record(
            source_record_id="source-a",
            turn_index=1,
            content_span_index=2,
            source_segment_index=2,
            text="tres cuatro",
        ),
        span_record(
            source_record_id="source-a",
            turn_index=1,
            content_span_index=3,
            source_segment_index=3,
            text=words(21, "cinco"),
        ),
    ]
    make_speaker_turn_document(
        root,
        turns=[
            turn_record(
                source_record_id="source-a",
                turn_index=1,
                text="",
                speech_word_count=25,
            )
        ],
        spans=spans,
    )
    metadata_path = write_metadata(tmp_path, [metadata_record("source-a")])
    overrides_path = write_overrides(tmp_path)

    run_export(tmp_path, root=root, metadata_path=metadata_path, overrides_path=overrides_path)
    source_turns = read_jsonl(tmp_path / "modeling" / "source_turns.jsonl")

    assert source_turns[0]["exact_retained_text"].count("\n\n") == 2
    assert [item["fragment_kind"] for item in source_turns[0]["provenance"]] == [
        "source_fragment",
        "synthetic_separator",
        "source_fragment",
        "synthetic_separator",
        "source_fragment",
    ]


def test_non_spoken_span_between_spoken_spans_is_omitted_but_token_separated(
    tmp_path: Path,
) -> None:
    root = tmp_path / "speaker_turns"
    spans = [
        span_record(
            source_record_id="source-a",
            turn_index=1,
            content_span_index=1,
            text="primera palabra",
        ),
        span_record(
            source_record_id="source-a",
            turn_index=1,
            content_span_index=2,
            text="(Aplausos.)",
            content_kind="stage_direction",
            include_in_speech=False,
        ),
        span_record(
            source_record_id="source-a",
            turn_index=1,
            content_span_index=3,
            text=words(23, "segunda"),
        ),
    ]
    make_speaker_turn_document(
        root,
        turns=[
            turn_record(
                source_record_id="source-a",
                turn_index=1,
                text="",
                speech_word_count=25,
            )
        ],
        spans=spans,
    )
    metadata_path = write_metadata(tmp_path, [metadata_record("source-a")])
    overrides_path = write_overrides(tmp_path)

    run_export(tmp_path, root=root, metadata_path=metadata_path, overrides_path=overrides_path)
    source_turn = read_jsonl(tmp_path / "modeling" / "source_turns.jsonl")[0]

    assert "(Aplausos.)" not in source_turn["exact_retained_text"]
    assert source_turn["exact_retained_text"].startswith("primera palabra\n\nsegunda")


def test_anchor_override_can_begin_in_later_spoken_span(tmp_path: Path) -> None:
    root = tmp_path / "speaker_turns"
    anchor_text = "Y la verdad " + words(25, "retained")
    spans = [
        span_record(
            source_record_id="source-a",
            turn_index=1,
            content_span_index=1,
            text="prefacio descartado",
        ),
        span_record(
            source_record_id="source-a",
            turn_index=1,
            content_span_index=2,
            text=anchor_text,
        ),
    ]
    make_speaker_turn_document(
        root,
        turns=[
            turn_record(
                source_record_id="source-a",
                turn_index=1,
                text="",
                speaker_family="chair",
                normalized_label="PRESIDENTE",
                raw_label="Presidente",
                speech_word_count=30,
            )
        ],
        spans=spans,
    )
    metadata_path = write_metadata(tmp_path, [metadata_record("source-a")])
    overrides_path = write_overrides(
        tmp_path,
        [
            {
                "action": "retain_from_anchor_and_relabel",
                "expected_anchor_occurrences": 1,
                "expected_speech_word_count": 30,
                "normalized_label": "PRESIDENTA DE LA NACION",
                "reason": "manual_retain",
                "session_date": "2020-01-01",
                "source_record_id": "source-a",
                "speaker_family": "executive_official",
                "start_anchor": "Y la verdad",
                "turn_index": 1,
            }
        ],
    )

    run_export(tmp_path, root=root, metadata_path=metadata_path, overrides_path=overrides_path)
    source_turn = read_jsonl(tmp_path / "modeling" / "source_turns.jsonl")[0]
    ledger = read_jsonl(tmp_path / "modeling" / "exclusion_ledger.jsonl")

    assert source_turn["exact_retained_text"].startswith("Y la verdad")
    assert source_turn["effective_speaker_family"] == "executive_official"
    assert ledger[0]["exact_excluded_text"] == "prefacio descartado\n\n"


def test_chunk_provenance_can_include_synthetic_separator(tmp_path: Path) -> None:
    root = tmp_path / "speaker_turns"
    spans = [
        span_record(
            source_record_id="source-a",
            turn_index=1,
            content_span_index=1,
            text="primera palabra",
        ),
        span_record(
            source_record_id="source-a",
            turn_index=1,
            content_span_index=2,
            text="segunda parte " + words(21, "resto"),
        ),
    ]
    make_speaker_turn_document(
        root,
        turns=[
            turn_record(
                source_record_id="source-a",
                turn_index=1,
                text="",
                speech_word_count=25,
            )
        ],
        spans=spans,
    )
    metadata_path = write_metadata(tmp_path, [metadata_record("source-a")])
    overrides_path = write_overrides(tmp_path)

    run_export(
        tmp_path,
        root=root,
        metadata_path=metadata_path,
        overrides_path=overrides_path,
        maximum_chunk_words=2,
    )
    first_document = read_jsonl(tmp_path / "modeling" / "documents.jsonl")[0]

    assert first_document["exact_text"] == "primera palabra\n\n"
    assert [item["fragment_kind"] for item in first_document["provenance"]] == [
        "source_fragment",
        "synthetic_separator",
    ]
    assert "source_start" not in first_document["provenance"][1]


def test_real_schema_multispan_fixture_with_nonempty_turn_segments(tmp_path: Path) -> None:
    root = tmp_path / "speaker_turns"
    first = " Con la presencia de 129 señores diputados."
    second = "–Puestos de pie los señores diputados."
    third = "2\nHIMNO NACIONAL ARGENTINO " + words(20, "himno")
    make_speaker_turn_document(
        root,
        source_record_id="8517a4af91d960b3040d",
        turns=[
            turn_record(
                source_record_id="8517a4af91d960b3040d",
                turn_index=1,
                text="",
                speaker_family="named_or_role_unspecified",
                normalized_label="MONZO",
                raw_label="Monzó",
                speech_word_count=len(first.split()) + len(second.split()) + len(third.split()),
            )
        ],
        spans=[
            span_record(
                source_record_id="8517a4af91d960b3040d",
                turn_index=1,
                content_span_index=1,
                source_segment_index=1,
                text=first,
            ),
            span_record(
                source_record_id="8517a4af91d960b3040d",
                turn_index=1,
                content_span_index=2,
                source_segment_index=2,
                text=second,
            ),
            span_record(
                source_record_id="8517a4af91d960b3040d",
                turn_index=1,
                content_span_index=3,
                source_segment_index=2,
                text="(Aplausos.)",
                content_kind="stage_direction",
                include_in_speech=False,
            ),
            span_record(
                source_record_id="8517a4af91d960b3040d",
                turn_index=1,
                content_span_index=4,
                source_segment_index=3,
                text=third,
            ),
        ],
        segments=[
            span_record(
                source_record_id="8517a4af91d960b3040d",
                turn_index=1,
                content_span_index=1,
                text=first,
            ),
            span_record(
                source_record_id="8517a4af91d960b3040d",
                turn_index=1,
                content_span_index=2,
                text=second + "(Aplausos.)",
            ),
            span_record(
                source_record_id="8517a4af91d960b3040d",
                turn_index=1,
                content_span_index=3,
                text=third,
            ),
        ],
    )
    metadata_path = write_metadata(
        tmp_path,
        [metadata_record("8517a4af91d960b3040d")],
    )
    overrides_path = write_overrides(tmp_path)

    run_export(tmp_path, root=root, metadata_path=metadata_path, overrides_path=overrides_path)
    source_turn = read_jsonl(tmp_path / "modeling" / "source_turns.jsonl")[0]

    assert source_turn["original_upstream_speech_word_count"] == (
        len(first.split()) + len(second.split()) + len(third.split())
    )
    assert "(Aplausos.)" not in source_turn["exact_retained_text"]
    assert source_turn["exact_retained_text"].count("\n\n") == 2


def test_pipeline_version_mismatch_fails(tmp_path: Path) -> None:
    root, metadata_path, overrides_path = single_turn_fixture(
        tmp_path,
        text=words(25),
        pipeline_version="1",
    )

    with pytest.raises(ModelingCorpusError, match="Pipeline version mismatch"):
        run_export(tmp_path, root=root, metadata_path=metadata_path, overrides_path=overrides_path)


def test_classifier_version_mismatch_fails(tmp_path: Path) -> None:
    root, metadata_path, overrides_path = single_turn_fixture(
        tmp_path,
        text=words(25),
        classifier_version="1",
    )

    with pytest.raises(ModelingCorpusError, match="Content classifier version mismatch"):
        run_export(tmp_path, root=root, metadata_path=metadata_path, overrides_path=overrides_path)


def test_missing_override_source_fails(tmp_path: Path) -> None:
    root, metadata_path, _ = single_turn_fixture(tmp_path, text=words(25))
    overrides_path = write_overrides(
        tmp_path,
        [
            {
                "action": "exclude_turn",
                "expected_speech_word_count": 25,
                "reason": "manual",
                "session_date": "2020-01-01",
                "source_record_id": "missing",
                "turn_index": 1,
            }
        ],
    )

    with pytest.raises(ModelingCorpusError, match="Override source_record_id"):
        run_export(tmp_path, root=root, metadata_path=metadata_path, overrides_path=overrides_path)


def test_missing_override_turn_fails(tmp_path: Path) -> None:
    root, metadata_path, _ = single_turn_fixture(tmp_path, text=words(25))
    overrides_path = write_overrides(
        tmp_path,
        [
            {
                "action": "exclude_turn",
                "expected_speech_word_count": 25,
                "reason": "manual",
                "session_date": "2020-01-01",
                "source_record_id": "source-a",
                "turn_index": 2,
            }
        ],
    )

    with pytest.raises(ModelingCorpusError, match="Override turn"):
        run_export(tmp_path, root=root, metadata_path=metadata_path, overrides_path=overrides_path)


def test_override_expected_word_count_mismatch_fails(tmp_path: Path) -> None:
    root, metadata_path, _ = single_turn_fixture(tmp_path, text=words(25))
    overrides_path = write_overrides(
        tmp_path,
        [
            {
                "action": "exclude_turn",
                "expected_speech_word_count": 24,
                "reason": "manual",
                "session_date": "2020-01-01",
                "source_record_id": "source-a",
                "turn_index": 1,
            }
        ],
    )

    with pytest.raises(ModelingCorpusError, match="expected_speech_word_count"):
        run_export(tmp_path, root=root, metadata_path=metadata_path, overrides_path=overrides_path)


def test_anchor_occurrence_mismatch_fails(tmp_path: Path) -> None:
    root, metadata_path, _ = single_turn_fixture(tmp_path, text="anchor " + words(25))
    overrides_path = write_overrides(
        tmp_path,
        [
            {
                "action": "retain_from_anchor_and_relabel",
                "expected_anchor_occurrences": 2,
                "expected_speech_word_count": 26,
                "normalized_label": "PRESIDENTA DE LA NACION",
                "reason": "manual",
                "session_date": "2020-01-01",
                "source_record_id": "source-a",
                "speaker_family": "executive_official",
                "start_anchor": "anchor",
                "turn_index": 1,
            }
        ],
    )

    with pytest.raises(ModelingCorpusError, match="anchor occurrence"):
        run_export(tmp_path, root=root, metadata_path=metadata_path, overrides_path=overrides_path)


def test_exclude_turn_is_recorded_exactly_once(tmp_path: Path) -> None:
    root, metadata_path, _ = single_turn_fixture(tmp_path, text=words(30))
    overrides_path = write_overrides(
        tmp_path,
        [
            {
                "action": "exclude_turn",
                "expected_speech_word_count": 30,
                "reason": "manual_exclude",
                "session_date": "2020-01-01",
                "source_record_id": "source-a",
                "turn_index": 1,
            }
        ],
    )

    manifest = run_export(
        tmp_path,
        root=root,
        metadata_path=metadata_path,
        overrides_path=overrides_path,
    )
    decisions = read_jsonl(tmp_path / "modeling" / "turn_decisions.jsonl")
    ledger = read_jsonl(tmp_path / "modeling" / "exclusion_ledger.jsonl")

    assert decisions[0]["decision"] == "excluded_by_override"
    assert len(ledger) == 1
    assert ledger[0]["exclusion_reason"] == "excluded_by_override"
    assert manifest["override_application_ledger"][0]["applied_count"] == 1


def test_retain_from_anchor_relabels_and_records_prefix(tmp_path: Path) -> None:
    text = "prefacio descartado " + "Y la verdad " + words(25)
    root, metadata_path, _ = single_turn_fixture(
        tmp_path,
        text=text,
        speaker_family="chair",
        normalized_label="PRESIDENTE",
        raw_label="Presidente",
    )
    overrides_path = write_overrides(
        tmp_path,
        [
            {
                "action": "retain_from_anchor_and_relabel",
                "expected_anchor_occurrences": 1,
                "expected_speech_word_count": len(text.split()),
                "normalized_label": "PRESIDENTA DE LA NACION",
                "reason": "manual_retain",
                "session_date": "2020-01-01",
                "source_record_id": "source-a",
                "speaker_family": "executive_official",
                "start_anchor": "Y la verdad",
                "turn_index": 1,
            }
        ],
    )

    run_export(tmp_path, root=root, metadata_path=metadata_path, overrides_path=overrides_path)
    source_turns = read_jsonl(tmp_path / "modeling" / "source_turns.jsonl")
    ledger = read_jsonl(tmp_path / "modeling" / "exclusion_ledger.jsonl")

    assert source_turns[0]["exact_retained_text"].startswith("Y la verdad")
    assert source_turns[0]["effective_speaker_family"] == "executive_official"
    assert source_turns[0]["effective_normalized_label"] == "PRESIDENTA DE LA NACION"
    assert len(ledger) == 1
    assert ledger[0]["exclusion_reason"] == "discarded_prefix_by_override"
    assert ledger[0]["exact_excluded_text"] == "prefacio descartado "


def test_chair_and_secretary_turns_are_excluded(tmp_path: Path) -> None:
    root = tmp_path / "speaker_turns"
    make_speaker_turn_document(
        root,
        source_record_id="source-a",
        turns=[
            turn_record(
                source_record_id="source-a",
                turn_index=1,
                text=words(30, "chair"),
                speaker_family="chair",
                normalized_label="PRESIDENTE",
            ),
            turn_record(
                source_record_id="source-a",
                turn_index=2,
                text=words(30, "sec"),
                speaker_family="chamber_secretary",
                normalized_label="SECRETARIO",
            ),
        ],
        spans=[
            span_record(
                source_record_id="source-a",
                turn_index=1,
                content_span_index=1,
                text=words(30, "chair"),
            ),
            span_record(
                source_record_id="source-a",
                turn_index=2,
                content_span_index=1,
                text=words(30, "sec"),
            ),
        ],
    )
    metadata_path = write_metadata(tmp_path, [metadata_record("source-a")])
    overrides_path = write_overrides(tmp_path)

    run_export(tmp_path, root=root, metadata_path=metadata_path, overrides_path=overrides_path)
    decisions = read_jsonl(tmp_path / "modeling" / "turn_decisions.jsonl")

    assert [decision["decision"] for decision in decisions] == [
        "excluded_speaker_family",
        "excluded_speaker_family",
    ]


def test_named_and_executive_turns_are_retained(tmp_path: Path) -> None:
    root = tmp_path / "speaker_turns"
    make_speaker_turn_document(
        root,
        source_record_id="source-a",
        turns=[
            turn_record(source_record_id="source-a", turn_index=1, text=words(30, "named")),
            turn_record(
                source_record_id="source-a",
                turn_index=2,
                text=words(30, "exec"),
                speaker_family="executive_official",
                normalized_label="JEFE DE GABINETE",
            ),
        ],
        spans=[
            span_record(
                source_record_id="source-a",
                turn_index=1,
                content_span_index=1,
                text=words(30, "named"),
            ),
            span_record(
                source_record_id="source-a",
                turn_index=2,
                content_span_index=1,
                text=words(30, "exec"),
            ),
        ],
    )
    metadata_path = write_metadata(tmp_path, [metadata_record("source-a")])
    overrides_path = write_overrides(tmp_path)

    run_export(tmp_path, root=root, metadata_path=metadata_path, overrides_path=overrides_path)
    source_turns = read_jsonl(tmp_path / "modeling" / "source_turns.jsonl")

    assert [record["effective_speaker_family"] for record in source_turns] == [
        "named_or_role_unspecified",
        "executive_official",
    ]


def test_zero_speech_turn_is_recorded_as_excluded_zero_speech(tmp_path: Path) -> None:
    root, metadata_path, overrides_path = single_turn_fixture(tmp_path, text="")

    run_export(tmp_path, root=root, metadata_path=metadata_path, overrides_path=overrides_path)
    decisions = read_jsonl(tmp_path / "modeling" / "turn_decisions.jsonl")

    assert decisions[0]["decision"] == "excluded_zero_speech"


def test_24_word_turn_is_excluded(tmp_path: Path) -> None:
    root, metadata_path, overrides_path = single_turn_fixture(tmp_path, text=words(24))

    run_export(
        tmp_path,
        root=root,
        metadata_path=metadata_path,
        overrides_path=overrides_path,
        minimum_words=25,
    )
    decisions = read_jsonl(tmp_path / "modeling" / "turn_decisions.jsonl")

    assert decisions[0]["decision"] == "excluded_below_minimum_words"


def test_25_word_turn_is_retained(tmp_path: Path) -> None:
    root, metadata_path, overrides_path = single_turn_fixture(tmp_path, text=words(25))

    run_export(
        tmp_path,
        root=root,
        metadata_path=metadata_path,
        overrides_path=overrides_path,
        minimum_words=25,
    )
    source_turns = read_jsonl(tmp_path / "modeling" / "source_turns.jsonl")

    assert len(source_turns) == 1
    assert source_turns[0]["post_override_modeling_word_count"] == 25


def test_under_300_words_creates_one_chunk(tmp_path: Path) -> None:
    root, metadata_path, overrides_path = single_turn_fixture(tmp_path, text=words(299))

    run_export(tmp_path, root=root, metadata_path=metadata_path, overrides_path=overrides_path)
    documents = read_jsonl(tmp_path / "modeling" / "documents.jsonl")

    assert len(documents) == 1
    assert documents[0]["chunk_count_for_turn"] == 1


def test_over_300_words_creates_multiple_chunks_and_caps_size(tmp_path: Path) -> None:
    root, metadata_path, overrides_path = single_turn_fixture(tmp_path, text=words(650))

    run_export(tmp_path, root=root, metadata_path=metadata_path, overrides_path=overrides_path)
    documents = read_jsonl(tmp_path / "modeling" / "documents.jsonl")

    assert len(documents) == 3
    assert max(document["word_count"] for document in documents) <= 300


def test_sentence_and_paragraph_boundaries_are_preferred(tmp_path: Path) -> None:
    text = "uno dos tres cuatro cinco seis.\n\nsiete ocho nueve diez once doce."
    root, metadata_path, overrides_path = single_turn_fixture(tmp_path, text=text)

    run_export(
        tmp_path,
        root=root,
        metadata_path=metadata_path,
        overrides_path=overrides_path,
        maximum_chunk_words=8,
    )
    documents = read_jsonl(tmp_path / "modeling" / "documents.jsonl")

    assert documents[0]["exact_text"] == "uno dos tres cuatro cinco seis.\n\n"
    assert documents[1]["exact_text"].startswith("siete ocho")


def test_late_sentence_boundary_beats_pathologically_early_paragraph(
    tmp_path: Path,
) -> None:
    text = (
        "uno dos.\n\n"
        "tres cuatro cinco seis siete ocho nueve diez once doce trece catorce."
        " quince dieciseis diecisiete"
    )
    root, metadata_path, overrides_path = single_turn_fixture(tmp_path, text=text)

    run_export(
        tmp_path,
        root=root,
        metadata_path=metadata_path,
        overrides_path=overrides_path,
        maximum_chunk_words=14,
    )
    documents = read_jsonl(tmp_path / "modeling" / "documents.jsonl")

    assert documents[0]["word_count"] == 14
    assert documents[0]["exact_text"].endswith("catorce. ")


def test_latter_half_sentence_beats_early_paragraph_when_no_final_quarter_boundary(
    tmp_path: Path,
) -> None:
    text = words(20, "early") + "\n\n" + words(180, "middle") + ". " + words(120, "tail")
    root, metadata_path, overrides_path = single_turn_fixture(
        tmp_path,
        text=text,
    )

    run_export(
        tmp_path,
        root=root,
        metadata_path=metadata_path,
        overrides_path=overrides_path,
        maximum_chunk_words=300,
    )
    documents = read_jsonl(tmp_path / "modeling" / "documents.jsonl")

    assert documents[0]["word_count"] == 200
    assert documents[0]["exact_text"].endswith("middle180. ")


def test_only_very_early_boundary_uses_hard_chunk_cap(
    tmp_path: Path,
) -> None:
    text = words(20, "early") + "\n\n" + words(320, "tail")
    root, metadata_path, overrides_path = single_turn_fixture(
        tmp_path,
        text=text,
    )

    run_export(
        tmp_path,
        root=root,
        metadata_path=metadata_path,
        overrides_path=overrides_path,
        maximum_chunk_words=300,
    )
    documents = read_jsonl(tmp_path / "modeling" / "documents.jsonl")

    assert documents[0]["word_count"] == 300
    assert documents[1]["word_count"] == 40


def test_final_quarter_paragraph_keeps_priority_over_later_sentence(
    tmp_path: Path,
) -> None:
    text = words(240, "paragraph") + "\n\n" + words(40, "sentence") + ". " + words(60, "tail")
    root, metadata_path, overrides_path = single_turn_fixture(
        tmp_path,
        text=text,
    )

    run_export(
        tmp_path,
        root=root,
        metadata_path=metadata_path,
        overrides_path=overrides_path,
        maximum_chunk_words=300,
    )
    documents = read_jsonl(tmp_path / "modeling" / "documents.jsonl")

    assert documents[0]["word_count"] == 240
    assert documents[0]["exact_text"].endswith("paragraph240\n\n")


def test_sentence_longer_than_limit_is_split_by_whitespace(tmp_path: Path) -> None:
    root, metadata_path, overrides_path = single_turn_fixture(tmp_path, text=words(12))

    run_export(
        tmp_path,
        root=root,
        metadata_path=metadata_path,
        overrides_path=overrides_path,
        maximum_chunk_words=5,
    )
    documents = read_jsonl(tmp_path / "modeling" / "documents.jsonl")

    assert [document["word_count"] for document in documents] == [5, 5, 2]


def test_chunks_exactly_reconstruct_source_turn(tmp_path: Path) -> None:
    text = "uno dos tres.\n\n" + words(12)
    root, metadata_path, overrides_path = single_turn_fixture(tmp_path, text=text)

    run_export(
        tmp_path,
        root=root,
        metadata_path=metadata_path,
        overrides_path=overrides_path,
        maximum_chunk_words=5,
    )
    source_turn = read_jsonl(tmp_path / "modeling" / "source_turns.jsonl")[0]
    documents = read_jsonl(tmp_path / "modeling" / "documents.jsonl")

    assert (
        "".join(document["exact_text"] for document in documents)
        == source_turn["exact_retained_text"]
    )


def test_different_turns_and_speakers_are_never_merged(tmp_path: Path) -> None:
    root = tmp_path / "speaker_turns"
    make_speaker_turn_document(
        root,
        source_record_id="source-a",
        turns=[
            turn_record(source_record_id="source-a", turn_index=1, text=words(5, "a")),
            turn_record(
                source_record_id="source-a",
                turn_index=2,
                text=words(5, "b"),
                normalized_label="BETA",
                raw_label="Beta",
            ),
        ],
        spans=[
            span_record(
                source_record_id="source-a",
                turn_index=1,
                content_span_index=1,
                text=words(5, "a"),
            ),
            span_record(
                source_record_id="source-a",
                turn_index=2,
                content_span_index=1,
                text=words(5, "b"),
            ),
        ],
    )
    metadata_path = write_metadata(tmp_path, [metadata_record("source-a")])
    overrides_path = write_overrides(tmp_path)

    run_export(
        tmp_path,
        root=root,
        metadata_path=metadata_path,
        overrides_path=overrides_path,
        maximum_chunk_words=20,
    )
    documents = read_jsonl(tmp_path / "modeling" / "documents.jsonl")

    assert len(documents) == 2
    assert {document["turn_index"] for document in documents} == {1, 2}
    assert {document["speaker_label"] for document in documents} == {"Alpha", "Beta"}


@pytest.mark.parametrize(
    ("session_date", "expected"),
    [
        ("2008-01-01", "2008-2011"),
        ("2011-12-31", "2008-2011"),
        ("2012-01-01", "2012-2015"),
        ("2015-12-31", "2012-2015"),
        ("2016-01-01", "2016-2019"),
        ("2019-12-31", "2016-2019"),
        ("2020-01-01", "2020-2023"),
        ("2023-12-31", "2020-2023"),
        ("2024-01-01", "2024-2025"),
        ("2025-12-31", "2024-2025"),
    ],
)
def test_temporal_period_boundaries(session_date: str, expected: str) -> None:
    assert _temporal_period(session_date)[1] == expected


def test_dates_outside_range_fail(tmp_path: Path) -> None:
    root, _, overrides_path = single_turn_fixture(tmp_path, text=words(25))
    metadata_path = write_metadata(tmp_path, [metadata_record("source-a", "2026-01-01")])

    with pytest.raises(ModelingCorpusError, match="outside 2008-2025"):
        run_export(tmp_path, root=root, metadata_path=metadata_path, overrides_path=overrides_path)


def test_output_ordering_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "speaker_turns"
    make_speaker_turn_document(
        root,
        source_record_id="source-b",
        turns=[turn_record(source_record_id="source-b", turn_index=2, text=words(3, "b"))],
        spans=[
            span_record(
                source_record_id="source-b",
                turn_index=2,
                content_span_index=1,
                text=words(3, "b"),
            )
        ],
    )
    make_speaker_turn_document(
        root,
        source_record_id="source-a",
        turns=[turn_record(source_record_id="source-a", turn_index=1, text=words(3, "a"))],
        spans=[
            span_record(
                source_record_id="source-a",
                turn_index=1,
                content_span_index=1,
                text=words(3, "a"),
            )
        ],
    )
    metadata_path = write_metadata(
        tmp_path,
        [
            metadata_record("source-b", "2020-01-02"),
            metadata_record("source-a", "2020-01-01"),
        ],
    )
    overrides_path = write_overrides(tmp_path)

    run_export(tmp_path, root=root, metadata_path=metadata_path, overrides_path=overrides_path)
    documents = read_jsonl(tmp_path / "modeling" / "documents.jsonl")

    assert [document["source_record_id"] for document in documents] == ["source-a", "source-b"]


def test_repeated_exports_are_byte_identical_except_manifest_timestamp(tmp_path: Path) -> None:
    root, metadata_path, overrides_path = single_turn_fixture(tmp_path, text=words(30))
    first = run_export(
        tmp_path,
        root=root,
        metadata_path=metadata_path,
        overrides_path=overrides_path,
    )
    first_bytes = {
        filename: (tmp_path / "modeling" / filename).read_bytes()
        for filename in (
            "documents.jsonl",
            "source_turns.jsonl",
            "turn_decisions.jsonl",
            "exclusion_ledger.jsonl",
        )
    }

    second = run_export(
        tmp_path,
        root=root,
        metadata_path=metadata_path,
        overrides_path=overrides_path,
        force=True,
    )

    for filename, first_file_bytes in first_bytes.items():
        assert first_file_bytes == (tmp_path / "modeling" / filename).read_bytes()

    first_without_time = dict(first)
    second_without_time = dict(second)
    first_without_time.pop("generated_at_utc")
    second_without_time.pop("generated_at_utc")
    assert first_without_time == second_without_time


def test_duplicate_document_ids_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, metadata_path, overrides_path = single_turn_fixture(tmp_path, text=words(650))

    monkeypatch.setattr(
        modeling_corpus,
        "_document_id",
        lambda source_record_id, turn_index, chunk_index: "duplicate",
    )

    with pytest.raises(ModelingCorpusError, match="Duplicate document_id"):
        run_export(tmp_path, root=root, metadata_path=metadata_path, overrides_path=overrides_path)


def test_output_directory_overwrite_protection(tmp_path: Path) -> None:
    root, metadata_path, overrides_path = single_turn_fixture(tmp_path, text=words(25))
    output_root = tmp_path / "modeling"
    output_root.mkdir()
    (output_root / "sentinel.txt").write_text("existing", encoding="utf-8")

    with pytest.raises(ModelingCorpusError, match="nonempty"):
        export_modeling_corpus(
            speaker_turn_root=root,
            overrides_path=overrides_path,
            metadata_summary_path=metadata_path,
            output_root=output_root,
        )


def test_transactional_failure_leaves_no_partial_final_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, metadata_path, overrides_path = single_turn_fixture(tmp_path, text=words(25))
    output_root = tmp_path / "modeling"
    original_replace = modeling_corpus._replace_path

    def fail_first_replace(source: Path, destination: Path) -> None:
        if destination.name == "documents.jsonl":
            raise OSError("synthetic failure")

        original_replace(source, destination)

    monkeypatch.setattr(modeling_corpus, "_replace_path", fail_first_replace)

    with pytest.raises(ModelingCorpusError, match="promote"):
        export_modeling_corpus(
            speaker_turn_root=root,
            overrides_path=overrides_path,
            metadata_summary_path=metadata_path,
            output_root=output_root,
        )

    assert not (output_root / "documents.jsonl").exists()
    assert not list(output_root.glob("*.part"))


def test_turn_decisions_reconcile_exactly(tmp_path: Path) -> None:
    root, metadata_path, overrides_path = single_turn_fixture(tmp_path, text=words(25))
    manifest = run_export(
        tmp_path,
        root=root,
        metadata_path=metadata_path,
        overrides_path=overrides_path,
    )
    decisions = read_jsonl(tmp_path / "modeling" / "turn_decisions.jsonl")

    assert len(decisions) == manifest["input_turn_count"]
    assert all(manifest["reconciliation_checks"].values())


def test_every_override_is_applied_once(tmp_path: Path) -> None:
    root, metadata_path, _ = single_turn_fixture(tmp_path, text=words(25))
    overrides_path = write_overrides(
        tmp_path,
        [
            {
                "action": "exclude_turn",
                "expected_speech_word_count": 25,
                "reason": "manual",
                "session_date": "2020-01-01",
                "source_record_id": "source-a",
                "turn_index": 1,
            }
        ],
    )

    manifest = run_export(
        tmp_path,
        root=root,
        metadata_path=metadata_path,
        overrides_path=overrides_path,
    )

    assert manifest["override_application_ledger"][0]["applied_count"] == 1


def test_manifest_hashes_match_emitted_files(tmp_path: Path) -> None:
    root, metadata_path, overrides_path = single_turn_fixture(tmp_path, text=words(25))
    manifest = run_export(
        tmp_path,
        root=root,
        metadata_path=metadata_path,
        overrides_path=overrides_path,
    )
    output_root = tmp_path / "modeling"

    for metadata in manifest["output_files"].values():
        path = Path(metadata["path"])
        assert path.parent == output_root
        assert metadata["sha256"] == sha256_file(path)
        assert metadata["size_bytes"] == path.stat().st_size

    assert (output_root / MANIFEST_FILENAME).is_file()
