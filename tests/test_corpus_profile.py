import csv
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from argentine_deputies_discursive_distance import corpus_profile
from argentine_deputies_discursive_distance.cli import app
from argentine_deputies_discursive_distance.corpus_profile import (
    CorpusProfileError,
    diagnostic_tokens,
    normalize_for_profile_tokenization,
    profile_modeling_corpus,
)
from argentine_deputies_discursive_distance.pdf_pipeline import sha256_file


@dataclass(frozen=True, slots=True)
class ProfilePaths:
    documents: Path
    export_manifest: Path
    corpus_lock: Path
    config: Path
    output_dir: Path


def temporal_period(year: int) -> str:
    if year <= 2011:
        return "2008-2011"
    if year <= 2015:
        return "2012-2015"
    if year <= 2019:
        return "2016-2019"
    if year <= 2023:
        return "2020-2023"
    return "2024-2025"


def write_json(path: Path, payload: dict[str, Any], *, bom: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8-sig" if bom else "utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def markdown_section(text: str, heading: str) -> str:
    return text.split(heading, maxsplit=1)[1].split("\n## ", maxsplit=1)[0]


def base_config(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "candidate_stopword_min_document_fraction": 0.5,
        "candidate_stopword_min_total_count": 2,
        "context_examples_per_token": 2,
        "minimum_bigram_token_length": 3,
        "primary_session_categories": ["legislative_debate"],
        "profile_version": "1",
        "random_seed": 42,
        "sample_documents_per_stratum": 10,
        "sample_strata": ["year", "session_category"],
        "top_bigram_count": 20,
        "top_token_count": 20,
    }
    payload.update(overrides)
    return payload


def document_record(
    document_id: str,
    text: str,
    *,
    source_record_id: str,
    turn_index: int,
    chunk_index: int = 1,
    session_category: str = "legislative_debate",
    speaker_family: str = "named_or_role_unspecified",
    year: int = 2020,
    word_count: int | None = None,
) -> dict[str, Any]:
    count = len(text.split()) if word_count is None else word_count
    return {
        "chunk_index": chunk_index,
        "document_id": document_id,
        "modeling_text": text,
        "modeling_word_count": count,
        "session_category": session_category,
        "session_date": f"{year}-05-01",
        "source_record_id": source_record_id,
        "speaker_family": speaker_family,
        "temporal_period": temporal_period(year),
        "turn_index": turn_index,
        "word_count": count,
        "year": year,
    }


def base_records() -> list[dict[str, Any]]:
    return [
        document_record(
            "doc-a",
            "Señor PRESIDENTE gracias ley 123 abc123 a boundaryend",
            source_record_id="session-a",
            turn_index=1,
            year=2020,
        ),
        document_record(
            "doc-b",
            "boundarystart café ﬁn abc123 aaaa mal� suave\u00ad Ã±",
            source_record_id="session-b",
            turn_index=2,
            session_category="informative",
            year=2014,
        ),
        document_record(
            "doc-c",
            "Diputada Cámara palabra presupuesto 2024 alpha beta",
            source_record_id="session-c",
            turn_index=3,
            speaker_family="executive_official",
            year=2009,
        ),
        document_record(
            "doc-d",
            "otra categoria numero 77 mezcla x9 cierre final",
            source_record_id="session-d",
            turn_index=4,
            session_category="preparatory",
            year=2024,
        ),
    ]


def write_profile_fixture(
    tmp_path: Path,
    records: list[dict[str, Any]] | None = None,
    *,
    raw_documents_text: str | None = None,
    config_payload: dict[str, Any] | None = None,
    config_bom: bool = False,
    manifest_document_count: int | None = None,
    manifest_word_total: int | None = None,
) -> ProfilePaths:
    root = tmp_path / "profile_fixture"
    documents_path = root / "documents.jsonl"
    manifest_path = root / "export_manifest.json"
    lock_path = root / "modeling_corpus_lock_v1.json"
    config_path = root / "corpus_profile_v1.json"
    output_dir = root / "outputs"
    root.mkdir(parents=True, exist_ok=True)

    actual_records = base_records() if records is None else records

    if raw_documents_text is None:
        documents_text = "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in actual_records
        )
    else:
        documents_text = raw_documents_text

    documents_path.write_text(documents_text, encoding="utf-8")
    document_count = (
        len(actual_records) if manifest_document_count is None else manifest_document_count
    )
    word_total = (
        sum(int(record.get("word_count", 0)) for record in actual_records)
        if manifest_word_total is None
        else manifest_word_total
    )
    source_turns = {
        (str(record.get("source_record_id")), int(record.get("turn_index", 0)))
        for record in actual_records
    }
    sessions = {str(record.get("source_record_id")) for record in actual_records}
    documents_metadata = {
        "path": str(documents_path),
        "sha256": sha256_file(documents_path),
        "size_bytes": documents_path.stat().st_size,
    }
    manifest = {
        "content_classifier_version": "2",
        "exclusion_ledger_count": 0,
        "exporter_version": "1",
        "input_session_count": len(sessions),
        "input_turn_count": len(source_turns),
        "modeling_document_count": document_count,
        "modeling_word_total": word_total,
        "output_files": {"documents": documents_metadata},
        "pipeline_version": "2",
        "positive_speech_turn_count": len(source_turns),
        "retained_source_turn_count": len(source_turns),
        "retained_source_turn_word_total": word_total,
    }
    write_json(manifest_path, manifest)
    lock = dict(manifest)
    lock.update(
        {
            "export_manifest_path": str(manifest_path),
            "export_manifest_sha256": sha256_file(manifest_path),
            "lock_version": "1",
        }
    )
    write_json(lock_path, lock)
    write_json(
        config_path,
        base_config() if config_payload is None else config_payload,
        bom=config_bom,
    )

    return ProfilePaths(
        documents=documents_path,
        export_manifest=manifest_path,
        corpus_lock=lock_path,
        config=config_path,
        output_dir=output_dir,
    )


def run_profile(paths: ProfilePaths, *, force: bool = False) -> dict[str, Any]:
    return profile_modeling_corpus(
        documents_path=paths.documents,
        export_manifest_path=paths.export_manifest,
        corpus_lock_path=paths.corpus_lock,
        config_path=paths.config,
        output_dir=paths.output_dir,
        force=force,
    )


def test_profile_semantics_grouping_tokenization_and_frequencies(tmp_path: Path) -> None:
    paths = write_profile_fixture(tmp_path)

    manifest = run_profile(paths)
    profile = read_json(paths.output_dir / "corpus_profile.json")
    token_rows = read_csv(paths.output_dir / "token_frequency.csv")
    histogram_rows = read_csv(paths.output_dir / "document_length_histogram.csv")
    category_rows = read_csv(paths.output_dir / "counts_by_session_category.csv")

    assert manifest["universes"]["all_sessions"]["documents"] == 4
    assert manifest["universes"]["primary"]["documents"] == 2
    assert profile["universes"]["all_sessions"]["counts"]["unique_sessions"] == 4
    assert profile["universes"]["primary"]["counts"]["unique_source_turns"] == 2

    assert any(
        row["universe"] == "all_sessions"
        and row["session_category"] == "informative"
        and row["document_count"] == "1"
        for row in category_rows
    )
    assert any(
        row["universe"] == "primary"
        and row["session_category"] == "legislative_debate"
        and row["document_count"] == "2"
        for row in category_rows
    )

    all_abc123 = next(
        row for row in token_rows if row["universe"] == "all_sessions" and row["token"] == "abc123"
    )
    primary_abc123 = next(
        row for row in token_rows if row["universe"] == "primary" and row["token"] == "abc123"
    )
    assert all_abc123["total_count"] == "2"
    assert all_abc123["document_count"] == "2"
    assert primary_abc123["document_count"] == "1"
    assert any(row["token"] == "café" for row in token_rows)
    assert any(row["token"] == "fin" for row in token_rows)
    assert "señor" in diagnostic_tokens("Señor")
    assert "café" in diagnostic_tokens("CAFÉ")
    assert normalize_for_profile_tokenization("ﬁn") == "fin"
    assert "sampled_document_membership" in {
        item["name"] for item in profile["statistics_scope"]["sampled_statistics_or_examples"]
    }
    assert "candidate_token_context_examples" in {
        item["name"] for item in profile["statistics_scope"]["sampled_statistics_or_examples"]
    }
    assert (
        "candidate_stopword_counts_and_reasons"
        in profile["statistics_scope"]["exact_full_corpus_statistics"]
    )

    all_histogram = [row for row in histogram_rows if row["universe"] == "all_sessions"]
    assert sum(int(row["document_count"]) for row in all_histogram) == 4
    assert (
        sum(int(row["word_total"]) for row in all_histogram)
        == manifest["universes"]["all_sessions"]["words"]
    )


def test_candidates_suspicious_tokens_bigrams_and_examples(tmp_path: Path) -> None:
    paths = write_profile_fixture(tmp_path)

    run_profile(paths)
    candidate_rows = read_csv(paths.output_dir / "candidate_stopwords.csv")
    suspicious_rows = read_csv(paths.output_dir / "suspicious_tokens.csv")
    bigram_rows = read_csv(paths.output_dir / "sampled_bigram_frequency.csv")
    examples = read_jsonl(paths.output_dir / "preprocessing_examples.jsonl")

    señor = next(
        row for row in candidate_rows if row["universe"] == "primary" and row["token"] == "señor"
    )
    assert "procedural_seed_match" in señor["candidate_reasons"]
    assert señor["selected_for_removal"] == "false"
    assert any(
        row["token"] == "a" and "very_short_alpha_token" in row["candidate_reasons"]
        for row in candidate_rows
    )
    assert any(row["reason"] == "replacement_character" for row in suspicious_rows)
    assert any(row["reason"] == "soft_hyphen" for row in suspicious_rows)
    assert any(row["reason"] == "mojibake_like_sequence" for row in suspicious_rows)
    assert any(row["reason"] == "mixed_letters_and_digits" for row in suspicious_rows)
    assert any(row["reason"] == "numeric_only_token" for row in suspicious_rows)
    assert any(
        row["reason"] == "repeated_identical_character_four_or_more" for row in suspicious_rows
    )
    assert not any(row["bigram"] == "boundaryend boundarystart" for row in bigram_rows)
    assert examples
    assert {
        "original_modeling_text",
        "nfkc_casefolded_text",
        "all_diagnostic_tokens",
        "alphabetic_tokens",
        "alphabetic_tokens_min_length_3",
    } <= set(examples[0])


def test_hapax_uses_total_frequency_and_document_frequency_singletons_are_separate(
    tmp_path: Path,
) -> None:
    paths = write_profile_fixture(
        tmp_path,
        [
            document_record(
                "doc-a",
                "repeat repeat repeat singleton",
                source_record_id="session-a",
                turn_index=1,
            ),
            document_record(
                "doc-b",
                "other other",
                source_record_id="session-b",
                turn_index=2,
            ),
        ],
    )

    run_profile(paths)
    profile = read_json(paths.output_dir / "corpus_profile.json")
    lexical = profile["universes"]["all_sessions"]["lexical"]

    assert lexical["hapax_count"] == 1
    assert lexical["tokens_appearing_in_exactly_1_document"] == 3


def test_markdown_report_separates_universes_and_escapes_candidate_reasons(
    tmp_path: Path,
) -> None:
    paths = write_profile_fixture(tmp_path)

    run_profile(paths)
    report = (paths.output_dir / "corpus_profile.md").read_text(encoding="utf-8")

    for heading in (
        "## Top frequent terms",
        "## Candidate stopwords",
        "## Suspicious-token summary",
        "## Sampled bigrams",
    ):
        section = report.split(heading, maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
        assert "### all_sessions" in section
        assert "### primary" in section

    assert "high_document_fraction\\|high_total_frequency" in report
    assert "| a |" in report


def test_suspicious_token_snippet_uses_normalized_text_for_token_diagnostics(
    tmp_path: Path,
) -> None:
    prefix = " ".join(f"prefijo{index}" for index in range(20))
    paths = write_profile_fixture(
        tmp_path,
        [
            document_record(
                "doc-normalized-snippet",
                f"{prefix} ＡBC123 final",
                source_record_id="session-a",
                turn_index=1,
            )
        ],
    )

    run_profile(paths)
    suspicious_rows = read_csv(paths.output_dir / "suspicious_tokens.csv")
    row = next(
        item
        for item in suspicious_rows
        if item["reason"] == "mixed_letters_and_digits" and item["token_or_anomaly"] == "abc123"
    )
    snippet = json.loads(row["example_snippets"])[0]

    assert row["snippet_text_kind"] == "normalized_nfkc_casefolded_modeling_text"
    assert "abc123" in snippet
    assert snippet.startswith("...")


def test_token_frequency_rows_are_streamed() -> None:
    assert inspect.isgeneratorfunction(corpus_profile._token_frequency_rows)


def test_suspicious_rows_and_markdown_section_are_order_independent(tmp_path: Path) -> None:
    first_paths = write_profile_fixture(tmp_path / "first", base_records())
    second_paths = write_profile_fixture(tmp_path / "second", list(reversed(base_records())))

    run_profile(first_paths)
    run_profile(second_paths)

    assert (first_paths.output_dir / "suspicious_tokens.csv").read_bytes() == (
        second_paths.output_dir / "suspicious_tokens.csv"
    ).read_bytes()
    first_report = (first_paths.output_dir / "corpus_profile.md").read_text(encoding="utf-8")
    second_report = (second_paths.output_dir / "corpus_profile.md").read_text(encoding="utf-8")

    assert markdown_section(first_report, "## Suspicious-token summary") == markdown_section(
        second_report,
        "## Suspicious-token summary",
    )


def test_deterministic_sample_is_stable_and_order_independent(tmp_path: Path) -> None:
    first_paths = write_profile_fixture(tmp_path / "first", base_records())
    second_paths = write_profile_fixture(tmp_path / "second", list(reversed(base_records())))

    run_profile(first_paths)
    run_profile(second_paths)

    first_sample = read_jsonl(first_paths.output_dir / "sampled_documents.jsonl")
    second_sample = read_jsonl(second_paths.output_dir / "sampled_documents.jsonl")

    assert [(row["universe"], row["document_id"], row["stable_hash"]) for row in first_sample] == [
        (row["universe"], row["document_id"], row["stable_hash"]) for row in second_sample
    ]


def test_primary_reconciliation_fails_when_primary_document_is_omitted(tmp_path: Path) -> None:
    paths = write_profile_fixture(tmp_path)
    config = corpus_profile._load_config(paths.config)
    export_manifest, corpus_lock, input_hashes = corpus_profile._validate_manifest_and_lock(
        documents_path=paths.documents,
        export_manifest_path=paths.export_manifest,
        corpus_lock_path=paths.corpus_lock,
    )
    run = corpus_profile._stream_profile(
        documents_path=paths.documents,
        export_manifest=export_manifest,
        corpus_lock=corpus_lock,
        config=config,
        input_hashes=input_hashes,
    )
    run.states[corpus_profile.UNIVERSE_PRIMARY] = corpus_profile.UniverseState(
        name=corpus_profile.UNIVERSE_PRIMARY
    )
    samples = {universe: run.samplers[universe].selected() for universe in corpus_profile.UNIVERSES}

    checks = corpus_profile._reconciliation_checks(run=run, samples=samples)

    assert checks["primary_universe_matches_configured_session_categories"] is False


def test_primary_reconciliation_fails_when_non_primary_document_is_included(
    tmp_path: Path,
) -> None:
    paths = write_profile_fixture(tmp_path)
    config = corpus_profile._load_config(paths.config)
    export_manifest, corpus_lock, input_hashes = corpus_profile._validate_manifest_and_lock(
        documents_path=paths.documents,
        export_manifest_path=paths.export_manifest,
        corpus_lock_path=paths.corpus_lock,
    )
    run = corpus_profile._stream_profile(
        documents_path=paths.documents,
        export_manifest=export_manifest,
        corpus_lock=corpus_lock,
        config=config,
        input_hashes=input_hashes,
    )
    run.states[corpus_profile.UNIVERSE_PRIMARY] = run.states[corpus_profile.UNIVERSE_ALL]
    samples = {universe: run.samplers[universe].selected() for universe in corpus_profile.UNIVERSES}

    checks = corpus_profile._reconciliation_checks(run=run, samples=samples)

    assert checks["primary_universe_matches_configured_session_categories"] is False


def test_outputs_are_deterministic_except_manifest_timestamp_and_hashes_match(
    tmp_path: Path,
) -> None:
    paths = write_profile_fixture(tmp_path)

    first_manifest = run_profile(paths)
    first_bytes = {
        path.name: path.read_bytes()
        for path in paths.output_dir.iterdir()
        if path.name != "profile_manifest.json"
    }
    second_manifest = run_profile(paths, force=True)

    for filename, content in first_bytes.items():
        assert (paths.output_dir / filename).read_bytes() == content

    first_without_time = dict(first_manifest)
    second_without_time = dict(second_manifest)
    first_without_time.pop("generated_at_utc")
    second_without_time.pop("generated_at_utc")
    assert first_without_time == second_without_time

    for metadata in second_manifest["output_files"].values():
        path = Path(metadata["path"])
        assert metadata["sha256"] == sha256_file(path)
        assert metadata["size_bytes"] == path.stat().st_size
        assert not path.read_bytes().startswith(b"\xef\xbb\xbf")


def test_bom_bearing_configuration_input_is_accepted(tmp_path: Path) -> None:
    paths = write_profile_fixture(tmp_path, config_bom=True)

    manifest = run_profile(paths)

    assert manifest["profile_version"] == "1"


def test_cli_profile_modeling_corpus_command(tmp_path: Path) -> None:
    paths = write_profile_fixture(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "profile-modeling-corpus",
            "--documents",
            str(paths.documents),
            "--export-manifest",
            str(paths.export_manifest),
            "--corpus-lock",
            str(paths.corpus_lock),
            "--config",
            str(paths.config),
            "--output-dir",
            str(paths.output_dir),
        ],
    )

    assert result.exit_code == 0
    assert (paths.output_dir / "profile_manifest.json").is_file()


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda paths: write_json(
                paths.corpus_lock,
                {
                    **read_json(paths.corpus_lock),
                    "export_manifest_sha256": "0" * 64,
                },
            ),
            "export manifest SHA-256",
        ),
        (
            lambda paths: paths.documents.write_text(
                paths.documents.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            ),
            "documents.jsonl SHA-256",
        ),
    ],
)
def test_locked_input_hash_mismatches_fail(
    tmp_path: Path,
    mutate: Any,
    match: str,
) -> None:
    paths = write_profile_fixture(tmp_path)
    mutate(paths)

    with pytest.raises(CorpusProfileError, match=match):
        run_profile(paths)


@pytest.mark.parametrize(
    ("target", "field_name", "match"),
    [
        ("lock", "modeling_word_total", "Missing required corpus lock total field"),
        ("manifest", "exporter_version", "Missing required export manifest version field"),
    ],
)
def test_missing_locked_total_or_version_fields_fail(
    tmp_path: Path,
    target: str,
    field_name: str,
    match: str,
) -> None:
    paths = write_profile_fixture(tmp_path)

    if target == "lock":
        payload = read_json(paths.corpus_lock)
        payload.pop(field_name)
        write_json(paths.corpus_lock, payload)
    else:
        manifest = read_json(paths.export_manifest)
        manifest.pop(field_name)
        write_json(paths.export_manifest, manifest)
        lock = read_json(paths.corpus_lock)
        lock["export_manifest_sha256"] = sha256_file(paths.export_manifest)
        write_json(paths.corpus_lock, lock)

    with pytest.raises(CorpusProfileError, match=match):
        run_profile(paths)


@pytest.mark.parametrize(
    ("records", "match"),
    [
        (
            [
                document_record(
                    "duplicate",
                    "uno dos tres",
                    source_record_id="session-a",
                    turn_index=1,
                ),
                document_record(
                    "duplicate",
                    "cuatro cinco seis",
                    source_record_id="session-b",
                    turn_index=2,
                ),
            ],
            "Duplicate document_id",
        ),
        (
            [
                {
                    key: value
                    for key, value in document_record(
                        "missing",
                        "uno dos tres",
                        source_record_id="session-a",
                        turn_index=1,
                    ).items()
                    if key != "modeling_text"
                }
            ],
            "Missing required document fields",
        ),
        (
            [
                document_record(
                    "bad-word-count",
                    "uno dos tres",
                    source_record_id="session-a",
                    turn_index=1,
                    word_count=0,
                )
            ],
            "Invalid word_count",
        ),
        (
            [
                document_record(
                    "bad-family",
                    "uno dos tres",
                    source_record_id="session-a",
                    turn_index=1,
                    speaker_family="chair",
                )
            ],
            "Invalid speaker_family",
        ),
        (
            [
                document_record(
                    "bad-text-word-count",
                    "uno dos",
                    source_record_id="session-a",
                    turn_index=1,
                    word_count=3,
                )
            ],
            "modeling_text whitespace word count",
        ),
    ],
)
def test_document_validation_failures(
    tmp_path: Path,
    records: list[dict[str, Any]],
    match: str,
) -> None:
    paths = write_profile_fixture(tmp_path, records)

    with pytest.raises(CorpusProfileError, match=match):
        run_profile(paths)


def test_malformed_jsonl_fails(tmp_path: Path) -> None:
    records = [
        document_record("synthetic", "uno dos tres", source_record_id="session-a", turn_index=1)
    ]
    paths = write_profile_fixture(
        tmp_path,
        records,
        raw_documents_text='{"document_id": "broken"\n',
        manifest_document_count=1,
        manifest_word_total=3,
    )

    with pytest.raises(CorpusProfileError, match="Malformed JSONL"):
        run_profile(paths)


def test_manifest_count_reconciliation_failure_rolls_back(tmp_path: Path) -> None:
    paths = write_profile_fixture(tmp_path, manifest_document_count=5)

    with pytest.raises(CorpusProfileError, match="Processed document count"):
        run_profile(paths)

    assert not any(paths.output_dir.glob("*")) if paths.output_dir.exists() else True


def test_overwrite_protection(tmp_path: Path) -> None:
    paths = write_profile_fixture(tmp_path)
    paths.output_dir.mkdir(parents=True)
    (paths.output_dir / "sentinel.txt").write_text("existing", encoding="utf-8")

    with pytest.raises(CorpusProfileError, match="nonempty"):
        run_profile(paths)


def test_overwrite_protection_fails_before_streaming(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = write_profile_fixture(tmp_path)
    paths.output_dir.mkdir(parents=True)
    (paths.output_dir / "sentinel.txt").write_text("existing", encoding="utf-8")

    def fail_if_called(**_kwargs: Any) -> None:
        raise AssertionError("_stream_profile should not be called")

    monkeypatch.setattr(corpus_profile, "_stream_profile", fail_if_called)

    with pytest.raises(CorpusProfileError, match="nonempty"):
        run_profile(paths)


def test_transactional_rollback_leaves_no_partial_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = write_profile_fixture(tmp_path)
    original_replace = corpus_profile._replace_path

    def fail_first_replace(source: Path, destination: Path) -> None:
        if destination.name == "candidate_stopwords.csv":
            raise OSError("synthetic failure")

        original_replace(source, destination)

    monkeypatch.setattr(corpus_profile, "_replace_path", fail_first_replace)

    with pytest.raises(CorpusProfileError, match="promote"):
        run_profile(paths)

    assert not (paths.output_dir / "candidate_stopwords.csv").exists()
    assert not list(paths.output_dir.glob("*.part"))


def test_force_mode_rollback_restores_existing_outputs_after_partial_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = write_profile_fixture(tmp_path)
    run_profile(paths)
    prior_bytes = {path.name: path.read_bytes() for path in paths.output_dir.iterdir()}
    original_replace = corpus_profile._replace_path

    def fail_after_one_promoted(source: Path, destination: Path) -> None:
        if source.name == "corpus_profile.json.part" and destination.name == "corpus_profile.json":
            raise OSError("synthetic failure after first promoted file")

        original_replace(source, destination)

    monkeypatch.setattr(corpus_profile, "_replace_path", fail_after_one_promoted)

    with pytest.raises(CorpusProfileError, match="promote"):
        run_profile(paths, force=True)

    assert {path.name: path.read_bytes() for path in paths.output_dir.iterdir()} == prior_bytes
    assert not list(paths.output_dir.glob("*.part"))
    assert not list(paths.output_dir.glob("*.bak"))
