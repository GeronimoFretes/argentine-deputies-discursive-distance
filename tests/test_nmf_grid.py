import csv
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from argentine_deputies_discursive_distance import corpus_profile, nmf_grid
from argentine_deputies_discursive_distance.cli import app
from argentine_deputies_discursive_distance.nmf_grid import (
    NmfGridError,
    fit_nmf_grid,
    npmi_from_counts,
    topic_diversity,
    topic_exclusivity,
    topic_redundancy,
)
from argentine_deputies_discursive_distance.pdf_pipeline import sha256_file


@dataclass(frozen=True, slots=True)
class GridFixture:
    documents: Path
    export_manifest: Path
    corpus_lock: Path
    profile_config: Path
    profile_manifest: Path
    grid_config: Path
    stopwords: Path
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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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


def document_record(
    document_id: str,
    text: str,
    *,
    source_record_id: str,
    turn_index: int,
    chunk_index: int = 1,
    session_category: str = "legislative_debate",
    year: int = 2020,
) -> dict[str, Any]:
    word_count = len(text.split())
    return {
        "chunk_index": chunk_index,
        "document_id": document_id,
        "modeling_text": text,
        "modeling_word_count": word_count,
        "session_category": session_category,
        "session_date": f"{year}-05-01",
        "source_record_id": source_record_id,
        "speaker_family": "named_or_role_unspecified",
        "temporal_period": temporal_period(year),
        "turn_index": turn_index,
        "word_count": word_count,
        "year": year,
    }


def smoke_records() -> list[dict[str, Any]]:
    return [
        document_record(
            "doc-health-1",
            "Señor presidente salud hospital medicina pacientes vacunas cuidado salud pública",
            source_record_id="session-health",
            turn_index=1,
            chunk_index=1,
        ),
        document_record(
            "doc-health-2",
            "Salud pública hospital enfermería pacientes sistema sanitario cuidado vacunas",
            source_record_id="session-health",
            turn_index=1,
            chunk_index=2,
        ),
        document_record(
            "doc-health-3",
            "Señora presidenta hospital salud medicina prevención cuidado comunitario",
            source_record_id="session-health-b",
            turn_index=2,
        ),
        document_record(
            "doc-education-1",
            "Educación universidad escuela docentes estudiantes becas ciencia educa- ción",
            source_record_id="session-education",
            turn_index=1,
        ),
        document_record(
            "doc-education-2",
            "Universidad escuela docentes aulas estudiantes libros ciencia investigación",
            source_record_id="session-education-b",
            turn_index=1,
        ),
        document_record(
            "doc-education-3",
            "Escuela educación docentes estudiantes becas formación conocimiento",
            source_record_id="session-education-c",
            turn_index=1,
        ),
        document_record(
            "doc-budget-1",
            "Presupuesto impuestos inflación crédito actividad productiva empleo",
            source_record_id="session-budget",
            turn_index=1,
        ),
        document_record(
            "doc-budget-2",
            "Actividad productiva crédito presupuesto impuestos desarrollo empleo",
            source_record_id="session-budget-b",
            turn_index=1,
        ),
        document_record(
            "doc-budget-3",
            "Inflación crédito empleo industria presupuesto inversión productiva",
            source_record_id="session-budget-c",
            turn_index=1,
        ),
        document_record(
            "doc-energy-1",
            "Ambiente energía renovable clima transición recursos territorio",
            source_record_id="session-energy",
            turn_index=1,
        ),
        document_record(
            "doc-energy-2",
            "Energía renovable ambiente territorio inversión recursos clima",
            source_record_id="session-energy-b",
            turn_index=1,
        ),
        document_record(
            "doc-soft-hyphen",
            "Provincia públi\u00ad ca consti\u00ad tucional bancario fundamen- talmente",
            source_record_id="session-soft",
            turn_index=1,
        ),
        document_record(
            "doc-nonprimary",
            "Informativa archivo administrativo registro mesa trámite",
            source_record_id="session-info",
            turn_index=1,
            session_category="informative",
        ),
    ]


def base_profile_config() -> dict[str, Any]:
    return {
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


def primary_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    primary = [record for record in records if record["session_category"] == "legislative_debate"]
    source_turns = {(record["source_record_id"], record["turn_index"]) for record in primary}
    sessions = {record["source_record_id"] for record in primary}
    return {
        "documents": len(primary),
        "sessions": len(sessions),
        "source_turns": len(source_turns),
        "words": sum(int(record["word_count"]) for record in primary),
    }


def grid_config(records: list[dict[str, Any]], *, stopword_variant: str = "P1") -> dict[str, Any]:
    counts = primary_counts(records)
    return {
        "cleaned_excerpt_characters": 120,
        "expected_primary_counts": counts,
        "expected_profile_counts": {
            "all_documents": len(records),
            "all_words": sum(int(record["word_count"]) for record in records),
        },
        "grid_version": "1",
        "metrics_top_n": 5,
        "nmf": {
            "alpha_H": 0.0,
            "alpha_W": 0.0,
            "beta_loss": "frobenius",
            "init": "nndsvda",
            "k_values": [2, 3],
            "l1_ratio": 0.0,
            "max_iter": 80,
            "random_state": 42,
            "solver": "cd",
            "tol": 0.0001,
        },
        "preprocessing_example_limit": 5,
        "primary_session_categories": ["legislative_debate"],
        "representative_documents_per_topic": 5,
        "stopword_variant": stopword_variant,
        "tfidf": {
            "dtype": "float32",
            "lowercase": False,
            "max_df": 1.0,
            "max_features": 200,
            "min_df": 1,
            "ngram_range": [1, 2],
            "norm": "l2",
            "smooth_idf": True,
            "strip_accents": None,
            "sublinear_tf": True,
        },
        "top_terms_per_topic": 8,
    }


def write_grid_fixture(
    root: Path,
    records: list[dict[str, Any]] | None = None,
    *,
    stopword_variant: str = "P1",
) -> GridFixture:
    actual_records = smoke_records() if records is None else records
    documents = root / "documents.jsonl"
    export_manifest = root / "export_manifest.json"
    corpus_lock = root / "modeling_corpus_lock_v1.json"
    profile_config = root / "corpus_profile_v1.json"
    profile_manifest = root / "profile_manifest.json"
    config = root / "nmf_grid_v1.json"
    output_dir = root / "outputs"
    stopwords = Path("config/topic_modeling/stopwords_es_p0_v1.txt")
    root.mkdir(parents=True, exist_ok=True)
    documents.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in actual_records
        ),
        encoding="utf-8",
    )
    source_turns = {
        (str(record["source_record_id"]), int(record["turn_index"])) for record in actual_records
    }
    sessions = {str(record["source_record_id"]) for record in actual_records}
    word_total = sum(int(record["word_count"]) for record in actual_records)
    documents_metadata = {
        "path": str(documents),
        "sha256": sha256_file(documents),
        "size_bytes": documents.stat().st_size,
    }
    manifest = {
        "content_classifier_version": "2",
        "exclusion_ledger_count": 0,
        "exporter_version": "1",
        "input_session_count": len(sessions),
        "input_turn_count": len(source_turns),
        "modeling_document_count": len(actual_records),
        "modeling_word_total": word_total,
        "output_files": {"documents": documents_metadata},
        "pipeline_version": "2",
        "positive_speech_turn_count": len(source_turns),
        "retained_source_turn_count": len(source_turns),
        "retained_source_turn_word_total": word_total,
    }
    write_json(export_manifest, manifest)
    lock = {
        **manifest,
        "export_manifest_path": str(export_manifest),
        "export_manifest_sha256": sha256_file(export_manifest),
        "lock_version": "1",
    }
    write_json(corpus_lock, lock)
    write_json(profile_config, base_profile_config())
    profile_config_model = corpus_profile._load_config(profile_config)
    canonical_profile_config = corpus_profile._json_text(profile_config_model.to_json())
    counts = primary_counts(actual_records)
    write_json(
        profile_manifest,
        {
            "canonical_configuration_sha256": hashlib.sha256(
                canonical_profile_config.encode("utf-8")
            ).hexdigest(),
            "input_sha256": {
                "config_sha256": sha256_file(profile_config),
                "corpus_lock_sha256": sha256_file(corpus_lock),
                "documents_sha256": sha256_file(documents),
                "export_manifest_sha256": sha256_file(export_manifest),
            },
            "reconciliation_checks": {
                "all_output_hashes_match_emitted_files": True,
                "primary_universe_matches_configured_session_categories": True,
            },
            "universes": {
                "all_sessions": {"documents": len(actual_records), "words": word_total},
                "primary": {
                    "documents": counts["documents"],
                    "unique_sessions": counts["sessions"],
                    "unique_source_turns": counts["source_turns"],
                    "words": counts["words"],
                },
            },
        },
    )
    write_json(config, grid_config(actual_records, stopword_variant=stopword_variant))
    return GridFixture(
        documents=documents,
        export_manifest=export_manifest,
        corpus_lock=corpus_lock,
        profile_config=profile_config,
        profile_manifest=profile_manifest,
        grid_config=config,
        stopwords=stopwords,
        output_dir=output_dir,
    )


def run_grid(paths: GridFixture, *, force: bool = False) -> dict[str, Any]:
    return fit_nmf_grid(
        documents_path=paths.documents,
        export_manifest_path=paths.export_manifest,
        corpus_lock_path=paths.corpus_lock,
        profile_manifest_path=paths.profile_manifest,
        profile_config_path=paths.profile_config,
        config_path=paths.grid_config,
        stopwords_path=paths.stopwords,
        output_dir=paths.output_dir,
        force=force,
    )


def test_bounded_smoke_outputs_every_artifact_type(tmp_path: Path) -> None:
    paths = write_grid_fixture(tmp_path / "smoke")

    manifest = run_grid(paths)

    assert manifest["primary_counts"]["documents"] == 12
    assert (paths.output_dir / "cleaned_primary_documents.jsonl").is_file()
    assert (paths.output_dir / "vectorizer.joblib").is_file()
    assert (paths.output_dir / "nmf_k002.joblib").is_file()
    assert (paths.output_dir / "nmf_k003.joblib").is_file()
    assert (paths.output_dir / "topic_terms_k002.csv").is_file()
    assert (paths.output_dir / "representative_documents_k003.jsonl").is_file()
    assert (paths.output_dir / "grid_report.md").is_file()

    vectorizer_summary = read_json(paths.output_dir / "vectorizer_summary.json")
    assert vectorizer_summary["settings"]["lowercase"] is False
    assert vectorizer_summary["settings"]["ngram_range"] == [1, 2]
    assert vectorizer_summary["sparse_matrix_format"] == "csr"
    assert vectorizer_summary["unigram_feature_count"] > 0
    assert vectorizer_summary["bigram_feature_count"] > 0
    assert vectorizer_summary["zero_tfidf_rows"] == 0
    assert "salud pública" in {row["term"] for row in read_csv(paths.output_dir / "vocabulary.csv")}
    assert len(read_csv(paths.output_dir / "grid_metrics.csv")) == 2
    assert len(read_jsonl(paths.output_dir / "representative_documents_k002.jsonl")) == 10


def test_cli_fit_nmf_grid_command(tmp_path: Path) -> None:
    paths = write_grid_fixture(tmp_path / "cli")
    result = CliRunner().invoke(
        app,
        [
            "fit-nmf-grid",
            "--documents",
            str(paths.documents),
            "--export-manifest",
            str(paths.export_manifest),
            "--corpus-lock",
            str(paths.corpus_lock),
            "--corpus-profile-manifest",
            str(paths.profile_manifest),
            "--corpus-profile-config",
            str(paths.profile_config),
            "--config",
            str(paths.grid_config),
            "--stopwords",
            str(paths.stopwords),
            "--output-dir",
            str(paths.output_dir),
        ],
    )

    assert result.exit_code == 0
    assert (paths.output_dir / "run_manifest.json").is_file()


def test_primary_filtering_and_preprocessing_examples_are_deterministic(tmp_path: Path) -> None:
    paths = write_grid_fixture(tmp_path / "preprocessing")

    run_grid(paths)

    summary = read_json(paths.output_dir / "preprocessing_summary.json")
    examples = read_jsonl(paths.output_dir / "preprocessing_examples.jsonl")

    assert summary["primary_counts"]["documents"] == 12
    assert summary["primary_counts"]["sessions"] == 11
    assert summary["soft_hyphens_removed"] == 2
    assert summary["explicit_hyphenation_joins"] == 2
    assert [row["document_id"] for row in examples] == sorted(
        row["document_id"] for row in examples
    )
    assert any(row["document_id"] == "doc-soft-hyphen" for row in examples)


def test_lineage_hash_mismatch_fails(tmp_path: Path) -> None:
    paths = write_grid_fixture(tmp_path / "lineage")
    profile = read_json(paths.profile_manifest)
    profile["input_sha256"]["documents_sha256"] = "0" * 64
    write_json(paths.profile_manifest, profile)

    with pytest.raises(NmfGridError, match="documents_sha256"):
        run_grid(paths)


def test_zero_token_document_failure(tmp_path: Path) -> None:
    records = [
        document_record(
            "empty-after-tokenization",
            "12 x9 ab",
            source_record_id="session-zero",
            turn_index=1,
        )
    ]
    paths = write_grid_fixture(tmp_path / "zero-token", records)

    with pytest.raises(NmfGridError, match="zero alphabetic lexical tokens"):
        run_grid(paths)


def test_zero_tfidf_row_failure(tmp_path: Path) -> None:
    records = [
        document_record(
            "only-p1-stopwords",
            "señor señora señores presidente presidenta",
            source_record_id="session-stop",
            turn_index=1,
        ),
        document_record(
            "substantive",
            "salud hospital medicina pacientes vacunas",
            source_record_id="session-substantive",
            turn_index=1,
        ),
    ]
    paths = write_grid_fixture(tmp_path / "zero-tfidf", records)
    config = read_json(paths.grid_config)
    config["nmf"]["k_values"] = [1]
    config["metrics_top_n"] = 2
    config["top_terms_per_topic"] = 3
    write_json(paths.grid_config, config)

    with pytest.raises(NmfGridError, match="zero rows"):
        run_grid(paths)


def test_overwrite_protection(tmp_path: Path) -> None:
    paths = write_grid_fixture(tmp_path / "overwrite")
    paths.output_dir.mkdir(parents=True)
    (paths.output_dir / "sentinel.txt").write_text("existing", encoding="utf-8")

    with pytest.raises(NmfGridError, match="nonempty"):
        run_grid(paths)


def test_transactional_rollback_leaves_no_partial_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = write_grid_fixture(tmp_path / "rollback")
    original_replace = nmf_grid._replace_path

    def fail_first_replace(source: Path, destination: Path) -> None:
        if destination.name == "grid_metrics.csv":
            raise OSError("synthetic failure")

        original_replace(source, destination)

    monkeypatch.setattr(nmf_grid, "_replace_path", fail_first_replace)

    with pytest.raises(NmfGridError, match="promote"):
        run_grid(paths)

    assert not (paths.output_dir / "grid_metrics.csv").exists()
    assert not list(paths.output_dir.glob("*.part"))
    assert not list(paths.output_dir.glob("*.bak"))


def test_deterministic_text_outputs_except_runtime_and_timestamp(tmp_path: Path) -> None:
    paths = write_grid_fixture(tmp_path / "deterministic")

    first_manifest = run_grid(paths)
    first_summary = read_json(paths.output_dir / "preprocessing_summary.json")
    first_examples = (paths.output_dir / "preprocessing_examples.jsonl").read_bytes()
    first_vocabulary = (paths.output_dir / "vocabulary.csv").read_bytes()
    first_terms = (paths.output_dir / "topic_terms_k002.csv").read_bytes()
    second_manifest = run_grid(paths, force=True)

    assert read_json(paths.output_dir / "preprocessing_summary.json") == first_summary
    assert (paths.output_dir / "preprocessing_examples.jsonl").read_bytes() == first_examples
    assert (paths.output_dir / "vocabulary.csv").read_bytes() == first_vocabulary
    assert (paths.output_dir / "topic_terms_k002.csv").read_bytes() == first_terms

    first_manifest.pop("generated_at_utc")
    second_manifest.pop("generated_at_utc")
    first_manifest.pop("output_files")
    second_manifest.pop("output_files")
    assert first_manifest["primary_counts"] == second_manifest["primary_counts"]


def test_metric_definitions_ranges_and_edge_cases() -> None:
    components = np.array([[2.0, 1.0, 0.0], [1.0, 0.0, 3.0]], dtype=np.float64)
    top_indices = ((0, 1), (2, 0))
    exclusivity_by_topic, exclusivity = topic_exclusivity(
        components=components,
        top_indices=top_indices,
    )
    redundancy = topic_redundancy(components)

    assert topic_diversity(top_indices=top_indices) == 0.75
    assert all(0.0 <= value <= 1.0 for value in exclusivity_by_topic)
    assert 0.0 <= exclusivity <= 1.0
    assert 0.0 <= redundancy["mean_off_diagonal_similarity"] <= 1.0
    assert topic_redundancy(np.array([[1.0, 0.0]], dtype=np.float64))["pair_count"] == 0
    assert (
        npmi_from_counts(
            cooccurrence_count=0,
            left_count=2,
            right_count=2,
            document_count=4,
        )
        == -1.0
    )
    assert (
        -1.0
        <= npmi_from_counts(
            cooccurrence_count=1,
            left_count=2,
            right_count=2,
            document_count=4,
        )
        <= 1.0
    )


def test_representative_documents_prefer_distinct_source_turns() -> None:
    metadata = (
        nmf_grid.DocumentMetadata(
            0,
            "a",
            "s1",
            1,
            1,
            2020,
            "2020-2023",
            "legislative_debate",
            "named_or_role_unspecified",
            5,
            "s1::turn_000001",
            "a",
        ),
        nmf_grid.DocumentMetadata(
            1,
            "b",
            "s1",
            1,
            2,
            2020,
            "2020-2023",
            "legislative_debate",
            "named_or_role_unspecified",
            5,
            "s1::turn_000001",
            "b",
        ),
        nmf_grid.DocumentMetadata(
            2,
            "c",
            "s2",
            1,
            1,
            2020,
            "2020-2023",
            "legislative_debate",
            "named_or_role_unspecified",
            5,
            "s2::turn_000001",
            "c",
        ),
    )
    selected = nmf_grid._select_representative_indices(
        candidates=[0, 1, 2],
        document_metadata=metadata,
        per_topic_count=2,
    )

    assert selected == [0, 2]
