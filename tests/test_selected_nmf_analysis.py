import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_nmf_grid import (
    document_record,
    read_json,
    run_grid,
    write_grid_fixture,
    write_json,
)
from typer.testing import CliRunner

from argentine_deputies_discursive_distance import selected_nmf_analysis
from argentine_deputies_discursive_distance.cli import app
from argentine_deputies_discursive_distance.pdf_pipeline import sha256_file
from argentine_deputies_discursive_distance.selected_nmf_analysis import (
    SelectedNmfAnalysisError,
    analyze_selected_nmf,
    load_config,
)


@dataclass(frozen=True, slots=True)
class AnalysisFixture:
    grid_dir: Path
    config_path: Path
    output_dir: Path


HEALTH_TEXT = (
    "salud hospital medicina vacunas pacientes cuidado enfermeria sanitario publico "
    "clinica terapia comunidad prevencion sistema"
)
BUDGET_TEXT = (
    "presupuesto impuestos inflacion credito empleo industria inversion productiva "
    "salarios mercado desarrollo fiscal"
)
COMMON_TEXT = (
    "debate camara proyecto ley articulo nacional provincial reforma derechos programa "
    "informe comision dictamen"
)


def analysis_records() -> list[dict[str, Any]]:
    records = [
        document_record(
            "multi-health-1",
            f"{HEALTH_TEXT} {HEALTH_TEXT} {COMMON_TEXT}",
            source_record_id="session-balance-2008",
            turn_index=1,
            chunk_index=1,
            year=2008,
        ),
        document_record(
            "multi-health-2",
            f"{HEALTH_TEXT} {HEALTH_TEXT} {COMMON_TEXT}",
            source_record_id="session-balance-2008",
            turn_index=1,
            chunk_index=2,
            year=2008,
        ),
        document_record(
            "multi-budget-1",
            f"{BUDGET_TEXT} {BUDGET_TEXT} {COMMON_TEXT}",
            source_record_id="session-balance-2008",
            turn_index=2,
            chunk_index=1,
            year=2008,
        ),
        document_record(
            "single-budget-2008",
            f"{BUDGET_TEXT} {BUDGET_TEXT} {COMMON_TEXT}",
            source_record_id="session-single-2008",
            turn_index=1,
            chunk_index=1,
            year=2008,
        ),
    ]

    for year in range(2009, 2026):
        topic_text = HEALTH_TEXT if year % 2 == 0 else BUDGET_TEXT
        records.append(
            document_record(
                f"coverage-{year}",
                f"{topic_text} {topic_text} {COMMON_TEXT}",
                source_record_id=f"session-coverage-{year}",
                turn_index=1,
                chunk_index=1,
                year=year,
            )
        )

    records.extend(
        [
            document_record(
                "zero-unique-a",
                "zzzaaaonly zzzbbbonly zzzccconly",
                source_record_id="session-zero-a",
                turn_index=1,
                chunk_index=1,
                year=2014,
            ),
            document_record(
                "zero-unique-b",
                "zzzuuuonly zzzvvvonly zzzwwwonly",
                source_record_id="session-zero-b",
                turn_index=1,
                chunk_index=1,
                year=2014,
            ),
        ]
    )
    return records


def write_selected_config(
    path: Path,
    *,
    grid_dir: Path,
    selected_k: int = 2,
    threshold: float = 0.01,
) -> None:
    vectorizer_summary = read_json(grid_dir / "vectorizer_summary.json")
    preprocessing_summary = read_json(grid_dir / "preprocessing_summary.json")
    write_json(
        path,
        {
            "aggregation_levels": ["document", "source_turn", "session"],
            "analysis_version": "1",
            "annual_year_end": 2025,
            "annual_year_start": 2008,
            "expected_modeled_documents": vectorizer_summary["modeled_document_count"],
            "expected_primary_documents": preprocessing_summary["primary_counts"]["documents"],
            "expected_zero_tfidf_exclusions": vectorizer_summary["zero_tfidf_rows_excluded"],
            "float_dtype": "float32",
            "grid_prevalence_max_absolute_difference": threshold,
            "main_aggregation": "source_turn",
            "selected_k": selected_k,
            "stopword_variant": "P1",
            "temporal_periods": [
                "2008-2011",
                "2012-2015",
                "2016-2019",
                "2020-2023",
                "2024-2025",
            ],
            "weight_normalization": "row_sum_one",
        },
    )


def prepare_analysis_fixture(root: Path) -> AnalysisFixture:
    paths = write_grid_fixture(root / "grid-source", analysis_records())
    grid_config = read_json(paths.grid_config)
    grid_config["metrics_top_n"] = 10
    grid_config["tfidf"]["min_df"] = 2
    grid_config["top_terms_per_topic"] = 20
    grid_config["zero_tfidf_policy"] = {
        "maximum_fraction": 0.2,
        "maximum_rows": 5,
    }
    write_json(paths.grid_config, grid_config)
    run_grid(paths)
    selected_config = root / "selected_nmf_config.json"
    write_selected_config(selected_config, grid_dir=paths.output_dir)
    return AnalysisFixture(
        grid_dir=paths.output_dir,
        config_path=selected_config,
        output_dir=root / "selected-output",
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    assert rows
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def update_manifest_artifact(grid_dir: Path, artifact_key: str, filename: str) -> None:
    manifest_path = grid_dir / "run_manifest.json"
    manifest = read_json(manifest_path)
    artifact_path = grid_dir / filename
    manifest["output_files"][artifact_key]["sha256"] = sha256_file(artifact_path)
    manifest["output_files"][artifact_key]["size_bytes"] = artifact_path.stat().st_size
    write_json(manifest_path, manifest)


def run_analysis(fixture: AnalysisFixture, *, force: bool = False) -> dict[str, Any]:
    return analyze_selected_nmf(
        grid_input_dir=fixture.grid_dir,
        config_path=fixture.config_path,
        output_dir=fixture.output_dir,
        force=force,
    )


def test_strict_config_validation(tmp_path: Path) -> None:
    fixture = prepare_analysis_fixture(tmp_path)
    config = read_json(fixture.config_path)
    config["aggregation_levels"] = ["document", "session", "source_turn"]
    invalid_path = tmp_path / "invalid_config.json"
    write_json(invalid_path, config)

    with pytest.raises(SelectedNmfAnalysisError, match="aggregation_levels"):
        load_config(invalid_path)

    config = read_json(fixture.config_path)
    config["extra"] = True
    write_json(invalid_path, config)

    with pytest.raises(SelectedNmfAnalysisError, match="unsupported fields"):
        load_config(invalid_path)


def test_grid_manifest_artifact_hash_mismatch_fails(tmp_path: Path) -> None:
    fixture = prepare_analysis_fixture(tmp_path)
    manifest = read_json(fixture.grid_dir / "run_manifest.json")
    manifest["output_files"]["vectorizer_summary"]["sha256"] = "0" * 64
    write_json(fixture.grid_dir / "run_manifest.json", manifest)

    with pytest.raises(SelectedNmfAnalysisError, match="SHA-256 mismatch"):
        run_analysis(fixture)


def test_nonconverged_selected_k_rejection(tmp_path: Path) -> None:
    fixture = prepare_analysis_fixture(tmp_path)
    rows = read_csv_rows(fixture.grid_dir / "grid_metrics.csv")

    for row in rows:
        if row["k"] == "2":
            row["converged"] = "False"

    write_csv_rows(fixture.grid_dir / "grid_metrics.csv", rows)
    update_manifest_artifact(fixture.grid_dir, "grid_metrics", "grid_metrics.csv")

    with pytest.raises(SelectedNmfAnalysisError, match="did not converge"):
        run_analysis(fixture)


def test_zero_row_ledger_reconciliation_is_exact(tmp_path: Path) -> None:
    fixture = prepare_analysis_fixture(tmp_path)
    ledger_path = fixture.grid_dir / "zero_tfidf_documents.jsonl"
    rows = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows[0]["document_id"] = "wrong-document-id"
    ledger_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    update_manifest_artifact(fixture.grid_dir, "zero_tfidf_documents", "zero_tfidf_documents.jsonl")

    with pytest.raises(SelectedNmfAnalysisError, match="zero TF-IDF rows"):
        run_analysis(fixture)


def test_success_outputs_align_and_rows_are_normalized(tmp_path: Path) -> None:
    fixture = prepare_analysis_fixture(tmp_path)
    manifest = run_analysis(fixture)
    document_metadata = read_csv_rows(fixture.output_dir / "document_topic_metadata.csv")
    document_assignments = read_csv_rows(fixture.output_dir / "document_topic_assignments.csv")
    document_npz = np.load(fixture.output_dir / "document_topic_weights.npz")
    document_weights = document_npz["topic_weights"]
    document_ids = list(document_npz["document_ids"])

    assert manifest["configuration"]["expected_modeled_documents"] == len(document_metadata)
    assert document_weights.dtype == np.float32
    assert document_weights.shape[0] == len(document_metadata) == len(document_assignments)
    assert np.allclose(document_weights.sum(axis=1), 1.0, atol=1e-5)
    assert document_ids == [row["document_id"] for row in document_metadata]
    assert "topic_0" not in document_assignments[0]
    assert document_assignments[0]["dominant_topic_index"] in {"0", "1"}

    for filename in (
        "selected_model_manifest.json",
        "selected_model_report.md",
        "topic_metadata.csv",
        "source_turn_topic_weights.npz",
        "source_turn_metadata.csv",
        "session_topic_weights.npz",
        "session_metadata.csv",
        "annual_topic_prevalence.csv",
        "period_topic_prevalence.csv",
        "temporal_denominators.csv",
        "topic_change_summary.csv",
        "grid_prevalence_comparison.csv",
    ):
        assert (fixture.output_dir / filename).is_file()


def test_source_turn_averages_multiple_chunks(tmp_path: Path) -> None:
    fixture = prepare_analysis_fixture(tmp_path)
    run_analysis(fixture)
    document_weights = np.load(fixture.output_dir / "document_topic_weights.npz")["topic_weights"]
    source_turn_weights = np.load(fixture.output_dir / "source_turn_topic_weights.npz")[
        "topic_weights"
    ]
    document_metadata = read_csv_rows(fixture.output_dir / "document_topic_metadata.csv")
    source_turn_metadata = read_csv_rows(fixture.output_dir / "source_turn_metadata.csv")
    target_key = "session-balance-2008::turn_000001"
    document_indices = [
        index for index, row in enumerate(document_metadata) if row["source_turn_key"] == target_key
    ]
    source_turn_index = next(
        index
        for index, row in enumerate(source_turn_metadata)
        if row["source_turn_key"] == target_key
    )

    assert len(document_indices) == 2
    assert source_turn_metadata[source_turn_index]["modeled_document_count"] == "2"
    assert np.allclose(
        source_turn_weights[source_turn_index],
        document_weights[document_indices].mean(axis=0),
        atol=1e-6,
    )


def test_session_averages_source_turns_not_documents(tmp_path: Path) -> None:
    fixture = prepare_analysis_fixture(tmp_path)
    run_analysis(fixture)
    document_weights = np.load(fixture.output_dir / "document_topic_weights.npz")["topic_weights"]
    source_turn_weights = np.load(fixture.output_dir / "source_turn_topic_weights.npz")[
        "topic_weights"
    ]
    session_weights = np.load(fixture.output_dir / "session_topic_weights.npz")["topic_weights"]
    document_metadata = read_csv_rows(fixture.output_dir / "document_topic_metadata.csv")
    source_turn_metadata = read_csv_rows(fixture.output_dir / "source_turn_metadata.csv")
    session_metadata = read_csv_rows(fixture.output_dir / "session_metadata.csv")
    source_record_id = "session-balance-2008"
    document_indices = [
        index
        for index, row in enumerate(document_metadata)
        if row["source_record_id"] == source_record_id
    ]
    source_turn_indices = [
        index
        for index, row in enumerate(source_turn_metadata)
        if row["source_record_id"] == source_record_id
    ]
    session_index = next(
        index
        for index, row in enumerate(session_metadata)
        if row["source_record_id"] == source_record_id
    )

    expected_from_source_turns = source_turn_weights[source_turn_indices].mean(axis=0)
    direct_document_average = document_weights[document_indices].mean(axis=0)

    assert len(document_indices) == 3
    assert len(source_turn_indices) == 2
    assert np.allclose(session_weights[session_index], expected_from_source_turns, atol=1e-6)
    assert not np.allclose(session_weights[session_index], direct_document_average, atol=1e-4)


def test_weighting_levels_share_reconciliation_and_coverage(tmp_path: Path) -> None:
    fixture = prepare_analysis_fixture(tmp_path)
    run_analysis(fixture)
    annual_rows = read_csv_rows(fixture.output_dir / "annual_topic_prevalence.csv")
    period_rows = read_csv_rows(fixture.output_dir / "period_topic_prevalence.csv")
    annual_totals: dict[tuple[str, str], float] = defaultdict(float)
    period_totals: dict[tuple[str, str], float] = defaultdict(float)

    for row in annual_rows:
        annual_totals[(row["aggregation_level"], row["year"])] += float(row["prevalence_share"])

    for row in period_rows:
        period_totals[(row["aggregation_level"], row["temporal_period"])] += float(
            row["prevalence_share"]
        )

    assert set(annual_totals) == {
        (level, str(year))
        for level in ("document", "source_turn", "session")
        for year in range(2008, 2026)
    }
    assert set(period_totals) == {
        (level, period)
        for level in ("document", "source_turn", "session")
        for period in ("2008-2011", "2012-2015", "2016-2019", "2020-2023", "2024-2025")
    }
    assert all(abs(total - 1.0) < 1e-5 for total in annual_totals.values())
    assert all(abs(total - 1.0) < 1e-5 for total in period_totals.values())

    year_2008_vectors = {
        row["aggregation_level"]: np.array(
            [
                float(item["prevalence_share"])
                for item in annual_rows
                if item["aggregation_level"] == row["aggregation_level"] and item["year"] == "2008"
            ]
        )
        for row in annual_rows
        if row["year"] == "2008"
    }
    assert not np.allclose(year_2008_vectors["document"], year_2008_vectors["source_turn"])
    assert not np.allclose(year_2008_vectors["source_turn"], year_2008_vectors["session"])


def test_temporal_denominators_include_zero_vector_documents(tmp_path: Path) -> None:
    fixture = prepare_analysis_fixture(tmp_path)
    run_analysis(fixture)
    denominator_rows = read_csv_rows(fixture.output_dir / "temporal_denominators.csv")
    year_2014 = next(
        row
        for row in denominator_rows
        if row["denominator_scope"] == "year" and row["year"] == "2014"
    )
    period_2012_2015 = next(
        row
        for row in denominator_rows
        if row["denominator_scope"] == "period" and row["temporal_period"] == "2012-2015"
    )

    assert year_2014["zero_tfidf_excluded_document_count"] == "2"
    assert int(year_2014["corpus_primary_document_count_including_zero_tfidf"]) == (
        int(year_2014["modeled_document_count"]) + 2
    )
    assert period_2012_2015["zero_tfidf_excluded_document_count"] == "2"
    assert "no modelled topic vector" in year_2014["zero_tfidf_note"]


def test_topic_change_summary_calculations(tmp_path: Path) -> None:
    fixture = prepare_analysis_fixture(tmp_path)
    run_analysis(fixture)
    annual_rows = read_csv_rows(fixture.output_dir / "annual_topic_prevalence.csv")
    period_rows = read_csv_rows(fixture.output_dir / "period_topic_prevalence.csv")
    change_rows = read_csv_rows(fixture.output_dir / "topic_change_summary.csv")
    topic_index = 0
    row = next(item for item in change_rows if item["topic_index"] == str(topic_index))
    annual_values = [
        (
            int(item["year"]),
            float(item["prevalence_share"]),
        )
        for item in annual_rows
        if item["aggregation_level"] == "source_turn" and item["topic_index"] == str(topic_index)
    ]
    baseline = next(
        float(item["prevalence_share"])
        for item in period_rows
        if item["aggregation_level"] == "source_turn"
        and item["topic_index"] == str(topic_index)
        and item["temporal_period"] == "2008-2011"
    )
    final = next(
        float(item["prevalence_share"])
        for item in period_rows
        if item["aggregation_level"] == "source_turn"
        and item["topic_index"] == str(topic_index)
        and item["temporal_period"] == "2024-2025"
    )
    increases = [
        (annual_values[index][0], annual_values[index][1] - annual_values[index - 1][1])
        for index in range(1, len(annual_values))
    ]
    max_year, max_value = max(annual_values, key=lambda item: (item[1], -item[0]))
    min_year, min_value = min(annual_values, key=lambda item: (item[1], item[0]))
    increase_year, increase = max(increases, key=lambda item: (item[1], -item[0]))
    decrease_year, decrease = min(increases, key=lambda item: (item[1], item[0]))

    assert row["baseline_period"] == "2008-2011"
    assert row["final_period"] == "2024-2025"
    assert float(row["absolute_change"]) == pytest.approx(final - baseline, abs=1e-9)
    assert row["maximum_prevalence_year"] == str(max_year)
    assert float(row["maximum_prevalence_value"]) == pytest.approx(max_value, abs=1e-9)
    assert row["minimum_prevalence_year"] == str(min_year)
    assert float(row["minimum_prevalence_value"]) == pytest.approx(min_value, abs=1e-9)
    assert row["largest_year_to_year_increase_ending_year"] == str(increase_year)
    assert float(row["largest_year_to_year_increase"]) == pytest.approx(increase, abs=1e-9)
    assert row["largest_year_to_year_decrease_ending_year"] == str(decrease_year)
    assert float(row["largest_year_to_year_decrease"]) == pytest.approx(decrease, abs=1e-9)


def test_grid_prevalence_comparison_threshold(tmp_path: Path) -> None:
    fixture = prepare_analysis_fixture(tmp_path)
    rows = read_csv_rows(fixture.grid_dir / "topic_terms_k002.csv")

    for row in rows:
        if row["topic_index"] == "0":
            row["prevalence"] = "0.999999"

    write_csv_rows(fixture.grid_dir / "topic_terms_k002.csv", rows)
    update_manifest_artifact(fixture.grid_dir, "topic_terms_k002", "topic_terms_k002.csv")

    with pytest.raises(SelectedNmfAnalysisError, match="beyond threshold"):
        run_analysis(fixture)


def test_overwrite_protection(tmp_path: Path) -> None:
    fixture = prepare_analysis_fixture(tmp_path)
    fixture.output_dir.mkdir(parents=True)
    (fixture.output_dir / "sentinel.txt").write_text("existing", encoding="utf-8")

    with pytest.raises(SelectedNmfAnalysisError, match="nonempty"):
        run_analysis(fixture)


def test_transactional_rollback_leaves_no_partial_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = prepare_analysis_fixture(tmp_path)
    original_replace = selected_nmf_analysis._replace_path

    def fail_on_report(source: Path, destination: Path) -> None:
        if destination.name == "selected_model_report.md":
            raise OSError("synthetic promotion failure")

        original_replace(source, destination)

    monkeypatch.setattr(selected_nmf_analysis, "_replace_path", fail_on_report)

    with pytest.raises(SelectedNmfAnalysisError, match="promote"):
        run_analysis(fixture)

    assert not (fixture.output_dir / "selected_model_report.md").exists()
    assert not list(fixture.output_dir.glob("*.part"))
    assert not list(fixture.output_dir.glob("*.bak"))


def test_deterministic_textual_outputs_except_timestamp_fields(tmp_path: Path) -> None:
    fixture = prepare_analysis_fixture(tmp_path)
    run_analysis(fixture)
    textual_files = (
        "selected_model_report.md",
        "topic_metadata.csv",
        "document_topic_metadata.csv",
        "document_topic_assignments.csv",
        "source_turn_metadata.csv",
        "session_metadata.csv",
        "annual_topic_prevalence.csv",
        "period_topic_prevalence.csv",
        "temporal_denominators.csv",
        "topic_change_summary.csv",
        "grid_prevalence_comparison.csv",
    )
    first_bytes = {name: (fixture.output_dir / name).read_bytes() for name in textual_files}
    first_manifest = read_json(fixture.output_dir / "selected_model_manifest.json")
    run_analysis(fixture, force=True)

    assert {name: (fixture.output_dir / name).read_bytes() for name in textual_files} == first_bytes
    second_manifest = read_json(fixture.output_dir / "selected_model_manifest.json")
    first_manifest.pop("generated_at_utc")
    second_manifest.pop("generated_at_utc")
    first_manifest.pop("output_files")
    second_manifest.pop("output_files")
    assert first_manifest == second_manifest


def test_cli_smoke_execution(tmp_path: Path) -> None:
    fixture = prepare_analysis_fixture(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "analyze-selected-nmf",
            "--grid-input-dir",
            str(fixture.grid_dir),
            "--config",
            str(fixture.config_path),
            "--output-dir",
            str(fixture.output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (fixture.output_dir / "selected_model_manifest.json").is_file()
