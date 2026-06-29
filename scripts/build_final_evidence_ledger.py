
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]

DOCS_DIR = ROOT / "docs"
P1_GRID_DIR = ROOT / "data" / "qa" / "topic_modeling" / "nmf_grid_v1"
P0_GRID_DIR = ROOT / "data" / "qa" / "topic_modeling" / "nmf_p0_k024_v1"
SELECTED_DIR = ROOT / "data" / "qa" / "topic_modeling" / "selected_nmf_k024_v1"

OUTPUT_PATH = DOCS_DIR / "final_evidence_ledger.csv"

P1_GRID_METRICS = P1_GRID_DIR / "grid_metrics.csv"
P1_RUN_MANIFEST = P1_GRID_DIR / "run_manifest.json"
P1_REPRESENTATIVES = P1_GRID_DIR / "representative_documents_k024.jsonl"

P0_GRID_METRICS = P0_GRID_DIR / "grid_metrics.csv"
P0_RUN_MANIFEST = P0_GRID_DIR / "run_manifest.json"

ANNUAL_PREVALENCE = SELECTED_DIR / "annual_topic_prevalence.csv"
PERIOD_PREVALENCE = SELECTED_DIR / "period_topic_prevalence.csv"
GRID_COMPARISON = SELECTED_DIR / "grid_prevalence_comparison.csv"
TEMPORAL_DENOMINATORS = SELECTED_DIR / "temporal_denominators.csv"
TOPIC_CHANGE_SUMMARY = SELECTED_DIR / "topic_change_summary.csv"

EXPECTED_PRIMARY_DOCUMENTS = 75_123
EXPECTED_MODELLED_DOCUMENTS = 75_121
EXPECTED_SOURCE_TURNS = 34_060
EXPECTED_SESSIONS = 243
EXPECTED_PRIMARY_WORDS = 15_901_236
EXPECTED_SELECTED_K = 24
EXPECTED_P0_STOPWORDS = 310
EXPECTED_P1_STOPWORDS = 315
EXPECTED_ZERO_TFIDF_P1 = 2
EXPECTED_ZERO_TFIDF_P0 = 1

TOPIC_LABELS = {
    0: "General argumentation and confrontation",
    1: "Criminal law and procedural codes",
    2: "Article-by-article drafting",
    3: "Committees, reports and parliamentary business",
    4: "Justice and the judiciary",
    5: "Provinces, Buenos Aires and federal territory",
    6: "Applause and reactions in the chamber",
    7: "Bills and legislative initiatives",
    8: "Productive development and the economy",
    9: "Rights, gender and reproductive health",
    10: "Labour, employment and working conditions",
    11: "Political identity and partisan historical memory",
    12: "Government-opposition positioning",
    13: "Bloc positions and explanations of votes",
    14: "Motions, standing orders and agenda management",
    15: "Amendments to bill text",
    16: "Executive powers, emergency decrees and constitutional authority",
    17: "Budget, public expenditure and inflation",
    18: "Pensions and the social-security system",
    19: "Debt, the IMF and financial policy",
    20: "Taxes and fiscal policy",
    21: "Chamber and session dynamics",
    22: "Insertion of speeches into the official record",
    23: "Questions of privilege",
}

FIELDNAMES = [
    "claim_id",
    "slide_target",
    "claim_type",
    "claim",
    "topic_id",
    "topic_label",
    "aggregation",
    "year_or_period",
    "raw_value",
    "display_value",
    "unit",
    "comparison_value",
    "source_file",
    "source_sha256",
    "source_filter",
    "representative_document_id",
    "representative_document_excerpt",
    "caveat",
    "status",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV does not exist: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        return list(csv.DictReader(input_file))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON does not exist: {path}")

    payload = json.loads(path.read_text(encoding="utf-8-sig"))

    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")

    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSONL does not exist: {path}")

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise TypeError(f"Expected a JSON object at {path}:{line_number}")

            rows.append(payload)

    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def relative_path(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def first_present(mapping: dict[str, Any] | dict[str, str], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    raise KeyError(f"None of these fields were found: {keys}")


def exact_row(
    rows: Iterable[dict[str, str]],
    **filters: object,
) -> dict[str, str]:
    matches: list[dict[str, str]] = []

    for row in rows:
        matched = True

        for field, expected in filters.items():
            actual = row.get(field)

            if isinstance(expected, int):
                try:
                    matched = int(actual or "") == expected
                except ValueError:
                    matched = False
            else:
                matched = actual == str(expected)

            if not matched:
                break

        if matched:
            matches.append(row)

    if len(matches) != 1:
        raise AssertionError(
            f"Expected exactly one row for filters {filters}, found {len(matches)}"
        )

    return matches[0]


def percentage(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def decimal(value: float, digits: int = 10) -> str:
    return f"{value:.{digits}f}"


def json_compact(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def bounded_excerpt(text: str, limit: int = 280) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "…"


def representative_excerpt(row: dict[str, Any]) -> str:
    for key in (
        "cleaned_text_excerpt",
        "modeling_text_excerpt",
        "exact_text_excerpt",
        "modeling_text",
        "exact_text",
        "text",
    ):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return bounded_excerpt(value)

    return ""


def representative_document_id(row: dict[str, Any]) -> str:
    for key in ("document_id", "doc_id", "id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)

    return ""


def make_row(
    *,
    claim_id: str,
    slide_target: str,
    claim_type: str,
    claim: str,
    source_path: Path | None,
    raw_value: str = "",
    display_value: str = "",
    unit: str = "",
    comparison_value: object | str = "",
    topic_id: int | None = None,
    aggregation: str = "",
    year_or_period: str = "",
    source_filter: str = "",
    representative_document: dict[str, Any] | None = None,
    caveat: str = "",
    status: str = "LOCKED",
) -> dict[str, str]:
    if isinstance(comparison_value, str):
        serialized_comparison = comparison_value
    else:
        serialized_comparison = json_compact(comparison_value)

    if representative_document is None:
        rep_id = ""
        rep_excerpt = ""
    else:
        rep_id = representative_document_id(representative_document)
        rep_excerpt = representative_excerpt(representative_document)

    return {
        "claim_id": claim_id,
        "slide_target": slide_target,
        "claim_type": claim_type,
        "claim": claim,
        "topic_id": "" if topic_id is None else str(topic_id),
        "topic_label": "" if topic_id is None else TOPIC_LABELS[topic_id],
        "aggregation": aggregation,
        "year_or_period": year_or_period,
        "raw_value": raw_value,
        "display_value": display_value,
        "unit": unit,
        "comparison_value": serialized_comparison,
        "source_file": "" if source_path is None else relative_path(source_path),
        "source_sha256": "" if source_path is None else sha256_file(source_path),
        "source_filter": source_filter,
        "representative_document_id": rep_id,
        "representative_document_excerpt": rep_excerpt,
        "caveat": caveat,
        "status": status,
    }


def main() -> None:
    required_paths = [
        DOCS_DIR / "FINAL_SCOPE_FREEZE.md",
        P1_GRID_METRICS,
        P1_RUN_MANIFEST,
        P1_REPRESENTATIVES,
        P0_GRID_METRICS,
        P0_RUN_MANIFEST,
        ANNUAL_PREVALENCE,
        PERIOD_PREVALENCE,
        GRID_COMPARISON,
        TEMPORAL_DENOMINATORS,
        TOPIC_CHANGE_SUMMARY,
    ]

    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Missing required Phase 2 artifacts:\n"
            f"{formatted}"
        )

    p1_manifest = read_json(P1_RUN_MANIFEST)
    p0_manifest = read_json(P0_RUN_MANIFEST)

    p1_grid_rows = read_csv_rows(P1_GRID_METRICS)
    p0_grid_rows = read_csv_rows(P0_GRID_METRICS)

    annual_rows = read_csv_rows(ANNUAL_PREVALENCE)
    period_rows = read_csv_rows(PERIOD_PREVALENCE)
    grid_comparison_rows = read_csv_rows(GRID_COMPARISON)
    temporal_denominator_rows = read_csv_rows(TEMPORAL_DENOMINATORS)
    topic_change_rows = read_csv_rows(TOPIC_CHANGE_SUMMARY)

    representative_rows = read_jsonl(P1_REPRESENTATIVES)

    # ------------------------------------------------------------------
    # Locked corpus validation
    # ------------------------------------------------------------------

    p1_primary_counts = p1_manifest["primary_counts"]
    p1_stopwords = p1_manifest["stopwords"]
    p0_stopwords = p0_manifest["stopwords"]

    assert int(p1_manifest["modeled_document_count"]) == EXPECTED_MODELLED_DOCUMENTS
    assert int(p1_primary_counts["documents"]) == EXPECTED_PRIMARY_DOCUMENTS
    assert int(p1_primary_counts["source_turns"]) == EXPECTED_SOURCE_TURNS
    assert int(p1_primary_counts["sessions"]) == EXPECTED_SESSIONS
    assert int(p1_primary_counts["words"]) == EXPECTED_PRIMARY_WORDS

    assert int(p1_stopwords["p0_count"]) == EXPECTED_P0_STOPWORDS
    assert int(p1_stopwords["p1_count"]) == EXPECTED_P1_STOPWORDS
    assert int(p1_stopwords["selected_count"]) == EXPECTED_P1_STOPWORDS
    assert p1_stopwords["variant"] == "P1"

    assert int(p0_stopwords["selected_count"]) == EXPECTED_P0_STOPWORDS
    assert p0_stopwords["variant"] == "P0"

    assert int(
        p1_manifest["zero_tfidf_exclusions"]["excluded_document_count"]
    ) == EXPECTED_ZERO_TFIDF_P1
    assert int(
        p0_manifest["zero_tfidf_exclusions"]["excluded_document_count"]
    ) == EXPECTED_ZERO_TFIDF_P0

    # ------------------------------------------------------------------
    # NMF grid validation
    # ------------------------------------------------------------------

    p1_k24 = exact_row(p1_grid_rows, k=EXPECTED_SELECTED_K)
    p0_k24 = exact_row(p0_grid_rows, k=EXPECTED_SELECTED_K)
    p1_k12 = exact_row(p1_grid_rows, k=12)

    assert p1_k24["converged"] == "True"
    assert p1_k12["converged"] == "False"
    assert p0_k24["converged"] == "True"

    converged_p1_rows = [
        row for row in p1_grid_rows if row["converged"] == "True"
    ]

    best_npmi_row = max(
        converged_p1_rows,
        key=lambda row: float(row["mean_npmi_coherence_top10"]),
    )
    best_diversity_row = max(
        converged_p1_rows,
        key=lambda row: float(row["topic_diversity_top10"]),
    )
    best_exclusivity_row = max(
        converged_p1_rows,
        key=lambda row: float(row["mean_topic_exclusivity_top10"]),
    )

    assert int(best_npmi_row["k"]) == EXPECTED_SELECTED_K
    assert int(best_diversity_row["k"]) == EXPECTED_SELECTED_K
    assert int(best_exclusivity_row["k"]) == EXPECTED_SELECTED_K

    # ------------------------------------------------------------------
    # Representative-document validation
    # ------------------------------------------------------------------

    representatives_by_topic: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for row in representative_rows:
        topic_index = int(first_present(row, "topic_index", "topic_id"))
        representatives_by_topic[topic_index].append(row)

    assert set(representatives_by_topic) == set(range(24))

    for topic_index, rows_for_topic in representatives_by_topic.items():
        rows_for_topic.sort(
            key=lambda row: int(row.get("rank", 999999))
        )
        assert len(rows_for_topic) == 5, (
            f"Expected five representative documents for topic "
            f"{topic_index}, found {len(rows_for_topic)}"
        )

    def representative(
        topic_index: int,
        *,
        preferred_year: int | None = None,
        preferred_rank: int | None = None,
    ) -> dict[str, Any]:
        candidates = representatives_by_topic[topic_index]

        if preferred_year is not None:
            year_matches = [
                row
                for row in candidates
                if str(row.get("year", "")) == str(preferred_year)
            ]
            if year_matches:
                return year_matches[0]

        if preferred_rank is not None:
            rank_matches = [
                row
                for row in candidates
                if str(row.get("rank", "")) == str(preferred_rank)
            ]
            if rank_matches:
                return rank_matches[0]

        return candidates[0]

    # ------------------------------------------------------------------
    # Temporal accessors and validations
    # ------------------------------------------------------------------

    def annual(
        topic_index: int,
        year: int,
        aggregation: str = "source_turn",
    ) -> dict[str, str]:
        return exact_row(
            annual_rows,
            topic_index=topic_index,
            year=year,
            aggregation_level=aggregation,
        )

    def period(
        topic_index: int,
        temporal_period: str,
        aggregation: str = "source_turn",
    ) -> dict[str, str]:
        return exact_row(
            period_rows,
            topic_index=topic_index,
            temporal_period=temporal_period,
            aggregation_level=aggregation,
        )

    def annual_value(
        topic_index: int,
        year: int,
        aggregation: str = "source_turn",
    ) -> float:
        return float(
            first_present(
                annual(topic_index, year, aggregation),
                "prevalence_share",
                "prevalence",
                "topic_prevalence",
            )
        )

    def period_value(
        topic_index: int,
        temporal_period: str,
        aggregation: str = "source_turn",
    ) -> float:
        return float(
            first_present(
                period(topic_index, temporal_period, aggregation),
                "prevalence_share",
                "prevalence",
                "topic_prevalence",
            )
        )

    def maximum_year(topic_index: int, aggregation: str) -> int:
        topic_rows = [
            row
            for row in annual_rows
            if int(row["topic_index"]) == topic_index
            and row["aggregation_level"] == aggregation
        ]

        maximum = max(
            topic_rows,
            key=lambda row: float(
                first_present(
                    row,
                    "prevalence_share",
                    "prevalence",
                    "topic_prevalence",
                )
            ),
        )
        return int(maximum["year"])

    for aggregation in ("document", "source_turn", "session"):
        assert maximum_year(4, aggregation) == 2013
        assert maximum_year(9, aggregation) == 2018
        assert maximum_year(17, aggregation) == 2021
        assert maximum_year(20, aggregation) == 2021

    maximum_grid_difference = max(
        float(
            first_present(
                row,
                "absolute_difference",
                "abs_difference",
                "difference_absolute",
            )
        )
        for row in grid_comparison_rows
    )

    assert maximum_grid_difference < 0.00002

    represented_years = {
        int(row["year"])
        for row in annual_rows
        if row["aggregation_level"] == "source_turn"
    }
    assert represented_years == set(range(2008, 2026))

    annual_denominators = [
        row
        for row in temporal_denominator_rows
        if row.get("denominator_scope") == "year"
    ]
    assert len(annual_denominators) == 18

    assert {
        int(row["topic_index"])
        for row in topic_change_rows
    } == set(range(24))

    # ------------------------------------------------------------------
    # Values used in the evidence ledger
    # ------------------------------------------------------------------

    p1_npmi = float(p1_k24["mean_npmi_coherence_top10"])
    p0_npmi = float(p0_k24["mean_npmi_coherence_top10"])

    p1_diversity = float(p1_k24["topic_diversity_top10"])
    p0_diversity = float(p0_k24["topic_diversity_top10"])

    p1_exclusivity = float(p1_k24["mean_topic_exclusivity_top10"])
    p0_exclusivity = float(p0_k24["mean_topic_exclusivity_top10"])

    p1_mean_redundancy = float(
        p1_k24["redundancy_mean_off_diagonal_cosine"]
    )
    p0_mean_redundancy = float(
        p0_k24["redundancy_mean_off_diagonal_cosine"]
    )

    pension_initial = period_value(18, "2008-2011")
    pension_final = period_value(18, "2024-2025")
    pension_change = pension_final - pension_initial
    pension_relative_change = pension_change / pension_initial

    privilege_initial = period_value(23, "2008-2011")
    privilege_final = period_value(23, "2024-2025")
    privilege_change = privilege_final - privilege_initial
    privilege_relative_change = privilege_change / privilege_initial

    justice_robustness = {
        aggregation: annual_value(4, 2013, aggregation)
        for aggregation in ("document", "source_turn", "session")
    }
    rights_robustness = {
        aggregation: annual_value(9, 2018, aggregation)
        for aggregation in ("document", "source_turn", "session")
    }
    budget_robustness = {
        aggregation: annual_value(17, 2021, aggregation)
        for aggregation in ("document", "source_turn", "session")
    }
    taxes_robustness = {
        aggregation: annual_value(20, 2021, aggregation)
        for aggregation in ("document", "source_turn", "session")
    }

    justice_2013 = annual_value(4, 2013)
    rights_2018 = annual_value(9, 2018)
    rights_2020 = annual_value(9, 2020)
    budget_2021 = annual_value(17, 2021)
    taxes_2021 = annual_value(20, 2021)
    taxes_2022 = annual_value(20, 2022)
    taxes_2023 = annual_value(20, 2023)

    rows: list[dict[str, str]] = []

    # ------------------------------------------------------------------
    # Corpus claims
    # ------------------------------------------------------------------

    rows.extend(
        [
            make_row(
                claim_id="C01",
                slide_target="S3",
                claim_type="corpus",
                claim="The primary modelling corpus contains 75,123 documents.",
                source_path=P1_RUN_MANIFEST,
                raw_value=str(EXPECTED_PRIMARY_DOCUMENTS),
                display_value="75,123",
                unit="documents",
                source_filter="primary_counts.documents",
                caveat=(
                    "Two primary documents remain in corpus denominators but "
                    "produce zero TF-IDF vectors under P1."
                ),
            ),
            make_row(
                claim_id="C02",
                slide_target="S3",
                claim_type="corpus",
                claim="The selected NMF model contains 75,121 modelled documents.",
                source_path=P1_RUN_MANIFEST,
                raw_value=str(EXPECTED_MODELLED_DOCUMENTS),
                display_value="75,121",
                unit="documents",
                source_filter="modeled_document_count",
            ),
            make_row(
                claim_id="C03",
                slide_target="S3",
                claim_type="corpus",
                claim="The primary corpus contains 34,060 retained source turns.",
                source_path=P1_RUN_MANIFEST,
                raw_value=str(EXPECTED_SOURCE_TURNS),
                display_value="34,060",
                unit="source_turns",
                source_filter="primary_counts.source_turns",
            ),
            make_row(
                claim_id="C04",
                slide_target="S3",
                claim_type="corpus",
                claim="The temporal analysis contains 243 legislative-debate sessions.",
                source_path=P1_RUN_MANIFEST,
                raw_value=str(EXPECTED_SESSIONS),
                display_value="243",
                unit="sessions",
                source_filter="primary_counts.sessions",
            ),
            make_row(
                claim_id="C05",
                slide_target="S3",
                claim_type="corpus",
                claim="The primary modelling corpus contains approximately 15.9 million words.",
                source_path=P1_RUN_MANIFEST,
                raw_value=str(EXPECTED_PRIMARY_WORDS),
                display_value="15.9 million",
                unit="words",
                source_filter="primary_counts.words",
            ),
            make_row(
                claim_id="C06",
                slide_target="S3",
                claim_type="corpus",
                claim="Only two P1 primary documents are excluded as zero TF-IDF rows.",
                source_path=P1_RUN_MANIFEST,
                raw_value=str(EXPECTED_ZERO_TFIDF_P1),
                display_value="2",
                unit="documents",
                source_filter="zero_tfidf_exclusions.excluded_document_count",
                caveat=(
                    "The excluded documents remain part of corpus denominator "
                    "reporting but receive no modelled topic vector."
                ),
            ),
        ]
    )

    # ------------------------------------------------------------------
    # Model-selection and sensitivity claims
    # ------------------------------------------------------------------

    rows.extend(
        [
            make_row(
                claim_id="C07",
                slide_target="S5",
                claim_type="model_selection",
                claim="The K=12 P1 model did not converge within 400 iterations.",
                source_path=P1_GRID_METRICS,
                raw_value="False",
                display_value="Did not converge",
                unit="convergence_status",
                source_filter="k=12",
                caveat=(
                    "The remaining candidate models converged under the "
                    "configured maximum iteration count."
                ),
            ),
            make_row(
                claim_id="C08",
                slide_target="S5",
                claim_type="model_selection",
                claim="K=24 has the highest mean NPMI among converged P1 grid models.",
                source_path=P1_GRID_METRICS,
                raw_value=decimal(p1_npmi),
                display_value=f"{p1_npmi:.3f}",
                unit="mean_npmi",
                comparison_value={
                    int(row["k"]): float(row["mean_npmi_coherence_top10"])
                    for row in converged_p1_rows
                },
                source_filter="converged=True;metric=mean_npmi_coherence_top10",
            ),
            make_row(
                claim_id="C09",
                slide_target="S5",
                claim_type="model_selection",
                claim="K=24 has the highest top-10 topic diversity among converged P1 models.",
                source_path=P1_GRID_METRICS,
                raw_value=decimal(p1_diversity),
                display_value=f"{p1_diversity:.3f}",
                unit="topic_diversity",
                comparison_value={
                    int(row["k"]): float(row["topic_diversity_top10"])
                    for row in converged_p1_rows
                },
                source_filter="converged=True;metric=topic_diversity_top10",
            ),
            make_row(
                claim_id="C10",
                slide_target="S5",
                claim_type="model_selection",
                claim="K=24 has the highest mean exclusivity among converged P1 models.",
                source_path=P1_GRID_METRICS,
                raw_value=decimal(p1_exclusivity),
                display_value=f"{p1_exclusivity:.3f}",
                unit="mean_exclusivity",
                comparison_value={
                    int(row["k"]): float(
                        row["mean_topic_exclusivity_top10"]
                    )
                    for row in converged_p1_rows
                },
                source_filter="converged=True;metric=mean_topic_exclusivity_top10",
            ),
            make_row(
                claim_id="C11",
                slide_target="S5",
                claim_type="sensitivity",
                claim="P1 improves mean NPMI relative to P0 at K=24.",
                source_path=P1_GRID_METRICS,
                raw_value=decimal(p1_npmi),
                display_value=f"{p0_npmi:.3f} -> {p1_npmi:.3f}",
                unit="mean_npmi",
                comparison_value={
                    "P0": p0_npmi,
                    "P1": p1_npmi,
                    "absolute_change": p1_npmi - p0_npmi,
                },
                source_filter="P0 and P1; k=24",
                caveat=(
                    "P0 and P1 are separately fitted feature spaces; this is "
                    "a preprocessing sensitivity comparison, not a one-to-one "
                    "topic-alignment test."
                ),
            ),
            make_row(
                claim_id="C12",
                slide_target="S5",
                claim_type="sensitivity",
                claim="P1 improves top-10 topic diversity relative to P0.",
                source_path=P1_GRID_METRICS,
                raw_value=decimal(p1_diversity),
                display_value=f"{p0_diversity:.3f} -> {p1_diversity:.3f}",
                unit="topic_diversity",
                comparison_value={
                    "P0": p0_diversity,
                    "P1": p1_diversity,
                    "absolute_change": p1_diversity - p0_diversity,
                },
                source_filter="P0 and P1; k=24",
            ),
            make_row(
                claim_id="C13",
                slide_target="S5",
                claim_type="sensitivity",
                claim="P1 improves mean topic exclusivity relative to P0.",
                source_path=P1_GRID_METRICS,
                raw_value=decimal(p1_exclusivity),
                display_value=f"{p0_exclusivity:.3f} -> {p1_exclusivity:.3f}",
                unit="mean_exclusivity",
                comparison_value={
                    "P0": p0_exclusivity,
                    "P1": p1_exclusivity,
                    "absolute_change": p1_exclusivity - p0_exclusivity,
                },
                source_filter="P0 and P1; k=24",
            ),
            make_row(
                claim_id="C14",
                slide_target="S5",
                claim_type="sensitivity",
                claim="P1 increases mean redundancy only slightly relative to P0.",
                source_path=P1_GRID_METRICS,
                raw_value=decimal(p1_mean_redundancy),
                display_value=(
                    f"{p0_mean_redundancy:.3f} -> "
                    f"{p1_mean_redundancy:.3f}"
                ),
                unit="mean_pairwise_cosine",
                comparison_value={
                    "P0": p0_mean_redundancy,
                    "P1": p1_mean_redundancy,
                    "absolute_change": (
                        p1_mean_redundancy - p0_mean_redundancy
                    ),
                },
                source_filter="P0 and P1; k=24",
                caveat=(
                    "The small redundancy increase is outweighed by larger "
                    "improvements in coherence, diversity and exclusivity."
                ),
            ),
            make_row(
                claim_id="C15",
                slide_target="S10",
                claim_type="reconciliation",
                claim="Selected-model prevalence reproduces grid prevalence to within 0.00001888.",
                source_path=GRID_COMPARISON,
                raw_value=decimal(maximum_grid_difference),
                display_value=f"{maximum_grid_difference:.8f}",
                unit="maximum_absolute_difference",
                source_filter="max(absolute_difference)",
                caveat=(
                    "This validates that the selected-model transform is "
                    "consistent with the original fitted-grid prevalence."
                ),
            ),
        ]
    )

    # ------------------------------------------------------------------
    # Temporal findings
    # ------------------------------------------------------------------

    rows.extend(
        [
            make_row(
                claim_id="C16",
                slide_target="S7",
                claim_type="temporal_finding",
                claim="Justice and judicial discourse reaches its maximum in 2013.",
                source_path=ANNUAL_PREVALENCE,
                raw_value=decimal(justice_2013),
                display_value=percentage(justice_2013),
                unit="prevalence_share",
                topic_id=4,
                aggregation="source_turn",
                year_or_period="2013",
                comparison_value=justice_robustness,
                source_filter="aggregation_level=source_turn;year=2013;topic_index=4",
                representative_document=representative(4),
                caveat=(
                    "Representative documents validate the topic identity but "
                    "do not establish the cause of the 2013 peak."
                ),
            ),
            make_row(
                claim_id="C17",
                slide_target="S7",
                claim_type="temporal_finding",
                claim="Rights, gender and reproductive-health discourse reaches its maximum in 2018.",
                source_path=ANNUAL_PREVALENCE,
                raw_value=decimal(rights_2018),
                display_value=percentage(rights_2018),
                unit="prevalence_share",
                topic_id=9,
                aggregation="source_turn",
                year_or_period="2018",
                comparison_value=rights_robustness,
                source_filter="aggregation_level=source_turn;year=2018;topic_index=9",
                representative_document=representative(9, preferred_year=2018),
                caveat=(
                    "The component is broader than abortion and also contains "
                    "women's rights, human rights, health, gender and violence."
                ),
            ),
            make_row(
                claim_id="C18",
                slide_target="S7",
                claim_type="temporal_finding",
                claim="Rights, gender and reproductive-health discourse rises again in 2020.",
                source_path=ANNUAL_PREVALENCE,
                raw_value=decimal(rights_2020),
                display_value=percentage(rights_2020),
                unit="prevalence_share",
                topic_id=9,
                aggregation="source_turn",
                year_or_period="2020",
                source_filter="aggregation_level=source_turn;year=2020;topic_index=9",
                representative_document=representative(9),
            ),
            make_row(
                claim_id="C19",
                slide_target="S8",
                claim_type="temporal_finding",
                claim="Budget, public-expenditure and inflation discourse reaches its maximum in 2021.",
                source_path=ANNUAL_PREVALENCE,
                raw_value=decimal(budget_2021),
                display_value=percentage(budget_2021),
                unit="prevalence_share",
                topic_id=17,
                aggregation="source_turn",
                year_or_period="2021",
                comparison_value=budget_robustness,
                source_filter="aggregation_level=source_turn;year=2021;topic_index=17",
                representative_document=representative(17),
                caveat=(
                    "Representative documents validate the topic identity but "
                    "do not establish the cause of the 2021 peak."
                ),
            ),
            make_row(
                claim_id="C20",
                slide_target="S8",
                claim_type="temporal_finding",
                claim="Tax and fiscal-policy discourse reaches its maximum in 2021.",
                source_path=ANNUAL_PREVALENCE,
                raw_value=decimal(taxes_2021),
                display_value=percentage(taxes_2021),
                unit="prevalence_share",
                topic_id=20,
                aggregation="source_turn",
                year_or_period="2021",
                comparison_value=taxes_robustness,
                source_filter="aggregation_level=source_turn;year=2021;topic_index=20",
                representative_document=representative(20, preferred_year=2021),
                caveat=(
                    "The topic includes income tax, personal-assets tax, "
                    "thresholds, internal taxes and broader fiscal policy."
                ),
            ),
            make_row(
                claim_id="C21",
                slide_target="S8",
                claim_type="temporal_finding",
                claim="Tax discourse remains elevated through 2022 and 2023.",
                source_path=ANNUAL_PREVALENCE,
                raw_value=decimal(taxes_2023),
                display_value=(
                    f"2022: {percentage(taxes_2022)}; "
                    f"2023: {percentage(taxes_2023)}"
                ),
                unit="prevalence_share",
                topic_id=20,
                aggregation="source_turn",
                year_or_period="2022-2023",
                comparison_value={
                    "2021": taxes_2021,
                    "2022": taxes_2022,
                    "2023": taxes_2023,
                },
                source_filter="aggregation_level=source_turn;years=2021,2022,2023;topic_index=20",
                representative_document=representative(20),
            ),
            make_row(
                claim_id="C22",
                slide_target="S9",
                claim_type="long_run_change",
                claim="Pension and social-security discourse rises from 3.20% to 5.51% between the first and final periods.",
                source_path=PERIOD_PREVALENCE,
                raw_value=decimal(pension_final),
                display_value=(
                    f"{percentage(pension_initial)} -> "
                    f"{percentage(pension_final)}"
                ),
                unit="prevalence_share",
                topic_id=18,
                aggregation="source_turn",
                year_or_period="2008-2011 vs 2024-2025",
                comparison_value={
                    "baseline": pension_initial,
                    "final": pension_final,
                    "absolute_change": pension_change,
                    "relative_change": pension_relative_change,
                },
                source_filter=(
                    "aggregation_level=source_turn;"
                    "periods=2008-2011,2024-2025;topic_index=18"
                ),
                representative_document=representative(18),
                caveat=(
                    "The period-level increase is more robust than selecting "
                    "one exact annual maximum."
                ),
            ),
            make_row(
                claim_id="C23",
                slide_target="APPENDIX",
                claim_type="procedural_change",
                claim="Questions of privilege approximately double between the first and final periods.",
                source_path=PERIOD_PREVALENCE,
                raw_value=decimal(privilege_final),
                display_value=(
                    f"{percentage(privilege_initial)} -> "
                    f"{percentage(privilege_final)}"
                ),
                unit="prevalence_share",
                topic_id=23,
                aggregation="source_turn",
                year_or_period="2008-2011 vs 2024-2025",
                comparison_value={
                    "baseline": privilege_initial,
                    "final": privilege_final,
                    "absolute_change": privilege_change,
                    "relative_change": privilege_relative_change,
                },
                source_filter=(
                    "aggregation_level=source_turn;"
                    "periods=2008-2011,2024-2025;topic_index=23"
                ),
                representative_document=representative(23),
                caveat=(
                    "This is a parliamentary-procedure finding, not a public-"
                    "policy agenda."
                ),
            ),
        ]
    )

    # ------------------------------------------------------------------
    # Robustness claims
    # ------------------------------------------------------------------

    rows.extend(
        [
            make_row(
                claim_id="C24",
                slide_target="S10",
                claim_type="robustness",
                claim="The 2013 justice peak is preserved under document, source-turn and session weighting.",
                source_path=ANNUAL_PREVALENCE,
                raw_value=decimal(justice_2013),
                display_value=(
                    f"Document {percentage(justice_robustness['document'])}; "
                    f"source turn {percentage(justice_robustness['source_turn'])}; "
                    f"session {percentage(justice_robustness['session'])}"
                ),
                unit="prevalence_share",
                topic_id=4,
                aggregation="three_specifications",
                year_or_period="2013",
                comparison_value=justice_robustness,
                source_filter="year=2013;topic_index=4;all aggregations",
                caveat=(
                    "Magnitudes differ because the weighting units differ; "
                    "the peak-year interpretation is unchanged."
                ),
            ),
            make_row(
                claim_id="C25",
                slide_target="S10",
                claim_type="robustness",
                claim="The 2018 rights and reproductive-health peak is preserved under all three weighting schemes.",
                source_path=ANNUAL_PREVALENCE,
                raw_value=decimal(rights_2018),
                display_value=(
                    f"Document {percentage(rights_robustness['document'])}; "
                    f"source turn {percentage(rights_robustness['source_turn'])}; "
                    f"session {percentage(rights_robustness['session'])}"
                ),
                unit="prevalence_share",
                topic_id=9,
                aggregation="three_specifications",
                year_or_period="2018",
                comparison_value=rights_robustness,
                source_filter="year=2018;topic_index=9;all aggregations",
            ),
            make_row(
                claim_id="C26",
                slide_target="S10",
                claim_type="robustness",
                claim="The 2021 budget peak is preserved under all three weighting schemes.",
                source_path=ANNUAL_PREVALENCE,
                raw_value=decimal(budget_2021),
                display_value=(
                    f"Document {percentage(budget_robustness['document'])}; "
                    f"source turn {percentage(budget_robustness['source_turn'])}; "
                    f"session {percentage(budget_robustness['session'])}"
                ),
                unit="prevalence_share",
                topic_id=17,
                aggregation="three_specifications",
                year_or_period="2021",
                comparison_value=budget_robustness,
                source_filter="year=2021;topic_index=17;all aggregations",
                caveat=(
                    "Only eight modelled sessions are present in 2021. "
                    "The peak remains consistent."
                ),
            ),
            make_row(
                claim_id="C27",
                slide_target="S10",
                claim_type="robustness",
                claim="The 2021 tax peak is preserved under all three weighting schemes.",
                source_path=ANNUAL_PREVALENCE,
                raw_value=decimal(taxes_2021),
                display_value=(
                    f"Document {percentage(taxes_robustness['document'])}; "
                    f"source turn {percentage(taxes_robustness['source_turn'])}; "
                    f"session {percentage(taxes_robustness['session'])}"
                ),
                unit="prevalence_share",
                topic_id=20,
                aggregation="three_specifications",
                year_or_period="2021",
                comparison_value=taxes_robustness,
                source_filter="year=2021;topic_index=20;all aggregations",
                caveat=(
                    "Only eight modelled sessions are present in 2021. "
                    "The peak remains consistent."
                ),
            ),
        ]
    )

    rows.append(
        make_row(
            claim_id="C28",
            slide_target="S10",
            claim_type="model_comparison",
            claim="Full-context BERTopic benchmark result.",
            source_path=None,
            unit="pending_benchmark",
            source_filter="pending teammate review bundle",
            caveat=(
                "This row may affect only the model-comparison section. "
                "It may not reopen the selected NMF model or temporal findings."
            ),
            status="PENDING_BERTOPIC",
        )
    )

    claim_ids = [row["claim_id"] for row in rows]
    duplicates = [
        claim_id
        for claim_id, count in Counter(claim_ids).items()
        if count > 1
    ]

    if duplicates:
        raise AssertionError(f"Duplicate claim IDs: {duplicates}")

    if len(rows) != 28:
        raise AssertionError(
            f"Expected 28 ledger rows, generated {len(rows)}"
        )

    statuses = Counter(row["status"] for row in rows)

    if statuses != Counter({"LOCKED": 27, "PENDING_BERTOPIC": 1}):
        raise AssertionError(
            f"Unexpected status distribution: {statuses}"
        )

    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    temporary_path = OUTPUT_PATH.with_suffix(".csv.part")

    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=FIELDNAMES,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    temporary_path.replace(OUTPUT_PATH)

    print("Final evidence ledger generated successfully.")
    print(f"Output: {relative_path(OUTPUT_PATH)}")
    print(f"Rows: {len(rows)}")
    print(f"Locked claims: {statuses['LOCKED']}")
    print(
        "Pending BERTopic claims: "
        f"{statuses['PENDING_BERTOPIC']}"
    )
    print(
        "Maximum grid prevalence difference: "
        f"{maximum_grid_difference:.10f}"
    )
    print(f"SHA-256: {sha256_file(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
