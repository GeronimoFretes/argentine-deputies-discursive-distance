
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


EXPECTED_DOCUMENTS = 75_121
EXPECTED_NATIVE_TOPICS = 124
EXPECTED_REDUCED_TOPICS = 23
EXPECTED_OUTLIERS = 29_169
EXPECTED_SELECTED = "primary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize the BERTopic integration: lock claim C28, write the "
            "structural model comparison, and generate Figure 9."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to the parent of scripts/.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_compact(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    payload = json.loads(path.read_text(encoding="utf-8-sig"))

    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")

    return payload


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_text(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def validate_assignments(
    frame: pd.DataFrame,
    *,
    expected_topics: int,
    label: str,
) -> dict[str, Any]:
    required_columns = {
        "document_id",
        "topic_id",
    }
    missing = required_columns - set(frame.columns)

    if missing:
        raise ValueError(
            f"{label} assignments are missing columns: {sorted(missing)}"
        )

    if len(frame) != EXPECTED_DOCUMENTS:
        raise AssertionError(
            f"{label}: expected {EXPECTED_DOCUMENTS} rows, found {len(frame)}"
        )

    if frame["document_id"].duplicated().any():
        raise AssertionError(f"{label}: duplicate document IDs found.")

    topic_ids = frame["topic_id"].astype(int)
    outlier_count = int((topic_ids == -1).sum())
    assigned_count = int((topic_ids != -1).sum())
    topic_count = int(topic_ids[topic_ids != -1].nunique())

    if topic_count != expected_topics:
        raise AssertionError(
            f"{label}: expected {expected_topics} non-outlier topics, "
            f"found {topic_count}"
        )

    if outlier_count != EXPECTED_OUTLIERS:
        raise AssertionError(
            f"{label}: expected {EXPECTED_OUTLIERS} outliers, "
            f"found {outlier_count}"
        )

    sizes = (
        topic_ids[topic_ids != -1]
        .value_counts()
        .sort_values(ascending=False)
    )
    top5_share = float(sizes.head(5).sum() / assigned_count)

    return {
        "documents": int(len(frame)),
        "assigned_documents": assigned_count,
        "outlier_count": outlier_count,
        "coverage_fraction": assigned_count / len(frame),
        "outlier_fraction": outlier_count / len(frame),
        "topic_count": topic_count,
        "top5_share_assigned": top5_share,
        "largest_topic_documents": int(sizes.iloc[0]),
        "median_topic_documents": float(sizes.median()),
    }


def update_evidence_ledger(
    *,
    ledger_path: Path,
    source_path: Path,
    native: dict[str, Any],
    reduced: dict[str, Any],
) -> None:
    if not ledger_path.is_file():
        raise FileNotFoundError(ledger_path)

    with ledger_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if fieldnames is None:
        raise ValueError("Evidence ledger has no header.")

    required_fields = {
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
    }
    missing_fields = required_fields - set(fieldnames)

    if missing_fields:
        raise ValueError(
            "Evidence ledger is missing fields: "
            f"{sorted(missing_fields)}"
        )

    c28_rows = [row for row in rows if row["claim_id"] == "C28"]

    if len(c28_rows) != 1:
        raise AssertionError(
            f"Expected exactly one C28 row, found {len(c28_rows)}"
        )

    coverage = native["coverage_fraction"]
    outlier = native["outlier_fraction"]

    replacement = {
        "claim_id": "C28",
        "slide_target": "S10",
        "claim_type": "model_comparison",
        "claim": (
            "Full-context BGE-M3 improves BERTopic coverage relative to the "
            "earlier truncated MiniLM benchmark, but 38.8% of documents "
            "remain outliers, so BERTopic remains exploratory."
        ),
        "topic_id": "",
        "topic_label": "",
        "aggregation": "document_assignment",
        "year_or_period": "2008-2025",
        "raw_value": f"{outlier:.10f}",
        "display_value": (
            f"{coverage * 100:.1f}% coverage; "
            f"{outlier * 100:.1f}% outliers; "
            f"{native['topic_count']} native topics -> "
            f"{reduced['topic_count']} reduced topics"
        ),
        "unit": "coverage_and_cluster_structure",
        "comparison_value": json_compact(
            {
                "documents": native["documents"],
                "native": native,
                "reduced": reduced,
                "decision": "exploratory_supporting_benchmark",
                "primary_model": "NMF P1 K=24",
            }
        ),
        "source_file": source_path.as_posix(),
        "source_sha256": sha256_file(source_path),
        "source_filter": (
            "topic_id=-1 for outliers; non-outlier unique topic IDs for "
            "topic count; all 75,121 rows"
        ),
        "representative_document_id": "",
        "representative_document_excerpt": "",
        "caveat": (
            "The improvement over the earlier MiniLM benchmark cannot be "
            "attributed exclusively to context length because both the "
            "embedding model and available context changed. BERTopic common "
            "lexical metrics are not compared directly with NMF because the "
            "existing implementations use different definitions."
        ),
        "status": "LOCKED",
    }

    updated_rows = [
        replacement if row["claim_id"] == "C28" else row
        for row in rows
    ]

    claim_ids = [row["claim_id"] for row in updated_rows]
    duplicates = [
        claim_id
        for claim_id, count in Counter(claim_ids).items()
        if count > 1
    ]

    if duplicates:
        raise AssertionError(f"Duplicate claim IDs: {duplicates}")

    statuses = Counter(row["status"] for row in updated_rows)

    if statuses != Counter({"LOCKED": 28}):
        raise AssertionError(
            f"Unexpected final ledger status distribution: {statuses}"
        )

    temporary = ledger_path.with_suffix(".csv.part")

    with temporary.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(updated_rows)

    temporary.replace(ledger_path)


def build_comparison_rows(
    native: dict[str, Any],
    reduced: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "model": "NMF P1 K=24",
            "documents": EXPECTED_DOCUMENTS,
            "coverage_fraction": 1.0,
            "coverage_percent": 100.0,
            "outlier_fraction": 0.0,
            "outlier_percent": 0.0,
            "topic_count": 24,
            "assignment_type": "Soft mixture over all 24 topics",
            "longitudinal_use": "Primary",
            "project_role": "Primary topic map and temporal prevalence",
        },
        {
            "model": "BERTopic BGE-M3 native",
            "documents": native["documents"],
            "coverage_fraction": native["coverage_fraction"],
            "coverage_percent": native["coverage_fraction"] * 100,
            "outlier_fraction": native["outlier_fraction"],
            "outlier_percent": native["outlier_fraction"] * 100,
            "topic_count": native["topic_count"],
            "assignment_type": "One cluster or outlier",
            "longitudinal_use": "Exploratory only",
            "project_role": "Semantic clustering benchmark",
        },
        {
            "model": "BERTopic BGE-M3 reduced",
            "documents": reduced["documents"],
            "coverage_fraction": reduced["coverage_fraction"],
            "coverage_percent": reduced["coverage_fraction"] * 100,
            "outlier_fraction": reduced["outlier_fraction"],
            "outlier_percent": reduced["outlier_fraction"] * 100,
            "topic_count": reduced["topic_count"],
            "assignment_type": "One reduced cluster or outlier",
            "longitudinal_use": "Not used",
            "project_role": "Interpretability diagnostic",
        },
    ]


def write_comparison_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model",
        "documents",
        "coverage_fraction",
        "coverage_percent",
        "outlier_fraction",
        "outlier_percent",
        "topic_count",
        "assignment_type",
        "longitudinal_use",
        "project_role",
    ]

    temporary = path.with_suffix(".csv.part")

    with temporary.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(path)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 18,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 11,
            "figure.titlesize": 18,
        }
    )


def build_figure(
    *,
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    configure_matplotlib()

    labels = [
        "NMF P1 K=24",
        "BERTopic nativo",
        "BERTopic reducido",
    ]
    coverage = [float(row["coverage_percent"]) for row in rows]
    outliers = [float(row["outlier_percent"]) for row in rows]
    topics = [int(row["topic_count"]) for row in rows]

    fig, ax = plt.subplots(
        figsize=(12, 6.75),
        constrained_layout=True,
    )

    y_positions = list(range(len(labels)))
    bars = ax.barh(
        y_positions,
        coverage,
        height=0.55,
    )

    ax.set_yticks(y_positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel("Cobertura documental (%)")
    ax.set_title(
        "NMF ofrece cobertura completa para el análisis longitudinal",
        pad=18,
    )
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right", "left"]].set_visible(False)

    for index, (bar, coverage_value, outlier_value, topic_count) in enumerate(
        zip(bars, coverage, outliers, topics, strict=True)
    ):
        ax.text(
            coverage_value + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{coverage_value:.1f}%",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

        detail = (
            f"{topic_count} temas · {outlier_value:.1f}% outliers"
            if index > 0
            else "24 temas · 0% outliers · mezcla temática completa"
        )

        ax.text(
            2,
            bar.get_y() + bar.get_height() / 2,
            detail,
            va="center",
            fontsize=10.5,
        )

    fig.text(
        0.5,
        0.035,
        (
            "BGE-M3 mejora el benchmark semántico, pero la cobertura "
            "incompleta impide reemplazar a NMF en el análisis temporal."
        ),
        ha="center",
        va="bottom",
        fontsize=11,
    )
    fig.text(
        0.01,
        0.01,
        (
            "Fuente: asignaciones BERTopic full-context y modelo NMF P1 K=24. "
            "Las métricas léxicas no se comparan porque usan definiciones "
            "distintas."
        ),
        ha="left",
        va="bottom",
        fontsize=8,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Figure was not created correctly: {path}")


def update_figure_manifests(
    *,
    figure_dir: Path,
    figure_filename: str,
) -> None:
    csv_path = figure_dir / "figure_manifest.csv"
    json_path = figure_dir / "figure_manifest.json"

    new_entry = {
        "file": figure_filename,
        "figure": "Comparación estructural NMF-BERTopic",
        "claims": ["C28"],
    }

    csv_rows: list[dict[str, str]] = []

    if csv_path.is_file():
        with csv_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as input_file:
            reader = csv.DictReader(input_file)
            csv_rows = [
                row
                for row in reader
                if row.get("file") != figure_filename
            ]

    csv_rows.append(
        {
            "file": new_entry["file"],
            "figure": new_entry["figure"],
            "claims": ",".join(new_entry["claims"]),
        }
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["file", "figure", "claims"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    json_rows: list[dict[str, Any]] = []

    if json_path.is_file():
        payload = json.loads(
            json_path.read_text(encoding="utf-8-sig")
        )
        if isinstance(payload, list):
            json_rows = [
                row
                for row in payload
                if row.get("file") != figure_filename
            ]

    json_rows.append(new_entry)
    atomic_write_json(json_path, json_rows)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    output_dir = root / "outputs" / "bertopic_full_context_v1"
    review_dir = root / "docs" / "bertopic_review"
    figure_dir = root / "docs" / "figures" / "final_presentation"

    ledger_path = root / "docs" / "final_evidence_ledger.csv"
    selected_path = (
        output_dir / "selected_native_document_assignments.csv"
    )
    reduced_path = output_dir / "reduced_document_assignments.csv"
    decision_path = output_dir / "selection_decision.json"
    corrected_manifest_path = (
        output_dir
        / "corrected_representative_documents_manifest.json"
    )

    required = [
        ledger_path,
        selected_path,
        reduced_path,
        decision_path,
        corrected_manifest_path,
    ]
    missing = [path for path in required if not path.is_file()]

    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Missing required BERTopic integration inputs:\n"
            f"{formatted}"
        )

    decision = read_json(decision_path)

    if decision.get("selected") != EXPECTED_SELECTED:
        raise AssertionError(
            "The selected BERTopic configuration is not primary."
        )

    selected_frame = read_csv(selected_path)
    reduced_frame = read_csv(reduced_path)

    selected_ids = selected_frame["document_id"].tolist()
    reduced_ids = reduced_frame["document_id"].tolist()

    if selected_ids != reduced_ids:
        raise AssertionError(
            "Native and reduced assignments are not aligned."
        )

    native = validate_assignments(
        selected_frame,
        expected_topics=EXPECTED_NATIVE_TOPICS,
        label="Selected-native",
    )
    reduced = validate_assignments(
        reduced_frame,
        expected_topics=EXPECTED_REDUCED_TOPICS,
        label="Reduced",
    )

    if abs(native["outlier_fraction"] - reduced["outlier_fraction"]) > 1e-12:
        raise AssertionError(
            "Native and reduced outlier fractions differ unexpectedly."
        )

    review_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    update_evidence_ledger(
        ledger_path=ledger_path,
        source_path=Path(
            "outputs/bertopic_full_context_v1/"
            "selected_native_document_assignments.csv"
        ),
        native=native,
        reduced=reduced,
    )

    comparison_rows = build_comparison_rows(native, reduced)

    comparison_csv = review_dir / "model_comparison_structural.csv"
    comparison_json = review_dir / "bertopic_integration_summary.json"

    write_comparison_csv(comparison_csv, comparison_rows)
    atomic_write_json(
        comparison_json,
        {
            "decision": (
                "BERTopic is retained as an exploratory supporting benchmark; "
                "NMF P1 K=24 remains the primary longitudinal model."
            ),
            "selected_configuration": decision,
            "native": native,
            "reduced": reduced,
            "comparison_rows": comparison_rows,
            "interpretation": (
                "Full-context BGE-M3 substantially improved native BERTopic "
                "coverage relative to the earlier truncated MiniLM benchmark. "
                "Since both the embedding model and available context changed, "
                "the improvement cannot be attributed exclusively to "
                "truncation. With 38.8% outliers, BERTopic remains exploratory."
            ),
            "metric_comparability_note": (
                "The current BERTopic lexical NPMI, exclusivity and redundancy "
                "implementations are not compared directly with NMF because "
                "their definitions differ."
            ),
        },
    )

    figure_path = figure_dir / "09_model_comparison.png"
    build_figure(
        path=figure_path,
        rows=comparison_rows,
    )
    update_figure_manifests(
        figure_dir=figure_dir,
        figure_filename=figure_path.name,
    )

    print("BERTopic integration finalized successfully.")
    print(
        f"Native: {native['topic_count']} topics, "
        f"{native['coverage_fraction'] * 100:.1f}% coverage, "
        f"{native['outlier_fraction'] * 100:.1f}% outliers"
    )
    print(
        f"Reduced: {reduced['topic_count']} topics, "
        f"{reduced['coverage_fraction'] * 100:.1f}% coverage, "
        f"{reduced['outlier_fraction'] * 100:.1f}% outliers, "
        f"top-5 share among assigned documents "
        f"{reduced['top5_share_assigned'] * 100:.1f}%"
    )
    print("Evidence ledger: 28 LOCKED claims")
    print(f"Comparison CSV: {comparison_csv}")
    print(f"Integration summary: {comparison_json}")
    print(f"Figure 9: {figure_path}")
    print("Figure manifests updated.")


if __name__ == "__main__":
    main()
