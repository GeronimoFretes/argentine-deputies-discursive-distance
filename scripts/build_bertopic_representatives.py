from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

EXPECTED_DOCUMENT_COUNT = 75_121
EXPECTED_EMBEDDING_DIM = 1_024
DEFAULT_TOP_N = 5
EXCERPT_CHARS = 700


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate BERTopic representative documents using cosine "
            "similarity to each topic's mean BGE-M3 embedding centroid."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to the parent of scripts/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "BERTopic output directory. Defaults to "
            "<root>/outputs/bertopic_full_context_v1."
        ),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help=(
            "Exact cleaned corpus JSONL. Defaults to "
            "<root>/tmp/bertopic_handoff/cleaned_primary_documents.jsonl."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Representative documents per non-outlier topic.",
    )
    return parser.parse_args()


def sha256_bytes(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text(value: str) -> str:
    return " ".join(value.replace("\x00", " ").split())


def bounded_excerpt(value: str, limit: int = EXCERPT_CHARS) -> str:
    compact = normalize_text(value)
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "…"


def read_document_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Document-ID file not found: {path}")

    document_ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    if len(document_ids) != EXPECTED_DOCUMENT_COUNT:
        raise AssertionError(
            f"Expected {EXPECTED_DOCUMENT_COUNT} document IDs, "
            f"found {len(document_ids)}."
        )

    if len(document_ids) != len(set(document_ids)):
        raise AssertionError("Duplicate document IDs found.")

    return document_ids


def read_assignments(
    path: Path,
    expected_ids: list[str],
) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Assignment file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        rows = list(reader)

    required_columns = {
        "document_id",
        "topic_id",
        "source_record_id",
        "turn_index",
        "chunk_index",
        "year",
        "session_category",
        "speaker_family",
        "word_count",
    }
    missing_columns = required_columns - set(reader.fieldnames or [])

    if missing_columns:
        raise ValueError(
            f"{path.name} is missing columns: {sorted(missing_columns)}"
        )

    if len(rows) != len(expected_ids):
        raise AssertionError(
            f"{path.name}: expected {len(expected_ids)} rows, "
            f"found {len(rows)}."
        )

    actual_ids = [row["document_id"] for row in rows]
    if actual_ids != expected_ids:
        raise AssertionError(
            f"{path.name}: document order does not match document_ids.txt."
        )

    return rows


def load_embeddings(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Embeddings file not found: {path}")

    embeddings = np.load(path, mmap_mode="r")

    if embeddings.shape != (
        EXPECTED_DOCUMENT_COUNT,
        EXPECTED_EMBEDDING_DIM,
    ):
        raise AssertionError(
            "Unexpected embedding shape: "
            f"{embeddings.shape}; expected "
            f"({EXPECTED_DOCUMENT_COUNT}, {EXPECTED_EMBEDDING_DIM})."
        )

    if embeddings.dtype != np.float32:
        raise AssertionError(
            f"Expected float32 embeddings, found {embeddings.dtype}."
        )

    sample_indices = np.linspace(
        0,
        len(embeddings) - 1,
        num=min(2_000, len(embeddings)),
        dtype=int,
    )
    sample = np.asarray(embeddings[sample_indices], dtype=np.float32)

    if not np.isfinite(sample).all():
        raise AssertionError("NaN or infinite values found in embeddings.")

    sample_norms = np.linalg.norm(sample, axis=1)
    if not np.allclose(sample_norms, 1.0, atol=1e-3):
        raise AssertionError(
            "Embeddings are not normalized within tolerance."
        )

    return embeddings


def load_corpus_records(
    path: Path,
    expected_ids: set[str],
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Corpus JSONL not found: {path}")

    records: dict[str, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8-sig") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from error

            if not isinstance(payload, dict):
                raise TypeError(
                    f"Expected object at {path}:{line_number}"
                )

            document_id = payload.get("document_id")
            if document_id in expected_ids:
                if document_id in records:
                    raise AssertionError(
                        f"Duplicate corpus document ID: {document_id}"
                    )
                records[str(document_id)] = payload

    missing_ids = expected_ids - set(records)
    if missing_ids:
        sample = sorted(missing_ids)[:10]
        raise AssertionError(
            f"Corpus is missing {len(missing_ids)} modelled documents. "
            f"First IDs: {sample}"
        )

    return records


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def source_turn_key(
    assignment: dict[str, str],
    corpus_record: dict[str, Any],
) -> tuple[str, str]:
    source_record_id = (
        assignment.get("source_record_id")
        or str(corpus_record.get("source_record_id", ""))
    )
    turn_index = (
        assignment.get("turn_index")
        or str(corpus_record.get("turn_index", ""))
    )
    return source_record_id, turn_index


def choose_representatives(
    *,
    solution_name: str,
    assignments: list[dict[str, str]],
    embeddings: np.ndarray,
    corpus_records: dict[str, dict[str, Any]],
    top_n: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    indices_by_topic: dict[int, list[int]] = defaultdict(list)

    for index, row in enumerate(assignments):
        topic_id = int(row["topic_id"])
        if topic_id != -1:
            indices_by_topic[topic_id].append(index)

    output_rows: list[dict[str, Any]] = []
    topic_summaries: list[dict[str, Any]] = []

    for topic_id in sorted(indices_by_topic):
        topic_indices = np.asarray(
            indices_by_topic[topic_id],
            dtype=np.int64,
        )

        topic_embeddings = np.asarray(
            embeddings[topic_indices],
            dtype=np.float32,
        )

        centroid_raw = topic_embeddings.mean(axis=0)
        centroid_norm = float(np.linalg.norm(centroid_raw))

        if not np.isfinite(centroid_norm) or centroid_norm <= 0:
            raise AssertionError(
                f"{solution_name} topic {topic_id} has invalid centroid norm."
            )

        centroid = centroid_raw / centroid_norm
        similarities = topic_embeddings @ centroid

        if not np.isfinite(similarities).all():
            raise AssertionError(
                f"{solution_name} topic {topic_id} produced invalid scores."
            )

        order = np.argsort(-similarities, kind="stable")

        selected_local_positions: list[int] = []
        seen_source_turns: set[tuple[str, str]] = set()

        # First pass: prefer distinct source turns.
        for local_position in order:
            global_index = int(topic_indices[int(local_position)])
            assignment = assignments[global_index]
            document_id = assignment["document_id"]
            corpus_record = corpus_records[document_id]
            key = source_turn_key(assignment, corpus_record)

            if key in seen_source_turns:
                continue

            selected_local_positions.append(int(local_position))
            seen_source_turns.add(key)

            if len(selected_local_positions) == top_n:
                break

        # Fallback: fill remaining slots with the next-best documents.
        if len(selected_local_positions) < top_n:
            selected_set = set(selected_local_positions)
            for local_position in order:
                local_position_int = int(local_position)
                if local_position_int in selected_set:
                    continue

                selected_local_positions.append(local_position_int)
                selected_set.add(local_position_int)

                if len(selected_local_positions) == top_n:
                    break

        distinct_turn_count = len(
            {
                source_turn_key(
                    assignments[int(global_index)],
                    corpus_records[
                        assignments[int(global_index)]["document_id"]
                    ],
                )
                for global_index in topic_indices
            }
        )

        for rank, local_position in enumerate(
            selected_local_positions,
            start=1,
        ):
            global_index = int(topic_indices[local_position])
            assignment = assignments[global_index]
            document_id = assignment["document_id"]
            record = corpus_records[document_id]

            modeling_text = str(record.get("modeling_text", ""))
            cleaned_text = str(record.get("cleaned_text", ""))

            if not modeling_text:
                raise AssertionError(
                    f"Empty modeling_text for document {document_id}."
                )

            source_record_id = (
                assignment.get("source_record_id")
                or str(record.get("source_record_id", ""))
            )
            turn_index = (
                assignment.get("turn_index")
                or str(record.get("turn_index", ""))
            )

            output_rows.append(
                {
                    "solution": solution_name,
                    "topic_id": topic_id,
                    "rank": rank,
                    "document_id": document_id,
                    "cosine_similarity_to_centroid": round(
                        float(similarities[local_position]),
                        8,
                    ),
                    "topic_document_count": int(len(topic_indices)),
                    "distinct_source_turns_available": int(
                        distinct_turn_count
                    ),
                    "centroid_norm_before_normalization": round(
                        centroid_norm,
                        8,
                    ),
                    "source_record_id": source_record_id,
                    "turn_index": safe_int(
                        turn_index,
                        default=safe_int(record.get("turn_index")),
                    ),
                    "chunk_index": safe_int(
                        assignment.get("chunk_index"),
                        default=safe_int(record.get("chunk_index")),
                    ),
                    "year": safe_int(
                        assignment.get("year"),
                        default=safe_int(record.get("year")),
                    ),
                    "temporal_period": str(
                        record.get("temporal_period", "")
                    ),
                    "session_category": (
                        assignment.get("session_category")
                        or str(record.get("session_category", ""))
                    ),
                    "speaker_family": (
                        assignment.get("speaker_family")
                        or str(record.get("speaker_family", ""))
                    ),
                    "word_count": safe_int(
                        assignment.get("word_count"),
                        default=safe_int(record.get("word_count")),
                    ),
                    "modeling_text_excerpt": bounded_excerpt(
                        modeling_text
                    ),
                    "cleaned_text_excerpt": bounded_excerpt(
                        cleaned_text
                    ),
                    "full_modeling_text_sha256": sha256_bytes(
                        modeling_text
                    ),
                }
            )

        topic_summaries.append(
            {
                "topic_id": topic_id,
                "document_count": int(len(topic_indices)),
                "distinct_source_turn_count": int(distinct_turn_count),
                "representatives_written": int(
                    len(selected_local_positions)
                ),
                "centroid_norm_before_normalization": round(
                    centroid_norm,
                    8,
                ),
                "best_similarity": round(
                    float(similarities[order[0]]),
                    8,
                ),
                "worst_selected_similarity": round(
                    float(
                        similarities[selected_local_positions[-1]]
                    ),
                    8,
                ),
            }
        )

    summary = {
        "solution": solution_name,
        "non_outlier_topic_count": len(indices_by_topic),
        "representatives_per_topic_requested": top_n,
        "representative_rows_written": len(output_rows),
        "topics": topic_summaries,
    }

    return output_rows, summary


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")

    with temporary.open("w", encoding="utf-8", newline="\n") as output_file:
        for row in rows:
            output_file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    temporary.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / "outputs" / "bertopic_full_context_v1"
    )
    corpus_path = (
        args.corpus.resolve()
        if args.corpus is not None
        else root
        / "tmp"
        / "bertopic_handoff"
        / "cleaned_primary_documents.jsonl"
    )

    if args.top_n < 1:
        raise ValueError("--top-n must be at least 1.")

    document_ids = read_document_ids(
        output_dir / "document_ids.txt"
    )
    embeddings = load_embeddings(
        output_dir / "embeddings.npy"
    )

    selected_assignments = read_assignments(
        output_dir / "selected_native_document_assignments.csv",
        document_ids,
    )
    reduced_assignments = read_assignments(
        output_dir / "reduced_document_assignments.csv",
        document_ids,
    )

    corpus_records = load_corpus_records(
        corpus_path,
        set(document_ids),
    )

    selected_rows, selected_summary = choose_representatives(
        solution_name="selected_native",
        assignments=selected_assignments,
        embeddings=embeddings,
        corpus_records=corpus_records,
        top_n=args.top_n,
    )
    reduced_rows, reduced_summary = choose_representatives(
        solution_name="reduced",
        assignments=reduced_assignments,
        embeddings=embeddings,
        corpus_records=corpus_records,
        top_n=args.top_n,
    )

    selected_path = (
        output_dir
        / "corrected_selected_native_representative_documents.jsonl"
    )
    reduced_path = (
        output_dir
        / "corrected_reduced_representative_documents.jsonl"
    )
    summary_path = (
        output_dir
        / "corrected_representative_documents_manifest.json"
    )

    write_jsonl(selected_path, selected_rows)
    write_jsonl(reduced_path, reduced_rows)
    write_json(
        summary_path,
        {
            "method": (
                "Normalized mean BGE-M3 embedding centroid per non-outlier "
                "topic; cosine similarity; distinct source turns preferred."
            ),
            "embedding_file": "embeddings.npy",
            "document_count": len(document_ids),
            "embedding_shape": list(embeddings.shape),
            "top_n": args.top_n,
            "selected_native": selected_summary,
            "reduced": reduced_summary,
        },
    )

    expected_selected_rows = (
        selected_summary["non_outlier_topic_count"] * args.top_n
    )
    expected_reduced_rows = (
        reduced_summary["non_outlier_topic_count"] * args.top_n
    )

    if len(selected_rows) != expected_selected_rows:
        raise AssertionError(
            "Selected-native representative count mismatch: "
            f"{len(selected_rows)} != {expected_selected_rows}"
        )
    if len(reduced_rows) != expected_reduced_rows:
        raise AssertionError(
            "Reduced representative count mismatch: "
            f"{len(reduced_rows)} != {expected_reduced_rows}"
        )

    print("Corrected BERTopic representatives generated successfully.")
    print(f"Selected-native topics: {selected_summary['non_outlier_topic_count']}")
    print(f"Selected-native rows: {len(selected_rows)}")
    print(f"Reduced topics: {reduced_summary['non_outlier_topic_count']}")
    print(f"Reduced rows: {len(reduced_rows)}")
    print(f"Selected output: {selected_path}")
    print(f"Reduced output: {reduced_path}")
    print(f"Manifest: {summary_path}")


if __name__ == "__main__":
    main()
