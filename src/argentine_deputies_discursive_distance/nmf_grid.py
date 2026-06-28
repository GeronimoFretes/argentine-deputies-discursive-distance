"""Preprocess the locked primary corpus and fit a bounded NMF grid."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import time
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import scipy.sparse as sp  # type: ignore[import-untyped]
from sklearn.decomposition import NMF  # type: ignore[import-untyped]
from sklearn.exceptions import ConvergenceWarning  # type: ignore[import-untyped]
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-untyped]

from . import corpus_profile
from .corpus_profile import CorpusProfileError, DocumentRecord
from .pdf_pipeline import sha256_file
from .topic_preprocessing import (
    CLEANING_VERSION,
    LEXICAL_TOKENIZER_VERSION,
    CleaningResult,
    StopwordSet,
    TopicPreprocessingError,
    bounded_excerpt,
    clean_natural_text,
    lexical_tokens,
    load_stopwords,
)

NMF_GRID_VERSION = "1"

DEFAULT_DOCUMENTS_PATH = corpus_profile.DEFAULT_DOCUMENTS_PATH
DEFAULT_EXPORT_MANIFEST_PATH = corpus_profile.DEFAULT_EXPORT_MANIFEST_PATH
DEFAULT_CORPUS_LOCK_PATH = corpus_profile.DEFAULT_CORPUS_LOCK_PATH
DEFAULT_PROFILE_CONFIG_PATH = corpus_profile.DEFAULT_CONFIG_PATH
DEFAULT_PROFILE_MANIFEST_PATH = Path(
    "data/qa/topic_modeling/corpus_profile_v1/profile_manifest.json"
)
DEFAULT_CONFIG_PATH = Path("config/topic_modeling/nmf_grid_v1.json")
DEFAULT_STOPWORDS_PATH = Path("config/topic_modeling/stopwords_es_p0_v1.txt")
DEFAULT_OUTPUT_DIR = Path("data/qa/topic_modeling/nmf_grid_v1")

RUN_MANIFEST_FILENAME = "run_manifest.json"
PREPROCESSING_SUMMARY_FILENAME = "preprocessing_summary.json"
PREPROCESSING_EXAMPLES_FILENAME = "preprocessing_examples.jsonl"
CLEANED_DOCUMENTS_FILENAME = "cleaned_primary_documents.jsonl"
VECTORIZER_SUMMARY_FILENAME = "vectorizer_summary.json"
VOCABULARY_FILENAME = "vocabulary.csv"
GRID_METRICS_FILENAME = "grid_metrics.csv"
GRID_REPORT_FILENAME = "grid_report.md"
VECTORIZER_ARTIFACT_FILENAME = "vectorizer.joblib"

CONFIG_FIELDS = frozenset(
    {
        "cleaned_excerpt_characters",
        "expected_primary_counts",
        "expected_profile_counts",
        "grid_version",
        "metrics_top_n",
        "nmf",
        "preprocessing_example_limit",
        "primary_session_categories",
        "representative_documents_per_topic",
        "stopword_variant",
        "tfidf",
        "top_terms_per_topic",
    }
)
EXPECTED_PROFILE_FIELDS = frozenset({"all_documents", "all_words"})
EXPECTED_PRIMARY_FIELDS = frozenset({"documents", "source_turns", "sessions", "words"})
TFIDF_FIELDS = frozenset(
    {
        "dtype",
        "lowercase",
        "max_df",
        "max_features",
        "min_df",
        "ngram_range",
        "norm",
        "smooth_idf",
        "strip_accents",
        "sublinear_tf",
    }
)
NMF_FIELDS = frozenset(
    {
        "alpha_H",
        "alpha_W",
        "beta_loss",
        "init",
        "k_values",
        "l1_ratio",
        "max_iter",
        "random_state",
        "solver",
        "tol",
    }
)


class NmfGridError(RuntimeError):
    """Raised when the NMF grid stage cannot complete safely."""


@dataclass(frozen=True, slots=True)
class ExpectedProfileCounts:
    """Profile-level totals expected by this grid configuration."""

    all_documents: int
    all_words: int

    def to_json(self) -> dict[str, int]:
        """Return JSON-serializable expected profile counts."""
        return {"all_documents": self.all_documents, "all_words": self.all_words}


@dataclass(frozen=True, slots=True)
class ExpectedPrimaryCounts:
    """Primary-universe totals expected by this grid configuration."""

    documents: int
    source_turns: int
    sessions: int
    words: int

    def to_json(self) -> dict[str, int]:
        """Return JSON-serializable expected primary counts."""
        return {
            "documents": self.documents,
            "sessions": self.sessions,
            "source_turns": self.source_turns,
            "words": self.words,
        }


@dataclass(frozen=True, slots=True)
class TfidfConfig:
    """Strict TF-IDF settings for the grid."""

    ngram_range: tuple[int, int]
    min_df: int
    max_df: float
    max_features: int
    sublinear_tf: bool
    smooth_idf: bool
    norm: str
    dtype: str
    lowercase: bool
    strip_accents: str | None

    def to_json(self) -> dict[str, Any]:
        """Return deterministic JSON settings."""
        return {
            "dtype": self.dtype,
            "lowercase": self.lowercase,
            "max_df": self.max_df,
            "max_features": self.max_features,
            "min_df": self.min_df,
            "ngram_range": list(self.ngram_range),
            "norm": self.norm,
            "smooth_idf": self.smooth_idf,
            "strip_accents": self.strip_accents,
            "sublinear_tf": self.sublinear_tf,
            "token_pattern": None,
            "tokenizer": LEXICAL_TOKENIZER_VERSION,
        }


@dataclass(frozen=True, slots=True)
class NmfConfig:
    """Strict NMF settings for the grid."""

    k_values: tuple[int, ...]
    solver: str
    beta_loss: str
    init: str
    random_state: int
    max_iter: int
    tol: float
    alpha_W: float
    alpha_H: float
    l1_ratio: float

    def to_json(self) -> dict[str, Any]:
        """Return deterministic JSON settings."""
        return {
            "alpha_H": self.alpha_H,
            "alpha_W": self.alpha_W,
            "beta_loss": self.beta_loss,
            "init": self.init,
            "k_values": list(self.k_values),
            "l1_ratio": self.l1_ratio,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "solver": self.solver,
            "tol": self.tol,
        }


@dataclass(frozen=True, slots=True)
class NmfGridConfig:
    """Strict configuration for the focused NMF grid stage."""

    grid_version: str
    primary_session_categories: tuple[str, ...]
    expected_profile_counts: ExpectedProfileCounts
    expected_primary_counts: ExpectedPrimaryCounts
    stopword_variant: str
    tfidf: TfidfConfig
    nmf: NmfConfig
    top_terms_per_topic: int
    metrics_top_n: int
    representative_documents_per_topic: int
    preprocessing_example_limit: int
    cleaned_excerpt_characters: int

    def to_json(self) -> dict[str, Any]:
        """Return deterministic JSON configuration."""
        return {
            "cleaned_excerpt_characters": self.cleaned_excerpt_characters,
            "expected_primary_counts": self.expected_primary_counts.to_json(),
            "expected_profile_counts": self.expected_profile_counts.to_json(),
            "grid_version": self.grid_version,
            "metrics_top_n": self.metrics_top_n,
            "nmf": self.nmf.to_json(),
            "preprocessing_example_limit": self.preprocessing_example_limit,
            "primary_session_categories": list(self.primary_session_categories),
            "representative_documents_per_topic": self.representative_documents_per_topic,
            "stopword_variant": self.stopword_variant,
            "tfidf": self.tfidf.to_json(),
            "top_terms_per_topic": self.top_terms_per_topic,
        }


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    """Metadata retained for representative-document output."""

    row_index: int
    document_id: str
    source_record_id: str
    turn_index: int
    chunk_index: int
    year: int
    temporal_period: str
    session_category: str
    speaker_family: str
    word_count: int
    source_turn_key: str
    cleaned_excerpt: str


@dataclass(frozen=True, slots=True)
class PreprocessingArtifacts:
    """Staged cleaned corpus metadata and diagnostics."""

    summary: dict[str, Any]
    examples: tuple[dict[str, Any], ...]
    primary_counts: dict[str, int]
    cleaned_documents_metadata: dict[str, Any]
    cleaned_documents_part_path: Path


@dataclass(frozen=True, slots=True)
class VectorizationResult:
    """Fitted TF-IDF matrix and reporting artifacts."""

    vectorizer: TfidfVectorizer
    matrix: Any
    document_metadata: tuple[DocumentMetadata, ...]
    summary: dict[str, Any]
    vocabulary_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class TopicModelResult:
    """Per-K model outputs and metrics before serialization."""

    k: int
    model: NMF
    topic_term_rows: tuple[dict[str, Any], ...]
    representative_rows: tuple[dict[str, Any], ...]
    metric_row: dict[str, Any]
    per_topic_metrics: tuple[dict[str, Any], ...]


def _json_text(payload: Mapping[str, Any]) -> str:
    """Return deterministic UTF-8 JSON text."""
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _json_compact(payload: object) -> str:
    """Return compact deterministic JSON for CSV cells."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    """Read a UTF-8 JSON object."""
    if not path.is_file():
        raise NmfGridError(f"{label} does not exist: {path}")

    try:
        payload: object = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise NmfGridError(f"Could not read {label}: {path}") from error

    if not isinstance(payload, dict):
        raise NmfGridError(f"Expected {label} to contain a JSON object: {path}")

    return {str(key): value for key, value in payload.items()}


def _safe_int(value: object, *, field_name: str) -> int:
    """Return a strict JSON integer, rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise NmfGridError(f"Invalid integer for {field_name}: {value!r}")

    return value


def _safe_float(value: object, *, field_name: str) -> float:
    """Return a strict JSON number, rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise NmfGridError(f"Invalid number for {field_name}: {value!r}")

    return float(value)


def _safe_bool(value: object, *, field_name: str) -> bool:
    """Return a strict JSON boolean."""
    if not isinstance(value, bool):
        raise NmfGridError(f"Invalid boolean for {field_name}: {value!r}")

    return value


def _required_string(payload: Mapping[str, Any], field_name: str) -> str:
    """Return a required nonempty string."""
    value = payload.get(field_name)

    if not isinstance(value, str) or not value:
        raise NmfGridError(f"Missing or invalid {field_name}.")

    return value


def _string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    """Return a nonempty tuple of strings."""
    if not isinstance(value, list) or not value:
        raise NmfGridError(f"Invalid string list for {field_name}.")

    values: list[str] = []

    for item in value:
        if not isinstance(item, str) or not item:
            raise NmfGridError(f"Invalid string in {field_name}: {item!r}")

        values.append(item)

    return tuple(values)


def _int_tuple(value: object, *, field_name: str) -> tuple[int, ...]:
    """Return a nonempty tuple of positive integers."""
    if not isinstance(value, list) or not value:
        raise NmfGridError(f"Invalid integer list for {field_name}.")

    values: list[int] = []

    for item in value:
        parsed = _safe_int(item, field_name=field_name)

        if parsed < 1:
            raise NmfGridError(f"{field_name} values must be positive: {parsed}")

        values.append(parsed)

    if len(set(values)) != len(values):
        raise NmfGridError(f"{field_name} values must be unique.")

    return tuple(values)


def load_config(path: Path) -> NmfGridConfig:
    """Load and strictly validate the versioned NMF grid configuration."""
    payload = _read_json_object(path, label="NMF grid configuration")
    missing = CONFIG_FIELDS - set(payload)
    unexpected = set(payload) - CONFIG_FIELDS

    if missing:
        raise NmfGridError(f"NMF grid configuration is missing fields: {sorted(missing)}")

    if unexpected:
        raise NmfGridError(f"NMF grid configuration has unsupported fields: {sorted(unexpected)}")

    grid_version = _required_string(payload, "grid_version")

    if grid_version != NMF_GRID_VERSION:
        raise NmfGridError(f"Unsupported grid_version: {grid_version}")

    tfidf_payload = _object_field(payload, "tfidf")
    nmf_payload = _object_field(payload, "nmf")
    expected_profile_payload = _object_field(payload, "expected_profile_counts")
    expected_primary_payload = _object_field(payload, "expected_primary_counts")
    _validate_exact_fields(tfidf_payload, expected=TFIDF_FIELDS, label="tfidf")
    _validate_exact_fields(nmf_payload, expected=NMF_FIELDS, label="nmf")
    _validate_exact_fields(
        expected_profile_payload,
        expected=EXPECTED_PROFILE_FIELDS,
        label="expected_profile_counts",
    )
    _validate_exact_fields(
        expected_primary_payload,
        expected=EXPECTED_PRIMARY_FIELDS,
        label="expected_primary_counts",
    )

    ngram_range = _ngram_range(tfidf_payload.get("ngram_range"))
    dtype = _required_string(tfidf_payload, "dtype")

    if dtype != "float32":
        raise NmfGridError("TF-IDF dtype must be float32 for this grid version.")

    strip_accents_value = tfidf_payload.get("strip_accents")

    if strip_accents_value is not None and not isinstance(strip_accents_value, str):
        raise NmfGridError("tfidf.strip_accents must be null or a string.")

    config = NmfGridConfig(
        grid_version=grid_version,
        primary_session_categories=_string_tuple(
            payload["primary_session_categories"],
            field_name="primary_session_categories",
        ),
        expected_profile_counts=ExpectedProfileCounts(
            all_documents=_safe_int(
                expected_profile_payload.get("all_documents"),
                field_name="expected_profile_counts.all_documents",
            ),
            all_words=_safe_int(
                expected_profile_payload.get("all_words"),
                field_name="expected_profile_counts.all_words",
            ),
        ),
        expected_primary_counts=ExpectedPrimaryCounts(
            documents=_safe_int(
                expected_primary_payload.get("documents"),
                field_name="expected_primary_counts.documents",
            ),
            source_turns=_safe_int(
                expected_primary_payload.get("source_turns"),
                field_name="expected_primary_counts.source_turns",
            ),
            sessions=_safe_int(
                expected_primary_payload.get("sessions"),
                field_name="expected_primary_counts.sessions",
            ),
            words=_safe_int(
                expected_primary_payload.get("words"),
                field_name="expected_primary_counts.words",
            ),
        ),
        stopword_variant=_required_string(payload, "stopword_variant"),
        tfidf=TfidfConfig(
            ngram_range=ngram_range,
            min_df=_safe_int(tfidf_payload.get("min_df"), field_name="tfidf.min_df"),
            max_df=_safe_float(tfidf_payload.get("max_df"), field_name="tfidf.max_df"),
            max_features=_safe_int(
                tfidf_payload.get("max_features"),
                field_name="tfidf.max_features",
            ),
            sublinear_tf=_safe_bool(
                tfidf_payload.get("sublinear_tf"),
                field_name="tfidf.sublinear_tf",
            ),
            smooth_idf=_safe_bool(
                tfidf_payload.get("smooth_idf"),
                field_name="tfidf.smooth_idf",
            ),
            norm=_required_string(tfidf_payload, "norm"),
            dtype=dtype,
            lowercase=_safe_bool(tfidf_payload.get("lowercase"), field_name="tfidf.lowercase"),
            strip_accents=strip_accents_value,
        ),
        nmf=NmfConfig(
            k_values=_int_tuple(nmf_payload.get("k_values"), field_name="nmf.k_values"),
            solver=_required_string(nmf_payload, "solver"),
            beta_loss=_required_string(nmf_payload, "beta_loss"),
            init=_required_string(nmf_payload, "init"),
            random_state=_safe_int(nmf_payload.get("random_state"), field_name="nmf.random_state"),
            max_iter=_safe_int(nmf_payload.get("max_iter"), field_name="nmf.max_iter"),
            tol=_safe_float(nmf_payload.get("tol"), field_name="nmf.tol"),
            alpha_W=_safe_float(nmf_payload.get("alpha_W"), field_name="nmf.alpha_W"),
            alpha_H=_safe_float(nmf_payload.get("alpha_H"), field_name="nmf.alpha_H"),
            l1_ratio=_safe_float(nmf_payload.get("l1_ratio"), field_name="nmf.l1_ratio"),
        ),
        top_terms_per_topic=_safe_int(
            payload["top_terms_per_topic"],
            field_name="top_terms_per_topic",
        ),
        metrics_top_n=_safe_int(payload["metrics_top_n"], field_name="metrics_top_n"),
        representative_documents_per_topic=_safe_int(
            payload["representative_documents_per_topic"],
            field_name="representative_documents_per_topic",
        ),
        preprocessing_example_limit=_safe_int(
            payload["preprocessing_example_limit"],
            field_name="preprocessing_example_limit",
        ),
        cleaned_excerpt_characters=_safe_int(
            payload["cleaned_excerpt_characters"],
            field_name="cleaned_excerpt_characters",
        ),
    )
    _validate_config_values(config)
    return config


def _object_field(payload: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    """Return a required object field."""
    value = payload.get(field_name)

    if not isinstance(value, dict):
        raise NmfGridError(f"Missing or invalid {field_name} object.")

    return {str(key): item for key, item in value.items()}


def _validate_exact_fields(
    payload: Mapping[str, Any],
    *,
    expected: frozenset[str],
    label: str,
) -> None:
    """Validate strict object fields."""
    missing = expected - set(payload)
    unexpected = set(payload) - expected

    if missing:
        raise NmfGridError(f"{label} is missing fields: {sorted(missing)}")

    if unexpected:
        raise NmfGridError(f"{label} has unsupported fields: {sorted(unexpected)}")


def _ngram_range(value: object) -> tuple[int, int]:
    """Return a strict two-item ngram range."""
    if not isinstance(value, list) or len(value) != 2:
        raise NmfGridError("tfidf.ngram_range must be a two-item list.")

    lower = _safe_int(value[0], field_name="tfidf.ngram_range[0]")
    upper = _safe_int(value[1], field_name="tfidf.ngram_range[1]")

    if lower < 1 or upper < lower:
        raise NmfGridError("tfidf.ngram_range must be positive and ordered.")

    return (lower, upper)


def _validate_config_values(config: NmfGridConfig) -> None:
    """Validate cross-field constraints."""
    if config.primary_session_categories != ("legislative_debate",):
        raise NmfGridError(
            "primary_session_categories must be exactly ['legislative_debate'] for grid version 1."
        )

    if config.stopword_variant not in {"P0", "P1"}:
        raise NmfGridError("stopword_variant must be P0 or P1.")

    if config.tfidf.ngram_range != (1, 2):
        raise NmfGridError("tfidf.ngram_range must be [1, 2] for grid version 1.")

    if config.tfidf.min_df < 1:
        raise NmfGridError("tfidf.min_df must be positive.")

    if not 0 < config.tfidf.max_df <= 1:
        raise NmfGridError("tfidf.max_df must be in (0, 1].")

    if config.tfidf.max_features < 1:
        raise NmfGridError("tfidf.max_features must be positive.")

    if config.tfidf.norm != "l2":
        raise NmfGridError("tfidf.norm must be 'l2'.")

    if config.tfidf.lowercase:
        raise NmfGridError("tfidf.lowercase must be false because text is already casefolded.")

    if config.tfidf.strip_accents is not None:
        raise NmfGridError("tfidf.strip_accents must be null to preserve accents.")

    if config.nmf.solver != "cd":
        raise NmfGridError("NMF solver must be coordinate descent ('cd').")

    if config.nmf.beta_loss != "frobenius":
        raise NmfGridError("NMF beta_loss must be 'frobenius'.")

    if config.nmf.init != "nndsvda":
        raise NmfGridError("NMF init must be 'nndsvda'.")

    if config.nmf.max_iter < 1:
        raise NmfGridError("NMF max_iter must be positive.")

    if config.nmf.tol <= 0:
        raise NmfGridError("NMF tol must be positive.")

    if config.nmf.alpha_W != 0.0 or config.nmf.alpha_H != 0.0 or config.nmf.l1_ratio != 0.0:
        raise NmfGridError("NMF regularization must remain disabled for grid version 1.")

    if config.top_terms_per_topic < config.metrics_top_n:
        raise NmfGridError("top_terms_per_topic must be at least metrics_top_n.")

    if config.metrics_top_n < 2:
        raise NmfGridError("metrics_top_n must be at least 2.")

    if config.representative_documents_per_topic < 1:
        raise NmfGridError("representative_documents_per_topic must be positive.")

    if config.preprocessing_example_limit < 1:
        raise NmfGridError("preprocessing_example_limit must be positive.")

    if config.cleaned_excerpt_characters < 50:
        raise NmfGridError("cleaned_excerpt_characters must be at least 50.")


def _part_path(path: Path) -> Path:
    """Return the transactional part path."""
    return path.with_suffix(f"{path.suffix}.part")


def _backup_path(path: Path) -> Path:
    """Return the transactional backup path."""
    return path.with_suffix(f"{path.suffix}.bak")


def _replace_path(source: Path, destination: Path) -> None:
    """Replace a path through a narrow patch point for failure tests."""
    source.replace(destination)


def _cleanup_transaction_paths(final_paths: Iterable[Path]) -> None:
    """Remove transaction sidecar files."""
    for final_path in final_paths:
        _part_path(final_path).unlink(missing_ok=True)
        _backup_path(final_path).unlink(missing_ok=True)


def _preflight_output_directory(output_dir: Path, *, force: bool) -> None:
    """Refuse nonempty output directories unless force is supplied."""
    if output_dir.exists() and not output_dir.is_dir():
        raise NmfGridError(f"Output path is not a directory: {output_dir}")

    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise NmfGridError(f"Output directory is nonempty; use --force: {output_dir}")


def _ensure_output_directory(output_dir: Path, *, force: bool) -> None:
    """Create the output directory after overwrite protection."""
    _preflight_output_directory(output_dir, force=force)
    output_dir.mkdir(parents=True, exist_ok=True)


def _promote_transaction(final_paths: tuple[Path, ...]) -> None:
    """Promote staged part files to final paths with rollback."""
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
            raise NmfGridError(
                "Could not restore prior NMF-grid outputs after promotion failure."
            ) from restore_error

        _cleanup_transaction_paths(final_paths)
        raise NmfGridError("Could not promote NMF-grid outputs safely.") from error

    _cleanup_transaction_paths(final_paths)


def _final_output_paths(output_dir: Path, *, k_values: Sequence[int]) -> dict[str, Path]:
    """Return canonical final output paths."""
    paths = {
        "cleaned_documents": output_dir / CLEANED_DOCUMENTS_FILENAME,
        "grid_metrics": output_dir / GRID_METRICS_FILENAME,
        "grid_report": output_dir / GRID_REPORT_FILENAME,
        "preprocessing_examples": output_dir / PREPROCESSING_EXAMPLES_FILENAME,
        "preprocessing_summary": output_dir / PREPROCESSING_SUMMARY_FILENAME,
        "run_manifest": output_dir / RUN_MANIFEST_FILENAME,
        "vectorizer": output_dir / VECTORIZER_ARTIFACT_FILENAME,
        "vectorizer_summary": output_dir / VECTORIZER_SUMMARY_FILENAME,
        "vocabulary": output_dir / VOCABULARY_FILENAME,
    }

    for k_value in k_values:
        key = _k_key(k_value)
        paths[f"nmf_model_{key}"] = output_dir / f"nmf_{key}.joblib"
        paths[f"topic_terms_{key}"] = output_dir / f"topic_terms_{key}.csv"
        paths[f"representatives_{key}"] = output_dir / f"representative_documents_{key}.jsonl"

    return paths


def _k_key(k_value: int) -> str:
    """Return a stable K label for filenames."""
    return f"k{k_value:03d}"


def _write_text_part(path: Path, text: str) -> dict[str, Any]:
    """Write UTF-8 text to a staged part file."""
    _part_path(path).write_text(text, encoding="utf-8", newline="")
    return _file_metadata(path)


def _write_json_part(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Write deterministic JSON to a staged part file."""
    return _write_text_part(path, _json_text(payload))


def _write_jsonl_part(path: Path, records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Write deterministic JSONL records to a staged part file."""
    part_path = _part_path(path)
    hasher = hashlib.sha256()
    size_bytes = 0
    record_count = 0

    with part_path.open("wb") as output_file:
        for record in records:
            data = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            output_file.write(data)
            hasher.update(data)
            size_bytes += len(data)
            record_count += 1

    return {
        "path": str(path),
        "record_count": record_count,
        "sha256": hasher.hexdigest(),
        "size_bytes": size_bytes,
    }


def _write_csv_part(
    path: Path,
    *,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Write deterministic CSV rows to a staged part file."""
    part_path = _part_path(path)
    record_count = 0

    with part_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(row)
            record_count += 1

    metadata_payload = _file_metadata(path)
    metadata_payload["record_count"] = record_count
    return metadata_payload


def _write_joblib_part(path: Path, payload: object) -> dict[str, Any]:
    """Write a joblib artifact to a staged part file."""
    joblib.dump(payload, _part_path(path))
    return _file_metadata(path)


def _file_metadata(path: Path) -> dict[str, Any]:
    """Return metadata for a staged part file addressed by final path."""
    part_path = _part_path(path)
    return {
        "path": str(path),
        "sha256": sha256_file(part_path),
        "size_bytes": part_path.stat().st_size,
    }


def _validate_output_metadata(output_files: Mapping[str, Mapping[str, Any]]) -> bool:
    """Validate staged output metadata against staged files."""
    for metadata_payload in output_files.values():
        path_value = metadata_payload.get("path")
        sha256_value = metadata_payload.get("sha256")
        size_value = metadata_payload.get("size_bytes")

        if not isinstance(path_value, str) or not isinstance(sha256_value, str):
            return False

        if isinstance(size_value, bool) or not isinstance(size_value, int):
            return False

        part_path = _part_path(Path(path_value))

        if not part_path.is_file():
            return False

        if part_path.stat().st_size != size_value:
            return False

        if sha256_file(part_path) != sha256_value:
            return False

    return True


def _validate_manifest_and_lock(
    *,
    documents_path: Path,
    export_manifest_path: Path,
    corpus_lock_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Run the existing strict locked-input validator and convert its error type."""
    try:
        return corpus_profile._validate_manifest_and_lock(
            documents_path=documents_path,
            export_manifest_path=export_manifest_path,
            corpus_lock_path=corpus_lock_path,
        )
    except CorpusProfileError as error:
        raise NmfGridError(str(error)) from error


def _iter_documents(path: Path) -> Iterator[DocumentRecord]:
    """Stream documents through the existing strict document validator."""
    try:
        yield from corpus_profile._iter_documents(path)
    except CorpusProfileError as error:
        raise NmfGridError(str(error)) from error


def _validate_profile_manifest(
    *,
    profile_manifest_path: Path,
    profile_config_path: Path,
    current_hashes: Mapping[str, str],
    config: NmfGridConfig,
) -> dict[str, Any]:
    """Validate corpus-profile lineage and exact profile totals before preprocessing."""
    profile_manifest = _read_json_object(profile_manifest_path, label="corpus-profile manifest")
    input_sha256 = _object_field(profile_manifest, "input_sha256")
    profile_config_sha256 = sha256_file(profile_config_path)

    required_hashes = {
        "config_sha256": profile_config_sha256,
        "corpus_lock_sha256": current_hashes["corpus_lock_sha256"],
        "documents_sha256": current_hashes["documents_sha256"],
        "export_manifest_sha256": current_hashes["export_manifest_sha256"],
    }

    for field_name, expected_hash in required_hashes.items():
        if input_sha256.get(field_name) != expected_hash:
            raise NmfGridError(f"Corpus-profile manifest hash mismatch: {field_name}")

    try:
        profile_config = corpus_profile._load_config(profile_config_path)
    except CorpusProfileError as error:
        raise NmfGridError(str(error)) from error

    canonical_profile_config = corpus_profile._json_text(profile_config.to_json())
    canonical_profile_hash = hashlib.sha256(canonical_profile_config.encode("utf-8")).hexdigest()

    if profile_manifest.get("canonical_configuration_sha256") != canonical_profile_hash:
        raise NmfGridError("Corpus-profile canonical configuration hash mismatch.")

    reconciliation_checks = _object_field(profile_manifest, "reconciliation_checks")
    failed_checks = sorted(
        key for key, passed in reconciliation_checks.items() if passed is not True
    )

    if failed_checks:
        raise NmfGridError(f"Corpus-profile reconciliation checks failed: {failed_checks}")

    universes = _object_field(profile_manifest, "universes")
    all_counts = _object_field(universes, corpus_profile.UNIVERSE_ALL)
    primary_counts = _object_field(universes, corpus_profile.UNIVERSE_PRIMARY)

    expected_all = config.expected_profile_counts
    expected_primary = config.expected_primary_counts
    observed_all = {
        "all_documents": _safe_int(all_counts.get("documents"), field_name="all documents"),
        "all_words": _safe_int(all_counts.get("words"), field_name="all words"),
    }
    observed_primary = _primary_counts_from_profile(primary_counts)

    if observed_all != expected_all.to_json():
        raise NmfGridError(
            f"Corpus-profile all-session totals mismatch: {observed_all} != "
            f"{expected_all.to_json()}"
        )

    if observed_primary != expected_primary.to_json():
        raise NmfGridError(
            f"Corpus-profile primary totals mismatch: {observed_primary} != "
            f"{expected_primary.to_json()}"
        )

    return {
        "canonical_configuration_sha256": canonical_profile_hash,
        "input_sha256": dict(sorted(input_sha256.items())),
        "profile_manifest_sha256": sha256_file(profile_manifest_path),
        "reconciliation_checks": dict(sorted(reconciliation_checks.items())),
        "universes": {
            "all_sessions": observed_all,
            "primary": observed_primary,
        },
    }


def _primary_counts_from_profile(primary_counts: Mapping[str, Any]) -> dict[str, int]:
    """Return primary counts from a corpus-profile manifest count object."""
    return {
        "documents": _safe_int(primary_counts.get("documents"), field_name="primary documents"),
        "sessions": _safe_int(
            primary_counts.get("unique_sessions"),
            field_name="primary unique_sessions",
        ),
        "source_turns": _safe_int(
            primary_counts.get("unique_source_turns"),
            field_name="primary unique_source_turns",
        ),
        "words": _safe_int(primary_counts.get("words"), field_name="primary words"),
    }


def _stage_cleaned_primary_documents(
    *,
    documents_path: Path,
    cleaned_documents_path: Path,
    export_manifest: Mapping[str, Any],
    config: NmfGridConfig,
) -> PreprocessingArtifacts:
    """Stream locked documents, clean only the primary universe, and stage JSONL."""
    part_path = _part_path(cleaned_documents_path)
    seen_document_ids: set[str] = set()
    processed_document_count = 0
    processed_word_total = 0
    primary_document_count = 0
    primary_word_total = 0
    primary_source_turns: set[str] = set()
    primary_sessions: set[str] = set()
    documents_changed_by_soft = 0
    soft_hyphens_removed = 0
    documents_changed_by_explicit = 0
    explicit_hyphenation_joins = 0
    documents_changed_by_either = 0
    zero_token_documents: list[str] = []
    examples: list[dict[str, Any]] = []
    hasher = hashlib.sha256()
    size_bytes = 0

    with part_path.open("wb") as output_file:
        for document in _iter_documents(documents_path):
            if document.document_id in seen_document_ids:
                raise NmfGridError(f"Duplicate document_id: {document.document_id}")

            seen_document_ids.add(document.document_id)
            processed_document_count += 1
            processed_word_total += document.word_count

            if document.session_category not in config.primary_session_categories:
                continue

            primary_document_count += 1
            primary_word_total += document.word_count
            primary_source_turns.add(document.source_turn_key)
            primary_sessions.add(document.source_record_id)
            cleaned = clean_natural_text(document.modeling_text)
            tokens = lexical_tokens(cleaned.cleaned_text)

            if cleaned.changed_by_soft_hyphen_repair:
                documents_changed_by_soft += 1

            if cleaned.changed_by_explicit_hyphenation_repair:
                documents_changed_by_explicit += 1

            if cleaned.changed_by_any_repair:
                documents_changed_by_either += 1
                _add_preprocessing_example(
                    examples=examples,
                    document=document,
                    cleaned=cleaned,
                    limit=config.preprocessing_example_limit,
                )

            soft_hyphens_removed += cleaned.soft_hyphen_removed_count
            explicit_hyphenation_joins += cleaned.explicit_hyphenation_join_count

            if not tokens:
                zero_token_documents.append(document.document_id)

            record = {
                "changed_by_explicit_hyphenation_repair": (
                    cleaned.changed_by_explicit_hyphenation_repair
                ),
                "changed_by_soft_hyphen_repair": cleaned.changed_by_soft_hyphen_repair,
                "chunk_index": document.chunk_index,
                "cleaned_text": cleaned.cleaned_text,
                "document_id": document.document_id,
                "explicit_hyphenation_join_count": cleaned.explicit_hyphenation_join_count,
                "lexical_token_count": len(tokens),
                "modeling_text": document.modeling_text,
                "session_category": document.session_category,
                "source_record_id": document.source_record_id,
                "speaker_family": document.speaker_family,
                "soft_hyphen_removed_count": cleaned.soft_hyphen_removed_count,
                "temporal_period": document.temporal_period,
                "turn_index": document.turn_index,
                "word_count": document.word_count,
                "year": document.year,
            }
            data = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
            output_file.write(data)
            hasher.update(data)
            size_bytes += len(data)

    manifest_document_count = _safe_int(
        export_manifest.get("modeling_document_count"),
        field_name="modeling_document_count",
    )
    manifest_word_total = _safe_int(
        export_manifest.get("modeling_word_total"),
        field_name="modeling_word_total",
    )

    if processed_document_count != manifest_document_count:
        raise NmfGridError(
            "Processed document count differs from export manifest: "
            f"{processed_document_count} != {manifest_document_count}"
        )

    if processed_word_total != manifest_word_total:
        raise NmfGridError(
            "Processed word total differs from export manifest: "
            f"{processed_word_total} != {manifest_word_total}"
        )

    primary_counts = {
        "documents": primary_document_count,
        "sessions": len(primary_sessions),
        "source_turns": len(primary_source_turns),
        "words": primary_word_total,
    }

    if primary_counts != config.expected_primary_counts.to_json():
        raise NmfGridError(
            f"Primary corpus totals mismatch: {primary_counts} != "
            f"{config.expected_primary_counts.to_json()}"
        )

    summary = {
        "cleaning_version": CLEANING_VERSION,
        "documents_changed_by_either_repair": documents_changed_by_either,
        "documents_changed_by_explicit_hyphenation_repair": documents_changed_by_explicit,
        "documents_changed_by_soft_hyphen_repair": documents_changed_by_soft,
        "documents_with_zero_alphabetic_lexical_tokens": len(zero_token_documents),
        "explicit_hyphenation_joins": explicit_hyphenation_joins,
        "primary_counts": primary_counts,
        "soft_hyphens_removed": soft_hyphens_removed,
        "tokenizer_version": LEXICAL_TOKENIZER_VERSION,
        "zero_token_document_ids": zero_token_documents[:50],
    }

    if zero_token_documents:
        raise NmfGridError(
            "Primary documents with zero alphabetic lexical tokens after cleaning: "
            f"{zero_token_documents[:10]}"
        )

    metadata_payload = {
        "path": str(cleaned_documents_path),
        "record_count": primary_document_count,
        "sha256": hasher.hexdigest(),
        "size_bytes": size_bytes,
    }
    return PreprocessingArtifacts(
        summary=summary,
        examples=tuple(sorted(examples, key=lambda row: str(row["document_id"]))),
        primary_counts=primary_counts,
        cleaned_documents_metadata=metadata_payload,
        cleaned_documents_part_path=part_path,
    )


def _add_preprocessing_example(
    *,
    examples: list[dict[str, Any]],
    document: DocumentRecord,
    cleaned: CleaningResult,
    limit: int,
) -> None:
    """Keep bounded deterministic before/after examples."""
    row = {
        "changed_by_explicit_hyphenation_repair": cleaned.changed_by_explicit_hyphenation_repair,
        "changed_by_soft_hyphen_repair": cleaned.changed_by_soft_hyphen_repair,
        "cleaned_text_excerpt": bounded_excerpt(cleaned.cleaned_text, character_limit=280),
        "document_id": document.document_id,
        "explicit_hyphenation_join_count": cleaned.explicit_hyphenation_join_count,
        "modeling_text_excerpt": bounded_excerpt(document.modeling_text, character_limit=280),
        "soft_hyphen_removed_count": cleaned.soft_hyphen_removed_count,
        "source_record_id": document.source_record_id,
        "turn_index": document.turn_index,
    }
    examples.append(row)
    examples.sort(key=lambda item: str(item["document_id"]))
    del examples[limit:]


def _iter_cleaned_records(path: Path) -> Iterator[dict[str, Any]]:
    """Stream staged or final cleaned JSONL records."""
    try:
        with path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    raise NmfGridError(f"Blank cleaned JSONL record at line {line_number}")

                payload = json.loads(line)

                if not isinstance(payload, dict):
                    raise NmfGridError(f"Cleaned JSONL record is not an object: {line_number}")

                yield {str(key): value for key, value in payload.items()}
    except json.JSONDecodeError as error:
        raise NmfGridError(f"Malformed cleaned JSONL at line {line_number}") from error
    except OSError as error:
        raise NmfGridError(f"Could not read cleaned documents JSONL: {path}") from error


def _cleaned_texts(path: Path) -> Iterator[str]:
    """Yield cleaned text in row order for vectorization."""
    for record in _iter_cleaned_records(path):
        cleaned_text = record.get("cleaned_text")

        if not isinstance(cleaned_text, str):
            raise NmfGridError("Cleaned document record has invalid cleaned_text.")

        yield cleaned_text


def _document_metadata(path: Path, *, excerpt_characters: int) -> tuple[DocumentMetadata, ...]:
    """Load compact document metadata for representative-document output."""
    records: list[DocumentMetadata] = []

    for row_index, record in enumerate(_iter_cleaned_records(path)):
        cleaned_text = _required_cleaned_string(record, "cleaned_text")
        source_record_id = _required_cleaned_string(record, "source_record_id")
        turn_index = _safe_int(record.get("turn_index"), field_name="turn_index")
        records.append(
            DocumentMetadata(
                row_index=row_index,
                document_id=_required_cleaned_string(record, "document_id"),
                source_record_id=source_record_id,
                turn_index=turn_index,
                chunk_index=_safe_int(record.get("chunk_index"), field_name="chunk_index"),
                year=_safe_int(record.get("year"), field_name="year"),
                temporal_period=_required_cleaned_string(record, "temporal_period"),
                session_category=_required_cleaned_string(record, "session_category"),
                speaker_family=_required_cleaned_string(record, "speaker_family"),
                word_count=_safe_int(record.get("word_count"), field_name="word_count"),
                source_turn_key=f"{source_record_id}::turn_{turn_index:06d}",
                cleaned_excerpt=bounded_excerpt(
                    cleaned_text,
                    character_limit=excerpt_characters,
                ),
            )
        )

    return tuple(records)


def _required_cleaned_string(payload: Mapping[str, Any], field_name: str) -> str:
    """Return a required nonempty string from a cleaned document record."""
    value = payload.get(field_name)

    if not isinstance(value, str) or not value:
        raise NmfGridError(f"Cleaned document record has invalid {field_name}.")

    return value


def _fit_tfidf(
    *,
    cleaned_documents_path: Path,
    config: NmfGridConfig,
    stopwords: StopwordSet,
) -> VectorizationResult:
    """Fit sparse TF-IDF once for the primary corpus."""
    document_metadata = _document_metadata(
        cleaned_documents_path,
        excerpt_characters=config.cleaned_excerpt_characters,
    )
    vectorizer = TfidfVectorizer(
        dtype=np.float32,
        lowercase=config.tfidf.lowercase,
        max_df=config.tfidf.max_df,
        max_features=config.tfidf.max_features,
        min_df=config.tfidf.min_df,
        ngram_range=config.tfidf.ngram_range,
        norm=config.tfidf.norm,
        smooth_idf=config.tfidf.smooth_idf,
        stop_words=sorted(stopwords.words),
        strip_accents=config.tfidf.strip_accents,
        sublinear_tf=config.tfidf.sublinear_tf,
        token_pattern=None,
        tokenizer=lexical_tokens,
    )

    try:
        matrix = vectorizer.fit_transform(_cleaned_texts(cleaned_documents_path))
    except ValueError as error:
        raise NmfGridError(f"TF-IDF fitting failed: {error}") from error

    if not sp.issparse(matrix):
        raise NmfGridError("TF-IDF vectorizer returned a dense matrix.")

    matrix = matrix.tocsr()

    if matrix.shape[0] != len(document_metadata):
        raise NmfGridError("TF-IDF row count does not match cleaned documents.")

    zero_row_indices = np.flatnonzero(np.diff(matrix.indptr) == 0)
    zero_tfidf_rows = int(zero_row_indices.size)

    if zero_tfidf_rows:
        zero_document_ids = [
            document_metadata[int(index)].document_id for index in zero_row_indices[:10]
        ]
        raise NmfGridError(
            f"TF-IDF produced {zero_tfidf_rows} zero rows; first document IDs: {zero_document_ids}"
        )

    feature_names = vectorizer.get_feature_names_out()
    vocabulary_rows = _vocabulary_rows(matrix=matrix, feature_names=feature_names)
    unigram_feature_count = sum(1 for feature in feature_names if " " not in feature)
    bigram_feature_count = len(feature_names) - unigram_feature_count
    nnz = int(matrix.nnz)
    density = round(float(nnz) / float(matrix.shape[0] * matrix.shape[1]), 12)
    storage_bytes = int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
    summary = {
        "bigram_feature_count": bigram_feature_count,
        "density": density,
        "documents_processed": matrix.shape[0],
        "dtype": str(matrix.dtype),
        "estimated_sparse_storage_bytes": storage_bytes,
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "nonzero_count": nnz,
        "settings": config.tfidf.to_json(),
        "sparse_matrix_format": matrix.getformat(),
        "stopword_file_sha256": stopwords.p0_sha256,
        "stopword_variant": stopwords.variant,
        "unigram_feature_count": unigram_feature_count,
        "vocabulary_size": int(matrix.shape[1]),
        "zero_tfidf_rows": zero_tfidf_rows,
    }
    return VectorizationResult(
        vectorizer=vectorizer,
        matrix=matrix,
        document_metadata=document_metadata,
        summary=summary,
        vocabulary_rows=tuple(vocabulary_rows),
    )


def _vocabulary_rows(*, matrix: Any, feature_names: np.ndarray[Any, Any]) -> list[dict[str, Any]]:
    """Return deterministic vocabulary rows with document frequencies."""
    document_frequencies = np.asarray(matrix.getnnz(axis=0)).ravel()
    rows: list[dict[str, Any]] = []

    for index, term in enumerate(feature_names):
        term_text = str(term)
        rows.append(
            {
                "document_frequency": int(document_frequencies[index]),
                "feature_index": index,
                "ngram_order": 2 if " " in term_text else 1,
                "term": term_text,
            }
        )

    return rows


def _fit_models(
    *,
    vectorization: VectorizationResult,
    config: NmfGridConfig,
) -> tuple[TopicModelResult, ...]:
    """Fit every configured NMF model against the same sparse TF-IDF matrix."""
    results: list[TopicModelResult] = []
    feature_names = tuple(str(term) for term in vectorization.vectorizer.get_feature_names_out())

    for k_value in config.nmf.k_values:
        if k_value > min(vectorization.matrix.shape):
            raise NmfGridError(
                f"K={k_value} exceeds the fitted matrix minimum dimension "
                f"{min(vectorization.matrix.shape)}."
            )

        model = NMF(
            alpha_H=config.nmf.alpha_H,
            alpha_W=config.nmf.alpha_W,
            beta_loss=config.nmf.beta_loss,
            init=config.nmf.init,
            l1_ratio=config.nmf.l1_ratio,
            max_iter=config.nmf.max_iter,
            n_components=k_value,
            random_state=config.nmf.random_state,
            solver=config.nmf.solver,
            tol=config.nmf.tol,
        )
        start = time.perf_counter()

        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            document_topic = model.fit_transform(vectorization.matrix)

        runtime_seconds = round(time.perf_counter() - start, 6)
        convergence_warnings = [
            str(item.message)
            for item in captured_warnings
            if issubclass(item.category, ConvergenceWarning)
        ]
        top_indices = _top_indices_by_topic(
            components=model.components_,
            feature_names=feature_names,
            top_n=config.top_terms_per_topic,
        )
        metric_top_indices = tuple(
            tuple(indices[: config.metrics_top_n]) for indices in top_indices
        )
        metric_payload = _topic_metrics(
            components=model.components_,
            top_indices=metric_top_indices,
            matrix=vectorization.matrix,
        )
        concentration_payload = _document_topic_concentration(document_topic)
        normalized_document_topic = _normalize_document_topic(document_topic)
        prevalence = normalized_document_topic.mean(axis=0) if document_topic.size else np.array([])
        per_topic_metrics = _per_topic_metric_rows(
            k_value=k_value,
            feature_names=feature_names,
            top_indices=metric_top_indices,
            npmi_by_topic=metric_payload["npmi_by_topic"],
            exclusivity_by_topic=metric_payload["exclusivity_by_topic"],
            prevalence=prevalence,
        )
        topic_term_rows = _topic_term_rows(
            k_value=k_value,
            components=model.components_,
            feature_names=feature_names,
            top_indices=top_indices,
            per_topic_metrics=per_topic_metrics,
        )
        representative_rows = _representative_rows(
            k_value=k_value,
            normalized_document_topic=normalized_document_topic,
            document_metadata=vectorization.document_metadata,
            per_topic_count=config.representative_documents_per_topic,
        )
        metric_row = {
            "convergence_warnings": _json_compact(convergence_warnings),
            "converged": not convergence_warnings,
            "document_topic_mean_dominant_weight": concentration_payload[
                "mean_dominant_topic_weight"
            ],
            "document_topic_mean_normalized_entropy": concentration_payload[
                "mean_normalized_topic_entropy"
            ],
            "document_topic_zero_weight_rows": concentration_payload["zero_weight_rows"],
            "k": k_value,
            "mean_npmi_coherence_top10": metric_payload["mean_npmi_coherence"],
            "mean_topic_exclusivity_top10": metric_payload["mean_exclusivity"],
            "median_npmi_coherence_top10": metric_payload["median_npmi_coherence"],
            "min_npmi_coherence_top10": metric_payload["min_npmi_coherence"],
            "n_iter": int(model.n_iter_),
            "reconstruction_error": round(float(model.reconstruction_err_), 10),
            "redundancy_max_cosine": metric_payload["redundancy_max_cosine"],
            "redundancy_mean_off_diagonal_cosine": metric_payload[
                "redundancy_mean_off_diagonal_cosine"
            ],
            "runtime_seconds": runtime_seconds,
            "topic_diversity_top10": metric_payload["topic_diversity"],
        }
        results.append(
            TopicModelResult(
                k=k_value,
                model=model,
                topic_term_rows=tuple(topic_term_rows),
                representative_rows=tuple(representative_rows),
                metric_row=metric_row,
                per_topic_metrics=per_topic_metrics,
            )
        )

    return tuple(results)


def _top_indices_by_topic(
    *,
    components: np.ndarray[Any, Any],
    feature_names: Sequence[str],
    top_n: int,
) -> tuple[tuple[int, ...], ...]:
    """Return top term indices per topic with deterministic tie-breaking."""
    rows: list[tuple[int, ...]] = []

    for topic_weights in components:
        indices = sorted(
            range(len(topic_weights)),
            key=lambda index: (-float(topic_weights[index]), feature_names[index], index),
        )
        rows.append(tuple(indices[:top_n]))

    return tuple(rows)


def _topic_metrics(
    *,
    components: np.ndarray[Any, Any],
    top_indices: Sequence[Sequence[int]],
    matrix: Any,
) -> dict[str, Any]:
    """Compute all exact topic-level metrics for one fitted model."""
    diversity = topic_diversity(top_indices=top_indices)
    exclusivity_by_topic, mean_exclusivity = topic_exclusivity(
        components=components,
        top_indices=top_indices,
    )
    redundancy = topic_redundancy(components)
    npmi_by_topic = npmi_coherence_by_topic(matrix=matrix, top_indices=top_indices)
    npmi_values = list(npmi_by_topic)
    return {
        "exclusivity_by_topic": exclusivity_by_topic,
        "mean_exclusivity": round(mean_exclusivity, 10),
        "mean_npmi_coherence": round(float(np.mean(npmi_values)), 10) if npmi_values else 0.0,
        "median_npmi_coherence": round(float(np.median(npmi_values)), 10) if npmi_values else 0.0,
        "min_npmi_coherence": round(float(np.min(npmi_values)), 10) if npmi_values else 0.0,
        "npmi_by_topic": tuple(npmi_by_topic),
        "redundancy_max_cosine": redundancy["max_off_diagonal_similarity"],
        "redundancy_mean_off_diagonal_cosine": redundancy["mean_off_diagonal_similarity"],
        "topic_diversity": round(diversity, 10),
    }


def topic_diversity(*, top_indices: Sequence[Sequence[int]]) -> float:
    """Unique top terms divided by K times selected top-N length."""
    denominator = sum(len(indices) for indices in top_indices)

    if denominator == 0:
        return 0.0

    unique_terms = {index for indices in top_indices for index in indices}
    return float(len(unique_terms)) / float(denominator)


def topic_exclusivity(
    *,
    components: np.ndarray[Any, Any],
    top_indices: Sequence[Sequence[int]],
) -> tuple[tuple[float, ...], float]:
    """Return per-topic and global mean exclusivity over selected top terms."""
    term_totals = components.sum(axis=0)
    per_topic: list[float] = []
    all_values: list[float] = []

    for topic_index, indices in enumerate(top_indices):
        topic_values: list[float] = []

        for term_index in indices:
            denominator = float(term_totals[term_index])
            value = (
                0.0
                if denominator == 0.0
                else float(components[topic_index, term_index]) / denominator
            )
            topic_values.append(value)
            all_values.append(value)

        per_topic.append(float(np.mean(topic_values)) if topic_values else 0.0)

    global_mean = float(np.mean(all_values)) if all_values else 0.0
    return tuple(round(value, 10) for value in per_topic), global_mean


def topic_redundancy(components: np.ndarray[Any, Any]) -> dict[str, float | int]:
    """Compute pairwise cosine redundancy between topic-term vectors."""
    topic_count = int(components.shape[0])

    if topic_count < 2:
        return {
            "max_off_diagonal_similarity": 0.0,
            "mean_off_diagonal_similarity": 0.0,
            "pair_count": 0,
        }

    norms = np.linalg.norm(components, axis=1)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    normalized = components / safe_norms[:, np.newaxis]
    similarities = normalized @ normalized.T
    values = similarities[np.triu_indices(topic_count, k=1)]
    return {
        "max_off_diagonal_similarity": round(float(np.max(values)), 10),
        "mean_off_diagonal_similarity": round(float(np.mean(values)), 10),
        "pair_count": int(values.size),
    }


def npmi_coherence_by_topic(
    *, matrix: Any, top_indices: Sequence[Sequence[int]]
) -> tuple[float, ...]:
    """Compute top-term NPMI using only the union of selected topic terms."""
    unique_indices = sorted({index for indices in top_indices for index in indices})

    if not unique_indices:
        return tuple(0.0 for _ in top_indices)

    binary = matrix[:, unique_indices].copy()
    binary.data = np.ones_like(binary.data)
    cooccurrence = (binary.T @ binary).toarray()
    document_count = int(matrix.shape[0])
    local_index = {feature_index: local for local, feature_index in enumerate(unique_indices)}
    topic_scores: list[float] = []

    for indices in top_indices:
        pair_scores: list[float] = []

        for left_position, left_feature in enumerate(indices):
            for right_feature in indices[left_position + 1 :]:
                left = local_index[left_feature]
                right = local_index[right_feature]
                cooccurrence_count = int(cooccurrence[left, right])
                left_count = int(cooccurrence[left, left])
                right_count = int(cooccurrence[right, right])
                pair_scores.append(
                    npmi_from_counts(
                        cooccurrence_count=cooccurrence_count,
                        left_count=left_count,
                        right_count=right_count,
                        document_count=document_count,
                    )
                )

        topic_scores.append(round(float(np.mean(pair_scores)), 10) if pair_scores else 0.0)

    return tuple(topic_scores)


def npmi_from_counts(
    *,
    cooccurrence_count: int,
    left_count: int,
    right_count: int,
    document_count: int,
) -> float:
    """Return NPMI for one unordered term pair from binary document counts."""
    if document_count < 1:
        raise NmfGridError("document_count must be positive for NPMI.")

    if cooccurrence_count == 0:
        return -1.0

    if left_count < cooccurrence_count or right_count < cooccurrence_count:
        raise NmfGridError("Invalid NPMI counts: co-occurrence exceeds marginal count.")

    p_ij = float(cooccurrence_count) / float(document_count)
    p_i = float(left_count) / float(document_count)
    p_j = float(right_count) / float(document_count)

    if p_i <= 0.0 or p_j <= 0.0:
        raise NmfGridError("Invalid NPMI counts: positive co-occurrence with zero marginal.")

    denominator = -math.log(p_ij)

    if denominator == 0.0:
        return 1.0 if p_i == p_j == p_ij else 0.0

    pmi = math.log(p_ij / (p_i * p_j))
    value = pmi / denominator
    return round(max(-1.0, min(1.0, value)), 10)


def _document_topic_concentration(document_topic: np.ndarray[Any, Any]) -> dict[str, Any]:
    """Return concentration metrics from non-normalized document-topic weights."""
    normalized = _normalize_document_topic(document_topic)
    row_sums = document_topic.sum(axis=1)
    nonzero_mask = row_sums > 0.0
    zero_weight_rows = int((~nonzero_mask).sum())

    if not np.any(nonzero_mask):
        return {
            "mean_dominant_topic_weight": 0.0,
            "mean_normalized_topic_entropy": 0.0,
            "zero_weight_rows": zero_weight_rows,
        }

    nonzero = normalized[nonzero_mask]
    dominant = nonzero.max(axis=1)

    if nonzero.shape[1] < 2:
        entropies = np.zeros(nonzero.shape[0])
    else:
        safe = np.where(nonzero > 0.0, nonzero, 1.0)
        entropies = -(np.where(nonzero > 0.0, nonzero * np.log(safe), 0.0).sum(axis=1))
        entropies = entropies / math.log(nonzero.shape[1])

    return {
        "mean_dominant_topic_weight": round(float(np.mean(dominant)), 10),
        "mean_normalized_topic_entropy": round(float(np.mean(entropies)), 10),
        "zero_weight_rows": zero_weight_rows,
    }


def _normalize_document_topic(document_topic: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Normalize nonzero document-topic rows to sum to one."""
    row_sums = document_topic.sum(axis=1)
    normalized = np.zeros_like(document_topic, dtype=np.float64)
    nonzero_mask = row_sums > 0.0

    if np.any(nonzero_mask):
        normalized[nonzero_mask] = document_topic[nonzero_mask] / row_sums[nonzero_mask, np.newaxis]

    return normalized


def _per_topic_metric_rows(
    *,
    k_value: int,
    feature_names: Sequence[str],
    top_indices: Sequence[Sequence[int]],
    npmi_by_topic: Sequence[float],
    exclusivity_by_topic: Sequence[float],
    prevalence: np.ndarray[Any, Any],
) -> tuple[dict[str, Any], ...]:
    """Return per-topic metrics used in interpretation outputs."""
    rows: list[dict[str, Any]] = []

    for topic_index, indices in enumerate(top_indices):
        top_terms = [feature_names[index] for index in indices]
        rows.append(
            {
                "k": k_value,
                "mean_exclusivity_top10": exclusivity_by_topic[topic_index],
                "npmi_coherence_top10": npmi_by_topic[topic_index],
                "prevalence": round(float(prevalence[topic_index]), 10),
                "top_terms_top10": _json_compact(top_terms),
                "topic_index": topic_index,
            }
        )

    return tuple(rows)


def _topic_term_rows(
    *,
    k_value: int,
    components: np.ndarray[Any, Any],
    feature_names: Sequence[str],
    top_indices: Sequence[Sequence[int]],
    per_topic_metrics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return top-term rows for one K."""
    rows: list[dict[str, Any]] = []
    metrics_by_topic = {
        int(metric["topic_index"]): metric
        for metric in per_topic_metrics
        if "topic_index" in metric
    }

    for topic_index, indices in enumerate(top_indices):
        topic_metrics = metrics_by_topic[topic_index]

        for rank, term_index in enumerate(indices, start=1):
            rows.append(
                {
                    "feature_index": term_index,
                    "k": k_value,
                    "mean_exclusivity_top10": topic_metrics["mean_exclusivity_top10"],
                    "npmi_coherence_top10": topic_metrics["npmi_coherence_top10"],
                    "prevalence": topic_metrics["prevalence"],
                    "rank": rank,
                    "term": feature_names[term_index],
                    "topic_index": topic_index,
                    "top_terms_top10": topic_metrics["top_terms_top10"],
                    "weight": round(float(components[topic_index, term_index]), 10),
                }
            )

    return rows


def _representative_rows(
    *,
    k_value: int,
    normalized_document_topic: np.ndarray[Any, Any],
    document_metadata: Sequence[DocumentMetadata],
    per_topic_count: int,
) -> list[dict[str, Any]]:
    """Return representative-document rows with source-turn diversity preference."""
    rows: list[dict[str, Any]] = []

    for topic_index in range(normalized_document_topic.shape[1]):
        candidates = sorted(
            range(normalized_document_topic.shape[0]),
            key=lambda row_index: (
                -float(normalized_document_topic[row_index, topic_index]),
                document_metadata[row_index].document_id,
            ),
        )
        selected = _select_representative_indices(
            candidates=candidates,
            document_metadata=document_metadata,
            per_topic_count=per_topic_count,
        )

        for rank, row_index in enumerate(selected, start=1):
            metadata_row = document_metadata[row_index]
            rows.append(
                {
                    "chunk_index": metadata_row.chunk_index,
                    "cleaned_text_excerpt": metadata_row.cleaned_excerpt,
                    "document_id": metadata_row.document_id,
                    "k": k_value,
                    "normalized_topic_weight": round(
                        float(normalized_document_topic[row_index, topic_index]),
                        10,
                    ),
                    "rank": rank,
                    "session_category": metadata_row.session_category,
                    "source_record_id": metadata_row.source_record_id,
                    "speaker_family": metadata_row.speaker_family,
                    "temporal_period": metadata_row.temporal_period,
                    "topic_index": topic_index,
                    "turn_index": metadata_row.turn_index,
                    "word_count": metadata_row.word_count,
                    "year": metadata_row.year,
                }
            )

    return rows


def _select_representative_indices(
    *,
    candidates: Sequence[int],
    document_metadata: Sequence[DocumentMetadata],
    per_topic_count: int,
) -> list[int]:
    """Prefer distinct source turns, then fill remaining slots deterministically."""
    selected: list[int] = []
    selected_rows: set[int] = set()
    used_source_turns: set[str] = set()

    for row_index in candidates:
        if len(selected) >= per_topic_count:
            break

        source_turn_key = document_metadata[row_index].source_turn_key

        if source_turn_key in used_source_turns:
            continue

        selected.append(row_index)
        selected_rows.add(row_index)
        used_source_turns.add(source_turn_key)

    for row_index in candidates:
        if len(selected) >= per_topic_count:
            break

        if row_index in selected_rows:
            continue

        selected.append(row_index)
        selected_rows.add(row_index)

    return selected


def _grid_report(
    *,
    config: NmfGridConfig,
    preprocessing_summary: Mapping[str, Any],
    vectorizer_summary: Mapping[str, Any],
    grid_rows: Sequence[Mapping[str, Any]],
    stopwords: StopwordSet,
) -> str:
    """Return a concise Markdown grid report."""
    lines = [
        "# NMF Grid Report",
        "",
        "This report fits the configured NMF grid without selecting a winning K.",
        "Serialized model bytes are not guaranteed to be reproducible across library "
        "versions or platforms.",
        "",
        "## Preprocessing",
        "",
        f"- Primary documents: {preprocessing_summary['primary_counts']['documents']:,}",
        f"- Primary words: {preprocessing_summary['primary_counts']['words']:,}",
        f"- Soft hyphens removed: {preprocessing_summary['soft_hyphens_removed']:,}",
        f"- Explicit hyphenation joins: {preprocessing_summary['explicit_hyphenation_joins']:,}",
        "",
        "## TF-IDF",
        "",
        f"- Stopwords: {stopwords.variant} "
        f"(P0 count {stopwords.p0_count:,}, P1 count {stopwords.p1_count:,})",
        f"- Stopword file SHA-256: `{stopwords.p0_sha256}`",
        f"- Vocabulary size: {vectorizer_summary['vocabulary_size']:,}",
        f"- Matrix shape: {vectorizer_summary['matrix_shape']}",
        f"- Nonzero count: {vectorizer_summary['nonzero_count']:,}",
        "",
        "## Metrics",
        "",
        "| K | diversity@10 | mean NPMI@10 | median NPMI@10 | mean exclusivity@10 | "
        "mean redundancy | max redundancy | reconstruction error | converged |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]

    for row in grid_rows:
        lines.append(
            "| "
            f"{row['k']} | "
            f"{float(row['topic_diversity_top10']):.4f} | "
            f"{float(row['mean_npmi_coherence_top10']):.4f} | "
            f"{float(row['median_npmi_coherence_top10']):.4f} | "
            f"{float(row['mean_topic_exclusivity_top10']):.4f} | "
            f"{float(row['redundancy_mean_off_diagonal_cosine']):.4f} | "
            f"{float(row['redundancy_max_cosine']):.4f} | "
            f"{float(row['reconstruction_error']):.4f} | "
            f"{row['converged']} |"
        )

    lines.extend(
        [
            "",
            "## Metric Definitions",
            "",
            f"- Topic diversity@10: unique terms across all topic top-10 lists divided by "
            f"`K * {config.metrics_top_n}`.",
            "- Topic exclusivity@10: for each selected topic-term pair, the topic weight "
            "divided by that term's summed weight across all topics; reported as per-topic "
            "means and a global mean.",
            "- Topic redundancy: pairwise cosine similarity between dense topic-term vectors.",
            "- NPMI coherence@10: binary document co-occurrence over only the union of selected "
            "top terms; zero co-occurrence pairs receive NPMI -1.",
            "- Document-topic concentration: NMF document weights normalized per nonzero row, "
            "with zero rows reported explicitly.",
            "",
        ]
    )
    return "\n".join(lines)


def _package_versions() -> dict[str, str]:
    """Return exact package versions used by this run."""
    packages = ("joblib", "numpy", "scikit-learn", "scipy")
    return {package: metadata.version(package) for package in packages}


def _manifest_payload(
    *,
    config: NmfGridConfig,
    input_paths: Mapping[str, Path],
    input_hashes: Mapping[str, str],
    profile_lineage: Mapping[str, Any],
    stopwords: StopwordSet,
    preprocessing_summary: Mapping[str, Any],
    vectorizer_summary: Mapping[str, Any],
    output_files: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the run manifest payload."""
    non_byte_deterministic = [
        {
            "path": metadata_payload["path"],
            "reason": "joblib serialized bytes may vary across library versions or platforms",
        }
        for key, metadata_payload in sorted(output_files.items())
        if key == "vectorizer" or key.startswith("nmf_model_")
    ]
    config_snapshot = config.to_json()
    canonical_config_text = _json_text(config_snapshot)
    return {
        "canonical_configuration_sha256": hashlib.sha256(
            canonical_config_text.encode("utf-8")
        ).hexdigest(),
        "configuration": config_snapshot,
        "corpus_profile_lineage": dict(profile_lineage),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_paths": {key: str(path) for key, path in sorted(input_paths.items())},
        "input_sha256": dict(sorted(input_hashes.items())),
        "manifest_self_hash_excluded": True,
        "nmf_grid_version": NMF_GRID_VERSION,
        "nmf_settings": config.nmf.to_json(),
        "non_byte_deterministic_serialized_artifacts": non_byte_deterministic,
        "output_files": dict(sorted(output_files.items())),
        "package_versions": {
            **_package_versions(),
            "python": sys.version.split()[0],
        },
        "primary_counts": preprocessing_summary["primary_counts"],
        "reconciliation_checks": {
            "all_output_hashes_match_emitted_files": True,
            "corpus_profile_hashes_match_current_inputs": True,
            "corpus_profile_reconciliation_checks_all_true": True,
            "locked_input_validation_reused": True,
            "no_complete_document_topic_assignments_emitted": True,
            "primary_counts_match_expected": True,
            "zero_alphabetic_token_primary_documents": (
                preprocessing_summary["documents_with_zero_alphabetic_lexical_tokens"] == 0
            ),
            "zero_tfidf_rows": vectorizer_summary["zero_tfidf_rows"] == 0,
        },
        "stopwords": {
            "p0_count": stopwords.p0_count,
            "p1_additions": list(stopwords.p1_additions),
            "p1_count": stopwords.p1_count,
            "selected_count": len(stopwords.words),
            "sha256": stopwords.p0_sha256,
            "variant": stopwords.variant,
        },
        "vectorizer_settings": config.tfidf.to_json(),
    }


def fit_nmf_grid(
    *,
    documents_path: Path = DEFAULT_DOCUMENTS_PATH,
    export_manifest_path: Path = DEFAULT_EXPORT_MANIFEST_PATH,
    corpus_lock_path: Path = DEFAULT_CORPUS_LOCK_PATH,
    profile_manifest_path: Path = DEFAULT_PROFILE_MANIFEST_PATH,
    profile_config_path: Path = DEFAULT_PROFILE_CONFIG_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
    stopwords_path: Path = DEFAULT_STOPWORDS_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    force: bool = False,
) -> dict[str, Any]:
    """Run the focused preprocessing and NMF-grid stage transactionally."""
    config = load_config(config_path)
    _preflight_output_directory(output_dir, force=force)

    export_manifest, _corpus_lock, locked_hashes = _validate_manifest_and_lock(
        documents_path=documents_path,
        export_manifest_path=export_manifest_path,
        corpus_lock_path=corpus_lock_path,
    )
    profile_lineage = _validate_profile_manifest(
        profile_manifest_path=profile_manifest_path,
        profile_config_path=profile_config_path,
        current_hashes=locked_hashes,
        config=config,
    )

    try:
        stopwords = load_stopwords(stopwords_path, variant=config.stopword_variant)
    except TopicPreprocessingError as error:
        raise NmfGridError(str(error)) from error

    _ensure_output_directory(output_dir, force=force)
    paths = _final_output_paths(output_dir, k_values=config.nmf.k_values)
    final_paths = tuple(paths.values())
    _cleanup_transaction_paths(final_paths)
    output_files: dict[str, Mapping[str, Any]] = {}

    input_paths = {
        "config": config_path,
        "corpus_lock": corpus_lock_path,
        "corpus_profile_config": profile_config_path,
        "corpus_profile_manifest": profile_manifest_path,
        "documents": documents_path,
        "export_manifest": export_manifest_path,
        "stopwords": stopwords_path,
    }
    input_hashes = {
        **locked_hashes,
        "corpus_profile_config_sha256": sha256_file(profile_config_path),
        "corpus_profile_manifest_sha256": sha256_file(profile_manifest_path),
        "nmf_grid_config_sha256": sha256_file(config_path),
        "stopwords_sha256": stopwords.p0_sha256,
    }

    try:
        preprocessing = _stage_cleaned_primary_documents(
            documents_path=documents_path,
            cleaned_documents_path=paths["cleaned_documents"],
            export_manifest=export_manifest,
            config=config,
        )
        output_files["cleaned_documents"] = preprocessing.cleaned_documents_metadata
        output_files["preprocessing_summary"] = _write_json_part(
            paths["preprocessing_summary"],
            preprocessing.summary,
        )
        output_files["preprocessing_examples"] = _write_jsonl_part(
            paths["preprocessing_examples"],
            preprocessing.examples,
        )
        vectorization = _fit_tfidf(
            cleaned_documents_path=preprocessing.cleaned_documents_part_path,
            config=config,
            stopwords=stopwords,
        )
        output_files["vectorizer_summary"] = _write_json_part(
            paths["vectorizer_summary"],
            vectorization.summary,
        )
        output_files["vocabulary"] = _write_csv_part(
            paths["vocabulary"],
            fieldnames=("feature_index", "term", "ngram_order", "document_frequency"),
            rows=vectorization.vocabulary_rows,
        )
        output_files["vectorizer"] = _write_joblib_part(
            paths["vectorizer"], vectorization.vectorizer
        )
        model_results = _fit_models(vectorization=vectorization, config=config)
        grid_rows = [result.metric_row for result in model_results]
        output_files["grid_metrics"] = _write_csv_part(
            paths["grid_metrics"],
            fieldnames=(
                "k",
                "runtime_seconds",
                "n_iter",
                "converged",
                "convergence_warnings",
                "reconstruction_error",
                "topic_diversity_top10",
                "mean_topic_exclusivity_top10",
                "redundancy_mean_off_diagonal_cosine",
                "redundancy_max_cosine",
                "mean_npmi_coherence_top10",
                "median_npmi_coherence_top10",
                "min_npmi_coherence_top10",
                "document_topic_mean_dominant_weight",
                "document_topic_mean_normalized_entropy",
                "document_topic_zero_weight_rows",
            ),
            rows=grid_rows,
        )
        output_files["grid_report"] = _write_text_part(
            paths["grid_report"],
            _grid_report(
                config=config,
                preprocessing_summary=preprocessing.summary,
                vectorizer_summary=vectorization.summary,
                grid_rows=grid_rows,
                stopwords=stopwords,
            ),
        )

        for result in model_results:
            key = _k_key(result.k)
            output_files[f"topic_terms_{key}"] = _write_csv_part(
                paths[f"topic_terms_{key}"],
                fieldnames=(
                    "k",
                    "topic_index",
                    "rank",
                    "term",
                    "weight",
                    "feature_index",
                    "npmi_coherence_top10",
                    "mean_exclusivity_top10",
                    "prevalence",
                    "top_terms_top10",
                ),
                rows=result.topic_term_rows,
            )
            output_files[f"representatives_{key}"] = _write_jsonl_part(
                paths[f"representatives_{key}"],
                result.representative_rows,
            )
            output_files[f"nmf_model_{key}"] = _write_joblib_part(
                paths[f"nmf_model_{key}"],
                result.model,
            )

        if not _validate_output_metadata(output_files):
            raise NmfGridError("Staged output hash validation failed.")

        manifest = _manifest_payload(
            config=config,
            input_paths=input_paths,
            input_hashes=input_hashes,
            profile_lineage=profile_lineage,
            stopwords=stopwords,
            preprocessing_summary=preprocessing.summary,
            vectorizer_summary=vectorization.summary,
            output_files=output_files,
        )
        _part_path(paths["run_manifest"]).write_text(
            _json_text(manifest),
            encoding="utf-8",
            newline="",
        )
    except Exception as error:
        _cleanup_transaction_paths(final_paths)

        if isinstance(error, NmfGridError):
            raise

        raise NmfGridError("Could not stage NMF-grid outputs.") from error

    _promote_transaction(final_paths)
    return cast(dict[str, Any], json.loads(paths["run_manifest"].read_text(encoding="utf-8")))
