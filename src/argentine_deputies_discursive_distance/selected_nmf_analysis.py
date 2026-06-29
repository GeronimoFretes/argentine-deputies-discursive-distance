"""Analyze the selected fitted NMF model without refitting topic models."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol, cast

import joblib  # type: ignore[import-untyped]
import numpy as np
import scipy.sparse as sp  # type: ignore[import-untyped]

from .pdf_pipeline import sha256_file

SELECTED_NMF_ANALYSIS_VERSION = "1"

DEFAULT_CONFIG_PATH = Path("config/topic_modeling/selected_nmf_k024_v1.json")
DEFAULT_GRID_INPUT_DIR = Path("data/qa/topic_modeling/nmf_grid_v1")
DEFAULT_OUTPUT_DIR = Path("data/qa/topic_modeling/selected_nmf_k024_v1")

RUN_MANIFEST_FILENAME = "run_manifest.json"
PREPROCESSING_SUMMARY_FILENAME = "preprocessing_summary.json"
VECTORIZER_SUMMARY_FILENAME = "vectorizer_summary.json"
GRID_METRICS_FILENAME = "grid_metrics.csv"
CLEANED_DOCUMENTS_FILENAME = "cleaned_primary_documents.jsonl"
ZERO_TFIDF_DOCUMENTS_FILENAME = "zero_tfidf_documents.jsonl"
VECTORIZER_ARTIFACT_FILENAME = "vectorizer.joblib"

SELECTED_MODEL_MANIFEST_FILENAME = "selected_model_manifest.json"
SELECTED_MODEL_REPORT_FILENAME = "selected_model_report.md"
TOPIC_METADATA_FILENAME = "topic_metadata.csv"
DOCUMENT_TOPIC_WEIGHTS_FILENAME = "document_topic_weights.npz"
DOCUMENT_TOPIC_METADATA_FILENAME = "document_topic_metadata.csv"
DOCUMENT_TOPIC_ASSIGNMENTS_FILENAME = "document_topic_assignments.csv"
SOURCE_TURN_TOPIC_WEIGHTS_FILENAME = "source_turn_topic_weights.npz"
SOURCE_TURN_METADATA_FILENAME = "source_turn_metadata.csv"
SESSION_TOPIC_WEIGHTS_FILENAME = "session_topic_weights.npz"
SESSION_METADATA_FILENAME = "session_metadata.csv"
ANNUAL_TOPIC_PREVALENCE_FILENAME = "annual_topic_prevalence.csv"
PERIOD_TOPIC_PREVALENCE_FILENAME = "period_topic_prevalence.csv"
TEMPORAL_DENOMINATORS_FILENAME = "temporal_denominators.csv"
TOPIC_CHANGE_SUMMARY_FILENAME = "topic_change_summary.csv"
GRID_PREVALENCE_COMPARISON_FILENAME = "grid_prevalence_comparison.csv"

CONFIG_FIELDS = frozenset(
    {
        "aggregation_levels",
        "analysis_version",
        "annual_year_end",
        "annual_year_start",
        "expected_modeled_documents",
        "expected_primary_documents",
        "expected_zero_tfidf_exclusions",
        "float_dtype",
        "grid_prevalence_max_absolute_difference",
        "main_aggregation",
        "selected_k",
        "stopword_variant",
        "temporal_periods",
        "weight_normalization",
    }
)
AGGREGATION_LEVELS = ("document", "source_turn", "session")
ROW_SUM_TOLERANCE = 1e-5
PREVALENCE_SUM_TOLERANCE = 1e-5
NEGATIVE_WEIGHT_TOLERANCE = -1e-8


class SelectedNmfAnalysisError(RuntimeError):
    """Raised when selected NMF analysis cannot complete safely."""


class TemporalMetadata(Protocol):
    """Common temporal fields used by all aggregation metadata rows."""

    year: int
    temporal_period: str


@dataclass(frozen=True, slots=True)
class SelectedNmfAnalysisConfig:
    """Strict configuration for selected-model inference and temporal analysis."""

    analysis_version: str
    selected_k: int
    expected_primary_documents: int
    expected_modeled_documents: int
    expected_zero_tfidf_exclusions: int
    stopword_variant: str
    main_aggregation: str
    aggregation_levels: tuple[str, ...]
    annual_year_start: int
    annual_year_end: int
    temporal_periods: tuple[str, ...]
    weight_normalization: str
    float_dtype: str
    grid_prevalence_max_absolute_difference: float

    def to_json(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready configuration snapshot."""
        return {
            "aggregation_levels": list(self.aggregation_levels),
            "analysis_version": self.analysis_version,
            "annual_year_end": self.annual_year_end,
            "annual_year_start": self.annual_year_start,
            "expected_modeled_documents": self.expected_modeled_documents,
            "expected_primary_documents": self.expected_primary_documents,
            "expected_zero_tfidf_exclusions": self.expected_zero_tfidf_exclusions,
            "float_dtype": self.float_dtype,
            "grid_prevalence_max_absolute_difference": (
                self.grid_prevalence_max_absolute_difference
            ),
            "main_aggregation": self.main_aggregation,
            "selected_k": self.selected_k,
            "stopword_variant": self.stopword_variant,
            "temporal_periods": list(self.temporal_periods),
            "weight_normalization": self.weight_normalization,
        }


@dataclass(frozen=True, slots=True)
class Period:
    """Inclusive year range encoded by a configured temporal period."""

    label: str
    start_year: int
    end_year: int


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    """Metadata retained for every modelled document row."""

    document_id: str
    original_row_index: int
    source_record_id: str
    turn_index: int
    chunk_index: int
    source_turn_key: str
    year: int
    temporal_period: str
    session_category: str
    speaker_family: str
    word_count: int


@dataclass(frozen=True, slots=True)
class SourceTurnMetadata:
    """Metadata retained for every source-turn row."""

    source_turn_row_index: int
    source_turn_key: str
    source_record_id: str
    turn_index: int
    first_original_row_index: int
    modeled_document_count: int
    chunk_count: int
    year: int
    temporal_period: str
    session_category: str
    speaker_family: str
    word_count: int


@dataclass(frozen=True, slots=True)
class SessionMetadata:
    """Metadata retained for every session row."""

    session_row_index: int
    source_record_id: str
    first_source_turn_row_index: int
    source_turn_count: int
    modeled_document_count: int
    year: int
    temporal_period: str
    session_category: str
    word_count: int


@dataclass(frozen=True, slots=True)
class TopicMetadata:
    """Grid-time topic metadata for the selected K."""

    topic_index: int
    top_terms_top20: tuple[str, ...]
    top_terms_top10: tuple[str, ...]
    grid_npmi_coherence_top10: float
    grid_mean_exclusivity_top10: float
    grid_overall_prevalence: float


@dataclass(frozen=True, slots=True)
class SelectedMatrixPayload:
    """Document-topic matrix and row metadata reconstructed from selected artifacts."""

    document_topic: np.ndarray[Any, Any]
    document_metadata: tuple[DocumentMetadata, ...]
    zero_ledger_rows: tuple[dict[str, Any], ...]
    transformed_zero_rows: tuple[tuple[int, str], ...]
    raw_weight_row_sum_minimum: float
    raw_weight_row_sum_maximum: float
    normalized_row_sum_max_abs_deviation: float
    transformed_tfidf_shape_before_exclusion: tuple[int, int]
    transformed_tfidf_shape_after_exclusion: tuple[int, int]


@dataclass(frozen=True, slots=True)
class AggregationPayload:
    """All row-normalized topic matrices and matching metadata."""

    source_turn_topic: np.ndarray[Any, Any]
    source_turn_metadata: tuple[SourceTurnMetadata, ...]
    session_topic: np.ndarray[Any, Any]
    session_metadata: tuple[SessionMetadata, ...]


@dataclass(frozen=True, slots=True)
class PrevalencePayload:
    """Long-format prevalence rows and lookup tables."""

    annual_rows: tuple[dict[str, Any], ...]
    period_rows: tuple[dict[str, Any], ...]
    annual_source_turn_by_topic: dict[tuple[int, int], float]
    period_source_turn_by_topic: dict[tuple[str, int], float]


def _json_text(payload: Mapping[str, Any]) -> str:
    """Return deterministic UTF-8 JSON text."""
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _json_compact(payload: object) -> str:
    """Return compact deterministic JSON for CSV cells."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _format_float(value: float) -> str:
    """Return a stable decimal representation for CSV outputs."""
    return f"{float(value):.10f}"


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    """Read a UTF-8 JSON object."""
    if not path.is_file():
        raise SelectedNmfAnalysisError(f"{label} does not exist: {path}")

    try:
        payload: object = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise SelectedNmfAnalysisError(f"Could not read {label}: {path}") from error

    if not isinstance(payload, dict):
        raise SelectedNmfAnalysisError(f"Expected {label} to contain a JSON object: {path}")

    return {str(key): value for key, value in payload.items()}


def _safe_int(value: object, *, field_name: str) -> int:
    """Return a strict JSON integer, rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise SelectedNmfAnalysisError(f"Invalid integer for {field_name}: {value!r}")

    return value


def _safe_float(value: object, *, field_name: str) -> float:
    """Return a strict JSON number, rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise SelectedNmfAnalysisError(f"Invalid number for {field_name}: {value!r}")

    parsed = float(value)

    if not math.isfinite(parsed):
        raise SelectedNmfAnalysisError(f"Invalid finite number for {field_name}: {value!r}")

    return parsed


def _required_string(payload: Mapping[str, Any], field_name: str) -> str:
    """Return a required nonempty string."""
    value = payload.get(field_name)

    if not isinstance(value, str) or not value:
        raise SelectedNmfAnalysisError(f"Missing or invalid {field_name}.")

    return value


def _string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    """Return a nonempty tuple of strings."""
    if not isinstance(value, list) or not value:
        raise SelectedNmfAnalysisError(f"Invalid string list for {field_name}.")

    values: list[str] = []

    for item in value:
        if not isinstance(item, str) or not item:
            raise SelectedNmfAnalysisError(f"Invalid string in {field_name}: {item!r}")

        values.append(item)

    return tuple(values)


def _validate_exact_fields(
    payload: Mapping[str, Any],
    *,
    expected: frozenset[str],
    label: str,
) -> None:
    """Require exactly the expected JSON-object fields."""
    missing = expected - set(payload)
    unexpected = set(payload) - expected

    if missing:
        raise SelectedNmfAnalysisError(f"{label} is missing fields: {sorted(missing)}")

    if unexpected:
        raise SelectedNmfAnalysisError(f"{label} has unsupported fields: {sorted(unexpected)}")


def load_config(path: Path) -> SelectedNmfAnalysisConfig:
    """Load and strictly validate the selected-NMF analysis configuration."""
    payload = _read_json_object(path, label="selected NMF analysis configuration")
    _validate_exact_fields(payload, expected=CONFIG_FIELDS, label="selected NMF configuration")

    config = SelectedNmfAnalysisConfig(
        analysis_version=_required_string(payload, "analysis_version"),
        selected_k=_safe_int(payload.get("selected_k"), field_name="selected_k"),
        expected_primary_documents=_safe_int(
            payload.get("expected_primary_documents"),
            field_name="expected_primary_documents",
        ),
        expected_modeled_documents=_safe_int(
            payload.get("expected_modeled_documents"),
            field_name="expected_modeled_documents",
        ),
        expected_zero_tfidf_exclusions=_safe_int(
            payload.get("expected_zero_tfidf_exclusions"),
            field_name="expected_zero_tfidf_exclusions",
        ),
        stopword_variant=_required_string(payload, "stopword_variant"),
        main_aggregation=_required_string(payload, "main_aggregation"),
        aggregation_levels=_string_tuple(
            payload.get("aggregation_levels"),
            field_name="aggregation_levels",
        ),
        annual_year_start=_safe_int(
            payload.get("annual_year_start"),
            field_name="annual_year_start",
        ),
        annual_year_end=_safe_int(payload.get("annual_year_end"), field_name="annual_year_end"),
        temporal_periods=_string_tuple(payload.get("temporal_periods"), field_name="periods"),
        weight_normalization=_required_string(payload, "weight_normalization"),
        float_dtype=_required_string(payload, "float_dtype"),
        grid_prevalence_max_absolute_difference=_safe_float(
            payload.get("grid_prevalence_max_absolute_difference"),
            field_name="grid_prevalence_max_absolute_difference",
        ),
    )
    _validate_config_values(config)
    return config


def _validate_config_values(config: SelectedNmfAnalysisConfig) -> None:
    """Validate cross-field constraints in the selected-NMF config."""
    if config.analysis_version != SELECTED_NMF_ANALYSIS_VERSION:
        raise SelectedNmfAnalysisError(f"Unsupported analysis_version: {config.analysis_version}")

    if config.selected_k < 1:
        raise SelectedNmfAnalysisError("selected_k must be positive.")

    if config.expected_primary_documents < 1:
        raise SelectedNmfAnalysisError("expected_primary_documents must be positive.")

    if config.expected_modeled_documents < 1:
        raise SelectedNmfAnalysisError("expected_modeled_documents must be positive.")

    if config.expected_zero_tfidf_exclusions < 0:
        raise SelectedNmfAnalysisError("expected_zero_tfidf_exclusions cannot be negative.")

    if (
        config.expected_modeled_documents + config.expected_zero_tfidf_exclusions
        != config.expected_primary_documents
    ):
        raise SelectedNmfAnalysisError(
            "expected_modeled_documents plus expected_zero_tfidf_exclusions must equal "
            "expected_primary_documents."
        )

    if config.stopword_variant != "P1":
        raise SelectedNmfAnalysisError("stopword_variant must be P1 for this selected model.")

    if config.main_aggregation != "source_turn":
        raise SelectedNmfAnalysisError("main_aggregation must be source_turn.")

    if config.aggregation_levels != AGGREGATION_LEVELS:
        raise SelectedNmfAnalysisError(
            f"aggregation_levels must be exactly {list(AGGREGATION_LEVELS)}."
        )

    if config.weight_normalization != "row_sum_one":
        raise SelectedNmfAnalysisError("weight_normalization must be row_sum_one.")

    if config.float_dtype != "float32":
        raise SelectedNmfAnalysisError("float_dtype must be float32.")

    if config.annual_year_start > config.annual_year_end:
        raise SelectedNmfAnalysisError("annual_year_start cannot exceed annual_year_end.")

    if config.grid_prevalence_max_absolute_difference < 0:
        raise SelectedNmfAnalysisError(
            "grid_prevalence_max_absolute_difference cannot be negative."
        )

    periods = _parse_periods(config.temporal_periods)

    if periods[0].start_year != config.annual_year_start:
        raise SelectedNmfAnalysisError("temporal_periods must start at annual_year_start.")

    if periods[-1].end_year != config.annual_year_end:
        raise SelectedNmfAnalysisError("temporal_periods must end at annual_year_end.")

    for left, right in zip(periods[:-1], periods[1:], strict=True):
        if left.end_year + 1 != right.start_year:
            raise SelectedNmfAnalysisError("temporal_periods must be contiguous and ordered.")


def _parse_periods(period_labels: Sequence[str]) -> tuple[Period, ...]:
    """Parse configured period labels of the form YYYY-YYYY."""
    periods: list[Period] = []

    for label in period_labels:
        parts = label.split("-")

        if len(parts) != 2 or not all(part.isdigit() and len(part) == 4 for part in parts):
            raise SelectedNmfAnalysisError(f"Invalid temporal period label: {label}")

        start_year = int(parts[0])
        end_year = int(parts[1])

        if start_year > end_year:
            raise SelectedNmfAnalysisError(f"Invalid descending temporal period: {label}")

        periods.append(Period(label=label, start_year=start_year, end_year=end_year))

    if len({period.label for period in periods}) != len(periods):
        raise SelectedNmfAnalysisError("temporal_periods must be unique.")

    return tuple(periods)


def _period_for_year(year: int, periods: Sequence[Period]) -> str:
    """Return the configured period label containing a year."""
    for period in periods:
        if period.start_year <= year <= period.end_year:
            return period.label

    raise SelectedNmfAnalysisError(f"Year is outside configured temporal periods: {year}")


def _k_key(k_value: int) -> str:
    """Return a stable K label for filenames and manifest keys."""
    return f"k{k_value:03d}"


def _selected_model_filename(k_value: int) -> str:
    """Return the saved model filename for a selected K."""
    return f"nmf_{_k_key(k_value)}.joblib"


def _selected_topic_terms_filename(k_value: int) -> str:
    """Return the topic-term filename for a selected K."""
    return f"topic_terms_{_k_key(k_value)}.csv"


def _required_grid_artifacts(config: SelectedNmfAnalysisConfig) -> dict[str, str]:
    """Return required grid artifact manifest keys and filenames."""
    key = _k_key(config.selected_k)
    return {
        "cleaned_documents": CLEANED_DOCUMENTS_FILENAME,
        "grid_metrics": GRID_METRICS_FILENAME,
        f"nmf_model_{key}": _selected_model_filename(config.selected_k),
        "preprocessing_summary": PREPROCESSING_SUMMARY_FILENAME,
        f"topic_terms_{key}": _selected_topic_terms_filename(config.selected_k),
        "vectorizer": VECTORIZER_ARTIFACT_FILENAME,
        "vectorizer_summary": VECTORIZER_SUMMARY_FILENAME,
        "zero_tfidf_documents": ZERO_TFIDF_DOCUMENTS_FILENAME,
    }


def _validate_grid_artifact_metadata(
    *,
    grid_input_dir: Path,
    manifest: Mapping[str, Any],
    required_artifacts: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Validate manifest-recorded hashes and byte sizes for required artifacts."""
    output_files = manifest.get("output_files")

    if not isinstance(output_files, dict):
        raise SelectedNmfAnalysisError("Grid run_manifest.json is missing output_files.")

    validated: dict[str, dict[str, Any]] = {}

    for artifact_key, filename in sorted(required_artifacts.items()):
        metadata_payload = output_files.get(artifact_key)

        if not isinstance(metadata_payload, dict):
            raise SelectedNmfAnalysisError(
                f"Grid manifest is missing metadata for required artifact: {artifact_key}"
            )

        expected_sha = metadata_payload.get("sha256")
        expected_size = metadata_payload.get("size_bytes")

        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise SelectedNmfAnalysisError(f"Invalid SHA-256 metadata for {artifact_key}.")

        if isinstance(expected_size, bool) or not isinstance(expected_size, int):
            raise SelectedNmfAnalysisError(f"Invalid size metadata for {artifact_key}.")

        artifact_path = grid_input_dir / filename

        if not artifact_path.is_file():
            raise SelectedNmfAnalysisError(
                f"Required grid artifact does not exist: {artifact_path}"
            )

        actual_size = artifact_path.stat().st_size
        actual_sha = sha256_file(artifact_path)

        if actual_size != expected_size:
            raise SelectedNmfAnalysisError(
                f"Grid artifact byte-size mismatch for {artifact_key}: "
                f"{actual_size} != {expected_size}"
            )

        if actual_sha != expected_sha:
            raise SelectedNmfAnalysisError(
                f"Grid artifact SHA-256 mismatch for {artifact_key}: {actual_sha} != {expected_sha}"
            )

        validated[artifact_key] = {
            "path": str(artifact_path),
            "sha256": actual_sha,
            "size_bytes": actual_size,
        }

        if "record_count" in metadata_payload:
            validated[artifact_key]["record_count"] = metadata_payload["record_count"]

    return validated


def _require_grid_reconciliation_checks(manifest: Mapping[str, Any]) -> None:
    """Require every grid reconciliation check to be boolean true."""
    checks = manifest.get("reconciliation_checks")

    if not isinstance(checks, dict) or not checks:
        raise SelectedNmfAnalysisError("Grid manifest has no reconciliation_checks object.")

    failing = [
        str(key)
        for key, value in checks.items()
        if not isinstance(value, bool) or value is not True
    ]

    if failing:
        raise SelectedNmfAnalysisError(
            f"Grid reconciliation checks are not all true: {sorted(failing)}"
        )


def _read_csv_rows(
    path: Path,
    *,
    required_columns: frozenset[str],
    label: str,
) -> tuple[dict[str, str], ...]:
    """Read CSV rows and reject missing, extra, or malformed records."""
    if not path.is_file():
        raise SelectedNmfAnalysisError(f"{label} does not exist: {path}")

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as input_file:
            reader = csv.DictReader(input_file)

            if reader.fieldnames is None:
                raise SelectedNmfAnalysisError(f"{label} has no header: {path}")

            missing = required_columns - set(reader.fieldnames)

            if missing:
                raise SelectedNmfAnalysisError(f"{label} is missing columns: {sorted(missing)}")

            rows: list[dict[str, str]] = []

            for row_number, row in enumerate(reader, start=2):
                if None in row:
                    raise SelectedNmfAnalysisError(
                        f"{label} has extra columns at row {row_number}: {path}"
                    )

                parsed: dict[str, str] = {}

                for column in reader.fieldnames:
                    value = row.get(column)

                    if value is None:
                        raise SelectedNmfAnalysisError(
                            f"{label} has missing value at row {row_number}: {column}"
                        )

                    parsed[str(column)] = value

                rows.append(parsed)
    except OSError as error:
        raise SelectedNmfAnalysisError(f"Could not read {label}: {path}") from error

    return tuple(rows)


def _parse_int_text(value: str, *, field_name: str) -> int:
    """Parse an integer CSV cell."""
    try:
        return int(value)
    except ValueError as error:
        raise SelectedNmfAnalysisError(f"Invalid integer for {field_name}: {value!r}") from error


def _parse_float_text(value: str, *, field_name: str) -> float:
    """Parse a finite float CSV cell."""
    try:
        parsed = float(value)
    except ValueError as error:
        raise SelectedNmfAnalysisError(f"Invalid float for {field_name}: {value!r}") from error

    if not math.isfinite(parsed):
        raise SelectedNmfAnalysisError(f"Invalid finite float for {field_name}: {value!r}")

    return parsed


def _parse_bool_text(value: str, *, field_name: str) -> bool:
    """Parse a strict True/False CSV cell."""
    if value == "True":
        return True

    if value == "False":
        return False

    raise SelectedNmfAnalysisError(f"Invalid boolean for {field_name}: {value!r}")


def _validate_grid_summaries(
    *,
    manifest: Mapping[str, Any],
    preprocessing_summary: Mapping[str, Any],
    vectorizer_summary: Mapping[str, Any],
    grid_metrics_rows: Sequence[Mapping[str, str]],
    config: SelectedNmfAnalysisConfig,
) -> dict[str, Any]:
    """Validate grid-level counts, selected K, and convergence metadata."""
    primary_counts = preprocessing_summary.get("primary_counts")

    if not isinstance(primary_counts, dict):
        raise SelectedNmfAnalysisError("preprocessing_summary is missing primary_counts.")

    primary_documents = _safe_int(primary_counts.get("documents"), field_name="primary documents")
    modeled_documents = _safe_int(
        vectorizer_summary.get("modeled_document_count"),
        field_name="modeled_document_count",
    )
    zero_tfidf_exclusions = _safe_int(
        vectorizer_summary.get("zero_tfidf_rows_excluded"),
        field_name="zero_tfidf_rows_excluded",
    )
    input_documents = _safe_int(
        vectorizer_summary.get("input_document_count"),
        field_name="input_document_count",
    )

    if primary_documents != config.expected_primary_documents:
        raise SelectedNmfAnalysisError(
            f"Primary document count mismatch: {primary_documents} != "
            f"{config.expected_primary_documents}"
        )

    if input_documents != config.expected_primary_documents:
        raise SelectedNmfAnalysisError(
            f"Vectorizer input document count mismatch: {input_documents} != "
            f"{config.expected_primary_documents}"
        )

    if modeled_documents != config.expected_modeled_documents:
        raise SelectedNmfAnalysisError(
            f"Modeled document count mismatch: {modeled_documents} != "
            f"{config.expected_modeled_documents}"
        )

    if zero_tfidf_exclusions != config.expected_zero_tfidf_exclusions:
        raise SelectedNmfAnalysisError(
            f"Zero TF-IDF exclusion count mismatch: {zero_tfidf_exclusions} != "
            f"{config.expected_zero_tfidf_exclusions}"
        )

    if modeled_documents + zero_tfidf_exclusions != primary_documents:
        raise SelectedNmfAnalysisError("Modeled plus excluded documents do not equal primary.")

    manifest_modeled = _safe_int(
        manifest.get("modeled_document_count"),
        field_name="manifest modeled_document_count",
    )

    if manifest_modeled != modeled_documents:
        raise SelectedNmfAnalysisError("Grid manifest modeled_document_count mismatch.")

    manifest_primary = manifest.get("primary_counts")

    if not isinstance(manifest_primary, dict):
        raise SelectedNmfAnalysisError("Grid manifest is missing primary_counts.")

    if _safe_int(manifest_primary.get("documents"), field_name="manifest primary documents") != (
        primary_documents
    ):
        raise SelectedNmfAnalysisError("Grid manifest primary document count mismatch.")

    stopword_variant = vectorizer_summary.get("stopword_variant")

    if stopword_variant != config.stopword_variant:
        raise SelectedNmfAnalysisError(
            f"Stopword variant mismatch: {stopword_variant!r} != {config.stopword_variant!r}"
        )

    manifest_stopwords = manifest.get("stopwords")

    if not isinstance(manifest_stopwords, dict) or manifest_stopwords.get("variant") != (
        config.stopword_variant
    ):
        raise SelectedNmfAnalysisError("Grid manifest stopword variant mismatch.")

    selected_rows = [
        row
        for row in grid_metrics_rows
        if _parse_int_text(row["k"], field_name="grid_metrics.k") == config.selected_k
    ]

    if len(selected_rows) != 1:
        raise SelectedNmfAnalysisError(
            f"Selected K={config.selected_k} is not present exactly once in grid_metrics.csv."
        )

    selected_grid_row = selected_rows[0]
    converged = _parse_bool_text(selected_grid_row["converged"], field_name="converged")

    if not converged:
        raise SelectedNmfAnalysisError(f"Selected K={config.selected_k} did not converge.")

    return {
        "grid_metrics_selected_row": dict(selected_grid_row),
        "input_documents": input_documents,
        "modeled_documents": modeled_documents,
        "primary_documents": primary_documents,
        "zero_tfidf_exclusions": zero_tfidf_exclusions,
    }


def _load_joblib_artifact(path: Path, *, label: str) -> Any:
    """Load a joblib artifact with a clear error boundary."""
    try:
        return joblib.load(path)
    except Exception as error:
        raise SelectedNmfAnalysisError(f"Could not load {label}: {path}") from error


def _shape_list(value: object, *, field_name: str) -> tuple[int, int]:
    """Return a strict two-integer shape list from JSON."""
    if not isinstance(value, list) or len(value) != 2:
        raise SelectedNmfAnalysisError(f"Invalid matrix shape for {field_name}: {value!r}")

    return (
        _safe_int(value[0], field_name=f"{field_name}[0]"),
        _safe_int(value[1], field_name=f"{field_name}[1]"),
    )


def _validate_vectorizer_and_model(
    *,
    vectorizer: Any,
    model: Any,
    vectorizer_summary: Mapping[str, Any],
    config: SelectedNmfAnalysisConfig,
) -> tuple[int, int]:
    """Validate selected model dimensions against vectorizer metadata."""
    components = getattr(model, "components_", None)

    if not isinstance(components, np.ndarray):
        raise SelectedNmfAnalysisError("Selected NMF model has no components_ array.")

    if components.ndim != 2:
        raise SelectedNmfAnalysisError("Selected NMF components_ must be two-dimensional.")

    if int(components.shape[0]) != config.selected_k:
        raise SelectedNmfAnalysisError(
            f"Selected model component count mismatch: {components.shape[0]} != {config.selected_k}"
        )

    try:
        feature_names = vectorizer.get_feature_names_out()
    except Exception as error:
        raise SelectedNmfAnalysisError("Vectorizer does not expose feature names.") from error

    feature_count = int(len(feature_names))
    component_width = int(components.shape[1])

    if feature_count != component_width:
        raise SelectedNmfAnalysisError(
            f"Vectorizer feature count and model component width differ: "
            f"{feature_count} != {component_width}"
        )

    matrix_shape = _shape_list(vectorizer_summary.get("matrix_shape"), field_name="matrix_shape")
    pre_shape = _shape_list(
        vectorizer_summary.get("matrix_shape_before_zero_row_exclusion"),
        field_name="matrix_shape_before_zero_row_exclusion",
    )
    vocabulary_size = _safe_int(
        vectorizer_summary.get("vocabulary_size"),
        field_name="vocabulary_size",
    )

    for width, label in (
        (matrix_shape[1], "matrix_shape"),
        (pre_shape[1], "matrix_shape_before_zero_row_exclusion"),
        (vocabulary_size, "vocabulary_size"),
    ):
        if width != feature_count:
            raise SelectedNmfAnalysisError(
                f"Vectorizer summary {label} width mismatch: {width} != {feature_count}"
            )

    return feature_count, component_width


def _iter_jsonl_objects(path: Path, *, label: str) -> Iterator[dict[str, Any]]:
    """Stream JSONL objects and fail on blank or malformed records."""
    try:
        with path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    raise SelectedNmfAnalysisError(f"Blank {label} record at line {line_number}.")

                payload = json.loads(line)

                if not isinstance(payload, dict):
                    raise SelectedNmfAnalysisError(
                        f"{label} record is not an object at line {line_number}."
                    )

                yield {str(key): value for key, value in payload.items()}
    except json.JSONDecodeError as error:
        raise SelectedNmfAnalysisError(f"Malformed {label} JSONL at line {line_number}.") from error
    except OSError as error:
        raise SelectedNmfAnalysisError(f"Could not read {label}: {path}") from error


def _required_record_string(record: Mapping[str, Any], field_name: str) -> str:
    """Return a required nonempty string from a JSONL record."""
    value = record.get(field_name)

    if not isinstance(value, str) or not value:
        raise SelectedNmfAnalysisError(f"Cleaned document record has invalid {field_name}.")

    return value


def _document_source_turn_key(source_record_id: str, turn_index: int) -> str:
    """Return the source-turn key used by the grid stage."""
    return f"{source_record_id}::turn_{turn_index:06d}"


def _cleaned_texts_and_metadata(
    path: Path,
    *,
    metadata_rows: list[DocumentMetadata],
) -> Iterator[str]:
    """Yield cleaned text while retaining compact row metadata."""
    for row_index, record in enumerate(_iter_jsonl_objects(path, label="cleaned documents")):
        cleaned_text = _required_record_string(record, "cleaned_text")
        source_record_id = _required_record_string(record, "source_record_id")
        turn_index = _safe_int(record.get("turn_index"), field_name="turn_index")
        year = _safe_int(record.get("year"), field_name="year")
        temporal_period = _required_record_string(record, "temporal_period")
        metadata_rows.append(
            DocumentMetadata(
                document_id=_required_record_string(record, "document_id"),
                original_row_index=row_index,
                source_record_id=source_record_id,
                turn_index=turn_index,
                chunk_index=_safe_int(record.get("chunk_index"), field_name="chunk_index"),
                source_turn_key=_document_source_turn_key(source_record_id, turn_index),
                year=year,
                temporal_period=temporal_period,
                session_category=_required_record_string(record, "session_category"),
                speaker_family=_required_record_string(record, "speaker_family"),
                word_count=_safe_int(record.get("word_count"), field_name="word_count"),
            )
        )
        yield cleaned_text


def _read_zero_ledger(path: Path) -> tuple[dict[str, Any], ...]:
    """Read the zero TF-IDF ledger with strict required fields."""
    rows: list[dict[str, Any]] = []

    for row in _iter_jsonl_objects(path, label="zero TF-IDF ledger"):
        rows.append(
            {
                **row,
                "document_id": _required_record_string(row, "document_id"),
                "original_row_index": _safe_int(
                    row.get("original_row_index"),
                    field_name="original_row_index",
                ),
                "source_record_id": _required_record_string(row, "source_record_id"),
                "turn_index": _safe_int(row.get("turn_index"), field_name="turn_index"),
                "chunk_index": _safe_int(row.get("chunk_index"), field_name="chunk_index"),
                "year": _safe_int(row.get("year"), field_name="year"),
                "temporal_period": _required_record_string(row, "temporal_period"),
                "word_count": _safe_int(row.get("word_count"), field_name="word_count"),
            }
        )

    return tuple(rows)


def _reconstruct_selected_document_matrix(
    *,
    cleaned_documents_path: Path,
    zero_tfidf_documents_path: Path,
    vectorizer: Any,
    model: Any,
    vectorizer_summary: Mapping[str, Any],
    config: SelectedNmfAnalysisConfig,
) -> SelectedMatrixPayload:
    """Transform cleaned texts with saved artifacts and normalize document-topic rows."""
    metadata_rows: list[DocumentMetadata] = []

    try:
        tfidf_matrix = vectorizer.transform(
            _cleaned_texts_and_metadata(cleaned_documents_path, metadata_rows=metadata_rows)
        )
    except SelectedNmfAnalysisError:
        raise
    except Exception as error:
        raise SelectedNmfAnalysisError("Saved vectorizer transform failed.") from error

    if not sp.issparse(tfidf_matrix):
        raise SelectedNmfAnalysisError("Saved vectorizer returned a dense TF-IDF matrix.")

    tfidf_matrix = tfidf_matrix.tocsr()
    expected_pre_shape = _shape_list(
        vectorizer_summary.get("matrix_shape_before_zero_row_exclusion"),
        field_name="matrix_shape_before_zero_row_exclusion",
    )

    if tuple(int(value) for value in tfidf_matrix.shape) != expected_pre_shape:
        raise SelectedNmfAnalysisError(
            f"Transformed TF-IDF pre-exclusion shape mismatch: {tfidf_matrix.shape} != "
            f"{expected_pre_shape}"
        )

    if tfidf_matrix.shape[0] != len(metadata_rows):
        raise SelectedNmfAnalysisError("Transformed TF-IDF rows do not match metadata rows.")

    zero_row_indices = np.flatnonzero(np.diff(tfidf_matrix.indptr) == 0)
    zero_ledger_rows = _read_zero_ledger(zero_tfidf_documents_path)
    ledger_pairs = tuple(
        (int(row["original_row_index"]), str(row["document_id"])) for row in zero_ledger_rows
    )
    transformed_pairs = tuple(
        (int(row_index), metadata_rows[int(row_index)].document_id)
        for row_index in zero_row_indices
    )

    if transformed_pairs != ledger_pairs:
        raise SelectedNmfAnalysisError(
            "Transformed zero TF-IDF rows do not exactly match zero_tfidf_documents.jsonl."
        )

    if len(zero_ledger_rows) != config.expected_zero_tfidf_exclusions:
        raise SelectedNmfAnalysisError("Zero TF-IDF ledger row count mismatch.")

    keep_mask = np.ones(tfidf_matrix.shape[0], dtype=bool)
    keep_mask[zero_row_indices] = False
    filtered_metadata = tuple(
        row for row_index, row in enumerate(metadata_rows) if bool(keep_mask[row_index])
    )
    tfidf_matrix = tfidf_matrix[keep_mask].tocsr()
    expected_model_shape = _shape_list(
        vectorizer_summary.get("matrix_shape"),
        field_name="matrix_shape",
    )

    if tuple(int(value) for value in tfidf_matrix.shape) != expected_model_shape:
        raise SelectedNmfAnalysisError(
            f"Filtered TF-IDF shape mismatch: {tfidf_matrix.shape} != {expected_model_shape}"
        )

    if tfidf_matrix.shape[0] != len(filtered_metadata):
        raise SelectedNmfAnalysisError("Filtered TF-IDF rows do not match filtered metadata rows.")

    if tfidf_matrix.shape[0] != config.expected_modeled_documents:
        raise SelectedNmfAnalysisError("Filtered TF-IDF row count does not match configuration.")

    try:
        raw_document_topic = model.transform(tfidf_matrix)
    except Exception as error:
        raise SelectedNmfAnalysisError("Saved NMF model transform failed.") from error

    del tfidf_matrix

    normalized, row_sum_min, row_sum_max, max_deviation = _normalize_document_topic_rows(
        raw_document_topic,
        expected_k=config.selected_k,
    )

    return SelectedMatrixPayload(
        document_topic=normalized,
        document_metadata=filtered_metadata,
        zero_ledger_rows=zero_ledger_rows,
        transformed_zero_rows=transformed_pairs,
        raw_weight_row_sum_minimum=row_sum_min,
        raw_weight_row_sum_maximum=row_sum_max,
        normalized_row_sum_max_abs_deviation=max_deviation,
        transformed_tfidf_shape_before_exclusion=expected_pre_shape,
        transformed_tfidf_shape_after_exclusion=expected_model_shape,
    )


def _normalize_document_topic_rows(
    raw_document_topic: Any,
    *,
    expected_k: int,
) -> tuple[np.ndarray[Any, Any], float, float, float]:
    """Normalize nonnegative document-topic rows to sum to one."""
    weights = np.asarray(raw_document_topic)

    if weights.ndim != 2 or int(weights.shape[1]) != expected_k:
        raise SelectedNmfAnalysisError(
            f"Document-topic matrix shape mismatch: {weights.shape} != (*, {expected_k})"
        )

    if not np.all(np.isfinite(weights)):
        raise SelectedNmfAnalysisError("Document-topic matrix contains non-finite weights.")

    if weights.size and float(np.min(weights)) < NEGATIVE_WEIGHT_TOLERANCE:
        raise SelectedNmfAnalysisError("Document-topic matrix contains negative weights.")

    weights = np.maximum(weights, 0.0).astype(np.float32, copy=False)
    row_sums = weights.sum(axis=1, dtype=np.float64)

    if np.any(row_sums <= 0.0):
        first_bad = int(np.flatnonzero(row_sums <= 0.0)[0])
        raise SelectedNmfAnalysisError(f"Document-topic row has nonpositive sum: {first_bad}")

    normalized = (weights / row_sums[:, np.newaxis]).astype(np.float32, copy=False)
    normalized_row_sums = normalized.sum(axis=1, dtype=np.float64)
    max_deviation = (
        float(np.max(np.abs(normalized_row_sums - 1.0))) if normalized_row_sums.size else 0.0
    )

    if max_deviation > ROW_SUM_TOLERANCE:
        raise SelectedNmfAnalysisError(
            f"Normalized document-topic row sums exceed tolerance: {max_deviation:.12g}"
        )

    return normalized, float(np.min(row_sums)), float(np.max(row_sums)), max_deviation


def _validate_matrix_row_sums(
    matrix: np.ndarray[Any, Any],
    *,
    label: str,
) -> float:
    """Validate each row in a normalized topic matrix sums to one."""
    row_sums = matrix.sum(axis=1, dtype=np.float64)

    if np.any(row_sums <= 0.0):
        raise SelectedNmfAnalysisError(f"{label} contains nonpositive row sums.")

    max_deviation = float(np.max(np.abs(row_sums - 1.0))) if row_sums.size else 0.0

    if max_deviation > ROW_SUM_TOLERANCE:
        raise SelectedNmfAnalysisError(f"{label} row sums exceed tolerance: {max_deviation:.12g}")

    return max_deviation


def _load_topic_metadata(
    *,
    topic_terms_path: Path,
    selected_k: int,
) -> tuple[TopicMetadata, ...]:
    """Load one topic metadata row per selected topic."""
    rows = _read_csv_rows(
        topic_terms_path,
        required_columns=frozenset(
            {
                "k",
                "topic_index",
                "rank",
                "term",
                "npmi_coherence_top10",
                "mean_exclusivity_top10",
                "prevalence",
            }
        ),
        label="selected topic terms",
    )
    by_topic: dict[int, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        k_value = _parse_int_text(row["k"], field_name="topic_terms.k")

        if k_value != selected_k:
            raise SelectedNmfAnalysisError(
                f"topic_terms file contains K={k_value}, expected K={selected_k}."
            )

        topic_index = _parse_int_text(row["topic_index"], field_name="topic_index")

        if not 0 <= topic_index < selected_k:
            raise SelectedNmfAnalysisError(f"Invalid topic_index: {topic_index}")

        by_topic[topic_index].append(row)

    if set(by_topic) != set(range(selected_k)):
        raise SelectedNmfAnalysisError("topic_terms file does not contain every selected topic.")

    topic_metadata: list[TopicMetadata] = []

    for topic_index in range(selected_k):
        topic_rows = sorted(
            by_topic[topic_index],
            key=lambda item: _parse_int_text(item["rank"], field_name="rank"),
        )
        ranks = [_parse_int_text(row["rank"], field_name="rank") for row in topic_rows]

        if ranks != list(range(1, len(ranks) + 1)):
            raise SelectedNmfAnalysisError(f"Topic {topic_index} has noncontiguous ranks.")

        terms = tuple(row["term"] for row in topic_rows)

        if len(terms) < 10:
            raise SelectedNmfAnalysisError(f"Topic {topic_index} has fewer than 10 top terms.")

        coherence_values = {
            _parse_float_text(row["npmi_coherence_top10"], field_name="npmi_coherence_top10")
            for row in topic_rows
        }
        exclusivity_values = {
            _parse_float_text(
                row["mean_exclusivity_top10"],
                field_name="mean_exclusivity_top10",
            )
            for row in topic_rows
        }
        prevalence_values = {
            _parse_float_text(row["prevalence"], field_name="prevalence") for row in topic_rows
        }

        if (
            len(coherence_values) != 1
            or len(exclusivity_values) != 1
            or len(prevalence_values) != 1
        ):
            raise SelectedNmfAnalysisError(
                f"Topic {topic_index} has inconsistent repeated grid metrics."
            )

        topic_metadata.append(
            TopicMetadata(
                topic_index=topic_index,
                top_terms_top20=terms[:20],
                top_terms_top10=terms[:10],
                grid_npmi_coherence_top10=next(iter(coherence_values)),
                grid_mean_exclusivity_top10=next(iter(exclusivity_values)),
                grid_overall_prevalence=next(iter(prevalence_values)),
            )
        )

    return tuple(topic_metadata)


def _grid_prevalence_comparison_rows(
    *,
    document_topic: np.ndarray[Any, Any],
    topic_metadata: Sequence[TopicMetadata],
    threshold: float,
) -> tuple[tuple[dict[str, Any], ...], float]:
    """Compare transform-time prevalence to grid fit_transform prevalence."""
    transform_prevalence = document_topic.mean(axis=0, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    max_abs_difference = 0.0

    for topic in topic_metadata:
        selected_prevalence = float(transform_prevalence[topic.topic_index])
        grid_prevalence = topic.grid_overall_prevalence
        difference = abs(selected_prevalence - grid_prevalence)
        max_abs_difference = max(max_abs_difference, difference)
        rows.append(
            {
                "absolute_difference": _format_float(difference),
                "grid_fit_transform_prevalence": _format_float(grid_prevalence),
                "selected_transform_prevalence": _format_float(selected_prevalence),
                "threshold": _format_float(threshold),
                "topic_index": topic.topic_index,
                "top_terms_top10": _json_compact(topic.top_terms_top10),
                "within_threshold": difference <= threshold,
            }
        )

    if max_abs_difference > threshold:
        raise SelectedNmfAnalysisError(
            "Selected transform prevalence differs from grid prevalence beyond threshold: "
            f"{max_abs_difference:.12g} > {threshold:.12g}"
        )

    return tuple(rows), max_abs_difference


def _build_aggregations(
    *,
    document_topic: np.ndarray[Any, Any],
    document_metadata: Sequence[DocumentMetadata],
) -> AggregationPayload:
    """Build source-turn and session topic matrices from normalized document rows."""
    source_turn_rows: list[SourceTurnMetadata] = []
    source_turn_vectors: list[np.ndarray[Any, Any]] = []
    source_turn_positions: dict[tuple[str, int], int] = {}
    source_turn_sums: list[np.ndarray[Any, Any]] = []
    source_turn_chunk_sets: list[set[int]] = []

    for document_row_index, document_row in enumerate(document_metadata):
        key = (document_row.source_record_id, document_row.turn_index)
        position = source_turn_positions.get(key)

        if position is None:
            position = len(source_turn_rows)
            source_turn_positions[key] = position
            source_turn_rows.append(
                SourceTurnMetadata(
                    source_turn_row_index=position,
                    source_turn_key=document_row.source_turn_key,
                    source_record_id=document_row.source_record_id,
                    turn_index=document_row.turn_index,
                    first_original_row_index=document_row.original_row_index,
                    modeled_document_count=0,
                    chunk_count=0,
                    year=document_row.year,
                    temporal_period=document_row.temporal_period,
                    session_category=document_row.session_category,
                    speaker_family=document_row.speaker_family,
                    word_count=0,
                )
            )
            source_turn_sums.append(np.zeros(document_topic.shape[1], dtype=np.float64))
            source_turn_chunk_sets.append(set())
        else:
            existing = source_turn_rows[position]
            _require_consistent_source_turn(existing, document_row)

        existing = source_turn_rows[position]
        source_turn_sums[position] += document_topic[document_row_index].astype(np.float64)
        source_turn_chunk_sets[position].add(document_row.chunk_index)
        source_turn_rows[position] = SourceTurnMetadata(
            source_turn_row_index=existing.source_turn_row_index,
            source_turn_key=existing.source_turn_key,
            source_record_id=existing.source_record_id,
            turn_index=existing.turn_index,
            first_original_row_index=existing.first_original_row_index,
            modeled_document_count=existing.modeled_document_count + 1,
            chunk_count=len(source_turn_chunk_sets[position]),
            year=existing.year,
            temporal_period=existing.temporal_period,
            session_category=existing.session_category,
            speaker_family=existing.speaker_family,
            word_count=existing.word_count + document_row.word_count,
        )

    for position, source_turn_row in enumerate(source_turn_rows):
        if source_turn_row.modeled_document_count < 1:
            raise SelectedNmfAnalysisError("Source-turn group has no modelled documents.")

        source_turn_vectors.append(
            (source_turn_sums[position] / float(source_turn_row.modeled_document_count)).astype(
                np.float32
            )
        )

    source_turn_topic = np.vstack(source_turn_vectors).astype(np.float32)
    _validate_matrix_row_sums(source_turn_topic, label="source-turn topic matrix")
    session_topic, session_metadata = _build_session_aggregation(
        source_turn_topic=source_turn_topic,
        source_turn_metadata=tuple(source_turn_rows),
    )
    return AggregationPayload(
        source_turn_topic=source_turn_topic,
        source_turn_metadata=tuple(source_turn_rows),
        session_topic=session_topic,
        session_metadata=session_metadata,
    )


def _require_consistent_source_turn(existing: SourceTurnMetadata, row: DocumentMetadata) -> None:
    """Validate fields that must be constant within a source turn."""
    checks = {
        "session_category": (existing.session_category, row.session_category),
        "source_turn_key": (existing.source_turn_key, row.source_turn_key),
        "speaker_family": (existing.speaker_family, row.speaker_family),
        "temporal_period": (existing.temporal_period, row.temporal_period),
        "year": (existing.year, row.year),
    }
    mismatches = [field for field, (left, right) in checks.items() if left != right]

    if mismatches:
        raise SelectedNmfAnalysisError(
            f"Inconsistent source-turn metadata for {existing.source_turn_key}: {mismatches}"
        )


def _build_session_aggregation(
    *,
    source_turn_topic: np.ndarray[Any, Any],
    source_turn_metadata: Sequence[SourceTurnMetadata],
) -> tuple[np.ndarray[Any, Any], tuple[SessionMetadata, ...]]:
    """Build session topic rows by averaging source-turn rows within sessions."""
    session_positions: dict[str, int] = {}
    session_rows: list[SessionMetadata] = []
    session_sums: list[np.ndarray[Any, Any]] = []

    for source_turn_row_index, source_turn_row in enumerate(source_turn_metadata):
        position = session_positions.get(source_turn_row.source_record_id)

        if position is None:
            position = len(session_rows)
            session_positions[source_turn_row.source_record_id] = position
            session_rows.append(
                SessionMetadata(
                    session_row_index=position,
                    source_record_id=source_turn_row.source_record_id,
                    first_source_turn_row_index=source_turn_row.source_turn_row_index,
                    source_turn_count=0,
                    modeled_document_count=0,
                    year=source_turn_row.year,
                    temporal_period=source_turn_row.temporal_period,
                    session_category=source_turn_row.session_category,
                    word_count=0,
                )
            )
            session_sums.append(np.zeros(source_turn_topic.shape[1], dtype=np.float64))
        else:
            existing = session_rows[position]
            _require_consistent_session(existing, source_turn_row)

        existing = session_rows[position]
        session_sums[position] += source_turn_topic[source_turn_row_index].astype(np.float64)
        session_rows[position] = SessionMetadata(
            session_row_index=existing.session_row_index,
            source_record_id=existing.source_record_id,
            first_source_turn_row_index=existing.first_source_turn_row_index,
            source_turn_count=existing.source_turn_count + 1,
            modeled_document_count=(
                existing.modeled_document_count + source_turn_row.modeled_document_count
            ),
            year=existing.year,
            temporal_period=existing.temporal_period,
            session_category=existing.session_category,
            word_count=existing.word_count + source_turn_row.word_count,
        )

    vectors: list[np.ndarray[Any, Any]] = []

    for position, session_row in enumerate(session_rows):
        if session_row.source_turn_count < 1:
            raise SelectedNmfAnalysisError("Session group has no source turns.")

        vectors.append(
            (session_sums[position] / float(session_row.source_turn_count)).astype(np.float32)
        )

    session_topic = np.vstack(vectors).astype(np.float32)
    _validate_matrix_row_sums(session_topic, label="session topic matrix")
    return session_topic, tuple(session_rows)


def _require_consistent_session(existing: SessionMetadata, row: SourceTurnMetadata) -> None:
    """Validate fields that must be constant within a session."""
    checks = {
        "session_category": (existing.session_category, row.session_category),
        "temporal_period": (existing.temporal_period, row.temporal_period),
        "year": (existing.year, row.year),
    }
    mismatches = [field for field, (left, right) in checks.items() if left != right]

    if mismatches:
        raise SelectedNmfAnalysisError(
            f"Inconsistent session metadata for {existing.source_record_id}: {mismatches}"
        )


def _topic_metadata_by_index(
    topic_metadata: Sequence[TopicMetadata],
) -> dict[int, TopicMetadata]:
    """Return topic metadata keyed by topic index."""
    return {topic.topic_index: topic for topic in topic_metadata}


def _metadata_year(row: TemporalMetadata) -> int:
    """Return the annual year for any row-metadata type."""
    return row.year


def _metadata_period(row: TemporalMetadata) -> str:
    """Return the temporal period for any row-metadata type."""
    return row.temporal_period


def _build_temporal_prevalence(
    *,
    config: SelectedNmfAnalysisConfig,
    document_topic: np.ndarray[Any, Any],
    document_metadata: Sequence[DocumentMetadata],
    source_turn_topic: np.ndarray[Any, Any],
    source_turn_metadata: Sequence[SourceTurnMetadata],
    session_topic: np.ndarray[Any, Any],
    session_metadata: Sequence[SessionMetadata],
    topic_metadata: Sequence[TopicMetadata],
) -> PrevalencePayload:
    """Build annual and period topic prevalence rows for all aggregation levels."""
    topic_lookup = _topic_metadata_by_index(topic_metadata)
    levels: tuple[tuple[str, np.ndarray[Any, Any], Sequence[TemporalMetadata]], ...] = (
        ("document", document_topic, cast(Sequence[TemporalMetadata], document_metadata)),
        ("source_turn", source_turn_topic, cast(Sequence[TemporalMetadata], source_turn_metadata)),
        ("session", session_topic, cast(Sequence[TemporalMetadata], session_metadata)),
    )
    years = tuple(range(config.annual_year_start, config.annual_year_end + 1))
    periods = _parse_periods(config.temporal_periods)
    annual_rows: list[dict[str, Any]] = []
    period_rows: list[dict[str, Any]] = []
    annual_source_turn_by_topic: dict[tuple[int, int], float] = {}
    period_source_turn_by_topic: dict[tuple[str, int], float] = {}

    for aggregation_level, matrix, rows_metadata in levels:
        for year in years:
            indices = [
                row_index
                for row_index, row in enumerate(rows_metadata)
                if _metadata_year(row) == year
            ]
            prevalence = _mean_prevalence(
                matrix=matrix,
                indices=indices,
                label=f"{aggregation_level} year {year}",
            )

            for topic_index, value in enumerate(prevalence):
                if aggregation_level == "source_turn":
                    annual_source_turn_by_topic[(year, topic_index)] = float(value)

                annual_rows.append(
                    {
                        "aggregation_level": aggregation_level,
                        "denominator_unit_count": len(indices),
                        "prevalence_share": _format_float(float(value)),
                        "topic_index": topic_index,
                        "topic_top_terms": _json_compact(topic_lookup[topic_index].top_terms_top10),
                        "year": year,
                    }
                )

        for period in periods:
            indices = [
                row_index
                for row_index, row in enumerate(rows_metadata)
                if _metadata_period(row) == period.label
            ]
            prevalence = _mean_prevalence(
                matrix=matrix,
                indices=indices,
                label=f"{aggregation_level} period {period.label}",
            )

            for topic_index, value in enumerate(prevalence):
                if aggregation_level == "source_turn":
                    period_source_turn_by_topic[(period.label, topic_index)] = float(value)

                period_rows.append(
                    {
                        "aggregation_level": aggregation_level,
                        "denominator_unit_count": len(indices),
                        "period_end_year": period.end_year,
                        "period_start_year": period.start_year,
                        "prevalence_share": _format_float(float(value)),
                        "temporal_period": period.label,
                        "topic_index": topic_index,
                        "topic_top_terms": _json_compact(topic_lookup[topic_index].top_terms_top10),
                    }
                )

    _validate_share_rows(annual_rows, time_column="year", label="annual")
    _validate_share_rows(period_rows, time_column="temporal_period", label="period")
    return PrevalencePayload(
        annual_rows=tuple(annual_rows),
        period_rows=tuple(period_rows),
        annual_source_turn_by_topic=annual_source_turn_by_topic,
        period_source_turn_by_topic=period_source_turn_by_topic,
    )


def _mean_prevalence(
    *,
    matrix: np.ndarray[Any, Any],
    indices: Sequence[int],
    label: str,
) -> np.ndarray[Any, Any]:
    """Return the mean prevalence vector for selected rows."""
    if not indices:
        raise SelectedNmfAnalysisError(f"No modelled units available for {label}.")

    prevalence = matrix[list(indices)].mean(axis=0, dtype=np.float64)
    total = float(np.sum(prevalence))

    if abs(total - 1.0) > PREVALENCE_SUM_TOLERANCE:
        raise SelectedNmfAnalysisError(f"Prevalence shares do not sum to one for {label}.")

    return np.asarray(prevalence, dtype=np.float64)


def _validate_share_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    time_column: str,
    label: str,
) -> None:
    """Validate long-format prevalence rows sum to one by aggregation/time group."""
    totals: dict[tuple[str, str], float] = defaultdict(float)

    for row in rows:
        aggregation_level = str(row["aggregation_level"])
        time_value = str(row[time_column])
        totals[(aggregation_level, time_value)] += float(row["prevalence_share"])

    bad = [
        (aggregation_level, time_value, total)
        for (aggregation_level, time_value), total in totals.items()
        if abs(total - 1.0) > PREVALENCE_SUM_TOLERANCE
    ]

    if bad:
        raise SelectedNmfAnalysisError(f"{label} prevalence rows do not sum to one: {bad[:5]}")


def _build_temporal_denominator_rows(
    *,
    config: SelectedNmfAnalysisConfig,
    document_metadata: Sequence[DocumentMetadata],
    source_turn_metadata: Sequence[SourceTurnMetadata],
    session_metadata: Sequence[SessionMetadata],
    zero_ledger_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Build denominator rows for every configured year and period."""
    rows: list[dict[str, Any]] = []
    periods = _parse_periods(config.temporal_periods)
    zero_by_year: dict[int, int] = defaultdict(int)
    zero_by_period: dict[str, int] = defaultdict(int)

    for row in zero_ledger_rows:
        year = _safe_int(row.get("year"), field_name="zero ledger year")
        period_label = _required_record_string(row, "temporal_period")
        zero_by_year[year] += 1
        zero_by_period[period_label] += 1

    for year in range(config.annual_year_start, config.annual_year_end + 1):
        modeled_document_count = sum(1 for row in document_metadata if row.year == year)
        modeled_source_turn_count = sum(1 for row in source_turn_metadata if row.year == year)
        modeled_session_count = sum(1 for row in session_metadata if row.year == year)
        total_words = sum(row.word_count for row in document_metadata if row.year == year)
        zero_count = zero_by_year[year]
        rows.append(
            {
                "corpus_primary_document_count_including_zero_tfidf": (
                    modeled_document_count + zero_count
                ),
                "denominator_scope": "year",
                "modeled_document_count": modeled_document_count,
                "modeled_session_count": modeled_session_count,
                "modeled_source_turn_count": modeled_source_turn_count,
                "temporal_period": _period_for_year(year, periods),
                "total_words_represented": total_words,
                "year": year,
                "zero_tfidf_excluded_document_count": zero_count,
                "zero_tfidf_note": (
                    "zero-vector primary documents remain corpus documents but have no "
                    "modelled topic vector"
                ),
            }
        )

    for period_row in periods:
        modeled_document_count = sum(
            1 for row in document_metadata if row.temporal_period == period_row.label
        )
        modeled_source_turn_count = sum(
            1 for row in source_turn_metadata if row.temporal_period == period_row.label
        )
        modeled_session_count = sum(
            1 for row in session_metadata if row.temporal_period == period_row.label
        )
        total_words = sum(
            row.word_count for row in document_metadata if row.temporal_period == period_row.label
        )
        zero_count = zero_by_period[period_row.label]
        rows.append(
            {
                "corpus_primary_document_count_including_zero_tfidf": (
                    modeled_document_count + zero_count
                ),
                "denominator_scope": "period",
                "modeled_document_count": modeled_document_count,
                "modeled_session_count": modeled_session_count,
                "modeled_source_turn_count": modeled_source_turn_count,
                "temporal_period": period_row.label,
                "total_words_represented": total_words,
                "year": "",
                "zero_tfidf_excluded_document_count": zero_count,
                "zero_tfidf_note": (
                    "zero-vector primary documents remain corpus documents but have no "
                    "modelled topic vector"
                ),
            }
        )

    return tuple(rows)


def _build_topic_change_summary_rows(
    *,
    config: SelectedNmfAnalysisConfig,
    source_turn_topic: np.ndarray[Any, Any],
    prevalence: PrevalencePayload,
    topic_metadata: Sequence[TopicMetadata],
) -> tuple[dict[str, Any], ...]:
    """Build source-turn-weighted topic-change summaries."""
    rows: list[dict[str, Any]] = []
    years = tuple(range(config.annual_year_start, config.annual_year_end + 1))
    baseline_period = config.temporal_periods[0]
    final_period = config.temporal_periods[-1]
    overall_prevalence = source_turn_topic.mean(axis=0, dtype=np.float64)

    for topic in topic_metadata:
        topic_index = topic.topic_index
        annual_values = [
            (year, prevalence.annual_source_turn_by_topic[(year, topic_index)]) for year in years
        ]
        baseline_value = prevalence.period_source_turn_by_topic[(baseline_period, topic_index)]
        final_value = prevalence.period_source_turn_by_topic[(final_period, topic_index)]
        absolute_change = final_value - baseline_value
        relative_change = (
            "" if baseline_value <= 0.0 else _format_float(absolute_change / baseline_value)
        )
        max_year, max_value = max(annual_values, key=lambda item: (item[1], -item[0]))
        min_year, min_value = min(annual_values, key=lambda item: (item[1], item[0]))
        increases = [
            (annual_values[index][0], annual_values[index][1] - annual_values[index - 1][1])
            for index in range(1, len(annual_values))
        ]
        largest_increase_year, largest_increase = max(
            increases,
            key=lambda item: (item[1], -item[0]),
        )
        largest_decrease_year, largest_decrease = min(
            increases,
            key=lambda item: (item[1], item[0]),
        )
        rows.append(
            {
                "absolute_change": _format_float(absolute_change),
                "baseline_period": baseline_period,
                "baseline_period_prevalence": _format_float(baseline_value),
                "final_period": final_period,
                "final_period_prevalence": _format_float(final_value),
                "largest_year_to_year_decrease": _format_float(largest_decrease),
                "largest_year_to_year_decrease_ending_year": largest_decrease_year,
                "largest_year_to_year_increase": _format_float(largest_increase),
                "largest_year_to_year_increase_ending_year": largest_increase_year,
                "maximum_prevalence_value": _format_float(max_value),
                "maximum_prevalence_year": max_year,
                "minimum_prevalence_value": _format_float(min_value),
                "minimum_prevalence_year": min_year,
                "overall_source_turn_weighted_prevalence": _format_float(
                    float(overall_prevalence[topic_index])
                ),
                "relative_change_from_baseline": relative_change,
                "topic_index": topic_index,
                "top_terms_top10": _json_compact(topic.top_terms_top10),
            }
        )

    return tuple(rows)


def _document_topic_metadata_rows(
    metadata_rows: Sequence[DocumentMetadata],
) -> tuple[dict[str, Any], ...]:
    """Return CSV rows matching document-topic matrix row order."""
    return tuple(
        {
            "chunk_index": row.chunk_index,
            "document_id": row.document_id,
            "document_topic_row_index": row_index,
            "original_row_index": row.original_row_index,
            "session_category": row.session_category,
            "source_record_id": row.source_record_id,
            "source_turn_key": row.source_turn_key,
            "speaker_family": row.speaker_family,
            "temporal_period": row.temporal_period,
            "turn_index": row.turn_index,
            "word_count": row.word_count,
            "year": row.year,
        }
        for row_index, row in enumerate(metadata_rows)
    )


def _document_topic_assignment_rows(
    *,
    document_topic: np.ndarray[Any, Any],
    metadata_rows: Sequence[DocumentMetadata],
) -> tuple[dict[str, Any], ...]:
    """Return dominant-topic assignment rows for every modelled document."""
    entropies = _normalized_entropy(document_topic)
    rows: list[dict[str, Any]] = []

    for row_index, metadata_row in enumerate(metadata_rows):
        weights = document_topic[row_index]
        dominant_topic_index = int(np.argmax(weights))
        rows.append(
            {
                "chunk_index": metadata_row.chunk_index,
                "document_id": metadata_row.document_id,
                "document_topic_row_index": row_index,
                "dominant_topic_index": dominant_topic_index,
                "dominant_topic_weight": _format_float(float(weights[dominant_topic_index])),
                "normalized_topic_entropy": _format_float(float(entropies[row_index])),
                "original_row_index": metadata_row.original_row_index,
                "session_category": metadata_row.session_category,
                "source_record_id": metadata_row.source_record_id,
                "source_turn_key": metadata_row.source_turn_key,
                "speaker_family": metadata_row.speaker_family,
                "temporal_period": metadata_row.temporal_period,
                "turn_index": metadata_row.turn_index,
                "word_count": metadata_row.word_count,
                "year": metadata_row.year,
            }
        )

    return tuple(rows)


def _normalized_entropy(matrix: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Return normalized entropy for each row in a probability matrix."""
    topic_count = matrix.shape[1]

    if topic_count <= 1:
        return np.zeros(matrix.shape[0], dtype=np.float64)

    safe = np.where(matrix > 0.0, matrix, 1.0)
    entropy = -np.sum(np.where(matrix > 0.0, matrix * np.log(safe), 0.0), axis=1)
    return entropy / math.log(topic_count)


def _source_turn_metadata_rows(
    metadata_rows: Sequence[SourceTurnMetadata],
) -> tuple[dict[str, Any], ...]:
    """Return CSV rows matching source-turn topic matrix row order."""
    return tuple(
        {
            "chunk_count": row.chunk_count,
            "first_original_row_index": row.first_original_row_index,
            "modeled_document_count": row.modeled_document_count,
            "session_category": row.session_category,
            "source_record_id": row.source_record_id,
            "source_turn_key": row.source_turn_key,
            "source_turn_row_index": row.source_turn_row_index,
            "speaker_family": row.speaker_family,
            "temporal_period": row.temporal_period,
            "turn_index": row.turn_index,
            "word_count": row.word_count,
            "year": row.year,
        }
        for row in metadata_rows
    )


def _session_metadata_rows(
    metadata_rows: Sequence[SessionMetadata],
) -> tuple[dict[str, Any], ...]:
    """Return CSV rows matching session topic matrix row order."""
    return tuple(
        {
            "first_source_turn_row_index": row.first_source_turn_row_index,
            "modeled_document_count": row.modeled_document_count,
            "session_category": row.session_category,
            "session_row_index": row.session_row_index,
            "source_record_id": row.source_record_id,
            "source_turn_count": row.source_turn_count,
            "temporal_period": row.temporal_period,
            "word_count": row.word_count,
            "year": row.year,
        }
        for row in metadata_rows
    )


def _topic_metadata_rows(topic_metadata: Sequence[TopicMetadata]) -> tuple[dict[str, Any], ...]:
    """Return one CSV row per topic."""
    return tuple(
        {
            "grid_mean_exclusivity_top10": _format_float(topic.grid_mean_exclusivity_top10),
            "grid_npmi_coherence_top10": _format_float(topic.grid_npmi_coherence_top10),
            "grid_overall_prevalence": _format_float(topic.grid_overall_prevalence),
            "top_terms_top10": _json_compact(topic.top_terms_top10),
            "top_terms_top20": _json_compact(topic.top_terms_top20),
            "topic_index": topic.topic_index,
        }
        for topic in topic_metadata
    )


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
        raise SelectedNmfAnalysisError(f"Output path is not a directory: {output_dir}")

    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise SelectedNmfAnalysisError(f"Output directory is nonempty; use --force: {output_dir}")


def _ensure_output_directory(output_dir: Path, *, force: bool) -> None:
    """Create the output directory after overwrite protection."""
    _preflight_output_directory(output_dir, force=force)
    output_dir.mkdir(parents=True, exist_ok=True)


def _final_output_paths(output_dir: Path) -> dict[str, Path]:
    """Return canonical selected-analysis output paths."""
    return {
        "annual_topic_prevalence": output_dir / ANNUAL_TOPIC_PREVALENCE_FILENAME,
        "document_topic_assignments": output_dir / DOCUMENT_TOPIC_ASSIGNMENTS_FILENAME,
        "document_topic_metadata": output_dir / DOCUMENT_TOPIC_METADATA_FILENAME,
        "document_topic_weights": output_dir / DOCUMENT_TOPIC_WEIGHTS_FILENAME,
        "grid_prevalence_comparison": output_dir / GRID_PREVALENCE_COMPARISON_FILENAME,
        "period_topic_prevalence": output_dir / PERIOD_TOPIC_PREVALENCE_FILENAME,
        "selected_model_manifest": output_dir / SELECTED_MODEL_MANIFEST_FILENAME,
        "selected_model_report": output_dir / SELECTED_MODEL_REPORT_FILENAME,
        "session_metadata": output_dir / SESSION_METADATA_FILENAME,
        "session_topic_weights": output_dir / SESSION_TOPIC_WEIGHTS_FILENAME,
        "source_turn_metadata": output_dir / SOURCE_TURN_METADATA_FILENAME,
        "source_turn_topic_weights": output_dir / SOURCE_TURN_TOPIC_WEIGHTS_FILENAME,
        "temporal_denominators": output_dir / TEMPORAL_DENOMINATORS_FILENAME,
        "topic_change_summary": output_dir / TOPIC_CHANGE_SUMMARY_FILENAME,
        "topic_metadata": output_dir / TOPIC_METADATA_FILENAME,
    }


def _write_text_part(path: Path, text: str) -> dict[str, Any]:
    """Write UTF-8 text to a staged part file."""
    _part_path(path).write_text(text, encoding="utf-8", newline="")
    return _file_metadata(path)


def _write_json_part(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Write deterministic JSON to a staged part file."""
    return _write_text_part(path, _json_text(payload))


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


def _write_npz_part(
    path: Path,
    *,
    weights: np.ndarray[Any, Any],
    identifier_key: str,
    identifiers: Sequence[str],
) -> dict[str, Any]:
    """Write a normalized dense topic matrix as an NPZ sidecar."""
    if weights.dtype != np.float32:
        raise SelectedNmfAnalysisError("NPZ weights must be float32.")

    part_path = _part_path(path)
    identifier_array = np.asarray(identifiers, dtype=str)

    with part_path.open("wb") as output_file:
        if identifier_key == "document_ids":
            np.savez_compressed(
                output_file,
                document_ids=identifier_array,
                topic_weights=weights,
            )
        elif identifier_key == "source_turn_keys":
            np.savez_compressed(
                output_file,
                source_turn_keys=identifier_array,
                topic_weights=weights,
            )
        elif identifier_key == "source_record_ids":
            np.savez_compressed(
                output_file,
                source_record_ids=identifier_array,
                topic_weights=weights,
            )
        else:
            raise SelectedNmfAnalysisError(f"Unsupported NPZ identifier key: {identifier_key}")

    metadata_payload = _file_metadata(path)
    metadata_payload["dtype"] = "float32"
    metadata_payload["keys"] = ["topic_weights", identifier_key]
    metadata_payload["record_count"] = int(weights.shape[0])
    metadata_payload["shape"] = [int(weights.shape[0]), int(weights.shape[1])]
    return metadata_payload


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
            raise SelectedNmfAnalysisError(
                "Could not restore prior selected-NMF outputs after promotion failure."
            ) from restore_error

        _cleanup_transaction_paths(final_paths)
        raise SelectedNmfAnalysisError("Could not promote selected-NMF outputs safely.") from error

    _cleanup_transaction_paths(final_paths)


def _selected_model_report(
    *,
    config: SelectedNmfAnalysisConfig,
    grid_input_dir: Path,
    output_dir: Path,
    selected_payload: SelectedMatrixPayload,
    aggregation_payload: AggregationPayload,
    grid_prevalence_max_difference: float,
) -> str:
    """Return a concise Markdown report for the selected NMF analysis."""
    return "\n".join(
        [
            "# Selected NMF Analysis",
            "",
            "This stage loads the saved TF-IDF vectorizer and fitted NMF model, then "
            "uses `transform()` to infer selected-model document weights against the "
            "fixed fitted topic-term matrix. It does not refit the vectorizer or NMF model.",
            "",
            "## Lineage",
            "",
            f"- Grid input directory: `{grid_input_dir}`",
            f"- Output directory: `{output_dir}`",
            f"- Analysis version: {config.analysis_version}",
            f"- Selected K: {config.selected_k}",
            f"- Stopword variant: {config.stopword_variant}",
            f"- Primary documents: {config.expected_primary_documents:,}",
            f"- Modelled documents: {config.expected_modeled_documents:,}",
            f"- Zero TF-IDF exclusions: {config.expected_zero_tfidf_exclusions:,}",
            f"- Grid prevalence maximum absolute difference: {grid_prevalence_max_difference:.10f}",
            "",
            "## Aggregation Definitions",
            "",
            "- Document weighting: average normalized document-topic vectors directly.",
            "- Source-turn weighting: average chunks within each source-record/turn, "
            "then average source turns equally within each year or period.",
            "- Session weighting: average source-turn vectors within each source record, "
            "then average sessions equally within each year or period.",
            "",
            "## Matrix Shapes",
            "",
            f"- Document topic matrix: {selected_payload.document_topic.shape}",
            f"- Source-turn topic matrix: {aggregation_payload.source_turn_topic.shape}",
            f"- Session topic matrix: {aggregation_payload.session_topic.shape}",
            "",
            "Zero-vector primary documents remain in the corpus denominator outputs but "
            "do not contribute a modelled topic vector.",
            "",
        ]
    )


def _package_versions() -> dict[str, str]:
    """Return exact package versions used by this run."""
    packages = ("joblib", "numpy", "scikit-learn", "scipy")
    return {package: metadata.version(package) for package in packages}


def _manifest_payload(
    *,
    config: SelectedNmfAnalysisConfig,
    config_path: Path,
    grid_input_dir: Path,
    grid_manifest: Mapping[str, Any],
    validated_grid_artifacts: Mapping[str, Mapping[str, Any]],
    selected_payload: SelectedMatrixPayload,
    aggregation_payload: AggregationPayload,
    output_files: Mapping[str, Mapping[str, Any]],
    grid_prevalence_max_difference: float,
    temporal_period_count: int,
    annual_group_count: int,
    period_group_count: int,
) -> dict[str, Any]:
    """Return the selected-model manifest payload."""
    canonical_config_text = _json_text(config.to_json())
    source_grid_generated_at = grid_manifest.get("generated_at_utc")
    return {
        "analysis_version": SELECTED_NMF_ANALYSIS_VERSION,
        "canonical_configuration_sha256": hashlib.sha256(
            canonical_config_text.encode("utf-8")
        ).hexdigest(),
        "configuration": config.to_json(),
        "configuration_path": str(config_path),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "grid_input_dir": str(grid_input_dir),
        "grid_manifest_generated_at_utc": source_grid_generated_at,
        "grid_manifest_path": str(grid_input_dir / RUN_MANIFEST_FILENAME),
        "grid_prevalence_comparison": {
            "maximum_absolute_difference": round(grid_prevalence_max_difference, 12),
            "threshold": config.grid_prevalence_max_absolute_difference,
            "within_threshold": (
                grid_prevalence_max_difference <= config.grid_prevalence_max_absolute_difference
            ),
        },
        "input_grid_artifacts": dict(sorted(validated_grid_artifacts.items())),
        "manifest_self_hash_excluded": True,
        "model_weight_semantics": (
            "selected-model inferred weights from saved vectorizer.transform() and "
            "saved NMF.transform() using the fixed fitted topic-term matrix"
        ),
        "output_files": dict(sorted(output_files.items())),
        "package_versions": {
            **_package_versions(),
            "python": sys.version.split()[0],
        },
        "reconciliation_checks": {
            "all_new_output_hashes_match_emitted_files": True,
            "all_required_grid_input_hashes_match": True,
            "annual_shares_sum_to_one": True,
            "configured_periods_appear_once_per_aggregation_topic": True,
            "document_matrix_rows_equal_document_metadata_rows": (
                selected_payload.document_topic.shape[0] == len(selected_payload.document_metadata)
            ),
            "document_topic_rows_sum_to_one": True,
            "grid_prevalence_comparison_within_threshold": (
                grid_prevalence_max_difference <= config.grid_prevalence_max_absolute_difference
            ),
            "modeled_documents": config.expected_modeled_documents,
            "modeled_plus_excluded_equals_primary": (
                config.expected_modeled_documents + config.expected_zero_tfidf_exclusions
                == config.expected_primary_documents
            ),
            "period_shares_sum_to_one": True,
            "primary_documents": config.expected_primary_documents,
            "session_matrix_rows_equal_session_metadata_rows": (
                aggregation_payload.session_topic.shape[0]
                == len(aggregation_payload.session_metadata)
            ),
            "session_topic_rows_sum_to_one": True,
            "selected_k_converged_in_source_grid": True,
            "source_turn_matrix_rows_equal_source_turn_metadata_rows": (
                aggregation_payload.source_turn_topic.shape[0]
                == len(aggregation_payload.source_turn_metadata)
            ),
            "source_turn_topic_rows_sum_to_one": True,
            "transformed_zero_rows_exactly_match_ledger": True,
            "years_cover_configured_range": True,
            "zero_tfidf_exclusions": config.expected_zero_tfidf_exclusions,
        },
        "row_ordering": {
            "document_topic_weights": (
                "row order matches document_topic_metadata.csv by document_topic_row_index"
            ),
            "session_topic_weights": "row order matches session_metadata.csv by session_row_index",
            "source_turn_topic_weights": (
                "row order matches source_turn_metadata.csv by source_turn_row_index"
            ),
        },
        "shapes": {
            "document_topic_weights": [
                int(selected_payload.document_topic.shape[0]),
                int(selected_payload.document_topic.shape[1]),
            ],
            "session_topic_weights": [
                int(aggregation_payload.session_topic.shape[0]),
                int(aggregation_payload.session_topic.shape[1]),
            ],
            "source_turn_topic_weights": [
                int(aggregation_payload.source_turn_topic.shape[0]),
                int(aggregation_payload.source_turn_topic.shape[1]),
            ],
            "transformed_tfidf_after_zero_exclusion": list(
                selected_payload.transformed_tfidf_shape_after_exclusion
            ),
            "transformed_tfidf_before_zero_exclusion": list(
                selected_payload.transformed_tfidf_shape_before_exclusion
            ),
        },
        "temporal_coverage": {
            "annual_aggregation_topic_groups": annual_group_count,
            "annual_year_end": config.annual_year_end,
            "annual_year_start": config.annual_year_start,
            "period_aggregation_topic_groups": period_group_count,
            "temporal_period_count": temporal_period_count,
            "temporal_periods": list(config.temporal_periods),
        },
        "weight_normalization": {
            "document_row_sum_max_abs_deviation": (
                selected_payload.normalized_row_sum_max_abs_deviation
            ),
            "raw_weight_row_sum_maximum": selected_payload.raw_weight_row_sum_maximum,
            "raw_weight_row_sum_minimum": selected_payload.raw_weight_row_sum_minimum,
            "row_sum_tolerance": ROW_SUM_TOLERANCE,
        },
        "zero_tfidf_exclusions": {
            "excluded_document_count": len(selected_payload.zero_ledger_rows),
            "transformed_zero_rows": [
                {"document_id": document_id, "original_row_index": row_index}
                for row_index, document_id in selected_payload.transformed_zero_rows
            ],
        },
    }


def analyze_selected_nmf(
    *,
    grid_input_dir: Path = DEFAULT_GRID_INPUT_DIR,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    force: bool = False,
) -> dict[str, Any]:
    """Run selected K NMF assignment and temporal analysis transactionally."""
    config = load_config(config_path)
    _preflight_output_directory(output_dir, force=force)
    periods = _parse_periods(config.temporal_periods)
    required_artifacts = _required_grid_artifacts(config)
    grid_manifest_path = grid_input_dir / RUN_MANIFEST_FILENAME
    grid_manifest = _read_json_object(grid_manifest_path, label="grid run manifest")
    _require_grid_reconciliation_checks(grid_manifest)
    validated_grid_artifacts = _validate_grid_artifact_metadata(
        grid_input_dir=grid_input_dir,
        manifest=grid_manifest,
        required_artifacts=required_artifacts,
    )
    preprocessing_summary = _read_json_object(
        grid_input_dir / PREPROCESSING_SUMMARY_FILENAME,
        label="preprocessing summary",
    )
    vectorizer_summary = _read_json_object(
        grid_input_dir / VECTORIZER_SUMMARY_FILENAME,
        label="vectorizer summary",
    )
    grid_metrics_rows = _read_csv_rows(
        grid_input_dir / GRID_METRICS_FILENAME,
        required_columns=frozenset({"k", "converged"}),
        label="grid metrics",
    )
    _validate_grid_summaries(
        manifest=grid_manifest,
        preprocessing_summary=preprocessing_summary,
        vectorizer_summary=vectorizer_summary,
        grid_metrics_rows=grid_metrics_rows,
        config=config,
    )
    vectorizer = _load_joblib_artifact(
        grid_input_dir / VECTORIZER_ARTIFACT_FILENAME,
        label="selected grid vectorizer",
    )
    model = _load_joblib_artifact(
        grid_input_dir / _selected_model_filename(config.selected_k),
        label=f"selected NMF K={config.selected_k} model",
    )
    _validate_vectorizer_and_model(
        vectorizer=vectorizer,
        model=model,
        vectorizer_summary=vectorizer_summary,
        config=config,
    )
    selected_payload = _reconstruct_selected_document_matrix(
        cleaned_documents_path=grid_input_dir / CLEANED_DOCUMENTS_FILENAME,
        zero_tfidf_documents_path=grid_input_dir / ZERO_TFIDF_DOCUMENTS_FILENAME,
        vectorizer=vectorizer,
        model=model,
        vectorizer_summary=vectorizer_summary,
        config=config,
    )
    topic_metadata = _load_topic_metadata(
        topic_terms_path=grid_input_dir / _selected_topic_terms_filename(config.selected_k),
        selected_k=config.selected_k,
    )
    grid_prevalence_rows, grid_prevalence_max_difference = _grid_prevalence_comparison_rows(
        document_topic=selected_payload.document_topic,
        topic_metadata=topic_metadata,
        threshold=config.grid_prevalence_max_absolute_difference,
    )
    aggregation_payload = _build_aggregations(
        document_topic=selected_payload.document_topic,
        document_metadata=selected_payload.document_metadata,
    )
    prevalence_payload = _build_temporal_prevalence(
        config=config,
        document_topic=selected_payload.document_topic,
        document_metadata=selected_payload.document_metadata,
        source_turn_topic=aggregation_payload.source_turn_topic,
        source_turn_metadata=aggregation_payload.source_turn_metadata,
        session_topic=aggregation_payload.session_topic,
        session_metadata=aggregation_payload.session_metadata,
        topic_metadata=topic_metadata,
    )
    denominator_rows = _build_temporal_denominator_rows(
        config=config,
        document_metadata=selected_payload.document_metadata,
        source_turn_metadata=aggregation_payload.source_turn_metadata,
        session_metadata=aggregation_payload.session_metadata,
        zero_ledger_rows=selected_payload.zero_ledger_rows,
    )
    topic_change_rows = _build_topic_change_summary_rows(
        config=config,
        source_turn_topic=aggregation_payload.source_turn_topic,
        prevalence=prevalence_payload,
        topic_metadata=topic_metadata,
    )

    _ensure_output_directory(output_dir, force=force)
    paths = _final_output_paths(output_dir)
    final_paths = tuple(paths.values())
    _cleanup_transaction_paths(final_paths)
    output_files: dict[str, Mapping[str, Any]] = {}

    try:
        output_files["topic_metadata"] = _write_csv_part(
            paths["topic_metadata"],
            fieldnames=(
                "topic_index",
                "top_terms_top20",
                "top_terms_top10",
                "grid_npmi_coherence_top10",
                "grid_mean_exclusivity_top10",
                "grid_overall_prevalence",
            ),
            rows=_topic_metadata_rows(topic_metadata),
        )
        output_files["document_topic_weights"] = _write_npz_part(
            paths["document_topic_weights"],
            weights=selected_payload.document_topic,
            identifier_key="document_ids",
            identifiers=[row.document_id for row in selected_payload.document_metadata],
        )
        output_files["document_topic_metadata"] = _write_csv_part(
            paths["document_topic_metadata"],
            fieldnames=(
                "document_topic_row_index",
                "document_id",
                "original_row_index",
                "source_record_id",
                "turn_index",
                "chunk_index",
                "source_turn_key",
                "year",
                "temporal_period",
                "session_category",
                "speaker_family",
                "word_count",
            ),
            rows=_document_topic_metadata_rows(selected_payload.document_metadata),
        )
        output_files["document_topic_assignments"] = _write_csv_part(
            paths["document_topic_assignments"],
            fieldnames=(
                "document_topic_row_index",
                "document_id",
                "original_row_index",
                "source_record_id",
                "turn_index",
                "chunk_index",
                "source_turn_key",
                "year",
                "temporal_period",
                "session_category",
                "speaker_family",
                "word_count",
                "dominant_topic_index",
                "dominant_topic_weight",
                "normalized_topic_entropy",
            ),
            rows=_document_topic_assignment_rows(
                document_topic=selected_payload.document_topic,
                metadata_rows=selected_payload.document_metadata,
            ),
        )
        output_files["source_turn_topic_weights"] = _write_npz_part(
            paths["source_turn_topic_weights"],
            weights=aggregation_payload.source_turn_topic,
            identifier_key="source_turn_keys",
            identifiers=[row.source_turn_key for row in aggregation_payload.source_turn_metadata],
        )
        output_files["source_turn_metadata"] = _write_csv_part(
            paths["source_turn_metadata"],
            fieldnames=(
                "source_turn_row_index",
                "source_turn_key",
                "source_record_id",
                "turn_index",
                "first_original_row_index",
                "modeled_document_count",
                "chunk_count",
                "year",
                "temporal_period",
                "session_category",
                "speaker_family",
                "word_count",
            ),
            rows=_source_turn_metadata_rows(aggregation_payload.source_turn_metadata),
        )
        output_files["session_topic_weights"] = _write_npz_part(
            paths["session_topic_weights"],
            weights=aggregation_payload.session_topic,
            identifier_key="source_record_ids",
            identifiers=[row.source_record_id for row in aggregation_payload.session_metadata],
        )
        output_files["session_metadata"] = _write_csv_part(
            paths["session_metadata"],
            fieldnames=(
                "session_row_index",
                "source_record_id",
                "first_source_turn_row_index",
                "source_turn_count",
                "modeled_document_count",
                "year",
                "temporal_period",
                "session_category",
                "word_count",
            ),
            rows=_session_metadata_rows(aggregation_payload.session_metadata),
        )
        output_files["annual_topic_prevalence"] = _write_csv_part(
            paths["annual_topic_prevalence"],
            fieldnames=(
                "aggregation_level",
                "year",
                "topic_index",
                "prevalence_share",
                "denominator_unit_count",
                "topic_top_terms",
            ),
            rows=prevalence_payload.annual_rows,
        )
        output_files["period_topic_prevalence"] = _write_csv_part(
            paths["period_topic_prevalence"],
            fieldnames=(
                "aggregation_level",
                "temporal_period",
                "period_start_year",
                "period_end_year",
                "topic_index",
                "prevalence_share",
                "denominator_unit_count",
                "topic_top_terms",
            ),
            rows=prevalence_payload.period_rows,
        )
        output_files["temporal_denominators"] = _write_csv_part(
            paths["temporal_denominators"],
            fieldnames=(
                "denominator_scope",
                "year",
                "temporal_period",
                "modeled_document_count",
                "modeled_source_turn_count",
                "modeled_session_count",
                "total_words_represented",
                "zero_tfidf_excluded_document_count",
                "corpus_primary_document_count_including_zero_tfidf",
                "zero_tfidf_note",
            ),
            rows=denominator_rows,
        )
        output_files["topic_change_summary"] = _write_csv_part(
            paths["topic_change_summary"],
            fieldnames=(
                "topic_index",
                "overall_source_turn_weighted_prevalence",
                "baseline_period",
                "baseline_period_prevalence",
                "final_period",
                "final_period_prevalence",
                "absolute_change",
                "relative_change_from_baseline",
                "maximum_prevalence_year",
                "maximum_prevalence_value",
                "minimum_prevalence_year",
                "minimum_prevalence_value",
                "largest_year_to_year_increase",
                "largest_year_to_year_increase_ending_year",
                "largest_year_to_year_decrease",
                "largest_year_to_year_decrease_ending_year",
                "top_terms_top10",
            ),
            rows=topic_change_rows,
        )
        output_files["grid_prevalence_comparison"] = _write_csv_part(
            paths["grid_prevalence_comparison"],
            fieldnames=(
                "topic_index",
                "selected_transform_prevalence",
                "grid_fit_transform_prevalence",
                "absolute_difference",
                "threshold",
                "within_threshold",
                "top_terms_top10",
            ),
            rows=grid_prevalence_rows,
        )
        output_files["selected_model_report"] = _write_text_part(
            paths["selected_model_report"],
            _selected_model_report(
                config=config,
                grid_input_dir=grid_input_dir,
                output_dir=output_dir,
                selected_payload=selected_payload,
                aggregation_payload=aggregation_payload,
                grid_prevalence_max_difference=grid_prevalence_max_difference,
            ),
        )

        if not _validate_output_metadata(output_files):
            raise SelectedNmfAnalysisError("Staged output hash validation failed.")

        annual_group_count = (
            len(config.aggregation_levels)
            * (config.annual_year_end - config.annual_year_start + 1)
            * config.selected_k
        )
        period_group_count = len(config.aggregation_levels) * len(periods) * config.selected_k
        manifest = _manifest_payload(
            config=config,
            config_path=config_path,
            grid_input_dir=grid_input_dir,
            grid_manifest=grid_manifest,
            validated_grid_artifacts=validated_grid_artifacts,
            selected_payload=selected_payload,
            aggregation_payload=aggregation_payload,
            output_files=output_files,
            grid_prevalence_max_difference=grid_prevalence_max_difference,
            temporal_period_count=len(periods),
            annual_group_count=annual_group_count,
            period_group_count=period_group_count,
        )
        _write_json_part(paths["selected_model_manifest"], manifest)
    except Exception:
        _cleanup_transaction_paths(final_paths)
        raise

    _promote_transaction(final_paths)
    return cast(
        dict[str, Any],
        json.loads(paths["selected_model_manifest"].read_text(encoding="utf-8")),
    )


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_GRID_INPUT_DIR",
    "DEFAULT_OUTPUT_DIR",
    "SelectedNmfAnalysisError",
    "analyze_selected_nmf",
    "load_config",
]
