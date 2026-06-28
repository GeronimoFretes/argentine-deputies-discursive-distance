"""Profile the locked modelling corpus without fitting topic models."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from statistics import median, pstdev
from typing import Any, cast

from .modeling_corpus import (
    INCLUDED_SPEAKER_FAMILIES,
    MODELING_CORPUS_EXPORTER_VERSION,
)
from .pdf_pipeline import sha256_file
from .speaker_turn_pipeline import SPEAKER_TURN_PIPELINE_VERSION
from .turn_content import TURN_CONTENT_CLASSIFIER_VERSION

PROFILE_VERSION = "1"
TOKENIZER_VERSION = "diagnostic_lexical_nfkc_casefold_alnum_v1"

DEFAULT_DOCUMENTS_PATH = Path("data/processed/modeling_corpus/documents.jsonl")
DEFAULT_EXPORT_MANIFEST_PATH = Path("data/processed/modeling_corpus/export_manifest.json")
DEFAULT_CORPUS_LOCK_PATH = Path("data/qa/modeling_corpus_lock_v1.json")
DEFAULT_CONFIG_PATH = Path("config/topic_modeling/corpus_profile_v1.json")
DEFAULT_OUTPUT_DIR = Path("data/qa/topic_modeling/corpus_profile_v1")

UNIVERSE_ALL = "all_sessions"
UNIVERSE_PRIMARY = "primary"
UNIVERSES = (UNIVERSE_ALL, UNIVERSE_PRIMARY)

PROFILE_MANIFEST_FILENAME = "profile_manifest.json"
CORPUS_PROFILE_JSON_FILENAME = "corpus_profile.json"
CORPUS_PROFILE_MD_FILENAME = "corpus_profile.md"
COUNTS_BY_YEAR_FILENAME = "counts_by_year.csv"
COUNTS_BY_TEMPORAL_PERIOD_FILENAME = "counts_by_temporal_period.csv"
COUNTS_BY_SESSION_CATEGORY_FILENAME = "counts_by_session_category.csv"
COUNTS_BY_SPEAKER_FAMILY_FILENAME = "counts_by_speaker_family.csv"
COUNTS_BY_YEAR_AND_CATEGORY_FILENAME = "counts_by_year_and_category.csv"
DOCUMENT_LENGTH_HISTOGRAM_FILENAME = "document_length_histogram.csv"
TOKEN_FREQUENCY_FILENAME = "token_frequency.csv"
CANDIDATE_STOPWORDS_FILENAME = "candidate_stopwords.csv"
SUSPICIOUS_TOKENS_FILENAME = "suspicious_tokens.csv"
SAMPLED_DOCUMENTS_FILENAME = "sampled_documents.jsonl"
SAMPLED_BIGRAM_FREQUENCY_FILENAME = "sampled_bigram_frequency.csv"
PREPROCESSING_EXAMPLES_FILENAME = "preprocessing_examples.jsonl"

REQUIRED_DOCUMENT_FIELDS = frozenset(
    {
        "document_id",
        "source_record_id",
        "turn_index",
        "chunk_index",
        "modeling_text",
        "word_count",
        "session_date",
        "year",
        "temporal_period",
        "session_category",
        "speaker_family",
    }
)
ALLOWED_SPEAKER_FAMILIES = frozenset(INCLUDED_SPEAKER_FAMILIES)
CONFIG_FIELDS = frozenset(
    {
        "profile_version",
        "primary_session_categories",
        "random_seed",
        "top_token_count",
        "top_bigram_count",
        "candidate_stopword_min_document_fraction",
        "candidate_stopword_min_total_count",
        "sample_documents_per_stratum",
        "sample_strata",
        "context_examples_per_token",
        "minimum_bigram_token_length",
    }
)
EXPECTED_SAMPLE_STRATA = ("year", "session_category")
PROCEDURAL_SEED_TOKENS = frozenset(
    {
        "señor",
        "señora",
        "presidente",
        "presidenta",
        "diputado",
        "diputada",
        "honorable",
        "cámara",
        "sesión",
        "palabra",
        "gracias",
    }
)
QUANTILES = (
    ("p01", 0.01),
    ("p05", 0.05),
    ("p10", 0.10),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
    ("p95", 0.95),
    ("p99", 0.99),
)
PERIOD_ORDER = ("2008-2011", "2012-2015", "2016-2019", "2020-2023", "2024-2025")
TOKEN_FREQUENCY_FIELDS = (
    "universe",
    "rank_by_total_frequency",
    "rank_by_document_frequency",
    "token",
    "token_class",
    "token_length",
    "total_count",
    "document_count",
    "document_fraction",
)
PREPROCESSING_EXAMPLE_LIMIT_PER_UNIVERSE = 25
SUSPICIOUS_EXAMPLES_PER_ENTRY = 3
SUSPICIOUS_LONG_TOKEN_MIN_LENGTH = 30
MOJIBAKE_PATTERNS = ("Ã", "Â", "â€", "â€“", "â€”", "â")
REPEATED_CHARACTER_PATTERN = re.compile(r"(.)\1{3,}")


class CorpusProfileError(RuntimeError):
    """Raised when corpus profiling cannot complete safely."""


@dataclass(frozen=True, slots=True)
class ProfileConfig:
    """Strict configuration for the corpus profiling stage."""

    profile_version: str
    primary_session_categories: tuple[str, ...]
    random_seed: int
    top_token_count: int
    top_bigram_count: int
    candidate_stopword_min_document_fraction: float
    candidate_stopword_min_total_count: int
    sample_documents_per_stratum: int
    sample_strata: tuple[str, ...]
    context_examples_per_token: int
    minimum_bigram_token_length: int

    def to_json(self) -> dict[str, Any]:
        """Return a deterministic JSON-serializable configuration snapshot."""
        return {
            "candidate_stopword_min_document_fraction": (
                self.candidate_stopword_min_document_fraction
            ),
            "candidate_stopword_min_total_count": self.candidate_stopword_min_total_count,
            "context_examples_per_token": self.context_examples_per_token,
            "minimum_bigram_token_length": self.minimum_bigram_token_length,
            "primary_session_categories": list(self.primary_session_categories),
            "profile_version": self.profile_version,
            "random_seed": self.random_seed,
            "sample_documents_per_stratum": self.sample_documents_per_stratum,
            "sample_strata": list(self.sample_strata),
            "top_bigram_count": self.top_bigram_count,
            "top_token_count": self.top_token_count,
        }


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """Validated modelling-corpus document record."""

    document_id: str
    source_record_id: str
    turn_index: int
    chunk_index: int
    modeling_text: str
    word_count: int
    session_date: str
    year: int
    temporal_period: str
    session_category: str
    speaker_family: str

    @property
    def source_turn_key(self) -> str:
        """Return a stable unique key for the source turn."""
        return f"{self.source_record_id}::turn_{self.turn_index:06d}"


@dataclass(slots=True)
class GroupAccumulator:
    """Counts for one categorical group."""

    document_count: int = 0
    word_total: int = 0
    source_turns: set[str] = field(default_factory=set)
    sessions: set[str] = field(default_factory=set)

    def add(self, document: DocumentRecord) -> None:
        """Add one document to the group."""
        self.document_count += 1
        self.word_total += document.word_count
        self.source_turns.add(document.source_turn_key)
        self.sessions.add(document.source_record_id)

    def to_json(self) -> dict[str, Any]:
        """Return a deterministic JSON-serializable group summary."""
        return {
            "document_count": self.document_count,
            "mean_document_length": _safe_ratio(self.word_total, self.document_count),
            "session_count": len(self.sessions),
            "source_turn_count": len(self.source_turns),
            "word_total": self.word_total,
        }


@dataclass(frozen=True, slots=True)
class CorpusRollup:
    """Independent corpus totals for reconciliation."""

    document_count: int
    word_total: int
    source_turns: frozenset[str]
    sessions: frozenset[str]


@dataclass(frozen=True, slots=True)
class SampledDocument:
    """One bounded deterministic sample member."""

    universe: str
    document_id: str
    source_record_id: str
    turn_index: int
    chunk_index: int
    year: int
    temporal_period: str
    session_category: str
    speaker_family: str
    word_count: int
    stable_hash: str
    modeling_text: str

    @property
    def stratum_key(self) -> tuple[str, str]:
        """Return the configured sample stratum key."""
        return (str(self.year), self.session_category)


@dataclass(slots=True)
class DeterministicSampler:
    """Bounded stable-hash sampler by configured stratum."""

    universe: str
    random_seed: int
    per_stratum: int
    samples_by_stratum: dict[tuple[str, str], list[SampledDocument]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def add(self, document: DocumentRecord) -> None:
        """Consider one document for the deterministic sample."""
        stable_hash = _stable_document_hash(
            seed=self.random_seed,
            document_id=document.document_id,
        )
        sampled = SampledDocument(
            universe=self.universe,
            document_id=document.document_id,
            source_record_id=document.source_record_id,
            turn_index=document.turn_index,
            chunk_index=document.chunk_index,
            year=document.year,
            temporal_period=document.temporal_period,
            session_category=document.session_category,
            speaker_family=document.speaker_family,
            word_count=document.word_count,
            stable_hash=stable_hash,
            modeling_text=document.modeling_text,
        )
        stratum = sampled.stratum_key
        stratum_samples = self.samples_by_stratum[stratum]
        stratum_samples.append(sampled)
        stratum_samples.sort(key=lambda item: (item.stable_hash, item.document_id))

        if len(stratum_samples) > self.per_stratum:
            del stratum_samples[self.per_stratum :]

    def selected(self) -> list[SampledDocument]:
        """Return all selected documents in deterministic output order."""
        selected_documents: list[SampledDocument] = []

        for stratum in sorted(self.samples_by_stratum):
            selected_documents.extend(self.samples_by_stratum[stratum])

        return sorted(
            selected_documents,
            key=lambda item: (item.universe, item.stratum_key, item.stable_hash, item.document_id),
        )

    def counts_by_stratum(self) -> list[dict[str, Any]]:
        """Return sample counts by stratum."""
        rows: list[dict[str, Any]] = []

        for stratum in sorted(self.samples_by_stratum):
            rows.append(
                {
                    "document_count": len(self.samples_by_stratum[stratum]),
                    "session_category": stratum[1],
                    "universe": self.universe,
                    "year": int(stratum[0]),
                }
            )

        return rows


@dataclass(frozen=True, slots=True)
class SuspiciousExample:
    """Deterministic evidence example for a suspicious-token row."""

    document_id: str
    snippet: str


@dataclass(slots=True)
class SuspiciousEntry:
    """Aggregated suspicious-token evidence."""

    total_count: int = 0
    document_count: int = 0
    examples: list[SuspiciousExample] = field(default_factory=list)

    def add(self, *, count: int, document_id: str, snippet: str) -> None:
        """Add one document-level observation."""
        self.total_count += count
        self.document_count += 1
        example = SuspiciousExample(document_id=document_id, snippet=snippet)

        if any(existing.document_id == document_id for existing in self.examples):
            return

        self.examples.append(example)
        self.examples.sort(key=lambda item: item.document_id)
        del self.examples[SUSPICIOUS_EXAMPLES_PER_ENTRY:]


@dataclass(slots=True)
class SuspiciousAccumulator:
    """Suspicious-token rows across universes."""

    entries: dict[tuple[str, str, str], SuspiciousEntry] = field(default_factory=dict)

    def add_document_observations(
        self,
        *,
        universe: str,
        document_id: str,
        document_count_denominator: int,
        observations: Mapping[tuple[str, str], int],
        snippets: Mapping[tuple[str, str], str],
    ) -> None:
        """Add all suspicious observations for one document and universe."""
        del document_count_denominator

        for (reason, value), count in observations.items():
            key = (universe, reason, value)
            entry = self.entries.setdefault(key, SuspiciousEntry())
            entry.add(
                count=count,
                document_id=document_id,
                snippet=snippets.get((reason, value), ""),
            )

    def row_count(self) -> int:
        """Return the suspicious output row count without materializing rows."""
        return len(self.entries)

    def iter_rows(self, *, denominators: Mapping[str, int]) -> Iterator[dict[str, Any]]:
        """Yield sorted suspicious-token CSV rows."""
        for universe, reason, value in sorted(
            self.entries,
            key=lambda key: _suspicious_entry_sort_key(key, self.entries[key]),
        ):
            entry = self.entries[(universe, reason, value)]
            yield {
                "document_count": entry.document_count,
                "document_fraction": _safe_ratio(
                    entry.document_count,
                    denominators.get(universe, 0),
                ),
                "example_document_ids": _json_compact(
                    [example.document_id for example in entry.examples]
                ),
                "example_snippets": _json_compact([example.snippet for example in entry.examples]),
                "reason": reason,
                "snippet_text_kind": _suspicious_snippet_text_kind(reason),
                "token_or_anomaly": value,
                "total_count": entry.total_count,
                "universe": universe,
            }

    def rows(self, *, denominators: Mapping[str, int]) -> list[dict[str, Any]]:
        """Return sorted suspicious-token CSV rows."""
        return list(self.iter_rows(denominators=denominators))


def _suspicious_entry_sort_key(
    key: tuple[str, str, str],
    entry: SuspiciousEntry,
) -> tuple[int, int, int, str, str]:
    """Return the required deterministic suspicious-row sort key."""
    universe, reason, value = key
    universe_index = UNIVERSES.index(universe) if universe in UNIVERSES else len(UNIVERSES)
    return (
        universe_index,
        -entry.total_count,
        -entry.document_count,
        reason,
        value,
    )


@dataclass(slots=True)
class UniverseState:
    """Streaming statistics for one analytical universe."""

    name: str
    document_count: int = 0
    word_total: int = 0
    source_turns: set[str] = field(default_factory=set)
    sessions: set[str] = field(default_factory=set)
    word_counts: list[int] = field(default_factory=list)
    histogram: Counter[int] = field(default_factory=Counter)
    counts_by_year: dict[int, GroupAccumulator] = field(default_factory=dict)
    counts_by_temporal_period: dict[str, GroupAccumulator] = field(default_factory=dict)
    counts_by_session_category: dict[str, GroupAccumulator] = field(default_factory=dict)
    counts_by_speaker_family: dict[str, GroupAccumulator] = field(default_factory=dict)
    counts_by_year_and_category: dict[tuple[int, str], GroupAccumulator] = field(
        default_factory=dict
    )
    token_total_counter: Counter[str] = field(default_factory=Counter)
    token_document_counter: Counter[str] = field(default_factory=Counter)
    token_class_counts: Counter[str] = field(default_factory=Counter)
    token_length_class_counts: Counter[str] = field(default_factory=Counter)
    tokens_containing_digits: int = 0
    tokens_containing_only_digits: int = 0
    year_token_sets: dict[int, set[str]] = field(default_factory=lambda: defaultdict(set))

    def add(self, *, document: DocumentRecord, tokens: Sequence[str]) -> None:
        """Add one document and its diagnostic tokens."""
        self.document_count += 1
        self.word_total += document.word_count
        self.source_turns.add(document.source_turn_key)
        self.sessions.add(document.source_record_id)
        self.word_counts.append(document.word_count)
        self.histogram[document.word_count] += 1

        _group_for(self.counts_by_year, document.year).add(document)
        _group_for(self.counts_by_temporal_period, document.temporal_period).add(document)
        _group_for(self.counts_by_session_category, document.session_category).add(document)
        _group_for(self.counts_by_speaker_family, document.speaker_family).add(document)
        _group_for(
            self.counts_by_year_and_category,
            (document.year, document.session_category),
        ).add(document)

        token_counts = Counter(tokens)
        self.token_total_counter.update(token_counts)
        self.token_document_counter.update(token_counts.keys())
        self.year_token_sets[document.year].update(token_counts.keys())

        for token, count in token_counts.items():
            token_class = classify_token(token)
            self.token_class_counts[token_class] += count
            self.token_length_class_counts[_token_length_class(token)] += count

            if any(character.isdigit() for character in token):
                self.tokens_containing_digits += count

            if token.isdigit():
                self.tokens_containing_only_digits += count

    def lexical_total(self) -> int:
        """Return total diagnostic lexical-token occurrences."""
        return self.token_total_counter.total()

    def to_json(self) -> dict[str, Any]:
        """Return deterministic JSON summary for this universe."""
        lexical_total = self.lexical_total()
        document_frequency_counts = Counter(self.token_document_counter.values())
        total_frequency_counts = Counter(self.token_total_counter.values())

        return {
            "counts": {
                "documents": self.document_count,
                "unique_sessions": len(self.sessions),
                "unique_source_turns": len(self.source_turns),
                "words": self.word_total,
            },
            "document_length": _length_statistics(self.word_counts),
            "grouped_counts": {
                "by_session_category": _group_json(self.counts_by_session_category),
                "by_speaker_family": _group_json(self.counts_by_speaker_family),
                "by_temporal_period": _group_json(self.counts_by_temporal_period),
                "by_year": _group_json(self.counts_by_year),
                "by_year_and_session_category": _year_category_group_json(
                    self.counts_by_year_and_category
                ),
            },
            "lexical": {
                "alphabetic_token_share": _safe_ratio(
                    self.token_class_counts["alphabetic"],
                    lexical_total,
                ),
                "hapax_count": total_frequency_counts.get(1, 0),
                "mixed_alphanumeric_token_share": _safe_ratio(
                    self.token_class_counts["mixed_alphanumeric"],
                    lexical_total,
                ),
                "numeric_token_share": _safe_ratio(
                    self.token_class_counts["numeric"],
                    lexical_total,
                ),
                "one_character_token_share": _safe_ratio(
                    self.token_length_class_counts["one_character"],
                    lexical_total,
                ),
                "tokens_appearing_in_exactly_1_document": document_frequency_counts.get(1, 0),
                "tokens_appearing_in_exactly_2_documents": document_frequency_counts.get(2, 0),
                "tokens_appearing_in_exactly_3_documents": document_frequency_counts.get(3, 0),
                "tokens_appearing_in_exactly_4_documents": document_frequency_counts.get(4, 0),
                "tokens_appearing_in_exactly_5_documents": document_frequency_counts.get(5, 0),
                "tokens_containing_digits": self.tokens_containing_digits,
                "tokens_containing_only_digits": self.tokens_containing_only_digits,
                "total_lexical_tokens": lexical_total,
                "two_character_token_share": _safe_ratio(
                    self.token_length_class_counts["two_character"],
                    lexical_total,
                ),
                "unique_lexical_tokens": len(self.token_total_counter),
                "vocabulary_growth_by_year": _vocabulary_growth(self.year_token_sets),
            },
        }


@dataclass(slots=True)
class ProfileRun:
    """Completed profile state before output writing."""

    config: ProfileConfig
    input_hashes: dict[str, str]
    export_manifest: dict[str, Any]
    corpus_lock: dict[str, Any]
    states: dict[str, UniverseState]
    samplers: dict[str, DeterministicSampler]
    suspicious: SuspiciousAccumulator
    processed_document_count: int
    processed_word_total: int


def _json_text(payload: Mapping[str, Any]) -> str:
    """Return deterministic pretty JSON text."""
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _json_compact(payload: object) -> str:
    """Return deterministic compact JSON text for CSV cells."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    """Return a rounded ratio, using zero when the denominator is zero."""
    if denominator == 0:
        return 0.0

    return round(float(numerator) / float(denominator), 10)


def _safe_int(value: object, *, field_name: str) -> int:
    """Return a strict JSON integer, rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise CorpusProfileError(f"Invalid integer for {field_name}: {value!r}")

    return value


def _safe_float(value: object, *, field_name: str) -> float:
    """Return a strict JSON number, rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise CorpusProfileError(f"Invalid number for {field_name}: {value!r}")

    return float(value)


def _required_string(payload: Mapping[str, Any], field_name: str) -> str:
    """Return a required nonempty string field."""
    value = payload.get(field_name)

    if not isinstance(value, str) or not value:
        raise CorpusProfileError(f"Missing or invalid {field_name}.")

    return value


def _string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    """Return a nonempty tuple of strings."""
    if not isinstance(value, list) or not value:
        raise CorpusProfileError(f"Invalid string list for {field_name}.")

    items: list[str] = []

    for item in value:
        if not isinstance(item, str) or not item:
            raise CorpusProfileError(f"Invalid string in {field_name}: {item!r}")

        items.append(item)

    return tuple(items)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    """Read a UTF-8 JSON object, accepting an external UTF-8 BOM."""
    if not path.is_file():
        raise CorpusProfileError(f"{label} does not exist: {path}")

    try:
        payload: object = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise CorpusProfileError(f"Could not read {label}: {path}") from error

    if not isinstance(payload, dict):
        raise CorpusProfileError(f"Expected {label} to contain a JSON object: {path}")

    return {str(key): value for key, value in payload.items()}


def _load_config(path: Path) -> ProfileConfig:
    """Load and strictly validate the versioned profile configuration."""
    payload = _read_json_object(path, label="profile configuration")
    missing = CONFIG_FIELDS - set(payload)
    unexpected = set(payload) - CONFIG_FIELDS

    if missing:
        raise CorpusProfileError(f"Profile configuration is missing fields: {sorted(missing)}")

    if unexpected:
        raise CorpusProfileError(
            f"Profile configuration has unsupported fields: {sorted(unexpected)}"
        )

    profile_version = _required_string(payload, "profile_version")

    if profile_version != PROFILE_VERSION:
        raise CorpusProfileError(f"Unsupported profile_version: {profile_version}")

    primary_session_categories = _string_tuple(
        payload["primary_session_categories"],
        field_name="primary_session_categories",
    )
    sample_strata = _string_tuple(payload["sample_strata"], field_name="sample_strata")

    if sample_strata != EXPECTED_SAMPLE_STRATA:
        raise CorpusProfileError(
            "sample_strata must be exactly ['year', 'session_category'] for this profile version."
        )

    config = ProfileConfig(
        profile_version=profile_version,
        primary_session_categories=primary_session_categories,
        random_seed=_safe_int(payload["random_seed"], field_name="random_seed"),
        top_token_count=_safe_int(payload["top_token_count"], field_name="top_token_count"),
        top_bigram_count=_safe_int(payload["top_bigram_count"], field_name="top_bigram_count"),
        candidate_stopword_min_document_fraction=_safe_float(
            payload["candidate_stopword_min_document_fraction"],
            field_name="candidate_stopword_min_document_fraction",
        ),
        candidate_stopword_min_total_count=_safe_int(
            payload["candidate_stopword_min_total_count"],
            field_name="candidate_stopword_min_total_count",
        ),
        sample_documents_per_stratum=_safe_int(
            payload["sample_documents_per_stratum"],
            field_name="sample_documents_per_stratum",
        ),
        sample_strata=sample_strata,
        context_examples_per_token=_safe_int(
            payload["context_examples_per_token"],
            field_name="context_examples_per_token",
        ),
        minimum_bigram_token_length=_safe_int(
            payload["minimum_bigram_token_length"],
            field_name="minimum_bigram_token_length",
        ),
    )

    if config.top_token_count < 1:
        raise CorpusProfileError("top_token_count must be positive.")
    if config.top_bigram_count < 1:
        raise CorpusProfileError("top_bigram_count must be positive.")
    if not 0 <= config.candidate_stopword_min_document_fraction <= 1:
        raise CorpusProfileError("candidate_stopword_min_document_fraction must be in [0, 1].")
    if config.candidate_stopword_min_total_count < 1:
        raise CorpusProfileError("candidate_stopword_min_total_count must be positive.")
    if config.sample_documents_per_stratum < 1:
        raise CorpusProfileError("sample_documents_per_stratum must be positive.")
    if config.context_examples_per_token < 0:
        raise CorpusProfileError("context_examples_per_token cannot be negative.")
    if config.minimum_bigram_token_length < 1:
        raise CorpusProfileError("minimum_bigram_token_length must be positive.")

    return config


def _temporal_period_for_year(year: int) -> str:
    """Return the locked modelling-corpus temporal period for a year."""
    if 2008 <= year <= 2011:
        return "2008-2011"
    if 2012 <= year <= 2015:
        return "2012-2015"
    if 2016 <= year <= 2019:
        return "2016-2019"
    if 2020 <= year <= 2023:
        return "2020-2023"
    if 2024 <= year <= 2025:
        return "2024-2025"

    raise CorpusProfileError(f"Session year outside 2008-2025: {year}")


def _validate_document_record(payload: Mapping[str, Any], *, line_number: int) -> DocumentRecord:
    """Strictly validate one modelling-corpus JSONL record."""
    missing = REQUIRED_DOCUMENT_FIELDS - set(payload)

    if missing:
        raise CorpusProfileError(
            f"Missing required document fields at line {line_number}: {sorted(missing)}"
        )

    document_id = _required_string(payload, "document_id")
    source_record_id = _required_string(payload, "source_record_id")
    modeling_text = _required_string(payload, "modeling_text")
    session_date = _required_string(payload, "session_date")
    temporal_period = _required_string(payload, "temporal_period")
    session_category = _required_string(payload, "session_category")
    speaker_family = _required_string(payload, "speaker_family")
    turn_index = _safe_int(payload["turn_index"], field_name="turn_index")
    chunk_index = _safe_int(payload["chunk_index"], field_name="chunk_index")
    word_count = _safe_int(payload["word_count"], field_name="word_count")
    year = _safe_int(payload["year"], field_name="year")

    if turn_index < 0:
        raise CorpusProfileError(f"Invalid turn_index at line {line_number}: {turn_index}")
    if chunk_index < 1:
        raise CorpusProfileError(f"Invalid chunk_index at line {line_number}: {chunk_index}")
    if not 1 <= word_count <= 300:
        raise CorpusProfileError(f"Invalid word_count at line {line_number}: {word_count}")
    if len(modeling_text.split()) != word_count:
        raise CorpusProfileError(
            f"modeling_text whitespace word count does not match word_count at line {line_number}"
        )
    if speaker_family not in ALLOWED_SPEAKER_FAMILIES:
        raise CorpusProfileError(f"Invalid speaker_family at line {line_number}: {speaker_family}")

    try:
        parsed_date = date.fromisoformat(session_date)
    except ValueError as error:
        raise CorpusProfileError(
            f"Invalid session_date at line {line_number}: {session_date}"
        ) from error

    if parsed_date.year != year:
        raise CorpusProfileError(
            f"session_date/year mismatch at line {line_number}: {session_date} vs {year}"
        )

    expected_period = _temporal_period_for_year(year)

    if temporal_period != expected_period:
        raise CorpusProfileError(
            f"temporal_period mismatch at line {line_number}: {temporal_period}"
        )

    modeling_word_count = payload.get("modeling_word_count")

    if (
        modeling_word_count is not None
        and _safe_int(
            modeling_word_count,
            field_name="modeling_word_count",
        )
        != word_count
    ):
        raise CorpusProfileError(f"modeling_word_count mismatch at line {line_number}")

    return DocumentRecord(
        document_id=document_id,
        source_record_id=source_record_id,
        turn_index=turn_index,
        chunk_index=chunk_index,
        modeling_text=modeling_text,
        word_count=word_count,
        session_date=session_date,
        year=year,
        temporal_period=temporal_period,
        session_category=session_category,
        speaker_family=speaker_family,
    )


def _validate_manifest_and_lock(
    *,
    documents_path: Path,
    export_manifest_path: Path,
    corpus_lock_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Validate locked inputs before any profiling work."""
    for label, path in (
        ("documents", documents_path),
        ("export manifest", export_manifest_path),
        ("corpus lock", corpus_lock_path),
    ):
        if not path.is_file():
            raise CorpusProfileError(f"{label} does not exist: {path}")

    export_manifest_sha256 = sha256_file(export_manifest_path)
    corpus_lock_sha256 = sha256_file(corpus_lock_path)
    export_manifest = _read_json_object(export_manifest_path, label="export manifest")
    corpus_lock = _read_json_object(corpus_lock_path, label="corpus lock")

    if corpus_lock.get("export_manifest_sha256") != export_manifest_sha256:
        raise CorpusProfileError("Corpus lock export manifest SHA-256 does not match.")

    locked_manifest_path = corpus_lock.get("export_manifest_path")

    if isinstance(locked_manifest_path, str) and locked_manifest_path:
        if Path(locked_manifest_path).name != export_manifest_path.name:
            raise CorpusProfileError("Corpus lock references a different export manifest.")

    output_files = export_manifest.get("output_files")

    if not isinstance(output_files, dict):
        raise CorpusProfileError("Export manifest has no valid output_files object.")

    documents_metadata = output_files.get("documents")

    if not isinstance(documents_metadata, dict):
        raise CorpusProfileError("Export manifest has no documents output metadata.")

    expected_documents_sha256 = documents_metadata.get("sha256")

    if not isinstance(expected_documents_sha256, str) or not expected_documents_sha256:
        raise CorpusProfileError("Export manifest has no valid documents SHA-256.")

    documents_sha256 = sha256_file(documents_path)

    if documents_sha256 != expected_documents_sha256:
        raise CorpusProfileError("documents.jsonl SHA-256 does not match export manifest.")

    lock_output_files = corpus_lock.get("output_files")

    if not isinstance(lock_output_files, dict):
        raise CorpusProfileError("Corpus lock has no valid output_files object.")

    lock_documents_metadata = lock_output_files.get("documents")

    if not isinstance(lock_documents_metadata, dict):
        raise CorpusProfileError("Corpus lock has no documents output metadata.")

    if lock_documents_metadata.get("sha256") != documents_sha256:
        raise CorpusProfileError("documents.jsonl SHA-256 does not match corpus lock.")

    _validate_locked_totals(export_manifest=export_manifest, corpus_lock=corpus_lock)
    _validate_expected_versions(export_manifest=export_manifest, corpus_lock=corpus_lock)

    return (
        export_manifest,
        corpus_lock,
        {
            "corpus_lock_sha256": corpus_lock_sha256,
            "documents_sha256": documents_sha256,
            "export_manifest_sha256": export_manifest_sha256,
        },
    )


def _validate_locked_totals(
    *,
    export_manifest: Mapping[str, Any],
    corpus_lock: Mapping[str, Any],
) -> None:
    """Validate that export manifest totals are locked unchanged."""
    total_fields = (
        "exclusion_ledger_count",
        "input_session_count",
        "input_turn_count",
        "modeling_document_count",
        "modeling_word_total",
        "positive_speech_turn_count",
        "retained_source_turn_count",
        "retained_source_turn_word_total",
    )

    for field_name in total_fields:
        export_value = _required_locked_int(export_manifest, field_name, label="export manifest")
        lock_value = _required_locked_int(corpus_lock, field_name, label="corpus lock")

        if export_value != lock_value:
            raise CorpusProfileError(
                f"Export manifest total does not match corpus lock: {field_name}"
            )


def _validate_expected_versions(
    *,
    export_manifest: Mapping[str, Any],
    corpus_lock: Mapping[str, Any],
) -> None:
    """Validate exporter, parser and classifier versions."""
    expected_versions = {
        "content_classifier_version": TURN_CONTENT_CLASSIFIER_VERSION,
        "exporter_version": MODELING_CORPUS_EXPORTER_VERSION,
        "pipeline_version": SPEAKER_TURN_PIPELINE_VERSION,
    }

    for field_name, expected in expected_versions.items():
        export_value = _required_locked_string(
            export_manifest,
            field_name,
            label="export manifest",
        )
        lock_value = _required_locked_string(corpus_lock, field_name, label="corpus lock")

        if export_value != expected:
            raise CorpusProfileError(f"Export manifest {field_name} mismatch: {export_value!r}")
        if lock_value != expected:
            raise CorpusProfileError(f"Corpus lock {field_name} mismatch: {lock_value!r}")


def _required_locked_int(payload: Mapping[str, Any], field_name: str, *, label: str) -> int:
    """Return a required locked integer field."""
    if field_name not in payload:
        raise CorpusProfileError(f"Missing required {label} total field: {field_name}")

    value = payload[field_name]

    if isinstance(value, bool) or not isinstance(value, int):
        raise CorpusProfileError(f"Invalid {label} total field {field_name}: {value!r}")

    return value


def _required_locked_string(payload: Mapping[str, Any], field_name: str, *, label: str) -> str:
    """Return a required locked string field."""
    if field_name not in payload:
        raise CorpusProfileError(f"Missing required {label} version field: {field_name}")

    value = payload[field_name]

    if not isinstance(value, str) or not value:
        raise CorpusProfileError(f"Invalid {label} version field {field_name}: {value!r}")

    return value


def normalize_for_profile_tokenization(text: str) -> str:
    """Apply the versioned diagnostic normalization policy."""
    return unicodedata.normalize("NFKC", text).casefold()


def diagnostic_tokens(text: str) -> list[str]:
    """Tokenize modelling text with the explicit diagnostic lexical policy.

    Policy: NFKC normalization, casefolding, Unicode alphanumeric runs as tokens,
    and deterministic splitting on non-alphanumeric boundaries. Accented letters,
    numeric tokens, and mixed alphanumeric tokens are preserved.
    """
    normalized = normalize_for_profile_tokenization(text)
    tokens: list[str] = []
    current: list[str] = []

    for character in normalized:
        if character.isalnum():
            current.append(character)
            continue

        if current:
            tokens.append("".join(current))
            current.clear()

    if current:
        tokens.append("".join(current))

    return tokens


def classify_token(token: str) -> str:
    """Return the diagnostic token class."""
    if token.isalpha():
        return "alphabetic"

    if token.isdigit():
        return "numeric"

    return "mixed_alphanumeric"


def _token_length_class(token: str) -> str:
    """Return the configured token-length class."""
    if len(token) == 1:
        return "one_character"
    if len(token) == 2:
        return "two_character"

    return "three_or_more_characters"


def _stable_document_hash(*, seed: int, document_id: str) -> str:
    """Return the stable sample rank hash for a document."""
    return hashlib.sha256(f"{seed}:{document_id}".encode()).hexdigest()


def _group_for[K](groups: dict[K, GroupAccumulator], key: K) -> GroupAccumulator:
    """Return the accumulator for a group, creating it when needed."""
    group = groups.get(key)

    if group is None:
        group = GroupAccumulator()
        groups[key] = group

    return group


def _length_statistics(values: Sequence[int]) -> dict[str, Any]:
    """Return exact document-length statistics for one universe."""
    if not values:
        return {
            "maximum": 0,
            "mean_words_per_document": 0.0,
            "median": 0.0,
            "minimum": 0,
            "quantiles": {name: 0.0 for name, _ in QUANTILES},
            "standard_deviation": 0.0,
        }

    sorted_values = sorted(values)

    return {
        "maximum": max(values),
        "mean_words_per_document": round(sum(values) / len(values), 10),
        "median": round(float(median(values)), 10),
        "minimum": min(values),
        "quantiles": {
            name: round(_percentile(sorted_values, probability), 10)
            for name, probability in QUANTILES
        },
        "standard_deviation": round(float(pstdev(values)), 10),
    }


def _percentile(sorted_values: Sequence[int], probability: float) -> float:
    """Return a deterministic linear-interpolated percentile."""
    if not sorted_values:
        return 0.0

    if len(sorted_values) == 1:
        return float(sorted_values[0])

    position = (len(sorted_values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return float(sorted_values[lower_index])

    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    weight = position - lower_index
    return float(lower + ((upper - lower) * weight))


SimpleGroupKey = Any
SimpleGroupMap = Mapping[SimpleGroupKey, GroupAccumulator]


def _group_json(groups: SimpleGroupMap) -> list[dict[str, Any]]:
    """Return sorted group JSON records."""
    rows: list[dict[str, Any]] = []

    for key in sorted(groups, key=_simple_group_sort_key):
        group_payload = groups[key].to_json()
        group_payload["group"] = key
        rows.append(group_payload)

    return rows


def _year_category_group_json(
    groups: Mapping[tuple[int, str], GroupAccumulator],
) -> list[dict[str, Any]]:
    """Return sorted year-category group JSON records."""
    rows: list[dict[str, Any]] = []

    for year, session_category in sorted(groups):
        group_payload = groups[(year, session_category)].to_json()
        group_payload["session_category"] = session_category
        group_payload["year"] = year
        rows.append(group_payload)

    return rows


def _simple_group_sort_key(value: SimpleGroupKey) -> tuple[int, str]:
    """Return a stable sort key for simple group values."""
    if isinstance(value, int):
        return (0, f"{value:04d}")

    if isinstance(value, str) and value in PERIOD_ORDER:
        return (1, f"{PERIOD_ORDER.index(value):02d}")

    return (2, str(value))


def _simple_group_map(state: UniverseState, group_name: str) -> SimpleGroupMap:
    """Return one typed simple group mapping by name."""
    if group_name == "year":
        return cast(SimpleGroupMap, state.counts_by_year)
    if group_name == "temporal_period":
        return cast(SimpleGroupMap, state.counts_by_temporal_period)
    if group_name == "session_category":
        return cast(SimpleGroupMap, state.counts_by_session_category)
    if group_name == "speaker_family":
        return cast(SimpleGroupMap, state.counts_by_speaker_family)

    raise CorpusProfileError(f"Unsupported group name: {group_name}")


def _vocabulary_growth(year_token_sets: Mapping[int, set[str]]) -> list[dict[str, Any]]:
    """Return cumulative vocabulary growth by year."""
    cumulative: set[str] = set()
    rows: list[dict[str, Any]] = []

    for year in sorted(year_token_sets):
        tokens = year_token_sets[year]
        new_tokens = tokens - cumulative
        cumulative.update(tokens)
        rows.append(
            {
                "cumulative_unique_tokens": len(cumulative),
                "new_unique_tokens": len(new_tokens),
                "unique_tokens_in_year": len(tokens),
                "year": year,
            }
        )

    return rows


def _snippet_from_text(text: str, *, value: str, max_length: int = 180) -> str:
    """Return a short snippet around a value in already prepared text."""
    flattened = " ".join(text.split())
    index = flattened.find(value)

    if index < 0:
        return flattened[:max_length]

    start = max(0, index - 60)
    end = min(len(flattened), index + len(value) + 60)
    snippet = flattened[start:end]

    if start > 0:
        snippet = "..." + snippet
    if end < len(flattened):
        snippet = snippet + "..."

    return snippet[:max_length]


def _raw_snippet(text: str, *, value: str) -> str:
    """Return a snippet from raw flattened modelling text."""
    return _snippet_from_text(text, value=value)


def _normalized_token_snippet(text: str, *, token: str) -> str:
    """Return a snippet from NFKC-casefolded modelling text for token diagnostics."""
    return _snippet_from_text(normalize_for_profile_tokenization(text), value=token)


def _suspicious_snippet_text_kind(reason: str) -> str:
    """Return the snippet text basis for a suspicious diagnostic reason."""
    if reason in {
        "mixed_letters_and_digits",
        "numeric_only_token",
        "one_character_alphabetic_token",
        "repeated_identical_character_four_or_more",
        "unusually_long_token",
    }:
        return "normalized_nfkc_casefolded_modeling_text"

    return "raw_modeling_text"


def _suspicious_observations(
    *,
    document: DocumentRecord,
    tokens: Sequence[str],
) -> tuple[Counter[tuple[str, str]], dict[tuple[str, str], str]]:
    """Return suspicious diagnostics for one document."""
    observations: Counter[tuple[str, str]] = Counter()
    snippets: dict[tuple[str, str], str] = {}
    text = document.modeling_text

    def add_raw(reason: str, value: str, count: int, snippet_value: str | None = None) -> None:
        key = (reason, value)
        observations[key] += count
        snippets.setdefault(key, _raw_snippet(text, value=snippet_value or value))

    def add_token(reason: str, token: str, count: int) -> None:
        key = (reason, token)
        observations[key] += count
        snippets.setdefault(key, _normalized_token_snippet(text, token=token))

    if "\ufffd" in text:
        add_raw("replacement_character", "U+FFFD", text.count("\ufffd"), "\ufffd")

    if "\u00ad" in text:
        add_raw("soft_hyphen", "U+00AD", text.count("\u00ad"), "\u00ad")

    for pattern in MOJIBAKE_PATTERNS:
        count = text.count(pattern)

        if count:
            add_raw("mojibake_like_sequence", pattern, count, pattern)

    raw_character_counts = Counter(text)

    for character, count in raw_character_counts.items():
        category = unicodedata.category(character)

        if category == "Cc" and character not in "\n\r\t":
            add_raw("control_character_in_raw_text", _unicode_label(character), count, character)

        if category in {"Cf", "Co", "Cs", "Cn"} and character != "\u00ad":
            add_raw(
                "unexpected_unicode_category_in_raw_text",
                f"{_unicode_label(character)}:{category}",
                count,
                character,
            )

    token_counts = Counter(tokens)

    for token, count in token_counts.items():
        token_class = classify_token(token)

        if (
            token_class == "mixed_alphanumeric"
            and any(character.isalpha() for character in token)
            and any(character.isdigit() for character in token)
        ):
            add_token("mixed_letters_and_digits", token, count)

        if token.isdigit():
            add_token("numeric_only_token", token, count)

        if token.isalpha() and len(token) == 1:
            add_token("one_character_alphabetic_token", token, count)

        if len(token) >= SUSPICIOUS_LONG_TOKEN_MIN_LENGTH:
            add_token("unusually_long_token", token, count)

        if REPEATED_CHARACTER_PATTERN.search(token) is not None:
            add_token("repeated_identical_character_four_or_more", token, count)

    return observations, snippets


def _unicode_label(character: str) -> str:
    """Return a stable Unicode code point label."""
    return f"U+{ord(character):04X}"


def _iter_documents(path: Path) -> Iterator[DocumentRecord]:
    """Stream and validate modelling documents."""
    try:
        with path.open("r", encoding="utf-8-sig") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    raise CorpusProfileError(f"Blank JSONL record at line {line_number}")

                try:
                    payload: object = json.loads(line)
                except json.JSONDecodeError as error:
                    raise CorpusProfileError(f"Malformed JSONL at line {line_number}") from error

                if not isinstance(payload, dict):
                    raise CorpusProfileError(f"JSONL record is not an object at line {line_number}")

                yield _validate_document_record(
                    {str(key): value for key, value in payload.items()},
                    line_number=line_number,
                )
    except OSError as error:
        raise CorpusProfileError(f"Could not read documents JSONL: {path}") from error


def _stream_profile(
    *,
    documents_path: Path,
    export_manifest: dict[str, Any],
    corpus_lock: dict[str, Any],
    config: ProfileConfig,
    input_hashes: dict[str, str],
) -> ProfileRun:
    """Stream documents into exact counters and bounded samples."""
    states = {
        UNIVERSE_ALL: UniverseState(name=UNIVERSE_ALL),
        UNIVERSE_PRIMARY: UniverseState(name=UNIVERSE_PRIMARY),
    }
    samplers = {
        UNIVERSE_ALL: DeterministicSampler(
            universe=UNIVERSE_ALL,
            random_seed=config.random_seed,
            per_stratum=config.sample_documents_per_stratum,
        ),
        UNIVERSE_PRIMARY: DeterministicSampler(
            universe=UNIVERSE_PRIMARY,
            random_seed=config.random_seed,
            per_stratum=config.sample_documents_per_stratum,
        ),
    }
    suspicious = SuspiciousAccumulator()
    seen_document_ids: set[str] = set()
    processed_document_count = 0
    processed_word_total = 0

    for document in _iter_documents(documents_path):
        if document.document_id in seen_document_ids:
            raise CorpusProfileError(f"Duplicate document_id: {document.document_id}")

        seen_document_ids.add(document.document_id)
        processed_document_count += 1
        processed_word_total += document.word_count
        tokens = diagnostic_tokens(document.modeling_text)
        observations, snippets = _suspicious_observations(document=document, tokens=tokens)

        universes_for_document = [UNIVERSE_ALL]

        if document.session_category in config.primary_session_categories:
            universes_for_document.append(UNIVERSE_PRIMARY)

        for universe in universes_for_document:
            states[universe].add(document=document, tokens=tokens)
            samplers[universe].add(document)
            suspicious.add_document_observations(
                universe=universe,
                document_id=document.document_id,
                document_count_denominator=states[universe].document_count,
                observations=observations,
                snippets=snippets,
            )

    manifest_document_count = _safe_int(
        export_manifest.get("modeling_document_count"),
        field_name="modeling_document_count",
    )
    manifest_word_total = _safe_int(
        export_manifest.get("modeling_word_total"),
        field_name="modeling_word_total",
    )

    if processed_document_count != manifest_document_count:
        raise CorpusProfileError(
            "Processed document count differs from export manifest: "
            f"{processed_document_count} != {manifest_document_count}"
        )

    if processed_word_total != manifest_word_total:
        raise CorpusProfileError(
            "Processed word total differs from export manifest: "
            f"{processed_word_total} != {manifest_word_total}"
        )

    return ProfileRun(
        config=config,
        input_hashes=input_hashes,
        export_manifest=export_manifest,
        corpus_lock=corpus_lock,
        states=states,
        samplers=samplers,
        suspicious=suspicious,
        processed_document_count=processed_document_count,
        processed_word_total=processed_word_total,
    )


TokenRankMaps = Mapping[str, tuple[dict[str, int], dict[str, int]]]


def _token_rank_maps(state: UniverseState) -> tuple[dict[str, int], dict[str, int]]:
    """Return total-frequency and document-frequency rank maps."""
    total_ranked = sorted(
        state.token_total_counter,
        key=lambda token: (-state.token_total_counter[token], token),
    )
    document_ranked = sorted(
        state.token_document_counter,
        key=lambda token: (-state.token_document_counter[token], token),
    )
    return (
        {token: rank for rank, token in enumerate(total_ranked, start=1)},
        {token: rank for rank, token in enumerate(document_ranked, start=1)},
    )


def _all_token_rank_maps(
    states: Mapping[str, UniverseState],
) -> dict[str, tuple[dict[str, int], dict[str, int]]]:
    """Return token rank maps for every universe."""
    return {universe: _token_rank_maps(states[universe]) for universe in UNIVERSES}


def _token_frequency_rows(
    states: Mapping[str, UniverseState],
    rank_maps: TokenRankMaps,
) -> Iterator[dict[str, Any]]:
    """Yield exact unigram rows for all universes."""
    for universe in UNIVERSES:
        state = states[universe]
        total_ranks, document_ranks = rank_maps[universe]

        for token in sorted(
            state.token_total_counter,
            key=lambda item: (total_ranks[item], item),
        ):
            yield {
                "document_count": state.token_document_counter[token],
                "document_fraction": _safe_ratio(
                    state.token_document_counter[token],
                    state.document_count,
                ),
                "rank_by_document_frequency": document_ranks[token],
                "rank_by_total_frequency": total_ranks[token],
                "token": token,
                "token_class": classify_token(token),
                "token_length": len(token),
                "total_count": state.token_total_counter[token],
                "universe": universe,
            }


def _top_token_rows(
    *,
    states: Mapping[str, UniverseState],
    rank_maps: TokenRankMaps,
    top_count: int,
) -> list[dict[str, Any]]:
    """Return top tokens by total frequency for compact reports."""
    rows: list[dict[str, Any]] = []

    for universe in UNIVERSES:
        state = states[universe]
        total_ranks, document_ranks = rank_maps[universe]

        for token in sorted(
            state.token_total_counter,
            key=lambda item: (total_ranks[item], item),
        )[:top_count]:
            rows.append(
                {
                    "document_count": state.token_document_counter[token],
                    "document_fraction": _safe_ratio(
                        state.token_document_counter[token],
                        state.document_count,
                    ),
                    "rank_by_document_frequency": document_ranks[token],
                    "rank_by_total_frequency": total_ranks[token],
                    "token": token,
                    "token_class": classify_token(token),
                    "token_length": len(token),
                    "total_count": state.token_total_counter[token],
                    "universe": universe,
                }
            )

    return rows


def _candidate_stopword_rows(
    *,
    states: Mapping[str, UniverseState],
    samples: Mapping[str, Sequence[SampledDocument]],
    config: ProfileConfig,
) -> list[dict[str, Any]]:
    """Return diagnostic candidate stopword rows."""
    rows: list[dict[str, Any]] = []

    for universe in UNIVERSES:
        state = states[universe]
        candidate_tokens: dict[str, set[str]] = {}

        for token, total_count in state.token_total_counter.items():
            reasons: set[str] = set()
            document_count = state.token_document_counter[token]
            document_fraction = _safe_ratio(document_count, state.document_count)

            if document_fraction >= config.candidate_stopword_min_document_fraction:
                reasons.add("high_document_fraction")

            if total_count >= config.candidate_stopword_min_total_count:
                reasons.add("high_total_frequency")

            if token.isalpha() and len(token) <= 2:
                reasons.add("very_short_alpha_token")

            if token in PROCEDURAL_SEED_TOKENS:
                reasons.add("procedural_seed_match")

            if reasons:
                candidate_tokens[token] = reasons

        contexts = _candidate_contexts(
            candidate_tokens=tuple(candidate_tokens),
            sample_documents=samples[universe],
            context_limit=config.context_examples_per_token,
        )

        for token in sorted(
            candidate_tokens,
            key=lambda item: (
                -state.token_document_counter[item],
                -state.token_total_counter[item],
                item,
            ),
        ):
            rows.append(
                {
                    "candidate_reasons": "|".join(sorted(candidate_tokens[token])),
                    "context_examples": _json_compact(contexts.get(token, [])),
                    "document_count": state.token_document_counter[token],
                    "document_fraction": _safe_ratio(
                        state.token_document_counter[token],
                        state.document_count,
                    ),
                    "selected_for_removal": "false",
                    "token": token,
                    "token_class": classify_token(token),
                    "token_length": len(token),
                    "total_count": state.token_total_counter[token],
                    "universe": universe,
                }
            )

    return rows


def _candidate_contexts(
    *,
    candidate_tokens: Sequence[str],
    sample_documents: Sequence[SampledDocument],
    context_limit: int,
) -> dict[str, list[dict[str, str]]]:
    """Return bounded deterministic contexts from the stable sample."""
    if context_limit == 0 or not candidate_tokens:
        return {}

    candidates = set(candidate_tokens)
    contexts: dict[str, list[dict[str, str]]] = {token: [] for token in candidates}

    for document in sorted(sample_documents, key=lambda item: (item.stable_hash, item.document_id)):
        normalized = normalize_for_profile_tokenization(document.modeling_text)
        document_tokens = set(diagnostic_tokens(document.modeling_text))

        for token in sorted(candidates & document_tokens):
            if len(contexts[token]) >= context_limit:
                continue

            contexts[token].append(
                {
                    "document_id": document.document_id,
                    "snippet": _snippet_from_text(normalized, value=token),
                }
            )

    return {token: examples for token, examples in contexts.items() if examples}


def _sampled_bigram_rows(
    *,
    universe: str,
    sample_documents: Sequence[SampledDocument],
    minimum_token_length: int,
    top_count: int,
) -> list[dict[str, Any]]:
    """Return exact sampled bigram counts for one universe."""
    bigram_counts: Counter[tuple[str, str]] = Counter()
    bigram_document_counts: Counter[tuple[str, str]] = Counter()

    for document in sample_documents:
        tokens = [
            token
            for token in diagnostic_tokens(document.modeling_text)
            if token.isalpha() and len(token) >= minimum_token_length
        ]
        document_bigrams = list(zip(tokens, tokens[1:], strict=False))
        bigram_counts.update(document_bigrams)
        bigram_document_counts.update(set(document_bigrams))

    sample_document_total = len(sample_documents)
    rows: list[dict[str, Any]] = []

    for rank, bigram in enumerate(
        sorted(bigram_counts, key=lambda item: (-bigram_counts[item], item[0], item[1]))[
            :top_count
        ],
        start=1,
    ):
        token_1, token_2 = bigram
        rows.append(
            {
                "bigram": f"{token_1} {token_2}",
                "rank": rank,
                "sample_count": bigram_counts[bigram],
                "sample_document_count": bigram_document_counts[bigram],
                "sample_document_fraction": _safe_ratio(
                    bigram_document_counts[bigram],
                    sample_document_total,
                ),
                "sample_document_total": sample_document_total,
                "token_1": token_1,
                "token_2": token_2,
                "universe": universe,
            }
        )

    return rows


def _all_sampled_bigram_rows(
    *,
    samples: Mapping[str, Sequence[SampledDocument]],
    config: ProfileConfig,
) -> list[dict[str, Any]]:
    """Return sampled bigram rows for all universes."""
    rows: list[dict[str, Any]] = []

    for universe in UNIVERSES:
        rows.extend(
            _sampled_bigram_rows(
                universe=universe,
                sample_documents=samples[universe],
                minimum_token_length=config.minimum_bigram_token_length,
                top_count=config.top_bigram_count,
            )
        )

    return rows


def _preprocessing_example_records(
    samples: Mapping[str, Sequence[SampledDocument]],
) -> Iterator[dict[str, Any]]:
    """Yield deterministic preprocessing examples from the bounded sample."""
    for universe in UNIVERSES:
        selected = sorted(samples[universe], key=lambda item: (item.stable_hash, item.document_id))[
            :PREPROCESSING_EXAMPLE_LIMIT_PER_UNIVERSE
        ]

        for document in selected:
            normalized = normalize_for_profile_tokenization(document.modeling_text)
            tokens = diagnostic_tokens(document.modeling_text)
            alphabetic_tokens = [token for token in tokens if token.isalpha()]
            yield {
                "alphabetic_tokens": alphabetic_tokens,
                "alphabetic_tokens_min_length_3": [
                    token for token in alphabetic_tokens if len(token) >= 3
                ],
                "all_diagnostic_tokens": tokens,
                "document_id": document.document_id,
                "nfkc_casefolded_text": normalized,
                "original_modeling_text": document.modeling_text,
                "session_category": document.session_category,
                "universe": universe,
                "year": document.year,
            }


def _sampled_document_records(
    samples: Mapping[str, Sequence[SampledDocument]],
) -> Iterator[dict[str, Any]]:
    """Yield exact deterministic sample membership records."""
    for universe in UNIVERSES:
        by_stratum: dict[tuple[str, str], list[SampledDocument]] = defaultdict(list)

        for document in samples[universe]:
            by_stratum[document.stratum_key].append(document)

        for stratum in sorted(by_stratum):
            stratum_documents = sorted(
                by_stratum[stratum],
                key=lambda item: (item.stable_hash, item.document_id),
            )

            for rank, document in enumerate(stratum_documents, start=1):
                yield {
                    "chunk_index": document.chunk_index,
                    "document_id": document.document_id,
                    "modeling_text": document.modeling_text,
                    "rank_within_stratum": rank,
                    "session_category": document.session_category,
                    "source_record_id": document.source_record_id,
                    "speaker_family": document.speaker_family,
                    "stable_hash": document.stable_hash,
                    "stratum": {
                        "session_category": stratum[1],
                        "year": int(stratum[0]),
                    },
                    "temporal_period": document.temporal_period,
                    "turn_index": document.turn_index,
                    "universe": universe,
                    "word_count": document.word_count,
                    "year": document.year,
                }


def _csv_group_rows(
    *,
    states: Mapping[str, UniverseState],
    group_name: str,
) -> list[dict[str, Any]]:
    """Return group CSV rows for simple one-dimensional groups."""
    rows: list[dict[str, Any]] = []

    for universe in UNIVERSES:
        groups = _simple_group_map(states[universe], group_name)

        for group_value in sorted(groups, key=_simple_group_sort_key):
            payload = groups[group_value].to_json()
            rows.append(
                {
                    "document_count": payload["document_count"],
                    group_name: group_value,
                    "mean_document_length": payload["mean_document_length"],
                    "session_count": payload["session_count"],
                    "source_turn_count": payload["source_turn_count"],
                    "universe": universe,
                    "word_total": payload["word_total"],
                }
            )

    return rows


def _year_category_csv_rows(states: Mapping[str, UniverseState]) -> list[dict[str, Any]]:
    """Return year and category grouped CSV rows."""
    rows: list[dict[str, Any]] = []

    for universe in UNIVERSES:
        groups = states[universe].counts_by_year_and_category

        for year, session_category in sorted(groups):
            payload = groups[(year, session_category)].to_json()
            rows.append(
                {
                    "document_count": payload["document_count"],
                    "mean_document_length": payload["mean_document_length"],
                    "session_category": session_category,
                    "session_count": payload["session_count"],
                    "source_turn_count": payload["source_turn_count"],
                    "universe": universe,
                    "word_total": payload["word_total"],
                    "year": year,
                }
            )

    return rows


def _histogram_rows(states: Mapping[str, UniverseState]) -> list[dict[str, Any]]:
    """Return exact 1-300 document-length histogram rows."""
    rows: list[dict[str, Any]] = []

    for universe in UNIVERSES:
        histogram = states[universe].histogram

        for word_count in range(1, 301):
            rows.append(
                {
                    "document_count": histogram.get(word_count, 0),
                    "universe": universe,
                    "word_count": word_count,
                    "word_total": word_count * histogram.get(word_count, 0),
                }
            )

    return rows


def _corpus_profile_payload(
    *,
    run: ProfileRun,
    samples: Mapping[str, Sequence[SampledDocument]],
    top_tokens: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    suspicious_token_row_count: int,
    sampled_bigram_row_count: int,
    reconciliation_checks: Mapping[str, bool],
) -> dict[str, Any]:
    """Return the deterministic machine-readable corpus profile."""
    return {
        "candidate_stopword_count": len(candidate_rows),
        "configuration": run.config.to_json(),
        "corpus_lock_counts": _locked_counts(run.corpus_lock),
        "preprocessing_decision_status": "no_preprocessing_decision_frozen",
        "profile_version": PROFILE_VERSION,
        "reconciliation_checks": dict(sorted(reconciliation_checks.items())),
        "sample": {
            universe: {
                "document_count": len(samples[universe]),
                "strata": run.samplers[universe].counts_by_stratum(),
            }
            for universe in UNIVERSES
        },
        "statistics_scope": _statistics_scope_metadata(),
        "sampled_bigram_count": sampled_bigram_row_count,
        "suspicious_token_row_count": suspicious_token_row_count,
        "tokenizer": _tokenizer_policy(),
        "top_tokens_by_total_frequency": list(top_tokens),
        "universes": {universe: run.states[universe].to_json() for universe in UNIVERSES},
    }


def _statistics_scope_metadata() -> dict[str, Any]:
    """Return explicit exact-versus-sampled scope metadata."""
    return {
        "exact_full_corpus_statistics": [
            "corpus_counts",
            "grouped_counts",
            "document_length_statistics",
            "document_length_histogram",
            "lexical_totals_and_classes",
            "vocabulary_growth_by_year",
            "unigram_total_frequency",
            "unigram_document_frequency",
            "candidate_stopword_counts_and_reasons",
            "suspicious_token_counts",
        ],
        "sampled_statistics_or_examples": [
            {
                "name": "sampled_document_membership",
                "output": SAMPLED_DOCUMENTS_FILENAME,
                "scope": "deterministic_stratified_sample",
            },
            {
                "name": "sampled_bigram_frequency",
                "output": SAMPLED_BIGRAM_FREQUENCY_FILENAME,
                "scope": "deterministic_stratified_sample",
            },
            {
                "name": "candidate_token_context_examples",
                "output": CANDIDATE_STOPWORDS_FILENAME,
                "scope": "deterministic_stratified_sample",
            },
            {
                "name": "preprocessing_examples",
                "output": PREPROCESSING_EXAMPLES_FILENAME,
                "scope": "deterministic_stratified_sample",
            },
        ],
    }


def _locked_counts(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return locked count fields from the corpus lock or manifest."""
    fields = (
        "exclusion_ledger_count",
        "input_session_count",
        "input_turn_count",
        "modeling_document_count",
        "modeling_word_total",
        "positive_speech_turn_count",
        "retained_source_turn_count",
        "retained_source_turn_word_total",
    )
    return {field_name: payload.get(field_name) for field_name in fields if field_name in payload}


def _tokenizer_policy() -> dict[str, Any]:
    """Return the versioned tokenizer policy description."""
    return {
        "case_normalization": "casefold",
        "input_field": "modeling_text",
        "lemmatization": "none",
        "modeling_text_mutated": False,
        "name": TOKENIZER_VERSION,
        "names_or_provinces_removed": False,
        "split_rule": "deterministic split on non-alphanumeric Unicode boundaries",
        "stopword_removal": "none",
        "token_classes": [
            "alphabetic",
            "numeric",
            "mixed_alphanumeric",
            "one-character",
            "two-character",
            "three-or-more characters",
        ],
        "unicode_normalization": "NFKC",
    }


def _reconciliation_checks(
    *,
    run: ProfileRun,
    samples: Mapping[str, Sequence[SampledDocument]],
) -> dict[str, bool]:
    """Return independent reconciliation checks required by the profile."""
    all_state = run.states[UNIVERSE_ALL]
    primary_state = run.states[UNIVERSE_PRIMARY]
    expected_primary = _session_category_rollup(
        all_state,
        categories=run.config.primary_session_categories,
        include_matching=True,
    )
    expected_non_primary = _session_category_rollup(
        all_state,
        categories=run.config.primary_session_categories,
        include_matching=False,
    )

    checks = {
        "all_output_hashes_match_emitted_files": True,
        "deterministic_sample_membership_has_no_duplicate_document_ids": all(
            len({document.document_id for document in samples[universe]}) == len(samples[universe])
            for universe in UNIVERSES
        ),
        "every_document_belongs_to_all_sessions": (
            all_state.document_count == run.processed_document_count
        ),
        "grouped_category_counts_reconcile": _groups_reconcile(
            all_state,
            "session_category",
        )
        and _groups_reconcile(primary_state, "session_category"),
        "grouped_period_counts_reconcile": _groups_reconcile(all_state, "temporal_period")
        and _groups_reconcile(primary_state, "temporal_period"),
        "grouped_speaker_family_counts_reconcile": _groups_reconcile(
            all_state,
            "speaker_family",
        )
        and _groups_reconcile(primary_state, "speaker_family"),
        "grouped_year_counts_reconcile": _groups_reconcile(all_state, "year")
        and _groups_reconcile(primary_state, "year"),
        "histogram_document_counts_reconcile": _histogram_documents_reconcile(all_state)
        and _histogram_documents_reconcile(primary_state),
        "histogram_word_total_reconciles": _histogram_words_reconcile(all_state)
        and _histogram_words_reconcile(primary_state),
        "configured_primary_and_non_primary_category_groups_reconcile_to_all_sessions": (
            _rollups_partition_all_sessions(
                primary=expected_primary,
                non_primary=expected_non_primary,
                all_state=all_state,
            )
        ),
        "primary_universe_matches_configured_session_categories": (
            _state_matches_rollup(primary_state, expected_primary)
        ),
        "processed_document_count_equals_export_manifest": (
            run.processed_document_count == run.export_manifest.get("modeling_document_count")
        ),
        "processed_word_total_equals_export_manifest": (
            run.processed_word_total == run.export_manifest.get("modeling_word_total")
        ),
    }
    return dict(sorted(checks.items()))


def _session_category_rollup(
    state: UniverseState,
    *,
    categories: Sequence[str],
    include_matching: bool,
) -> CorpusRollup:
    """Derive a corpus rollup from all-session category accumulators."""
    category_set = set(categories)
    document_count = 0
    word_total = 0
    source_turns: set[str] = set()
    sessions: set[str] = set()

    for category, group in state.counts_by_session_category.items():
        category_matches = category in category_set

        if category_matches != include_matching:
            continue

        document_count += group.document_count
        word_total += group.word_total
        source_turns.update(group.source_turns)
        sessions.update(group.sessions)

    return CorpusRollup(
        document_count=document_count,
        word_total=word_total,
        source_turns=frozenset(source_turns),
        sessions=frozenset(sessions),
    )


def _state_matches_rollup(state: UniverseState, rollup: CorpusRollup) -> bool:
    """Return whether a universe state matches an independently derived rollup."""
    return (
        state.document_count == rollup.document_count
        and state.word_total == rollup.word_total
        and state.source_turns == set(rollup.source_turns)
        and state.sessions == set(rollup.sessions)
    )


def _rollups_partition_all_sessions(
    *,
    primary: CorpusRollup,
    non_primary: CorpusRollup,
    all_state: UniverseState,
) -> bool:
    """Return whether primary and non-primary category rollups partition all sessions."""
    return (
        primary.document_count + non_primary.document_count == all_state.document_count
        and primary.word_total + non_primary.word_total == all_state.word_total
        and set(primary.source_turns).union(non_primary.source_turns) == all_state.source_turns
        and set(primary.source_turns).isdisjoint(non_primary.source_turns)
        and set(primary.sessions).union(non_primary.sessions) == all_state.sessions
    )


def _groups_reconcile(state: UniverseState, group_name: str) -> bool:
    """Return whether a one-dimensional grouping reconciles to the universe."""
    groups = _simple_group_map(state, group_name)
    document_count = sum(group.document_count for group in groups.values())
    word_total = sum(group.word_total for group in groups.values())
    source_turns = (
        set[str]().union(*(group.source_turns for group in groups.values())) if groups else set()
    )
    sessions = set[str]().union(*(group.sessions for group in groups.values())) if groups else set()

    return (
        document_count == state.document_count
        and word_total == state.word_total
        and source_turns == state.source_turns
        and sessions == state.sessions
    )


def _histogram_documents_reconcile(state: UniverseState) -> bool:
    """Return whether histogram document counts reconcile."""
    return sum(state.histogram.values()) == state.document_count


def _histogram_words_reconcile(state: UniverseState) -> bool:
    """Return whether histogram word totals reconcile."""
    return (
        sum(word_count * count for word_count, count in state.histogram.items()) == state.word_total
    )


def _markdown_report(
    *,
    profile: Mapping[str, Any],
    top_tokens: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    suspicious_rows: Sequence[Mapping[str, Any]],
    suspicious_token_row_count: int,
    bigram_rows: Sequence[Mapping[str, Any]],
) -> str:
    """Return a concise human-readable corpus profile report."""
    universes = profile["universes"]
    all_counts = universes[UNIVERSE_ALL]["counts"]
    primary_counts = universes[UNIVERSE_PRIMARY]["counts"]
    all_length = universes[UNIVERSE_ALL]["document_length"]
    primary_length = universes[UNIVERSE_PRIMARY]["document_length"]
    all_lexical = universes[UNIVERSE_ALL]["lexical"]
    primary_lexical = universes[UNIVERSE_PRIMARY]["lexical"]

    lines = [
        "# Corpus Profile v1",
        "",
        "This report profiles the locked modelling corpus for preprocessing evidence only. "
        "No NMF, LDA, BERTopic, embeddings or topic model is fitted here.",
        "",
        "## Corpus overview",
        "",
        (
            f"- all_sessions: {all_counts['documents']:,} documents, "
            f"{all_counts['unique_source_turns']:,} source turns, "
            f"{all_counts['unique_sessions']:,} sessions, {all_counts['words']:,} words."
        ),
        (
            f"- primary: {primary_counts['documents']:,} documents, "
            f"{primary_counts['unique_source_turns']:,} source turns, "
            f"{primary_counts['unique_sessions']:,} sessions, {primary_counts['words']:,} words."
        ),
        "",
        "## Primary versus all",
        "",
        (
            "Primary documents are "
            f"{_safe_ratio(primary_counts['documents'], all_counts['documents']):.4f} "
            "of all modelling documents and primary words are "
            f"{_safe_ratio(primary_counts['words'], all_counts['words']):.4f} "
            "of all modelling words. Rows in tabular outputs identify their universe."
        ),
        "",
        "## Document length",
        "",
        _document_length_finding(UNIVERSE_ALL, all_length),
        _document_length_finding(UNIVERSE_PRIMARY, primary_length),
        "",
        "## Year and category imbalance",
        "",
        _year_volume_finding(UNIVERSE_ALL, universes[UNIVERSE_ALL]),
        _year_volume_finding(UNIVERSE_PRIMARY, universes[UNIVERSE_PRIMARY]),
        _period_volume_finding(UNIVERSE_ALL, universes[UNIVERSE_ALL]),
        _period_volume_finding(UNIVERSE_PRIMARY, universes[UNIVERSE_PRIMARY]),
        _category_distribution_finding(UNIVERSE_ALL, universes[UNIVERSE_ALL]),
        _category_distribution_finding(UNIVERSE_PRIMARY, universes[UNIVERSE_PRIMARY]),
        "These are volume diagnostics only, not substantive political interpretations.",
        "",
        "## Lexical overview",
        "",
        (
            f"- all_sessions: {all_lexical['total_lexical_tokens']:,} lexical-token "
            f"occurrences and {all_lexical['unique_lexical_tokens']:,} unique tokens; "
            f"numeric share {all_lexical['numeric_token_share']:.4f}, "
            f"mixed-alphanumeric share {all_lexical['mixed_alphanumeric_token_share']:.4f}."
        ),
        (
            f"- primary: {primary_lexical['total_lexical_tokens']:,} lexical-token "
            f"occurrences and {primary_lexical['unique_lexical_tokens']:,} unique tokens; "
            f"numeric share {primary_lexical['numeric_token_share']:.4f}, "
            f"mixed-alphanumeric share {primary_lexical['mixed_alphanumeric_token_share']:.4f}."
        ),
        "",
        "## Top frequent terms",
        "",
    ]
    lines.extend(_markdown_top_rows(top_tokens, count=10))
    lines.extend(
        [
            "",
            "## Candidate stopwords",
            "",
            f"{len(candidate_rows):,} diagnostic candidates were emitted. "
            "They are evidence rows only; selected_for_removal is false for every row.",
            "",
        ]
    )
    lines.extend(_markdown_candidate_rows(candidate_rows, count=10))
    lines.extend(
        [
            "",
            "## Suspicious-token summary",
            "",
            f"{suspicious_token_row_count:,} suspicious-token or raw-text anomaly rows "
            "were emitted.",
            "",
        ]
    )
    lines.extend(_markdown_suspicious_rows(suspicious_rows, count=10))
    lines.extend(
        [
            "",
            "## Sampled bigrams",
            "",
            "Bigram counts are exact only over the deterministic stratified sample, "
            "not the full corpus.",
            "",
        ]
    )
    lines.extend(_markdown_bigram_rows(bigram_rows, count=10))
    lines.extend(
        [
            "",
            "## Important cautions",
            "",
            "- The diagnostic tokenizer preserves stopwords, names, provinces, numbers "
            "and mixed tokens.",
            "- Candidate stopwords and suspicious-token rows are not automatic exclusions.",
            "- No preprocessing decision has yet been frozen.",
            "",
        ]
    )

    return "\n".join(lines)


def _markdown_top_rows(rows: Sequence[Mapping[str, Any]], *, count: int) -> list[str]:
    """Return compact top-token report rows."""
    output: list[str] = []

    for universe in UNIVERSES:
        output.extend(
            [
                f"### {universe}",
                "",
                "| rank | token | total_count | document_fraction |",
                "| ---: | --- | ---: | ---: |",
            ]
        )

        for row in _rows_for_universe(rows, universe, count):
            output.append(
                f"| {row['rank_by_total_frequency']} | {_markdown_escape(row['token'])} | "
                f"{row['total_count']} | {float(row['document_fraction']):.4f} |"
            )

        output.append("")

    return output


def _markdown_candidate_rows(rows: Sequence[Mapping[str, Any]], *, count: int) -> list[str]:
    """Return compact candidate rows."""
    output: list[str] = []

    for universe in UNIVERSES:
        output.extend(
            [
                f"### {universe}",
                "",
                "| token | reasons | document_fraction |",
                "| --- | --- | ---: |",
            ]
        )

        for row in _rows_for_universe(rows, universe, count):
            output.append(
                f"| {_markdown_escape(row['token'])} | "
                f"{_markdown_escape(row['candidate_reasons'])} | "
                f"{float(row['document_fraction']):.4f} |"
            )

        output.append("")

    return output


def _markdown_suspicious_rows(rows: Sequence[Mapping[str, Any]], *, count: int) -> list[str]:
    """Return compact suspicious-token rows."""
    output: list[str] = []

    for universe in UNIVERSES:
        output.extend(
            [
                f"### {universe}",
                "",
                "| reason | value | snippet_text_kind | total_count |",
                "| --- | --- | --- | ---: |",
            ]
        )

        for row in _rows_for_universe(rows, universe, count):
            output.append(
                f"| {_markdown_escape(row['reason'])} | "
                f"{_markdown_escape(row['token_or_anomaly'])} | "
                f"{_markdown_escape(row['snippet_text_kind'])} | {row['total_count']} |"
            )

        output.append("")

    return output


def _markdown_bigram_rows(rows: Sequence[Mapping[str, Any]], *, count: int) -> list[str]:
    """Return compact sampled-bigram rows."""
    output: list[str] = []

    for universe in UNIVERSES:
        output.extend(
            [
                f"### {universe}",
                "",
                "| rank | bigram | sample_count |",
                "| ---: | --- | ---: |",
            ]
        )

        for row in _rows_for_universe(rows, universe, count):
            output.append(
                f"| {row['rank']} | {_markdown_escape(row['bigram'])} | {row['sample_count']} |"
            )

        output.append("")

    return output


def _rows_for_universe(
    rows: Sequence[Mapping[str, Any]],
    universe: str,
    count: int,
) -> list[Mapping[str, Any]]:
    """Return the first bounded rows for one universe."""
    return [row for row in rows if row.get("universe") == universe][:count]


def _bounded_rows_by_universe(
    rows: Iterable[Mapping[str, Any]],
    *,
    count: int,
) -> list[Mapping[str, Any]]:
    """Collect at most count rows per universe from an iterator."""
    selected: dict[str, list[Mapping[str, Any]]] = {universe: [] for universe in UNIVERSES}

    for row in rows:
        universe = row.get("universe")

        if universe not in selected:
            continue

        universe_rows = selected[str(universe)]

        if len(universe_rows) < count:
            universe_rows.append(row)

    output: list[Mapping[str, Any]] = []

    for universe in UNIVERSES:
        output.extend(selected[universe])

    return output


def _markdown_escape(value: object) -> str:
    """Escape a value for a Markdown table cell."""
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _document_length_finding(universe: str, length: Mapping[str, Any]) -> str:
    """Return a document-length finding for one universe."""
    quantiles = length["quantiles"]
    return (
        f"- {universe}: mean {length['mean_words_per_document']:.2f}, "
        f"p05 {quantiles['p05']:.2f}, p50 {quantiles['p50']:.2f}, "
        f"p95 {quantiles['p95']:.2f}, p99 {quantiles['p99']:.2f}, "
        f"max {length['maximum']} words."
    )


def _year_volume_finding(universe: str, universe_payload: Mapping[str, Any]) -> str:
    """Return highest and lowest volume years for one universe."""
    rows = universe_payload["grouped_counts"]["by_year"]

    if not rows:
        return f"- {universe}: no year counts."

    highest = max(rows, key=lambda row: (row["word_total"], row["document_count"]))
    lowest = min(rows, key=lambda row: (row["word_total"], row["document_count"]))
    return (
        f"- {universe}: highest-volume year {highest['group']} "
        f"({highest['word_total']:,} words, {highest['document_count']:,} documents); "
        f"lowest-volume year {lowest['group']} "
        f"({lowest['word_total']:,} words, {lowest['document_count']:,} documents)."
    )


def _period_volume_finding(universe: str, universe_payload: Mapping[str, Any]) -> str:
    """Return temporal-period volume range for one universe."""
    rows = universe_payload["grouped_counts"]["by_temporal_period"]

    if not rows:
        return f"- {universe}: no temporal-period counts."

    highest = max(rows, key=lambda row: (row["word_total"], row["document_count"]))
    lowest = min(rows, key=lambda row: (row["word_total"], row["document_count"]))
    return (
        f"- {universe}: highest-volume temporal period {highest['group']} "
        f"({highest['word_total']:,} words); lowest-volume period {lowest['group']} "
        f"({lowest['word_total']:,} words)."
    )


def _category_distribution_finding(universe: str, universe_payload: Mapping[str, Any]) -> str:
    """Return category distribution finding for one universe."""
    rows = universe_payload["grouped_counts"]["by_session_category"]

    if not rows:
        return f"- {universe}: no session-category counts."

    top_category = max(rows, key=lambda row: (row["document_count"], row["word_total"]))
    total_documents = universe_payload["counts"]["documents"]
    return (
        f"- {universe}: largest session category {top_category['group']} "
        f"with {top_category['document_count']:,} documents "
        f"({_safe_ratio(top_category['document_count'], total_documents):.4f} of documents)."
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
    """Remove temporary transaction files."""
    for final_path in final_paths:
        _part_path(final_path).unlink(missing_ok=True)
        _backup_path(final_path).unlink(missing_ok=True)


def _promote_transaction(final_paths: tuple[Path, ...]) -> None:
    """Promote part files to final outputs with rollback."""
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
            raise CorpusProfileError(
                "Could not restore prior corpus-profile outputs after promotion failure."
            ) from restore_error

        _cleanup_transaction_paths(final_paths)
        raise CorpusProfileError("Could not promote corpus-profile outputs safely.") from error

    _cleanup_transaction_paths(final_paths)


def _ensure_output_directory(output_dir: Path, *, force: bool) -> None:
    """Enforce overwrite protection."""
    _preflight_output_directory(output_dir, force=force)
    output_dir.mkdir(parents=True, exist_ok=True)


def _preflight_output_directory(output_dir: Path, *, force: bool) -> None:
    """Check output overwrite protection without creating or modifying outputs."""
    if output_dir.exists() and not output_dir.is_dir():
        raise CorpusProfileError(f"Output path is not a directory: {output_dir}")

    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise CorpusProfileError(f"Output directory is nonempty; use --force: {output_dir}")


def _final_output_paths(output_dir: Path) -> dict[str, Path]:
    """Return canonical output paths."""
    return {
        "candidate_stopwords": output_dir / CANDIDATE_STOPWORDS_FILENAME,
        "corpus_profile_json": output_dir / CORPUS_PROFILE_JSON_FILENAME,
        "corpus_profile_md": output_dir / CORPUS_PROFILE_MD_FILENAME,
        "counts_by_session_category": output_dir / COUNTS_BY_SESSION_CATEGORY_FILENAME,
        "counts_by_speaker_family": output_dir / COUNTS_BY_SPEAKER_FAMILY_FILENAME,
        "counts_by_temporal_period": output_dir / COUNTS_BY_TEMPORAL_PERIOD_FILENAME,
        "counts_by_year": output_dir / COUNTS_BY_YEAR_FILENAME,
        "counts_by_year_and_category": output_dir / COUNTS_BY_YEAR_AND_CATEGORY_FILENAME,
        "document_length_histogram": output_dir / DOCUMENT_LENGTH_HISTOGRAM_FILENAME,
        "preprocessing_examples": output_dir / PREPROCESSING_EXAMPLES_FILENAME,
        "profile_manifest": output_dir / PROFILE_MANIFEST_FILENAME,
        "sampled_bigram_frequency": output_dir / SAMPLED_BIGRAM_FREQUENCY_FILENAME,
        "sampled_documents": output_dir / SAMPLED_DOCUMENTS_FILENAME,
        "suspicious_tokens": output_dir / SUSPICIOUS_TOKENS_FILENAME,
        "token_frequency": output_dir / TOKEN_FREQUENCY_FILENAME,
    }


def _write_text_part(path: Path, text: str) -> dict[str, Any]:
    """Write a UTF-8 text part file and return file metadata."""
    part_path = _part_path(path)
    part_path.write_text(text, encoding="utf-8", newline="")
    return _file_metadata(path)


def _write_jsonl_part(path: Path, records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Write deterministic JSONL records to a part file."""
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
    """Write deterministic CSV rows to a part file."""
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

    metadata = _file_metadata(path)
    metadata["record_count"] = record_count
    return metadata


def _file_metadata(path: Path) -> dict[str, Any]:
    """Return metadata for a staged part file addressed by final path."""
    part_path = _part_path(path)
    return {
        "path": str(path),
        "sha256": sha256_file(part_path),
        "size_bytes": part_path.stat().st_size,
    }


def _validate_output_metadata(output_files: Mapping[str, Mapping[str, Any]]) -> bool:
    """Validate staged output metadata against the staged files."""
    for metadata in output_files.values():
        path_value = metadata.get("path")
        sha256_value = metadata.get("sha256")
        size_value = metadata.get("size_bytes")

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


def _write_outputs(
    *,
    run: ProfileRun,
    output_dir: Path,
    force: bool,
    input_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Write all profile outputs transactionally."""
    _ensure_output_directory(output_dir, force=force)
    paths = _final_output_paths(output_dir)
    final_paths = tuple(paths.values())
    _cleanup_transaction_paths(final_paths)

    samples = {universe: run.samplers[universe].selected() for universe in UNIVERSES}
    token_rank_maps = _all_token_rank_maps(run.states)
    top_tokens = _top_token_rows(
        states=run.states,
        rank_maps=token_rank_maps,
        top_count=run.config.top_token_count,
    )
    candidate_rows = _candidate_stopword_rows(
        states=run.states,
        samples=samples,
        config=run.config,
    )
    suspicious_denominators = {
        universe: run.states[universe].document_count for universe in UNIVERSES
    }
    suspicious_report_rows = _bounded_rows_by_universe(
        run.suspicious.iter_rows(denominators=suspicious_denominators),
        count=10,
    )
    suspicious_token_row_count = run.suspicious.row_count()
    bigram_rows = _all_sampled_bigram_rows(samples=samples, config=run.config)
    reconciliation_checks = _reconciliation_checks(run=run, samples=samples)

    if not all(reconciliation_checks.values()):
        failed = sorted(key for key, passed in reconciliation_checks.items() if not passed)
        raise CorpusProfileError(f"Corpus profile reconciliation failed: {failed}")

    output_files: dict[str, Mapping[str, Any]] = {}

    try:
        output_files["corpus_profile_json"] = _write_text_part(
            paths["corpus_profile_json"],
            _json_text(
                _corpus_profile_payload(
                    run=run,
                    samples=samples,
                    top_tokens=top_tokens,
                    candidate_rows=candidate_rows,
                    suspicious_token_row_count=suspicious_token_row_count,
                    sampled_bigram_row_count=len(bigram_rows),
                    reconciliation_checks=reconciliation_checks,
                )
            ),
        )
        output_files["corpus_profile_md"] = _write_text_part(
            paths["corpus_profile_md"],
            _markdown_report(
                profile=json.loads(
                    _part_path(paths["corpus_profile_json"]).read_text(encoding="utf-8")
                ),
                top_tokens=top_tokens,
                candidate_rows=candidate_rows,
                suspicious_rows=suspicious_report_rows,
                suspicious_token_row_count=suspicious_token_row_count,
                bigram_rows=bigram_rows,
            ),
        )
        output_files["counts_by_year"] = _write_csv_part(
            paths["counts_by_year"],
            fieldnames=_group_csv_fields("year"),
            rows=_csv_group_rows(states=run.states, group_name="year"),
        )
        output_files["counts_by_temporal_period"] = _write_csv_part(
            paths["counts_by_temporal_period"],
            fieldnames=_group_csv_fields("temporal_period"),
            rows=_csv_group_rows(states=run.states, group_name="temporal_period"),
        )
        output_files["counts_by_session_category"] = _write_csv_part(
            paths["counts_by_session_category"],
            fieldnames=_group_csv_fields("session_category"),
            rows=_csv_group_rows(states=run.states, group_name="session_category"),
        )
        output_files["counts_by_speaker_family"] = _write_csv_part(
            paths["counts_by_speaker_family"],
            fieldnames=_group_csv_fields("speaker_family"),
            rows=_csv_group_rows(states=run.states, group_name="speaker_family"),
        )
        output_files["counts_by_year_and_category"] = _write_csv_part(
            paths["counts_by_year_and_category"],
            fieldnames=(
                "universe",
                "year",
                "session_category",
                "document_count",
                "source_turn_count",
                "session_count",
                "word_total",
                "mean_document_length",
            ),
            rows=_year_category_csv_rows(run.states),
        )
        output_files["document_length_histogram"] = _write_csv_part(
            paths["document_length_histogram"],
            fieldnames=("universe", "word_count", "document_count", "word_total"),
            rows=_histogram_rows(run.states),
        )
        output_files["token_frequency"] = _write_csv_part(
            paths["token_frequency"],
            fieldnames=TOKEN_FREQUENCY_FIELDS,
            rows=_token_frequency_rows(run.states, token_rank_maps),
        )
        output_files["candidate_stopwords"] = _write_csv_part(
            paths["candidate_stopwords"],
            fieldnames=(
                "universe",
                "token",
                "total_count",
                "document_count",
                "document_fraction",
                "token_length",
                "token_class",
                "candidate_reasons",
                "selected_for_removal",
                "context_examples",
            ),
            rows=candidate_rows,
        )
        output_files["suspicious_tokens"] = _write_csv_part(
            paths["suspicious_tokens"],
            fieldnames=(
                "universe",
                "reason",
                "token_or_anomaly",
                "total_count",
                "document_count",
                "document_fraction",
                "snippet_text_kind",
                "example_document_ids",
                "example_snippets",
            ),
            rows=run.suspicious.iter_rows(denominators=suspicious_denominators),
        )
        output_files["sampled_documents"] = _write_jsonl_part(
            paths["sampled_documents"],
            _sampled_document_records(samples),
        )
        output_files["sampled_bigram_frequency"] = _write_csv_part(
            paths["sampled_bigram_frequency"],
            fieldnames=(
                "universe",
                "rank",
                "token_1",
                "token_2",
                "bigram",
                "sample_count",
                "sample_document_count",
                "sample_document_fraction",
                "sample_document_total",
            ),
            rows=bigram_rows,
        )
        output_files["preprocessing_examples"] = _write_jsonl_part(
            paths["preprocessing_examples"],
            _preprocessing_example_records(samples),
        )

        if not _validate_output_metadata(output_files):
            raise CorpusProfileError("Staged output hash validation failed.")

        manifest = _profile_manifest(
            run=run,
            input_paths=input_paths,
            samples=samples,
            output_files=output_files,
            reconciliation_checks={
                **reconciliation_checks,
                "all_output_hashes_match_emitted_files": True,
            },
        )
        _part_path(paths["profile_manifest"]).write_text(
            _json_text(manifest),
            encoding="utf-8",
            newline="",
        )
    except Exception as error:
        _cleanup_transaction_paths(final_paths)

        if isinstance(error, CorpusProfileError):
            raise

        raise CorpusProfileError("Could not stage corpus-profile outputs.") from error

    _promote_transaction(final_paths)
    return manifest


def _group_csv_fields(group_name: str) -> tuple[str, ...]:
    """Return fields for a simple grouped-count CSV."""
    return (
        "universe",
        group_name,
        "document_count",
        "source_turn_count",
        "session_count",
        "word_total",
        "mean_document_length",
    )


def _profile_manifest(
    *,
    run: ProfileRun,
    input_paths: Mapping[str, Path],
    samples: Mapping[str, Sequence[SampledDocument]],
    output_files: Mapping[str, Mapping[str, Any]],
    reconciliation_checks: Mapping[str, bool],
) -> dict[str, Any]:
    """Return the profile manifest."""
    config_snapshot = run.config.to_json()
    canonical_config_text = _json_text(config_snapshot)

    return {
        "canonical_configuration_sha256": hashlib.sha256(
            canonical_config_text.encode("utf-8")
        ).hexdigest(),
        "configuration": config_snapshot,
        "corpus_lock_hash": run.input_hashes["corpus_lock_sha256"],
        "export_manifest_hash": run.input_hashes["export_manifest_sha256"],
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_paths": {key: str(path) for key, path in sorted(input_paths.items())},
        "input_sha256": dict(sorted(run.input_hashes.items())),
        "locked_corpus_counts": _locked_counts(run.corpus_lock),
        "output_files": dict(sorted(output_files.items())),
        "primary_counts": run.states[UNIVERSE_PRIMARY].to_json()["counts"],
        "processed_corpus_counts": {
            "documents": run.processed_document_count,
            "words": run.processed_word_total,
        },
        "profile_version": PROFILE_VERSION,
        "reconciliation_checks": dict(sorted(reconciliation_checks.items())),
        "sample_counts": {
            universe: {
                "documents": len(samples[universe]),
                "strata": run.samplers[universe].counts_by_stratum(),
            }
            for universe in UNIVERSES
        },
        "tokenizer_version": TOKENIZER_VERSION,
        "universes": {
            UNIVERSE_ALL: run.states[UNIVERSE_ALL].to_json()["counts"],
            UNIVERSE_PRIMARY: run.states[UNIVERSE_PRIMARY].to_json()["counts"],
        },
    }


def profile_modeling_corpus(
    *,
    documents_path: Path = DEFAULT_DOCUMENTS_PATH,
    export_manifest_path: Path = DEFAULT_EXPORT_MANIFEST_PATH,
    corpus_lock_path: Path = DEFAULT_CORPUS_LOCK_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    force: bool = False,
) -> dict[str, Any]:
    """Profile the locked modelling corpus transactionally."""
    config = _load_config(config_path)
    _preflight_output_directory(output_dir, force=force)
    export_manifest, corpus_lock, input_hashes = _validate_manifest_and_lock(
        documents_path=documents_path,
        export_manifest_path=export_manifest_path,
        corpus_lock_path=corpus_lock_path,
    )
    input_hashes["config_sha256"] = sha256_file(config_path)
    run = _stream_profile(
        documents_path=documents_path,
        export_manifest=export_manifest,
        corpus_lock=corpus_lock,
        config=config,
        input_hashes=input_hashes,
    )
    return _write_outputs(
        run=run,
        output_dir=output_dir,
        force=force,
        input_paths={
            "config": config_path,
            "corpus_lock": corpus_lock_path,
            "documents": documents_path,
            "export_manifest": export_manifest_path,
        },
    )
