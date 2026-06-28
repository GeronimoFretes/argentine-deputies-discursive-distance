"""Focused text cleaning and lexical tokenization for topic modelling."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .pdf_pipeline import sha256_file

SOFT_HYPHEN = "\u00ad"
SOFT_HYPHEN_JOIN_PATTERN = re.compile(r"(?<=[^\W\d_])\u00ad\s*(?=[^\W\d_])")
EXPLICIT_HYPHENATION_PATTERN = re.compile(r"(?<=[^\W\d_])-\s+(?=[^\W\d_])")
LEXICAL_TOKENIZER_VERSION = "unicode_alpha_runs_nfkc_casefold_min3_v1"
CLEANING_VERSION = "soft_hyphen_linebreak_hyphen_nfkc_casefold_v1"

P1_ADDITIONS = frozenset({"señor", "señora", "señores", "presidente", "presidenta"})
PROTECTED_SUBSTANTIVE_TERMS = frozenset(
    {
        "artículo",
        "comisión",
        "democracia",
        "derecho",
        "derechos",
        "educación",
        "ejecutivo",
        "estado",
        "gobierno",
        "justicia",
        "ley",
        "leyes",
        "nación",
        "nacional",
        "poder",
        "política",
        "políticas",
        "presupuesto",
        "provincia",
        "provincias",
        "proyecto",
        "proyectos",
        "pública",
        "público",
        "salud",
        "social",
        "trabajo",
    }
)


class TopicPreprocessingError(RuntimeError):
    """Raised when topic preprocessing configuration is invalid."""


@dataclass(frozen=True, slots=True)
class CleaningResult:
    """Cleaned text plus exact repair diagnostics for one document."""

    cleaned_text: str
    soft_hyphen_join_count: int
    soft_hyphen_removed_count: int
    explicit_hyphenation_join_count: int
    changed_by_soft_hyphen_repair: bool
    changed_by_explicit_hyphenation_repair: bool

    @property
    def changed_by_any_repair(self) -> bool:
        """Return whether either repair stage changed the pre-normalized text."""
        return self.changed_by_soft_hyphen_repair or self.changed_by_explicit_hyphenation_repair


@dataclass(frozen=True, slots=True)
class StopwordSet:
    """Loaded frozen Spanish stopwords for a configured variant."""

    variant: str
    words: frozenset[str]
    p0_words: frozenset[str]
    p0_count: int
    p1_count: int
    p0_sha256: str
    p1_additions: tuple[str, ...]


def clean_natural_text(text: str) -> CleaningResult:
    """Apply the locked repair and normalization sequence to modelling text."""
    soft_hyphen_removed_count = text.count(SOFT_HYPHEN)
    after_soft_join, soft_hyphen_join_count = SOFT_HYPHEN_JOIN_PATTERN.subn("", text)
    after_soft = after_soft_join.replace(SOFT_HYPHEN, "")
    changed_by_soft = after_soft != text

    explicit_join_count = 0
    repaired = after_soft

    while True:
        next_repaired, join_count = EXPLICIT_HYPHENATION_PATTERN.subn("", repaired)
        explicit_join_count += join_count

        if join_count == 0:
            break

        repaired = next_repaired

    normalized = unicodedata.normalize("NFKC", repaired).casefold()
    return CleaningResult(
        cleaned_text=normalized,
        soft_hyphen_join_count=soft_hyphen_join_count,
        soft_hyphen_removed_count=soft_hyphen_removed_count,
        explicit_hyphenation_join_count=explicit_join_count,
        changed_by_soft_hyphen_repair=changed_by_soft,
        changed_by_explicit_hyphenation_repair=explicit_join_count > 0,
    )


def lexical_tokens(cleaned_text: str) -> list[str]:
    """Tokenize NFKC/casefolded text into Unicode alphabetic runs of length at least 3."""
    tokens: list[str] = []
    current: list[str] = []

    for character in cleaned_text:
        if character.isalpha() or character.isdigit():
            current.append(character)
            continue

        _append_lexical_cluster(tokens=tokens, cluster=current)
        current = []

    _append_lexical_cluster(tokens=tokens, cluster=current)
    return tokens


def _append_lexical_cluster(*, tokens: list[str], cluster: list[str]) -> None:
    """Append a cluster only when it is purely alphabetic and long enough."""
    if len(cluster) >= 3 and all(character.isalpha() for character in cluster):
        tokens.append("".join(cluster))


def normalize_stopword(term: str) -> str:
    """Return the stopword normalization used by the runtime."""
    return unicodedata.normalize("NFKC", term.strip()).casefold()


def load_stopwords(path: Path, *, variant: str) -> StopwordSet:
    """Load the local P0 stoplist and apply the requested frozen variant."""
    if variant not in {"P0", "P1"}:
        raise TopicPreprocessingError(f"Unsupported stopword variant: {variant}")

    if not path.is_file():
        raise TopicPreprocessingError(f"Stopword file does not exist: {path}")

    p0_words: set[str] = set()

    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as error:
        raise TopicPreprocessingError(f"Could not read stopword file: {path}") from error

    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()

        if not stripped or stripped.startswith("#"):
            continue

        normalized = normalize_stopword(stripped)

        if not normalized:
            raise TopicPreprocessingError(f"Blank stopword at line {line_number}")

        if normalized in P1_ADDITIONS:
            raise TopicPreprocessingError(
                f"P0 stopword file must not contain P1-only term: {normalized}"
            )

        p0_words.add(normalized)

    protected_present = sorted(PROTECTED_SUBSTANTIVE_TERMS & p0_words)

    if protected_present:
        raise TopicPreprocessingError(
            f"P0 stopwords contain protected substantive terms: {protected_present}"
        )

    selected_words = frozenset(p0_words | (set(P1_ADDITIONS) if variant == "P1" else set()))
    p1_words = p0_words | set(P1_ADDITIONS)
    return StopwordSet(
        variant=variant,
        words=selected_words,
        p0_words=frozenset(p0_words),
        p0_count=len(p0_words),
        p1_count=len(p1_words),
        p0_sha256=sha256_file(path),
        p1_additions=tuple(sorted(P1_ADDITIONS)),
    )


def bounded_excerpt(text: str, *, character_limit: int) -> str:
    """Return a single-line excerpt bounded by character count."""
    compact = " ".join(text.split())

    if len(compact) <= character_limit:
        return compact

    return compact[:character_limit].rstrip()
