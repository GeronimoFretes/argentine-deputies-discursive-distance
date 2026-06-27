"""Detection of explicit parliamentary speaker markers."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum


class SpeakerLabelFamily(StrEnum):
    """Coarse family inferred from an explicit speaker label."""

    CHAIR = "chair"
    CHAMBER_SECRETARY = "chamber_secretary"
    EXECUTIVE_OFFICIAL = "executive_official"
    COLLECTIVE_OR_ANONYMOUS = "collective_or_anonymous"
    NAMED_OR_ROLE_UNSPECIFIED = "named_or_role_unspecified"


class MarkerSeparatorKind(StrEnum):
    """Separator joining a speaker label to its speech."""

    DOT_DASH = "dot_dash"
    DASH_ONLY = "dash_only"


class MarkerPosition(StrEnum):
    """Location of a marker inside its extraction block."""

    BLOCK_START = "block_start"
    EMBEDDED = "embedded"


@dataclass(frozen=True, slots=True)
class SpeakerMarker:
    """One explicit speaker marker and its exact source offsets."""

    start: int
    end: int
    raw_marker: str
    raw_title: str
    normalized_title: str
    raw_label: str
    normalized_label: str
    family: SpeakerLabelFamily
    separator: str
    separator_kind: MarkerSeparatorKind
    position: MarkerPosition
    is_multiline: bool
    detection_method: str
    detection_confidence: float


@dataclass(frozen=True, slots=True)
class _Separator:
    """Internal candidate marker separator."""

    start: int
    end: int
    raw: str
    kind: MarkerSeparatorKind


MAX_HONORIFIC_TO_SEPARATOR = 140
MAX_LABEL_CHARACTERS = 100
MAX_LABEL_WORDS = 12
MAX_NO_TITLE_PERIOD_LABEL_WORDS = 6

HONORIFIC_PATTERN = re.compile(
    r"(?<!\w)"
    r"(?P<title>Sra?\.?)"
    r"(?=\s)",
    flags=re.IGNORECASE,
)

DOT_DASH_SEPARATOR_PATTERN = re.compile(r"\.\s*[-–—]")

DASH_ONLY_SEPARATOR_PATTERN = re.compile(r"\s+[-–—](?=\s|$)")

PLAUSIBLE_LABEL_PATTERN = re.compile(
    r"^["
    r"A-Za-zÁÉÍÓÚÜÑ"
    r"áéíóúüñ"
    r"0-9"
    r" .,'’()/-"
    r"]+$"
)


def normalize_visible_text(
    value: str,
) -> str:
    """Collapse whitespace while retaining visible spelling."""
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize(
            "NFKC",
            value,
        ),
    ).strip()


def fold_text(
    value: str,
) -> str:
    """Return uppercase accent-insensitive text."""
    decomposed = unicodedata.normalize(
        "NFKD",
        value,
    )
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )

    return (
        re.sub(
            r"\s+",
            " ",
            without_marks,
        )
        .strip()
        .upper()
    )


def normalize_speaker_label(
    value: str,
) -> str:
    """Normalize a raw speaker label without repairing spelling."""
    return fold_text(value).strip(" .,:;–—-")


def classify_speaker_label(
    normalized_label: str,
) -> SpeakerLabelFamily:
    """Assign a conservative family from the printed label."""
    if normalized_label in {
        "PRESIDENTE DE LA NACION",
        "PRESIDENTA DE LA NACION",
    }:
        return SpeakerLabelFamily.EXECUTIVE_OFFICIAL

    if re.search(
        r"\b"
        r"(?:PRESIDENTE|PRESIDENTA|"
        r"VICEPRESIDENTE|VICEPRESIDENTA)"
        r"\b",
        normalized_label,
    ):
        return SpeakerLabelFamily.CHAIR

    if re.match(
        r"^SECRETARI[OA](?:\s*\(|$)",
        normalized_label,
    ):
        return SpeakerLabelFamily.CHAMBER_SECRETARY

    if re.search(
        r"\b"
        r"(?:JEFE\s+DE\s+GABINETE|"
        r"MINISTR[OA]|"
        r"SUBSECRETARI[OA]|"
        r"SECRETARI[OA]\s+DE)"
        r"\b",
        normalized_label,
    ):
        return SpeakerLabelFamily.EXECUTIVE_OFFICIAL

    if re.search(
        r"\b"
        r"(?:VARIOS|VARIAS|VOCES|"
        r"DIPUTADOS|DIPUTADAS)"
        r"\b",
        normalized_label,
    ):
        return SpeakerLabelFamily.COLLECTIVE_OR_ANONYMOUS

    return SpeakerLabelFamily.NAMED_OR_ROLE_UNSPECIFIED


def _ranges_overlap(
    *,
    first_start: int,
    first_end: int,
    second_start: int,
    second_end: int,
) -> bool:
    return first_start < second_end and second_start < first_end


def _collect_separators(
    text: str,
) -> tuple[_Separator, ...]:
    separators = [
        _Separator(
            start=match.start(),
            end=match.end(),
            raw=match.group(0),
            kind=(MarkerSeparatorKind.DOT_DASH),
        )
        for match in (DOT_DASH_SEPARATOR_PATTERN.finditer(text))
    ]

    for match in DASH_ONLY_SEPARATOR_PATTERN.finditer(text):
        if any(
            _ranges_overlap(
                first_start=match.start(),
                first_end=match.end(),
                second_start=(separator.start),
                second_end=separator.end,
            )
            for separator in separators
        ):
            continue

        separators.append(
            _Separator(
                start=match.start(),
                end=match.end(),
                raw=match.group(0),
                kind=(MarkerSeparatorKind.DASH_ONLY),
            )
        )

    return tuple(
        sorted(
            separators,
            key=lambda separator: (
                separator.start,
                separator.end,
            ),
        )
    )


def _plausible_label(
    *,
    raw_label: str,
    raw_title: str,
) -> bool:
    label = normalize_visible_text(raw_label)

    if not label:
        return False

    if len(label) > MAX_LABEL_CHARACTERS:
        return False

    if len(label.split()) > MAX_LABEL_WORDS:
        return False

    if not PLAUSIBLE_LABEL_PATTERN.fullmatch(label):
        return False

    if not any(character.isalpha() for character in label):
        return False

    if not raw_title.endswith(".") and len(label.split()) > MAX_NO_TITLE_PERIOD_LABEL_WORDS:
        return False

    return True


def _normalized_title(
    raw_title: str,
) -> str:
    folded = fold_text(raw_title)

    if folded.startswith("SRA"):
        return "SRA."

    return "SR."


def _detection_metadata(
    *,
    raw_title: str,
    separator_kind: MarkerSeparatorKind,
) -> tuple[str, float]:
    method_parts = [
        "explicit_honorific",
        separator_kind.value,
    ]
    confidence = 1.0

    if not raw_title.endswith("."):
        method_parts.append("title_without_period")
        confidence = min(
            confidence,
            0.95,
        )

    if separator_kind == MarkerSeparatorKind.DASH_ONLY:
        confidence = min(
            confidence,
            0.95,
        )

    return (
        "_".join(method_parts),
        confidence,
    )


def find_speaker_markers(
    text: str,
) -> tuple[SpeakerMarker, ...]:
    """Find conservative explicit Sr./Sra. markers in one block."""
    honorifics = list(HONORIFIC_PATTERN.finditer(text))
    separators = _collect_separators(text)
    used_honorific_starts: set[int] = set()
    markers = []

    for separator in separators:
        candidates = [
            honorific
            for honorific in honorifics
            if (
                honorific.end() <= separator.start
                and (separator.start - honorific.end() <= MAX_HONORIFIC_TO_SEPARATOR)
                and honorific.start() not in used_honorific_starts
            )
        ]

        selected_honorific = None
        selected_raw_label = ""

        for honorific in reversed(candidates):
            raw_title = honorific.group("title")
            raw_label = text[honorific.end() : separator.start]

            if _plausible_label(
                raw_label=raw_label,
                raw_title=raw_title,
            ):
                selected_honorific = honorific
                selected_raw_label = raw_label
                break

        if selected_honorific is None:
            continue

        marker_start = selected_honorific.start()
        marker_end = separator.end
        raw_title = selected_honorific.group("title")
        visible_label = normalize_visible_text(selected_raw_label)
        normalized_label = normalize_speaker_label(visible_label)
        raw_marker = text[marker_start:marker_end]

        method, confidence = _detection_metadata(
            raw_title=raw_title,
            separator_kind=(separator.kind),
        )

        markers.append(
            SpeakerMarker(
                start=marker_start,
                end=marker_end,
                raw_marker=raw_marker,
                raw_title=raw_title,
                normalized_title=(_normalized_title(raw_title)),
                raw_label=visible_label,
                normalized_label=(normalized_label),
                family=(classify_speaker_label(normalized_label)),
                separator=(separator.raw),
                separator_kind=(separator.kind),
                position=(
                    MarkerPosition.BLOCK_START
                    if not text[:marker_start].strip()
                    else MarkerPosition.EMBEDDED
                ),
                is_multiline=("\n" in raw_marker or "\r" in raw_marker),
                detection_method=method,
                detection_confidence=(confidence),
            )
        )

        used_honorific_starts.add(marker_start)

    return tuple(
        sorted(
            markers,
            key=lambda marker: (
                marker.start,
                marker.end,
            ),
        )
    )
