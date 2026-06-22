"""Lossless content classification inside parsed parliamentary turns."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum

from .speaker import SpeakerLabelFamily
from .speaker_turns import (
    SegmentAttributionMethod,
    SpeakerTurn,
    SpeakerTurnSegment,
)


class TurnContentError(RuntimeError):
    """Raised when turn content cannot be classified safely."""


class TurnContentKind(StrEnum):
    """Functional kind of an exact source span."""

    SPOKEN_TEXT = "spoken_text"
    DOCUMENTARY_INSERT = "documentary_insert"
    STAGE_DIRECTION = "stage_direction"
    EDITORIAL_NOTE = "editorial_note"
    UNATTRIBUTED_TEXT = "unattributed_text"


class DocumentaryCue(StrEnum):
    """Audited cue establishing a documentary boundary."""

    ORDER_OF_DAY = "order_of_day_heading"
    CODED_PROJECT = "coded_project_heading"
    SECRETARY_EXPEDIENTE = "secretary_expediente_before_project"


@dataclass(frozen=True, slots=True)
class DocumentaryBoundary:
    """One high-confidence documentary transition inside a turn."""

    turn_offset: int
    page_number: int
    reading_order: int
    source_offset: int
    cue: DocumentaryCue
    matched_text: str
    classification_method: str
    classification_confidence: float
    documentary_word_count: int

    @property
    def source_reference(self) -> str:
        """Return the source block reference."""
        return f"p{self.page_number}:b{self.reading_order}"


@dataclass(frozen=True, slots=True)
class TurnContentSpan:
    """One exact source span with a functional classification."""

    page_number: int
    reading_order: int
    start: int
    end: int
    text: str
    content_kind: TurnContentKind
    include_in_speech: bool
    classification_method: str
    classification_confidence: float
    attribution_method: SegmentAttributionMethod

    @property
    def block_reference(self) -> str:
        """Return the source block reference."""
        return f"p{self.page_number}:b{self.reading_order}"

    @property
    def word_count(self) -> int:
        """Return a simple whitespace-token count."""
        return len(self.text.split())


@dataclass(frozen=True, slots=True)
class TurnContentResult:
    """Complete lossless classification for one parsed turn."""

    turn_index: int
    documentary_boundary: DocumentaryBoundary | None
    spans: tuple[TurnContentSpan, ...]

    @property
    def speech_word_count(self) -> int:
        """Return words eligible for speaker-level analysis."""
        return sum(span.word_count for span in self.spans if span.include_in_speech)

    @property
    def documentary_word_count(self) -> int:
        """Return words classified as documentary material."""
        return sum(
            span.word_count
            for span in self.spans
            if (span.content_kind == TurnContentKind.DOCUMENTARY_INSERT)
        )


@dataclass(frozen=True, slots=True)
class _JoinedSegment:
    """Map one source segment into the synthetic joined-turn text."""

    segment: SpeakerTurnSegment
    joined_start: int
    joined_end: int


@dataclass(frozen=True, slots=True)
class _PhysicalLine:
    """One physical source line and its exact offsets."""

    start: int
    full_end: int
    visible_end: int
    text: str


@dataclass(frozen=True, slots=True)
class _NonSpeechCandidate:
    """One exact local range classified as non-spoken content."""

    start: int
    end: int
    content_kind: TurnContentKind
    classification_method: str
    classification_confidence: float


MIN_DOCUMENTARY_WORDS = 200
MAX_FORMAL_BODY_LOOKAHEAD = 5000
MAX_EXPEDIENTE_TO_PROJECT_CHARACTERS = 3000

ELIGIBLE_DOCUMENTARY_FAMILIES = {
    SpeakerLabelFamily.CHAIR,
    SpeakerLabelFamily.CHAMBER_SECRETARY,
}

ORDER_OF_DAY_PATTERN = re.compile(
    r"\(\s*"
    r"ORDEN\s+DEL\s+D[IÍ]A\s+"
    r"(?:N(?:[.º°]|O)?\s*)?"
    r"\d[\d.]*"
    r"(?:\s*(?:Y|,)\s*\d[\d.]*)?"
    r"\s*\)",
    flags=re.IGNORECASE,
)

CODED_PROJECT_PATTERN = re.compile(
    r"\(\s*"
    r"\d[\d.]*-[A-Z]\.-\d+"
    r"\s*\)"
    r"\s*"
    r"PROYECTO\s+DE\s+"
    r"(?:LEY|RESOLUCI[ÓO]N|DECLARACI[ÓO]N)"
    r"\b",
    flags=re.IGNORECASE,
)

EXPEDIENTE_START_PATTERN = re.compile(
    r"(?m)^[ \t]*"
    r"EXPEDIENTES?\s+"
    r"\d[\d.]*-[A-Z]\.-\d+"
    r"\b",
    flags=re.IGNORECASE,
)

FORMAL_BODY_PATTERN = re.compile(
    r"(?im)^[ \t]*"
    r"(?:(?:I|II|III|IV|V|VI|VII|VIII|IX|X)"
    r"[ \t]+)?"
    r"(?:"
    r"DICTAMEN\b"
    r"|HONORABLE\s+C[AÁ]MARA\s*:"
    r"|PROYECTO\s+DE\s+"
    r"(?:LEY|RESOLUCI[ÓO]N|DECLARACI[ÓO]N)"
    r"|EL\s+SENADO\s+Y\s+C[AÁ]MARA\s+"
    r"DE\s+DIPUTADOS"
    r"|LA\s+C[AÁ]MARA\s+DE\s+DIPUTADOS\s+"
    r"DE\s+LA\s+NACI[ÓO]N"
    r"|FUNDAMENTOS\b"
    r")"
)

PROJECT_BODY_PATTERN = re.compile(
    r"\b"
    r"(?:"
    r"LA\s+C[AÁ]MARA\s+DE\s+DIPUTADOS\s+"
    r"DE\s+LA\s+NACI[ÓO]N"
    r"|EL\s+SENADO\s+Y\s+C[AÁ]MARA\s+"
    r"DE\s+DIPUTADOS"
    r")"
    r"\b",
    flags=re.IGNORECASE,
)

PARENTHETICAL_PATTERN = re.compile(
    r"\("
    r"(?P<body>[^()]{1,500})"
    r"\)",
    flags=re.DOTALL,
)

SIMPLE_STAGE_PARENTHETICAL_PATTERN = re.compile(
    r"^"
    r"(?:APLAUSOS?|RISAS?)"
    r"(?:\s+Y\s+"
    r"(?:"
    r"APLAUSOS?"
    r"|RISAS?"
    r"|MANIFESTACIONES"
    r"(?:\s+EN\s+"
    r"(?:LAS\s+GALERIAS|LAS\s+BANCAS|LA\s+SALA)"
    r")?"
    r")"
    r")?"
    r"(?:\s+PROLONGADOS?)?"
    r"\.?"
    r"$"
)

MULTI_EVENT_PARENTHETICAL_PATTERN = re.compile(
    r"^"
    r"APLAUSOS?\.\s+"
    r"VARIOS\s+SENORES\s+DIPUTADOS\s+"
    r"(?:"
    r"HABLAN\s+A\s+LA\s+VEZ"
    r"|RODEAN\s+Y\s+FELICITAN\s+AL\s+ORADOR"
    r")"
    r"\.?"
    r"$"
)

EDITORIAL_LINE_START_PATTERN = re.compile(
    r"^[ \t]*"
    r"(?:\d+[ \t]*[.)]?[ \t]+)?"
    r"V[EÉ]ASE\b",
    flags=re.IGNORECASE,
)

SPEAKER_LINE_START_PATTERN = re.compile(
    r"^[ \t]*"
    r"Sr(?:a)?\.?"
    r"(?=\s)",
    flags=re.IGNORECASE,
)

MAX_WRAPPED_NON_SPEECH_LINES = 6
MAX_WRAPPED_NON_SPEECH_CHARACTERS = 800
MAX_CROSS_SEGMENT_OPEN_STAGE_LINES = 10

CROSS_SEGMENT_LOWERCASE_PATTERN = re.compile(r"^[\s\u00a0]*[a-záéíóúüñ]")

OPEN_PRESIDENCY_PREFIXES = {
    "OCUPA LA PRESIDENCIA EL SENOR",
    "OCUPA LA PRESIDENCIA LA SENORA",
}

OPEN_STAGE_ACTION_PATTERN = re.compile(
    r"^(?:LA|EL)\s+"
    r"SENOR(?:A)?\s+DIPUTAD[OA]\b"
    r".*\s+(?:HACE|REALIZA)$"
)

OUTSIDE_MICROPHONE_STAGE_PATTERN = re.compile(
    r"^[ \t]*"
    r"(?:[-–—]+[ \t]*)+"
    r"(?:La|El)\s+"
    r"señor(?:a)?\s+diputad[oa]\b"
    r".{0,800}?"
    r"(?:"
    r"fuera\s+de\s+micr[óo]fono"
    r"|no\s+se\s+alcanzan\s+a\s+percibir"
    r")"
    r".{0,400}?"
    r"[.!?]"
    r"(?=[ \t]*(?:\r?\n|$))",
    flags=(re.IGNORECASE | re.MULTILINE | re.DOTALL),
)


def _joined_segments(
    turn: SpeakerTurn,
) -> tuple[_JoinedSegment, ...]:
    """Map source segments into `SpeakerTurn.text` offsets."""
    mappings = []
    cursor = 0

    for segment_index, segment in enumerate(turn.segments):
        joined_start = cursor
        joined_end = joined_start + len(segment.text)

        mappings.append(
            _JoinedSegment(
                segment=segment,
                joined_start=joined_start,
                joined_end=joined_end,
            )
        )

        cursor = joined_end

        if segment_index < len(turn.segments) - 1:
            cursor += 1

    if cursor != len(turn.text):
        raise TurnContentError(
            f"Joined turn text does not match its source segments for turn {turn.turn_index}."
        )

    return tuple(mappings)


def _source_for_turn_offset(
    *,
    turn: SpeakerTurn,
    offset: int,
) -> tuple[
    int,
    int,
    int,
]:
    """Map a joined-turn offset to an exact source offset."""
    mappings = _joined_segments(turn)

    for mapping in mappings:
        if mapping.joined_start <= offset < mapping.joined_end:
            local_offset = offset - mapping.joined_start
            segment = mapping.segment

            return (
                segment.page_number,
                segment.reading_order,
                segment.start + local_offset,
            )

    raise TurnContentError(
        f"Documentary boundary does not map to source text in turn {turn.turn_index}: {offset}"
    )


def _has_formal_body(
    *,
    text: str,
    candidate_start: int,
    cue: DocumentaryCue,
) -> bool:
    """Return whether the candidate has formal-document support."""
    suffix = text[candidate_start:]

    if len(suffix.split()) < MIN_DOCUMENTARY_WORDS:
        return False

    lookahead = suffix[:MAX_FORMAL_BODY_LOOKAHEAD]

    if cue == DocumentaryCue.ORDER_OF_DAY:
        return FORMAL_BODY_PATTERN.search(lookahead) is not None

    if cue == DocumentaryCue.CODED_PROJECT:
        return PROJECT_BODY_PATTERN.search(lookahead) is not None

    if cue == DocumentaryCue.SECRETARY_EXPEDIENTE:
        project_match = CODED_PROJECT_PATTERN.search(
            suffix[:(MAX_EXPEDIENTE_TO_PROJECT_CHARACTERS)]
        )

        if project_match is None:
            return False

        return PROJECT_BODY_PATTERN.search(lookahead) is not None

    return False


def find_documentary_boundary(
    turn: SpeakerTurn,
) -> DocumentaryBoundary | None:
    """Find one conservative documentary transition in a turn."""
    if turn.speaker_family not in ELIGIBLE_DOCUMENTARY_FAMILIES:
        return None

    text = turn.text

    candidates: list[
        tuple[
            int,
            DocumentaryCue,
            str,
        ]
    ] = []

    for match in ORDER_OF_DAY_PATTERN.finditer(text):
        candidates.append(
            (
                match.start(),
                DocumentaryCue.ORDER_OF_DAY,
                match.group(0),
            )
        )

    for match in CODED_PROJECT_PATTERN.finditer(text):
        candidates.append(
            (
                match.start(),
                DocumentaryCue.CODED_PROJECT,
                match.group(0),
            )
        )

    if turn.speaker_family == SpeakerLabelFamily.CHAMBER_SECRETARY:
        project_matches = tuple(CODED_PROJECT_PATTERN.finditer(text))

        for expediente_match in EXPEDIENTE_START_PATTERN.finditer(text):
            has_nearby_project = any(
                (
                    expediente_match.start()
                    <= project_match.start()
                    <= (expediente_match.start() + (MAX_EXPEDIENTE_TO_PROJECT_CHARACTERS))
                )
                for project_match in project_matches
            )

            if has_nearby_project:
                candidates.append(
                    (
                        expediente_match.start(),
                        (DocumentaryCue.SECRETARY_EXPEDIENTE),
                        expediente_match.group(0),
                    )
                )

    for start, cue, matched_text in sorted(
        candidates,
        key=lambda candidate: (
            candidate[0],
            candidate[1].value,
        ),
    ):
        if not _has_formal_body(
            text=text,
            candidate_start=start,
            cue=cue,
        ):
            continue

        (
            page_number,
            reading_order,
            source_offset,
        ) = _source_for_turn_offset(
            turn=turn,
            offset=start,
        )

        return DocumentaryBoundary(
            turn_offset=start,
            page_number=page_number,
            reading_order=reading_order,
            source_offset=source_offset,
            cue=cue,
            matched_text=matched_text,
            classification_method=("eligible_role_formal_cue_formal_body_minimum_length"),
            classification_confidence=1.0,
            documentary_word_count=len(text[start:].split()),
        )

    return None


def _default_kind(
    turn: SpeakerTurn,
) -> TurnContentKind:
    """Return the default kind before any documentary transition."""
    if turn.is_unattributed:
        return TurnContentKind.UNATTRIBUTED_TEXT

    return TurnContentKind.SPOKEN_TEXT


def _append_content_span(
    *,
    output: list[TurnContentSpan],
    segment: SpeakerTurnSegment,
    start: int,
    end: int,
    kind: TurnContentKind,
    method: str,
    confidence: float,
) -> None:
    """Append one non-empty exact source span."""
    if not (segment.start <= start <= end <= segment.end):
        raise TurnContentError(
            f"Invalid source content span at {segment.block_reference}:{start}:{end}"
        )

    text_start = start - segment.start
    text_end = end - segment.start
    span_text = segment.text[text_start:text_end]

    if not span_text:
        return

    output.append(
        TurnContentSpan(
            page_number=segment.page_number,
            reading_order=(segment.reading_order),
            start=start,
            end=end,
            text=span_text,
            content_kind=kind,
            include_in_speech=(kind == TurnContentKind.SPOKEN_TEXT),
            classification_method=method,
            classification_confidence=(confidence),
            attribution_method=(segment.attribution_method),
        )
    )


def _fold_non_speech_text(
    value: str,
) -> str:
    """Return normalized uppercase accent-insensitive text."""
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


def _ranges_overlap(
    *,
    first_start: int,
    first_end: int,
    second_start: int,
    second_end: int,
) -> bool:
    """Return whether two half-open ranges overlap."""
    return first_start < second_end and second_start < first_end


def _physical_lines(
    text: str,
) -> tuple[_PhysicalLine, ...]:
    """Return exact physical-line ranges."""
    lines = []
    cursor = 0

    for raw_line in text.splitlines(keepends=True):
        full_end = cursor + len(raw_line)
        visible_end = full_end

        while visible_end > cursor and text[visible_end - 1] in "\r\n":
            visible_end -= 1

        lines.append(
            _PhysicalLine(
                start=cursor,
                full_end=full_end,
                visible_end=visible_end,
                text=text[cursor:visible_end],
            )
        )
        cursor = full_end

    return tuple(lines)


def _line_ends_direction(
    value: str,
) -> bool:
    """Return whether a wrapped note appears complete."""
    stripped = value.rstrip()

    if not stripped:
        return False

    if stripped.endswith("-"):
        return False

    return stripped.endswith(
        (
            ".",
            ":",
            ";",
            "!",
            "?",
            ")",
        )
    )


def _strip_stage_prefix(
    value: str,
) -> tuple[str, bool]:
    """Remove leading chamber dashes and optional page markers."""
    folded = _fold_non_speech_text(value)
    dash_match = re.match(
        r"^[ \t]*(?:[-–—]+[ \t]*)+",
        folded,
    )
    has_dash = dash_match is not None

    if dash_match is not None:
        folded = folded[dash_match.end() :]

    page_marker = re.match(
        r"^\d+[ \t]*[-–—][ \t]*",
        folded,
    )

    if page_marker is not None:
        folded = folded[page_marker.end() :]

    return folded.strip(), has_dash


def _stage_line_start(
    value: str,
) -> tuple[bool, bool]:
    """Return whether a line begins an audited stage direction.

    The second result indicates whether only the physical line
    should be consumed.
    """
    folded, has_dash = _strip_stage_prefix(value)

    original = value.strip()
    has_letters = any(character.isalpha() for character in original)
    is_uppercase_line = has_letters and original.upper() == original

    if folded in {
        "MANIFESTACIONES",
        "MANIFESTACIONES.",
        "MANIFESTACIONES EN LAS BANCAS.",
        "MANIFESTACIONES EN LA SALA.",
        "MURMULLOS EN EL RECINTO.",
    } and (has_dash or is_uppercase_line):
        return True, True

    strict_prefixes = (
        "OCUPA LA PRESIDENCIA ",
        "MIENTRAS SE PRACTICA LA VOTACION",
        "MIENTRAS SE REALIZA LA VOTACION",
        "VARIOS SENORES DIPUTADOS HABLAN A LA VEZ",
    )

    if folded.startswith(strict_prefixes):
        return True, False

    outside_microphone = re.match(
        r"^(?:LA|EL)\s+"
        r"SENOR(?:A)?\s+DIPUTAD[OA]\b"
        r".*"
        r"(?:"
        r"FUERA\s+DE\s+MICROFONO"
        r"|NO\s+SE\s+ALCANZAN\s+A\s+PERCIBIR"
        r")",
        folded,
    )

    if outside_microphone is not None:
        return True, False

    if has_dash and (OPEN_STAGE_ACTION_PATTERN.fullmatch(folded) is not None):
        return True, False

    if not has_dash:
        return False, False

    dash_required_prefixes = (
        "LUEGO DE UNOS INSTANTES",
        "DESPUES DE UNOS INSTANTES",
        "TRANSCURRIDOS UNOS INSTANTES",
        "A LA HORA ",
        "SE PASA A CUARTO INTERMEDIO",
        "SE DISPONE UN CUARTO INTERMEDIO",
        "SE REANUDA LA SESION",
        "SE REANUDA EL ACTO",
        "SE SUSPENDE LA SESION",
        "MANIFESTACIONES EN LAS BANCAS",
        "MANIFESTACIONES EN LA SALA",
        "MURMULLOS EN EL RECINTO",
    )

    return (
        folded.startswith(dash_required_prefixes),
        False,
    )


def _is_stage_parenthetical(
    body: str,
) -> bool:
    """Return whether a parenthetical is an audited chamber event."""
    folded = _fold_non_speech_text(body).strip()

    return bool(
        SIMPLE_STAGE_PARENTHETICAL_PATTERN.fullmatch(folded)
        or (MULTI_EVENT_PARENTHETICAL_PATTERN.fullmatch(folded))
    )


def _line_starts_new_item(
    value: str,
) -> bool:
    """Return whether a wrapped candidate must stop before this line."""
    stripped = value.strip()

    if not stripped:
        return True

    if SPEAKER_LINE_START_PATTERN.match(stripped):
        return True

    if EDITORIAL_LINE_START_PATTERN.match(value):
        return True

    is_stage, _ = _stage_line_start(value)
    return is_stage


def _extend_wrapped_line(
    *,
    text: str,
    lines: tuple[_PhysicalLine, ...],
    start_index: int,
    single_line: bool,
) -> tuple[int, int, bool]:
    """Extend a known note through PDF-wrapped continuation lines."""
    first = lines[start_index]
    end = first.visible_end
    consumed_lines = 1

    if single_line:
        return first.start, end, False

    while (
        consumed_lines < MAX_WRAPPED_NON_SPEECH_LINES
        and (start_index + consumed_lines < len(lines))
        and (end - first.start < MAX_WRAPPED_NON_SPEECH_CHARACTERS)
    ):
        current = lines[start_index + consumed_lines - 1]

        if _line_ends_direction(current.text):
            break

        next_line = lines[start_index + consumed_lines]

        if _line_starts_new_item(next_line.text):
            break

        end = next_line.visible_end
        consumed_lines += 1

    return (
        first.start,
        end,
        consumed_lines > 1,
    )


def _deduplicate_non_speech_candidates(
    candidates: list[_NonSpeechCandidate],
) -> tuple[
    _NonSpeechCandidate,
    ...,
]:
    """Return ordered, non-overlapping exact candidates."""
    priority = {
        TurnContentKind.STAGE_DIRECTION: 0,
        TurnContentKind.EDITORIAL_NOTE: 1,
    }

    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.start,
            priority[candidate.content_kind],
            candidate.end,
        ),
    )
    accepted: list[_NonSpeechCandidate] = []

    for candidate in ordered:
        if any(
            _ranges_overlap(
                first_start=(candidate.start),
                first_end=candidate.end,
                second_start=(existing.start),
                second_end=existing.end,
            )
            for existing in accepted
        ):
            continue

        accepted.append(candidate)

    return tuple(
        sorted(
            accepted,
            key=lambda candidate: (
                candidate.start,
                candidate.end,
            ),
        )
    )


def _find_non_speech_candidates(
    text: str,
) -> tuple[
    _NonSpeechCandidate,
    ...,
]:
    """Find conservative exact non-speech spans."""
    candidates = []

    for match in OUTSIDE_MICROPHONE_STAGE_PATTERN.finditer(text):
        candidates.append(
            _NonSpeechCandidate(
                start=match.start(),
                end=match.end(),
                content_kind=(TurnContentKind.STAGE_DIRECTION),
                classification_method=("exact_multiline_outside_microphone_stage_direction"),
                classification_confidence=1.0,
            )
        )

    for match in PARENTHETICAL_PATTERN.finditer(text):
        if not _is_stage_parenthetical(match.group("body")):
            continue

        candidates.append(
            _NonSpeechCandidate(
                start=match.start(),
                end=match.end(),
                content_kind=(TurnContentKind.STAGE_DIRECTION),
                classification_method=("exact_parenthetical_stage_direction"),
                classification_confidence=1.0,
            )
        )

    lines = _physical_lines(text)

    for line_index, line in enumerate(lines):
        if EDITORIAL_LINE_START_PATTERN.match(line.text):
            start, end, wrapped = _extend_wrapped_line(
                text=text,
                lines=lines,
                start_index=line_index,
                single_line=False,
            )
            candidates.append(
                _NonSpeechCandidate(
                    start=start,
                    end=end,
                    content_kind=(TurnContentKind.EDITORIAL_NOTE),
                    classification_method=(
                        "anchored_vease_" + ("wrapped_line" if wrapped else "line")
                    ),
                    classification_confidence=1.0,
                )
            )
            continue

        is_stage, single_line = _stage_line_start(line.text)

        if not is_stage:
            continue

        start, end, wrapped = _extend_wrapped_line(
            text=text,
            lines=lines,
            start_index=line_index,
            single_line=single_line,
        )
        candidates.append(
            _NonSpeechCandidate(
                start=start,
                end=end,
                content_kind=(TurnContentKind.STAGE_DIRECTION),
                classification_method=("anchored_stage_" + ("wrapped_line" if wrapped else "line")),
                classification_confidence=(0.95 if wrapped else 1.0),
            )
        )

    return _deduplicate_non_speech_candidates(candidates)


def _append_refined_content_span(
    *,
    output: list[TurnContentSpan],
    source_span: TurnContentSpan,
    local_start: int,
    local_end: int,
    kind: TurnContentKind,
    method: str,
    confidence: float,
) -> None:
    """Append one exact subspan from an existing content span."""
    if not (0 <= local_start <= local_end <= len(source_span.text)):
        raise TurnContentError(
            "Invalid refined content range at "
            f"{source_span.block_reference}:"
            f"{local_start}:{local_end}"
        )

    if local_start == local_end:
        return

    output.append(
        TurnContentSpan(
            page_number=(source_span.page_number),
            reading_order=(source_span.reading_order),
            start=(source_span.start + local_start),
            end=(source_span.start + local_end),
            text=source_span.text[local_start:local_end],
            content_kind=kind,
            include_in_speech=(kind == TurnContentKind.SPOKEN_TEXT),
            classification_method=method,
            classification_confidence=(confidence),
            attribution_method=(source_span.attribution_method),
        )
    )


def _refine_non_speech_content(
    span: TurnContentSpan,
) -> tuple[
    TurnContentSpan,
    ...,
]:
    """Split exact stage and editorial subspans from base content."""
    if span.content_kind not in {
        TurnContentKind.SPOKEN_TEXT,
        TurnContentKind.UNATTRIBUTED_TEXT,
    }:
        return (span,)

    candidates = _find_non_speech_candidates(span.text)

    if not candidates:
        return (span,)

    output: list[TurnContentSpan] = []
    cursor = 0

    for candidate in candidates:
        _append_refined_content_span(
            output=output,
            source_span=span,
            local_start=cursor,
            local_end=candidate.start,
            kind=span.content_kind,
            method=(span.classification_method),
            confidence=(span.classification_confidence),
        )
        _append_refined_content_span(
            output=output,
            source_span=span,
            local_start=candidate.start,
            local_end=candidate.end,
            kind=candidate.content_kind,
            method=(candidate.classification_method),
            confidence=(candidate.classification_confidence),
        )
        cursor = candidate.end

    _append_refined_content_span(
        output=output,
        source_span=span,
        local_start=cursor,
        local_end=len(span.text),
        kind=span.content_kind,
        method=(span.classification_method),
        confidence=(span.classification_confidence),
    )

    return tuple(output)


def _cross_segment_continuation_end(
    text: str,
    *,
    max_lines: int = MAX_WRAPPED_NON_SPEECH_LINES,
) -> int:
    """Find the end of a wrapped non-speech continuation."""
    lines = _physical_lines(text)

    if not lines:
        return 0

    end = lines[0].visible_end
    consumed_lines = 1

    while (
        consumed_lines < max_lines
        and consumed_lines < len(lines)
        and end < MAX_WRAPPED_NON_SPEECH_CHARACTERS
    ):
        current = lines[consumed_lines - 1]

        if _line_ends_direction(current.text):
            break

        next_line = lines[consumed_lines]

        if _line_starts_new_item(next_line.text):
            break

        end = next_line.visible_end
        consumed_lines += 1

    return end


def _cross_segment_extension_method(
    span: TurnContentSpan,
) -> str | None:
    """Return the method for a supported cross-segment continuation."""
    if span.content_kind not in {
        TurnContentKind.STAGE_DIRECTION,
        TurnContentKind.EDITORIAL_NOTE,
    }:
        return None

    if span.text.rstrip().endswith("-"):
        return "cross_segment_hyphenated_" + span.content_kind.value

    if span.content_kind != TurnContentKind.STAGE_DIRECTION:
        return None

    folded, _ = _strip_stage_prefix(span.text)

    if folded in OPEN_PRESIDENCY_PREFIXES:
        return "cross_segment_open_stage_direction"

    if OPEN_STAGE_ACTION_PATTERN.fullmatch(folded) is not None:
        return "cross_segment_open_stage_direction"

    return None


def _extend_cross_segment_non_speech_content(
    spans: tuple[
        TurnContentSpan,
        ...,
    ],
) -> tuple[
    TurnContentSpan,
    ...,
]:
    """Extend hyphenated non-speech material into the next source span."""
    output: list[TurnContentSpan] = []
    span_index = 0

    while span_index < len(spans):
        current = spans[span_index]
        output.append(current)

        extension_method = _cross_segment_extension_method(current)

        if extension_method is None or span_index + 1 >= len(spans):
            span_index += 1
            continue

        following = spans[span_index + 1]

        if following.content_kind not in {
            TurnContentKind.SPOKEN_TEXT,
            TurnContentKind.UNATTRIBUTED_TEXT,
        } or (CROSS_SEGMENT_LOWERCASE_PATTERN.match(following.text) is None):
            span_index += 1
            continue

        max_lines = (
            MAX_CROSS_SEGMENT_OPEN_STAGE_LINES
            if (extension_method == "cross_segment_open_stage_direction")
            else MAX_WRAPPED_NON_SPEECH_LINES
        )

        continuation_end = _cross_segment_continuation_end(
            following.text,
            max_lines=max_lines,
        )

        if continuation_end <= 0:
            span_index += 1
            continue

        method = extension_method

        _append_refined_content_span(
            output=output,
            source_span=following,
            local_start=0,
            local_end=continuation_end,
            kind=current.content_kind,
            method=method,
            confidence=(current.classification_confidence),
        )

        _append_refined_content_span(
            output=output,
            source_span=following,
            local_start=continuation_end,
            local_end=len(following.text),
            kind=following.content_kind,
            method=(following.classification_method),
            confidence=(following.classification_confidence),
        )

        span_index += 2

    return tuple(output)


def classify_speaker_turn_content(
    turn: SpeakerTurn,
) -> TurnContentResult:
    """Classify every source character assigned to one turn."""
    boundary = find_documentary_boundary(turn)
    default_kind = _default_kind(turn)
    output: list[TurnContentSpan] = []

    if turn.is_unattributed:
        default_method = "unattributed_turn"
    else:
        default_method = "speaker_turn_default"

    for mapping in _joined_segments(turn):
        segment = mapping.segment

        if boundary is None:
            _append_content_span(
                output=output,
                segment=segment,
                start=segment.start,
                end=segment.end,
                kind=default_kind,
                method=default_method,
                confidence=1.0,
            )
            continue

        boundary_offset = boundary.turn_offset

        if mapping.joined_end <= boundary_offset:
            _append_content_span(
                output=output,
                segment=segment,
                start=segment.start,
                end=segment.end,
                kind=default_kind,
                method=default_method,
                confidence=1.0,
            )
            continue

        if mapping.joined_start >= boundary_offset:
            _append_content_span(
                output=output,
                segment=segment,
                start=segment.start,
                end=segment.end,
                kind=(TurnContentKind.DOCUMENTARY_INSERT),
                method=(boundary.classification_method),
                confidence=(boundary.classification_confidence),
            )
            continue

        local_boundary = segment.start + boundary_offset - mapping.joined_start

        _append_content_span(
            output=output,
            segment=segment,
            start=segment.start,
            end=local_boundary,
            kind=default_kind,
            method=default_method,
            confidence=1.0,
        )
        _append_content_span(
            output=output,
            segment=segment,
            start=local_boundary,
            end=segment.end,
            kind=(TurnContentKind.DOCUMENTARY_INSERT),
            method=(boundary.classification_method),
            confidence=(boundary.classification_confidence),
        )

    refined_output = [
        refined_span for span in output for refined_span in _refine_non_speech_content(span)
    ]

    cross_segment_output = _extend_cross_segment_non_speech_content(tuple(refined_output))

    return TurnContentResult(
        turn_index=turn.turn_index,
        documentary_boundary=boundary,
        spans=cross_segment_output,
    )
