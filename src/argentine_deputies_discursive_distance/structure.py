"""Block-level structural segmentation of parliamentary transcripts."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum


class StructuralSegmentationError(RuntimeError):
    """Raised when reliable structural boundaries cannot be identified."""


class StructuralZone(StrEnum):
    """Document-wide structural location of a text block."""

    FRONT_MATTER = "front_matter"
    PROCEEDINGS = "proceedings"
    POST_PROCEEDINGS = "post_proceedings"
    UNKNOWN = "unknown"


class ContentRole(StrEnum):
    """Functional content role independent of structural location."""

    COVER = "cover"
    ATTENDANCE = "attendance"
    SUMMARY = "summary"
    TRANSCRIPT = "transcript"
    PROCEDURAL = "procedural"
    VOTE_RECORD = "vote_record"
    SANCTION_TEXT = "sanction_text"
    INSERTION = "insertion"
    RUNNING_HEADER = "running_header"
    RUNNING_FOOTER = "running_footer"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class StructuralInputBlock:
    """Minimal extracted-block representation required for segmentation."""

    page_number: int
    reading_order: int
    region: str
    y0: float
    y1: float
    text: str

    @property
    def reference(self) -> str:
        """Return a stable human-readable page/block reference."""
        return f"p{self.page_number}:b{self.reading_order}"


@dataclass(frozen=True, slots=True)
class StructuralAnchor:
    """One detected structural transition anchor."""

    page_number: int
    reading_order: int
    method: str
    matched_text: str

    @property
    def reference(self) -> str:
        """Return a human-readable page/block reference."""
        return f"p{self.page_number}:b{self.reading_order}"


@dataclass(frozen=True, slots=True)
class BlockClassification:
    """Structural classification assigned to one extracted block."""

    page_number: int
    reading_order: int
    structural_zone: StructuralZone
    content_role: ContentRole
    include_in_discourse: bool
    exclusion_reason: str | None
    classification_method: str
    classification_confidence: float

    @property
    def reference(self) -> str:
        """Return a human-readable page/block reference."""
        return f"p{self.page_number}:b{self.reading_order}"


@dataclass(frozen=True, slots=True)
class StructuralSegmentationResult:
    """Complete structural-boundary and block-classification result."""

    start_anchor: StructuralAnchor
    end_anchor: StructuralAnchor | None
    post_start_anchor: StructuralAnchor | None
    end_method: str
    classifications: tuple[BlockClassification, ...]


CHAIR_PATTERN = re.compile(r"(?m)^\s*(?:SR|SRA)\.\s+PRESIDENT(?:E|A)\b")

NON_CHAIR_SPEAKER_PATTERN = re.compile(
    r"(?m)^\s*(?:SR|SRA)\.\s+"
    r"(?!PRESIDENT(?:E|A)\b)[A-Z]"
)

OPENING_PATTERN = re.compile(
    r"\bQUEDA\s+ABIERTA\s+LA\s+SESION\b"
    r"|\bSE\s+ABRE\s+LA\s+SESION\b"
    r"|\bSE\s+DECLARA\s+ABIERTA\s+LA\s+SESION\b"
    r"|\bDECLARO\s+ABIERTA\s+LA\s+SESION\b"
)

CLOSING_PATTERN = re.compile(
    r"\bQUEDA\s+LEVANTADA\s+LA\s+SESION\b"
    r"|\bSE\s+LEVANTA\s+LA\s+SESION\b"
    r"|\bSE\s+CIERRA\s+LA\s+SESION\b"
    r"|\bSE\s+DA\s+POR\s+LEVANTADA\s+LA\s+SESION\b"
)

INTERMISSION_PATTERN = re.compile(r"\bSE\s+PASA\s+A\s+CUARTO\s+INTERMEDIO\b")

RESUMPTION_PATTERN = re.compile(
    r"\bCONTINUA\s+LA\s+SESION\b"
    r"|\bSE\s+REANUDA\s+LA\s+SESION\b"
)

TIME_PATTERN = re.compile(r"(?m)^\s*[-–]?\s*ES\s+LA\s+HORA\b")

DIRECTOR_PATTERN = re.compile(r"\bDIRECTOR\s+DEL\s+CUERPO\s+DE\s+TAQUIGRAFOS\b")

VOTE_RECORD_PATTERN = re.compile(
    r"^\s*[-–]?\s*"
    r"(?:SE\s+PRACTICA|FINALIZADA)\s+"
    r"LA\s+VOTACION\s+NOMINAL\b"
    r"|^\s*[-–]?\s*VOTAN\s+POR\s+LA\s+"
    r"(?:AFIRMATIVA|NEGATIVA)\b"
    r"|\bCONFORME\s+AL\s+TABLERO\s+ELECTRONICO\b",
    flags=re.MULTILINE,
)

ATTENDANCE_PATTERN = re.compile(
    r"(?m)^\s*(?:PRESENTES|AUSENTES)"
    r"(?:,\s+CON\s+(?:AVISO|LICENCIA))?\s*:"
)


def fold_text(value: str) -> str:
    """Return uppercase accent-insensitive text while retaining line breaks."""
    decomposed = unicodedata.normalize("NFKD", value)
    without_marks = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )

    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in without_marks.splitlines()]

    return "\n".join(lines).upper()


def _folded_lines(value: str) -> list[str]:
    return [line.strip() for line in fold_text(value).splitlines() if line.strip()]


def _contains_exact_line(
    *,
    text: str,
    pattern: re.Pattern[str],
) -> bool:
    return any(pattern.fullmatch(line) is not None for line in _folded_lines(text))


APPENDIX_LINE_PATTERN = re.compile(r"(?:\d+\s+)?APENDICE")

SANCTIONS_LINE_PATTERN = re.compile(
    r"(?:(?:[IVXLCDM]+|[A-Z])\.\s+)?"
    r"SANCIONES\s+DE\s+LA\s+HONORABLE\s+CAMARA"
)

VOTE_SECTION_LINE_PATTERN = re.compile(
    r"(?:(?:[IVXLCDM]+|[A-Z])\.\s+)?"
    r"ACTAS?\s+DE\s+VOTACIONES?\s+NOMINALES?"
    r"(?:\s+NUMEROS?.*)?"
)

INSERTION_LINE_PATTERN = re.compile(
    r"(?:(?:[IVXLCDM]+|[A-Z])\.\s+)?"
    r"INSERCIONES"
    r"(?:\s+SOLICITADAS|\s+DE\s+LOS.*)?"
)

SUMMARY_LINE_PATTERN = re.compile(r"SUMARIO")


def _is_appendix_heading(text: str) -> bool:
    return _contains_exact_line(
        text=text,
        pattern=APPENDIX_LINE_PATTERN,
    )


def _is_sanctions_heading(text: str) -> bool:
    return _contains_exact_line(
        text=text,
        pattern=SANCTIONS_LINE_PATTERN,
    )


def _is_vote_section_heading(text: str) -> bool:
    return _contains_exact_line(
        text=text,
        pattern=VOTE_SECTION_LINE_PATTERN,
    )


def _is_insertion_heading(text: str) -> bool:
    return _contains_exact_line(
        text=text,
        pattern=INSERTION_LINE_PATTERN,
    )


def _is_summary_heading(text: str) -> bool:
    return _contains_exact_line(
        text=text,
        pattern=SUMMARY_LINE_PATTERN,
    )


def _is_running_header(
    *,
    block: StructuralInputBlock,
    page_height: float,
) -> bool:
    folded = re.sub(
        r"\s+",
        " ",
        fold_text(block.text),
    ).strip()

    if not folded or len(folded) > 150:
        return False

    is_page_number = folded.isdigit()
    contains_chamber_name = "CAMARA DE DIPUTADOS DE LA NACION" in folded
    contains_meeting_label = "REUNION " in folded

    if block.region == "header":
        return is_page_number or contains_chamber_name or contains_meeting_label

    is_near_top = page_height > 0 and block.y1 <= page_height * 0.18

    return is_near_top and contains_chamber_name


def _is_running_footer(
    *,
    block: StructuralInputBlock,
) -> bool:
    folded = re.sub(
        r"\s+",
        " ",
        fold_text(block.text),
    ).strip()

    return block.region == "footer" and folded.isdigit()


def _has_non_chair_speaker(text: str) -> bool:
    return NON_CHAIR_SPEAKER_PATTERN.search(fold_text(text)) is not None


def _is_pure_procedural(
    text: str,
) -> bool:
    folded = fold_text(text)

    if _has_non_chair_speaker(text):
        return False

    return bool(
        INTERMISSION_PATTERN.search(folded)
        or TIME_PATTERN.search(folded)
        or DIRECTOR_PATTERN.search(folded)
        or OPENING_PATTERN.search(folded)
        or CLOSING_PATTERN.search(folded)
    )


def _is_vote_record(
    text: str,
) -> bool:
    if _has_non_chair_speaker(text):
        return False

    return VOTE_RECORD_PATTERN.search(fold_text(text)) is not None


def _find_first_index(
    *,
    blocks: Sequence[StructuralInputBlock],
    pattern: re.Pattern[str],
    minimum_index: int = 0,
) -> int | None:
    for index in range(
        minimum_index,
        len(blocks),
    ):
        if pattern.search(fold_text(blocks[index].text)):
            return index

    return None


def _find_first_appendix_index(
    *,
    blocks: Sequence[StructuralInputBlock],
    minimum_index: int,
) -> int | None:
    for index in range(
        minimum_index,
        len(blocks),
    ):
        if _is_appendix_heading(blocks[index].text):
            return index

    return None


def _anchor(
    *,
    block: StructuralInputBlock,
    method: str,
    matched_text: str,
) -> StructuralAnchor:
    return StructuralAnchor(
        page_number=block.page_number,
        reading_order=block.reading_order,
        method=method,
        matched_text=matched_text,
    )


def _procedural_tail_start(
    *,
    blocks: Sequence[StructuralInputBlock],
    start_index: int,
    appendix_index: int | None,
    closing_index: int | None,
) -> int | None:
    if appendix_index is None or closing_index is not None:
        return None

    intermission_indices = [
        index
        for index in range(
            start_index,
            appendix_index,
        )
        if INTERMISSION_PATTERN.search(fold_text(blocks[index].text))
    ]

    if not intermission_indices:
        return None

    candidate = intermission_indices[-1]

    resumed_after_candidate = any(
        RESUMPTION_PATTERN.search(fold_text(blocks[index].text)) is not None
        for index in range(
            candidate + 1,
            appendix_index,
        )
    )

    if resumed_after_candidate:
        return None

    return candidate


def _exclusion_reason(
    *,
    zone: StructuralZone,
    role: ContentRole,
    included: bool,
) -> str | None:
    if included:
        return None

    if role in {
        ContentRole.RUNNING_HEADER,
        ContentRole.RUNNING_FOOTER,
        ContentRole.PROCEDURAL,
        ContentRole.VOTE_RECORD,
    }:
        return role.value

    return zone.value


def classify_structural_blocks(
    *,
    blocks: Sequence[StructuralInputBlock],
    page_heights: Mapping[int, float],
) -> StructuralSegmentationResult:
    """Classify blocks using deterministic document-level boundaries."""
    if not blocks:
        raise StructuralSegmentationError("Cannot segment an empty block sequence.")

    ordered_blocks = tuple(
        sorted(
            blocks,
            key=lambda block: (
                block.page_number,
                block.reading_order,
            ),
        )
    )

    references = [
        (
            block.page_number,
            block.reading_order,
        )
        for block in ordered_blocks
    ]

    if len(references) != len(set(references)):
        raise StructuralSegmentationError("Duplicate page and reading-order references.")

    missing_page_heights = sorted(
        {block.page_number for block in ordered_blocks if block.page_number not in page_heights}
    )

    if missing_page_heights:
        raise StructuralSegmentationError(f"Missing page heights for pages: {missing_page_heights}")

    start_index = _find_first_index(
        blocks=ordered_blocks,
        pattern=CHAIR_PATTERN,
    )

    if start_index is None:
        raise StructuralSegmentationError("No chair intervention was found.")

    closing_index = _find_first_index(
        blocks=ordered_blocks,
        pattern=CLOSING_PATTERN,
        minimum_index=start_index,
    )
    appendix_index = _find_first_appendix_index(
        blocks=ordered_blocks,
        minimum_index=start_index,
    )

    if closing_index is not None:
        post_start_index = closing_index + 1
        end_method = "explicit_closing"
        end_anchor = _anchor(
            block=ordered_blocks[closing_index],
            method=end_method,
            matched_text="closing_formula",
        )
    elif appendix_index is not None:
        post_start_index = appendix_index
        end_method = "appendix_fallback"
        end_anchor = _anchor(
            block=ordered_blocks[appendix_index],
            method=end_method,
            matched_text="appendix_heading",
        )
    else:
        post_start_index = len(ordered_blocks)
        end_method = "document_end"
        end_anchor = None

    post_start_anchor = (
        _anchor(
            block=ordered_blocks[post_start_index],
            method=end_method,
            matched_text="post_proceedings_start",
        )
        if post_start_index < len(ordered_blocks)
        else None
    )

    start_anchor = _anchor(
        block=ordered_blocks[start_index],
        method="first_chair_intervention",
        matched_text="chair_intervention",
    )

    procedural_tail_index = _procedural_tail_start(
        blocks=ordered_blocks,
        start_index=start_index,
        appendix_index=appendix_index,
        closing_index=closing_index,
    )

    end_confidence = {
        "explicit_closing": 1.0,
        "appendix_fallback": 0.95,
        "document_end": 0.8,
    }[end_method]

    post_role = ContentRole.OTHER
    classifications = []

    for index, block in enumerate(ordered_blocks):
        if index < start_index:
            zone = StructuralZone.FRONT_MATTER
            zone_method = "before_first_chair"
        elif index < post_start_index:
            zone = StructuralZone.PROCEEDINGS
            zone_method = f"proceedings_{end_method}"
        else:
            zone = StructuralZone.POST_PROCEEDINGS
            zone_method = f"post_{end_method}"

        page_height = float(page_heights[block.page_number])

        if _is_running_header(
            block=block,
            page_height=page_height,
        ):
            role = ContentRole.RUNNING_HEADER
            role_method = "geometric_running_header"
            role_confidence = 0.99

        elif _is_running_footer(block=block):
            role = ContentRole.RUNNING_FOOTER
            role_method = "geometric_running_footer"
            role_confidence = 0.99

        elif zone == StructuralZone.FRONT_MATTER:
            if block.page_number == 1:
                role = ContentRole.COVER
                role_method = "first_page"
                role_confidence = 0.9
            elif _is_summary_heading(block.text):
                role = ContentRole.SUMMARY
                role_method = "summary_heading"
                role_confidence = 1.0
            elif ATTENDANCE_PATTERN.search(fold_text(block.text)):
                role = ContentRole.ATTENDANCE
                role_method = "attendance_heading"
                role_confidence = 0.95
            else:
                role = ContentRole.OTHER
                role_method = "front_matter_default"
                role_confidence = 0.8

        elif zone == StructuralZone.PROCEEDINGS:
            in_procedural_tail = (
                procedural_tail_index is not None and index >= procedural_tail_index
            )

            if in_procedural_tail or _is_pure_procedural(block.text):
                role = ContentRole.PROCEDURAL
                role_method = "procedural_marker"
                role_confidence = 0.98
            elif _is_vote_record(block.text):
                role = ContentRole.VOTE_RECORD
                role_method = "vote_record_marker"
                role_confidence = 0.95
            else:
                role = ContentRole.TRANSCRIPT
                role_method = "proceedings_default"
                role_confidence = 0.9

        else:
            if _is_appendix_heading(block.text):
                role = ContentRole.PROCEDURAL
                role_method = "appendix_heading"
                role_confidence = 1.0
            elif _is_sanctions_heading(block.text):
                post_role = ContentRole.SANCTION_TEXT
                role = post_role
                role_method = "sanctions_section_heading"
                role_confidence = 1.0
            elif _is_vote_section_heading(block.text):
                post_role = ContentRole.VOTE_RECORD
                role = post_role
                role_method = "vote_section_heading"
                role_confidence = 1.0
            elif _is_insertion_heading(block.text):
                post_role = ContentRole.INSERTION
                role = post_role
                role_method = "insertion_section_heading"
                role_confidence = 1.0
            else:
                role = post_role
                role_method = "post_section_state"
                role_confidence = 0.95 if post_role != ContentRole.OTHER else 0.75

        included = zone == StructuralZone.PROCEEDINGS and role == ContentRole.TRANSCRIPT

        classifications.append(
            BlockClassification(
                page_number=block.page_number,
                reading_order=(block.reading_order),
                structural_zone=zone,
                content_role=role,
                include_in_discourse=(included),
                exclusion_reason=(
                    _exclusion_reason(
                        zone=zone,
                        role=role,
                        included=included,
                    )
                ),
                classification_method=(f"{zone_method}+{role_method}"),
                classification_confidence=round(
                    min(
                        end_confidence,
                        role_confidence,
                    ),
                    3,
                ),
            )
        )

    return StructuralSegmentationResult(
        start_anchor=start_anchor,
        end_anchor=end_anchor,
        post_start_anchor=post_start_anchor,
        end_method=end_method,
        classifications=tuple(classifications),
    )
