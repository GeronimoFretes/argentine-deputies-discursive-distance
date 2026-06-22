"""Construction of speaker turns from ordered parliamentary blocks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from .speaker import (
    SpeakerLabelFamily,
    SpeakerMarker,
    find_speaker_markers,
)


class SpeakerTurnError(RuntimeError):
    """Raised when speaker turns cannot be constructed safely."""


class SegmentAttributionMethod(StrEnum):
    """How a discourse segment was assigned to a turn."""

    EXPLICIT_MARKER = "explicit_marker"
    CARRIED_FORWARD = "carried_forward"
    UNATTRIBUTED = "unattributed"


@dataclass(frozen=True, slots=True)
class SpeakerTurnInputBlock:
    """Minimal structural block required for speaker-turn parsing."""

    page_number: int
    reading_order: int
    structural_zone: str
    content_role: str
    include_in_discourse: bool
    text: str

    @property
    def reference(self) -> str:
        """Return a stable page/block reference."""
        return f"p{self.page_number}:b{self.reading_order}"


@dataclass(frozen=True, slots=True)
class SpeakerTurnSegment:
    """One exact text span assigned to a speaker turn."""

    page_number: int
    reading_order: int
    start: int
    end: int
    text: str
    attribution_method: SegmentAttributionMethod

    @property
    def block_reference(self) -> str:
        """Return the source page/block reference."""
        return f"p{self.page_number}:b{self.reading_order}"


@dataclass(frozen=True, slots=True)
class SpeakerTurn:
    """One explicit or unattributed parliamentary speaker turn."""

    turn_index: int
    marker: SpeakerMarker | None
    marker_page_number: int | None
    marker_reading_order: int | None
    marker_block_included: bool | None
    segments: tuple[
        SpeakerTurnSegment,
        ...,
    ]

    @property
    def marker_block_reference(
        self,
    ) -> str | None:
        """Return the block containing the explicit marker."""
        if self.marker_page_number is None or self.marker_reading_order is None:
            return None

        return f"p{self.marker_page_number}:b{self.marker_reading_order}"

    @property
    def normalized_label(
        self,
    ) -> str | None:
        """Return the normalized printed speaker label."""
        if self.marker is None:
            return None

        return self.marker.normalized_label

    @property
    def speaker_family(
        self,
    ) -> SpeakerLabelFamily | None:
        """Return the marker-derived speaker family."""
        if self.marker is None:
            return None

        return self.marker.family

    @property
    def is_unattributed(self) -> bool:
        """Return whether the turn has no explicit speaker context."""
        return self.marker is None

    @property
    def text(self) -> str:
        """Return turn text joined across source segments."""
        return "\n".join(segment.text for segment in self.segments)

    @property
    def word_count(self) -> int:
        """Return a simple whitespace-token count."""
        return len(self.text.split())

    @property
    def character_count(self) -> int:
        """Return source characters assigned to the turn."""
        return sum(len(segment.text) for segment in self.segments)


@dataclass(frozen=True, slots=True)
class SpeakerTurnParseResult:
    """Complete result of deterministic speaker-turn construction."""

    turns: tuple[SpeakerTurn, ...]
    explicit_marker_count: int
    unattributed_turn_count: int
    unattributed_segment_count: int
    barrier_reset_count: int
    assigned_segment_count: int
    assigned_character_count: int


@dataclass(slots=True)
class _TurnBuilder:
    """Mutable internal representation while constructing turns."""

    turn_index: int
    marker: SpeakerMarker | None
    marker_page_number: int | None
    marker_reading_order: int | None
    marker_block_included: bool | None
    segments: list[SpeakerTurnSegment] = field(default_factory=list)


IGNORED_CONTENT_ROLES = {
    "running_header",
    "running_footer",
}

BARRIER_CONTENT_ROLES = {
    "procedural",
    "vote_record",
}


def _validate_blocks(
    blocks: Sequence[SpeakerTurnInputBlock],
) -> tuple[
    SpeakerTurnInputBlock,
    ...,
]:
    """Order blocks and reject invalid or duplicate references."""
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
        raise SpeakerTurnError("Duplicate page and reading-order references.")

    for block in ordered_blocks:
        if block.structural_zone != "proceedings":
            raise SpeakerTurnError(
                f"Speaker-turn parsing received a non-proceedings block: {block.reference}"
            )

        if block.page_number < 1:
            raise SpeakerTurnError(f"Invalid page number in {block.reference}.")

        if block.reading_order < 1:
            raise SpeakerTurnError(f"Invalid reading order in {block.reference}.")

    return ordered_blocks


def _validate_marker_ranges(
    *,
    block: SpeakerTurnInputBlock,
    markers: tuple[
        SpeakerMarker,
        ...,
    ],
) -> None:
    """Ensure marker offsets are ordered and valid for their block."""
    previous_end = 0

    for marker in markers:
        if not (0 <= marker.start < marker.end <= len(block.text)):
            raise SpeakerTurnError(
                f"Invalid marker range in {block.reference}: {marker.start}:{marker.end}"
            )

        if marker.start < previous_end:
            raise SpeakerTurnError(f"Overlapping speaker markers in {block.reference}.")

        if block.text[marker.start : marker.end] != marker.raw_marker:
            raise SpeakerTurnError(
                f"Marker text does not match its source range in {block.reference}."
            )

        previous_end = marker.end


def parse_speaker_turns(
    blocks: Sequence[SpeakerTurnInputBlock],
) -> SpeakerTurnParseResult:
    """Build speaker turns from ordered proceedings blocks."""
    ordered_blocks = _validate_blocks(blocks)

    builders: list[_TurnBuilder] = []
    active_turn: _TurnBuilder | None = None
    barrier_reset_count = 0

    def create_explicit_turn(
        *,
        marker: SpeakerMarker,
        block: SpeakerTurnInputBlock,
    ) -> _TurnBuilder:
        builder = _TurnBuilder(
            turn_index=len(builders) + 1,
            marker=marker,
            marker_page_number=(block.page_number),
            marker_reading_order=(block.reading_order),
            marker_block_included=(block.include_in_discourse),
        )
        builders.append(builder)
        return builder

    def create_unattributed_turn() -> _TurnBuilder:
        builder = _TurnBuilder(
            turn_index=len(builders) + 1,
            marker=None,
            marker_page_number=None,
            marker_reading_order=None,
            marker_block_included=None,
        )
        builders.append(builder)
        return builder

    def append_span(
        *,
        block: SpeakerTurnInputBlock,
        start: int,
        end: int,
        method: (SegmentAttributionMethod),
    ) -> None:
        nonlocal active_turn

        if not (0 <= start <= end <= len(block.text)):
            raise SpeakerTurnError(f"Invalid discourse span in {block.reference}: {start}:{end}")

        span_text = block.text[start:end]

        if not span_text.strip():
            return

        if active_turn is None:
            active_turn = create_unattributed_turn()

        actual_method = (
            SegmentAttributionMethod.UNATTRIBUTED if active_turn.marker is None else method
        )

        active_turn.segments.append(
            SpeakerTurnSegment(
                page_number=(block.page_number),
                reading_order=(block.reading_order),
                start=start,
                end=end,
                text=span_text,
                attribution_method=(actual_method),
            )
        )

    for block in ordered_blocks:
        markers = find_speaker_markers(block.text)
        _validate_marker_ranges(
            block=block,
            markers=markers,
        )

        if block.content_role in IGNORED_CONTENT_ROLES:
            continue

        if not block.include_in_discourse:
            if markers:
                for marker in markers:
                    active_turn = create_explicit_turn(
                        marker=marker,
                        block=block,
                    )
            elif (
                block.content_role in BARRIER_CONTENT_ROLES
                or block.content_role not in IGNORED_CONTENT_ROLES
            ):
                if active_turn is not None:
                    barrier_reset_count += 1

                active_turn = None

            continue

        cursor = 0
        span_method = SegmentAttributionMethod.CARRIED_FORWARD

        for marker in markers:
            append_span(
                block=block,
                start=cursor,
                end=marker.start,
                method=span_method,
            )

            active_turn = create_explicit_turn(
                marker=marker,
                block=block,
            )
            cursor = marker.end
            span_method = SegmentAttributionMethod.EXPLICIT_MARKER

        append_span(
            block=block,
            start=cursor,
            end=len(block.text),
            method=span_method,
        )

    turns = tuple(
        SpeakerTurn(
            turn_index=builder.turn_index,
            marker=builder.marker,
            marker_page_number=(builder.marker_page_number),
            marker_reading_order=(builder.marker_reading_order),
            marker_block_included=(builder.marker_block_included),
            segments=tuple(builder.segments),
        )
        for builder in builders
    )

    unattributed_turns = [turn for turn in turns if turn.is_unattributed]
    all_segments = [segment for turn in turns for segment in turn.segments]

    return SpeakerTurnParseResult(
        turns=turns,
        explicit_marker_count=sum(turn.marker is not None for turn in turns),
        unattributed_turn_count=len(unattributed_turns),
        unattributed_segment_count=sum(len(turn.segments) for turn in unattributed_turns),
        barrier_reset_count=(barrier_reset_count),
        assigned_segment_count=len(all_segments),
        assigned_character_count=sum(len(segment.text) for segment in all_segments),
    )
