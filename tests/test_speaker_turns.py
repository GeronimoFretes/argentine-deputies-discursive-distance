import pytest

from argentine_deputies_discursive_distance.speaker_turns import (
    SegmentAttributionMethod,
    SpeakerTurnError,
    SpeakerTurnInputBlock,
    parse_speaker_turns,
)


def make_block(
    *,
    page: int,
    order: int,
    text: str,
    role: str = "transcript",
    included: bool = True,
) -> SpeakerTurnInputBlock:
    return SpeakerTurnInputBlock(
        page_number=page,
        reading_order=order,
        structural_zone="proceedings",
        content_role=role,
        include_in_discourse=included,
        text=text,
    )


def test_splits_multiple_markers_inside_one_block() -> None:
    text = "Sr. Alpha. – Primera intervención.\nSr. Beta.- Segunda intervención."

    result = parse_speaker_turns(
        [
            make_block(
                page=1,
                order=1,
                text=text,
            )
        ]
    )

    assert result.explicit_marker_count == 2
    assert len(result.turns) == 2

    first, second = result.turns

    assert first.normalized_label == "ALPHA"
    assert second.normalized_label == "BETA"
    assert first.text.strip() == ("Primera intervención.")
    assert second.text.strip() == ("Segunda intervención.")
    assert first.segments[0].attribution_method == SegmentAttributionMethod.EXPLICIT_MARKER


def test_prefix_before_embedded_marker_continues_previous_turn() -> None:
    blocks = [
        make_block(
            page=1,
            order=1,
            text=("Sr. Alpha. – Primera parte."),
        ),
        make_block(
            page=1,
            order=2,
            text=("Continuación de Alpha.\nSr. Beta. – Intervención de Beta."),
        ),
    ]

    result = parse_speaker_turns(blocks)

    assert len(result.turns) == 2
    alpha, beta = result.turns

    assert alpha.normalized_label == ("ALPHA")
    assert len(alpha.segments) == 2
    assert alpha.segments[1].text.strip() == "Continuación de Alpha."
    assert alpha.segments[1].attribution_method == SegmentAttributionMethod.CARRIED_FORWARD
    assert beta.normalized_label == ("BETA")


def test_markerless_block_continues_active_turn() -> None:
    result = parse_speaker_turns(
        [
            make_block(
                page=1,
                order=1,
                text=("Sr. Alpha. – Primera parte."),
            ),
            make_block(
                page=1,
                order=2,
                text="Segunda parte.",
            ),
        ]
    )

    assert len(result.turns) == 1
    turn = result.turns[0]

    assert len(turn.segments) == 2
    assert turn.segments[1].attribution_method == SegmentAttributionMethod.CARRIED_FORWARD


def test_running_header_does_not_break_continuity() -> None:
    result = parse_speaker_turns(
        [
            make_block(
                page=1,
                order=1,
                text=("Sr. Alpha. – Primera parte."),
            ),
            make_block(
                page=2,
                order=1,
                text=("CÁMARA DE DIPUTADOS DE LA NACIÓN"),
                role="running_header",
                included=False,
            ),
            make_block(
                page=2,
                order=2,
                text="Segunda parte.",
            ),
        ]
    )

    assert len(result.turns) == 1
    assert len(result.turns[0].segments) == 2


def test_procedural_barrier_without_marker_resets_speaker() -> None:
    result = parse_speaker_turns(
        [
            make_block(
                page=1,
                order=1,
                text=("Sr. Alpha. – Primera parte."),
            ),
            make_block(
                page=1,
                order=2,
                text=("–Se pasa a cuarto intermedio."),
                role="procedural",
                included=False,
            ),
            make_block(
                page=1,
                order=3,
                text=("Texto sin nuevo marcador."),
            ),
        ]
    )

    assert result.barrier_reset_count == 1
    assert len(result.turns) == 2

    unattributed = result.turns[1]

    assert unattributed.is_unattributed
    assert unattributed.text.strip() == "Texto sin nuevo marcador."


def test_marker_in_procedural_block_seeds_next_discourse_block() -> None:
    result = parse_speaker_turns(
        [
            make_block(
                page=1,
                order=1,
                text=("Sr. Presidente. – Queda abierta la sesión."),
                role="procedural",
                included=False,
            ),
            make_block(
                page=1,
                order=2,
                text=("Continúa la Presidencia."),
            ),
        ]
    )

    assert len(result.turns) == 1
    turn = result.turns[0]

    assert turn.normalized_label == ("PRESIDENTE")
    assert turn.marker_block_included is False
    assert turn.text.strip() == "Continúa la Presidencia."
    assert turn.segments[0].attribution_method == SegmentAttributionMethod.CARRIED_FORWARD


def test_unattributed_initial_text_is_preserved() -> None:
    result = parse_speaker_turns(
        [
            make_block(
                page=1,
                order=1,
                text=("Texto previo sin marcador.\nSr. Alpha. – Intervención."),
            )
        ]
    )

    assert len(result.turns) == 2
    assert result.turns[0].is_unattributed
    assert result.turns[0].text.strip() == "Texto previo sin marcador."
    assert result.unattributed_turn_count == 1


def test_repeated_same_label_starts_new_turn() -> None:
    result = parse_speaker_turns(
        [
            make_block(
                page=1,
                order=1,
                text=("Sr. Alpha. – Uno.\nSr. Alpha. – Dos."),
            )
        ]
    )

    assert len(result.turns) == 2
    assert [turn.normalized_label for turn in result.turns] == [
        "ALPHA",
        "ALPHA",
    ]


def test_rejects_non_proceedings_blocks() -> None:
    block = SpeakerTurnInputBlock(
        page_number=1,
        reading_order=1,
        structural_zone=("front_matter"),
        content_role="other",
        include_in_discourse=False,
        text="SUMARIO",
    )

    with pytest.raises(
        SpeakerTurnError,
        match="non-proceedings",
    ):
        parse_speaker_turns([block])
