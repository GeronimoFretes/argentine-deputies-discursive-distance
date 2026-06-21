from datetime import date

from argentine_deputies_discursive_distance.identifiers import (
    build_source_record_id,
    fold_for_matching,
    normalize_display_text,
)


def test_normalize_display_text_preserves_accents() -> None:
    value = "  Sesión   Ordinaria\nEspecial  "

    assert normalize_display_text(value) == "Sesión Ordinaria Especial"


def test_fold_for_matching_removes_accents_and_punctuation() -> None:
    value = "Sesión Ordinaria — Prórroga"

    assert fold_for_matching(value) == "SESION ORDINARIA PRORROGA"


def test_meeting_identifier_is_deterministic() -> None:
    identifier = build_source_record_id(
        period=144,
        session_date=date(2026, 5, 20),
        meeting_number=3,
        title="3° Reunión - 3° Sesión Ordinaria Especial",
    )

    repeated_identifier = build_source_record_id(
        period=144,
        session_date=date(2026, 5, 20),
        meeting_number=3,
        title="Changed display title",
    )

    assert identifier == repeated_identifier
    assert len(identifier) == 20


def test_non_meeting_identifier_uses_normalized_title() -> None:
    first = build_source_record_id(
        period=144,
        session_date=date(2026, 3, 1),
        meeting_number=None,
        title="Asamblea Legislativa",
    )
    second = build_source_record_id(
        period=144,
        session_date=date(2026, 3, 1),
        meeting_number=None,
        title="  ASAMBLEA   LEGISLATIVA ",
    )

    assert first == second
