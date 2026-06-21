import pytest

from argentine_deputies_discursive_distance.structure import (
    BlockClassification,
    ContentRole,
    StructuralInputBlock,
    StructuralSegmentationError,
    StructuralSegmentationResult,
    StructuralZone,
    classify_structural_blocks,
)


def make_block(
    *,
    page: int,
    order: int,
    text: str,
    region: str = "body_full",
    y0: float = 100,
    y1: float = 200,
) -> StructuralInputBlock:
    return StructuralInputBlock(
        page_number=page,
        reading_order=order,
        region=region,
        y0=y0,
        y1=y1,
        text=text,
    )


def classification_by_reference(
    result: StructuralSegmentationResult,
) -> dict[str, BlockClassification]:
    return {classification.reference: classification for classification in result.classifications}


def test_explicit_closing_starts_post_proceedings_after_block() -> None:
    blocks = [
        make_block(
            page=1,
            order=1,
            text="SUMARIO",
        ),
        make_block(
            page=2,
            order=1,
            text=("Sr. Presidente (Monzó). – Continúa la sesión."),
        ),
        make_block(
            page=2,
            order=2,
            text=("Sr. Diputado. – Intervención."),
        ),
        make_block(
            page=2,
            order=3,
            text=("Habiéndose cumplido el objeto, queda levantada la sesión."),
        ),
        make_block(
            page=2,
            order=4,
            text="–Es la hora 17 y 28.",
        ),
        make_block(
            page=2,
            order=5,
            text="APÉNDICE",
        ),
    ]

    result = classify_structural_blocks(
        blocks=blocks,
        page_heights={
            1: 700,
            2: 700,
        },
    )
    records = classification_by_reference(result)

    assert result.end_method == ("explicit_closing")
    assert result.end_anchor is not None
    assert result.end_anchor.reference == ("p2:b3")
    assert result.post_start_anchor is not None
    assert result.post_start_anchor.reference == "p2:b4"
    assert records["p2:b3"].structural_zone == StructuralZone.PROCEEDINGS
    assert records["p2:b4"].structural_zone == StructuralZone.POST_PROCEEDINGS


def test_appendix_fallback_marks_final_intermission_tail_procedural() -> None:
    blocks = [
        make_block(
            page=5,
            order=1,
            text=("Sr. Presidente (Fellner). – Queda abierta la sesión."),
        ),
        make_block(
            page=325,
            order=10,
            text=("La Presidencia invita a pasar a cuarto intermedio."),
        ),
        make_block(
            page=325,
            order=11,
            text=("–Se pasa a cuarto intermedio a la hora 2 y 21."),
        ),
        make_block(
            page=325,
            order=12,
            text="HORACIO M. GONZÁLEZ.",
        ),
        make_block(
            page=325,
            order=13,
            text=("Director del Cuerpo de Taquígrafos."),
        ),
        make_block(
            page=325,
            order=14,
            text="APÉNDICE",
        ),
    ]

    result = classify_structural_blocks(
        blocks=blocks,
        page_heights={
            5: 700,
            325: 700,
        },
    )
    records = classification_by_reference(result)

    assert result.end_method == ("appendix_fallback")
    assert result.end_anchor is not None
    assert result.end_anchor.reference == ("p325:b14")
    assert records["p325:b11"].content_role == ContentRole.PROCEDURAL
    assert records["p325:b12"].content_role == ContentRole.PROCEDURAL
    assert records["p325:b13"].content_role == ContentRole.PROCEDURAL
    assert records["p325:b14"].structural_zone == StructuralZone.POST_PROCEEDINGS


def test_closing_block_with_non_chair_speech_remains_transcript() -> None:
    blocks = [
        make_block(
            page=2,
            order=1,
            text=("Sr. Presidente (Menem). – Queda abierta la sesión."),
        ),
        make_block(
            page=256,
            order=4,
            text=(
                "Sr. Marino. – Esta es mi "
                "intervención final.\n"
                "Sr. Presidente (Menem). – "
                "Queda levantada la sesión."
            ),
        ),
        make_block(
            page=256,
            order=5,
            text="–Es la hora 22 y 28.",
        ),
    ]

    result = classify_structural_blocks(
        blocks=blocks,
        page_heights={
            2: 800,
            256: 800,
        },
    )
    records = classification_by_reference(result)

    assert records["p256:b4"].content_role == ContentRole.TRANSCRIPT
    assert records["p256:b4"].include_in_discourse is True


def test_old_style_top_chamber_line_is_running_header() -> None:
    blocks = [
        make_block(
            page=5,
            order=1,
            text=("Junio 30 de 2010\nCÁMARA DE DIPUTADOS DE LA NACIÓN\n5"),
            y0=112,
            y1=125,
        ),
        make_block(
            page=5,
            order=2,
            text=("Sr. Presidente (Fellner). – Queda abierta la sesión."),
            y0=200,
            y1=300,
        ),
    ]

    result = classify_structural_blocks(
        blocks=blocks,
        page_heights={5: 842},
    )
    records = classification_by_reference(result)

    assert records["p5:b1"].content_role == ContentRole.RUNNING_HEADER
    assert records["p5:b1"].include_in_discourse is False


def test_chamber_name_inside_body_is_not_running_header() -> None:
    blocks = [
        make_block(
            page=2,
            order=1,
            text=("Sr. Presidente (Menem). – Queda abierta la sesión."),
            y0=200,
            y1=350,
        ),
        make_block(
            page=2,
            order=2,
            text=("La Honorable Cámara de Diputados de la Nación considerará el expediente."),
            y0=400,
            y1=650,
        ),
    ]

    result = classify_structural_blocks(
        blocks=blocks,
        page_heights={2: 842},
    )
    records = classification_by_reference(result)

    assert records["p2:b2"].content_role == ContentRole.TRANSCRIPT


def test_post_proceedings_section_roles_persist() -> None:
    blocks = [
        make_block(
            page=1,
            order=1,
            text=("Sr. Presidente. – Queda abierta la sesión."),
        ),
        make_block(
            page=2,
            order=1,
            text="Queda levantada la sesión.",
        ),
        make_block(
            page=3,
            order=1,
            text="APÉNDICE",
        ),
        make_block(
            page=3,
            order=2,
            text=("I. SANCIONES DE LA HONORABLE CÁMARA"),
        ),
        make_block(
            page=3,
            order=3,
            text="Artículo 1° – Apruébase...",
        ),
        make_block(
            page=4,
            order=1,
            text=("II. ACTAS DE VOTACIONES NOMINALES"),
        ),
        make_block(
            page=4,
            order=2,
            text="Votos afirmativos...",
        ),
        make_block(
            page=5,
            order=1,
            text="III. INSERCIONES",
        ),
        make_block(
            page=5,
            order=2,
            text="Texto insertado...",
        ),
    ]

    result = classify_structural_blocks(
        blocks=blocks,
        page_heights={
            1: 700,
            2: 700,
            3: 700,
            4: 700,
            5: 700,
        },
    )
    records = classification_by_reference(result)

    assert records["p3:b3"].content_role == ContentRole.SANCTION_TEXT
    assert records["p4:b2"].content_role == ContentRole.VOTE_RECORD
    assert records["p5:b2"].content_role == ContentRole.INSERTION


def test_missing_chair_anchor_is_rejected() -> None:
    blocks = [
        make_block(
            page=1,
            order=1,
            text="Document without proceedings.",
        )
    ]

    with pytest.raises(
        StructuralSegmentationError,
        match="No chair intervention",
    ):
        classify_structural_blocks(
            blocks=blocks,
            page_heights={1: 700},
        )


def test_pure_opening_formula_is_procedural() -> None:
    blocks = [
        make_block(
            page=1,
            order=1,
            text="SUMARIO",
        ),
        make_block(
            page=2,
            order=1,
            text=("Sr. Presidente. – Queda abierta la sesión."),
        ),
        make_block(
            page=2,
            order=2,
            text=("Sr. Diputado. – Intervención sustantiva."),
        ),
    ]

    result = classify_structural_blocks(
        blocks=blocks,
        page_heights={
            1: 700,
            2: 700,
        },
    )
    records = classification_by_reference(result)

    assert records["p2:b1"].content_role == ContentRole.PROCEDURAL
    assert records["p2:b1"].include_in_discourse is False
    assert records["p2:b2"].content_role == ContentRole.TRANSCRIPT
    assert records["p2:b2"].include_in_discourse is True


def test_non_chair_speech_prevents_structural_role_exclusion() -> None:
    blocks = [
        make_block(
            page=1,
            order=1,
            text=("Sr. Presidente. – Queda abierta la sesión."),
        ),
        make_block(
            page=1,
            order=2,
            text=(
                "Sr. Diputado. – "
                "Esta es una intervención "
                "sustantiva.\n"
                "–Se pasa a cuarto intermedio."
            ),
        ),
        make_block(
            page=1,
            order=3,
            text=("Sr. Diputada. – Explicaré mi voto.\nFinalizada la votación nominal."),
        ),
    ]

    result = classify_structural_blocks(
        blocks=blocks,
        page_heights={1: 700},
    )
    records = classification_by_reference(result)

    assert records["p1:b2"].content_role == ContentRole.TRANSCRIPT
    assert records["p1:b2"].include_in_discourse is True
    assert records["p1:b3"].content_role == ContentRole.TRANSCRIPT
    assert records["p1:b3"].include_in_discourse is True
