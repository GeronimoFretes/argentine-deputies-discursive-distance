from argentine_deputies_discursive_distance.speaker_turns import (
    SpeakerTurnInputBlock,
    parse_speaker_turns,
)
from argentine_deputies_discursive_distance.turn_content import (
    DocumentaryCue,
    TurnContentKind,
    classify_speaker_turn_content,
    find_documentary_boundary,
)


def make_turn(
    *texts: str,
    role: str = "transcript",
):
    blocks = [
        SpeakerTurnInputBlock(
            page_number=1,
            reading_order=index,
            structural_zone="proceedings",
            content_role=role,
            include_in_discourse=True,
            text=text,
        )
        for index, text in enumerate(
            texts,
            start=1,
        )
    ]

    result = parse_speaker_turns(blocks)

    assert len(result.turns) == 1

    return result.turns[0]


def formal_order_document(
    *,
    body_words: int = 250,
) -> str:
    body = " ".join(f"palabra{index}" for index in range(body_words))

    return f"(Orden del Día N° 25)\nI Dictamen de mayoría\nHonorable Cámara:\n{body}"


def repeated_body(
    *,
    body_words: int = 220,
) -> str:
    return " ".join(f"contenido{index}" for index in range(body_words))


def formal_legislative_packet(
    *,
    body_words: int = 220,
) -> str:
    return (
        "DICTAMEN DE MAYORIA\n"
        "HONORABLE CAMARA:\n"
        "PROYECTO DE LEY\n"
        "EL SENADO Y CAMARA DE DIPUTADOS\n"
        f"{repeated_body(body_words=body_words)}"
    )


def formal_resolution_packet(
    *,
    body_words: int = 220,
) -> str:
    return (
        "PROYECTO DE RESOLUCION\n"
        "LA CAMARA DE DIPUTADOS DE LA NACION\n"
        "RESUELVE:\n"
        f"{repeated_body(body_words=body_words)}"
    )


def formal_judicial_packet(
    *,
    body_words: int = 220,
) -> str:
    return (
        f"PODER JUDICIAL DE LA NACION\nY VISTOS:\nRESUELVE:\n{repeated_body(body_words=body_words)}"
    )


def intro_packet(
    *,
    body_words: int = 220,
) -> str:
    return f"I\nDictamen de mayoria\nHonorable Camara:\n{repeated_body(body_words=body_words)}"


def assert_fully_spoken(turn) -> None:
    result = classify_speaker_turn_content(turn)

    assert result.documentary_boundary is None
    assert result.speech_word_count == turn.word_count
    assert result.documentary_word_count == 0
    assert all(span.content_kind == TurnContentKind.SPOKEN_TEXT for span in result.spans)
    assert_exact_segment_reconstruction(turn, result)


def assert_exact_segment_reconstruction(turn, result) -> None:
    for segment in turn.segments:
        spans = [
            span
            for span in result.spans
            if (
                span.page_number == segment.page_number
                and span.reading_order == segment.reading_order
                and segment.start <= span.start
                and span.end <= segment.end
            )
        ]

        assert spans
        assert spans[0].start == segment.start
        assert spans[-1].end == segment.end

        cursor = segment.start

        for span in spans:
            assert span.start == cursor
            cursor = span.end

        assert "".join(span.text for span in spans) == segment.text


def test_classifies_normal_turn_as_spoken_text() -> None:
    turn = make_turn("Sr. Alpha. – Esta es una intervención.")

    result = classify_speaker_turn_content(turn)

    assert result.documentary_boundary is None
    assert len(result.spans) == 1
    assert result.spans[0].content_kind == TurnContentKind.SPOKEN_TEXT
    assert result.spans[0].include_in_speech is True


def test_accepts_order_of_day_with_formal_body() -> None:
    turn = make_turn(
        "Sr. Presidente. – Corresponde considerar el asunto.\n" + formal_order_document()
    )

    boundary = find_documentary_boundary(turn)
    result = classify_speaker_turn_content(turn)

    assert boundary is not None
    assert boundary.cue == DocumentaryCue.ORDER_OF_DAY
    assert boundary.matched_text == "(Orden del Día N° 25)"
    assert [span.content_kind for span in result.spans] == [
        TurnContentKind.SPOKEN_TEXT,
        TurnContentKind.DOCUMENTARY_INSERT,
    ]
    assert result.spans[0].text.strip() == "Corresponde considerar el asunto."
    assert result.spans[1].text.startswith("(Orden del Día N° 25)")


def test_chair_intro_followed_by_formal_legislative_bundle() -> None:
    turn = make_turn(
        "Sr. Presidente.- Corresponde considerar el asunto.\n" + formal_legislative_packet()
    )

    result = classify_speaker_turn_content(turn)

    assert result.documentary_boundary is not None
    assert result.documentary_boundary.cue == (DocumentaryCue.FORMAL_AGENDA_DOCUMENT_BUNDLE)
    assert [span.content_kind for span in result.spans] == [
        TurnContentKind.SPOKEN_TEXT,
        TurnContentKind.DOCUMENTARY_INSERT,
    ]
    assert result.spans[0].text.strip() == "Corresponde considerar el asunto."
    assert result.spans[1].text.startswith("DICTAMEN DE MAYORIA")
    assert_exact_segment_reconstruction(turn, result)


def test_named_legislator_short_intro_followed_by_order_of_day_packet() -> None:
    turn = make_turn("Sr. Alpha.- Solicito mi abstencion.\n" + formal_order_document())

    result = classify_speaker_turn_content(turn)

    assert result.documentary_boundary is not None
    assert result.documentary_boundary.cue == DocumentaryCue.ORDER_OF_DAY
    assert result.spans[0].text.strip() == "Solicito mi abstencion."
    assert result.spans[1].content_kind == TurnContentKind.DOCUMENTARY_INSERT
    assert_exact_segment_reconstruction(turn, result)


def test_named_legislator_short_intro_followed_by_generic_committee_report() -> None:
    turn = make_turn("Sr. Alpha.- Solicito mi abstencion.\n" + formal_legislative_packet())

    result = classify_speaker_turn_content(turn)

    assert result.documentary_boundary is not None
    assert result.documentary_boundary.cue == (DocumentaryCue.FORMAL_AGENDA_DOCUMENT_BUNDLE)
    assert result.spans[0].text.strip() == "Solicito mi abstencion."
    assert result.spans[1].text.startswith("DICTAMEN DE MAYORIA")
    assert_exact_segment_reconstruction(turn, result)


def test_secretary_dice_asi_followed_by_formal_resolution() -> None:
    turn = make_turn("Sr. Secretario.- Dice asi:\n" + formal_resolution_packet())

    result = classify_speaker_turn_content(turn)

    assert result.documentary_boundary is not None
    assert result.documentary_boundary.cue == (DocumentaryCue.FORMAL_AGENDA_DOCUMENT_BUNDLE)
    assert result.spans[0].text.strip() == "Dice asi:"
    assert result.spans[1].text.startswith("PROYECTO DE RESOLUCION")
    assert_exact_segment_reconstruction(turn, result)


def test_chair_intro_followed_by_formal_judicial_record_bundle() -> None:
    turn = make_turn("Sr. Presidente.- Dese cuenta del expediente.\n" + formal_judicial_packet())

    result = classify_speaker_turn_content(turn)

    assert result.documentary_boundary is not None
    assert result.documentary_boundary.cue == (
        DocumentaryCue.FORMAL_JUDICIAL_ELECTORAL_RECORD_BUNDLE
    )
    assert result.spans[0].text.strip() == "Dese cuenta del expediente."
    assert result.spans[1].text.startswith("PODER JUDICIAL DE LA NACION")
    assert_exact_segment_reconstruction(turn, result)


def test_rejects_short_order_of_day_reference() -> None:
    turn = make_turn("Sr. Presidente. – Se va a votar el dictamen (Orden del Día N° 25).")

    result = classify_speaker_turn_content(turn)

    assert result.documentary_boundary is None
    assert result.speech_word_count > 0
    assert result.documentary_word_count == 0


def test_rejects_order_of_day_placeholder() -> None:
    turn = make_turn(
        "Sr. Presidente. – "
        "Corresponde considerar el asunto. "
        "(Orden del Día N° 25). "
        "AQUÍ ORDEN DEL DÍA N° 25"
    )

    assert find_documentary_boundary(turn) is None


def test_accepts_coded_project_document() -> None:
    body = " ".join("contenido" for _ in range(250))
    turn = make_turn(
        "Sr. Presidente. – "
        "Corresponde considerar el proyecto.\n"
        "(2.875-D.-17) "
        "Proyecto de declaración\n"
        "La Cámara de Diputados de la Nación "
        "DECLARA:\n"
        f"{body}"
    )

    boundary = find_documentary_boundary(turn)

    assert boundary is not None
    assert boundary.cue == DocumentaryCue.CODED_PROJECT


def test_accepts_secretary_expediente_sequence() -> None:
    body = " ".join("contenido" for _ in range(250))
    turn = make_turn(
        "Sr. Secretario (Figueroa). – "
        "Expediente 3.370-D.-24, "
        "del diputado Julio Cobos.\n"
        "(3.370-D.-24) "
        "Proyecto de resolución\n"
        "La Cámara de Diputados de la Nación "
        "RESUELVE:\n"
        f"{body}"
    )

    boundary = find_documentary_boundary(turn)

    assert boundary is not None
    assert boundary.cue == (DocumentaryCue.SECRETARY_EXPEDIENTE)


def test_accepts_named_legislator_document_cue_after_short_prefix() -> None:
    turn = make_turn(
        "Sr. Alpha. – Quiero referirme al asunto.\n" + formal_order_document(body_words=300)
    )

    assert find_documentary_boundary(turn) is not None


def test_wrapped_lowercase_dictamen_de_mayoria_remains_spoken() -> None:
    turn = make_turn(
        "Sr. Cacace.- escuchamos atentamente el informe que se hizo sobre el\n"
        "dictamen de mayor\u00eda.\n"
        "En respuesta a ese informe seguimos tratando el proyecto, el anexo y la ley. "
        f"{repeated_body(body_words=260)}"
    )

    assert_fully_spoken(turn)


def test_wrapped_lowercase_dictamen_final_remains_spoken() -> None:
    turn = make_turn(
        "Sr. Tunessi.- cuando se vuelve a emitir el\n"
        "dictamen final de unificacion de los codigos, en este debate "
        "tambien se menciona un proyecto, un informe, un anexo y una ley. "
        f"{repeated_body(body_words=260)}"
    )

    assert_fully_spoken(turn)


def test_wrapped_lowercase_fundamentos_remain_spoken() -> None:
    turn = make_turn(
        "Sr. Pais.- el eje es el que planteaban en sus\n"
        "fundamentos los proyectos de la oposicion, junto con cada informe, "
        "anexo y ley citada en el debate. "
        f"{repeated_body(body_words=260)}"
    )

    assert_fully_spoken(turn)


def test_wrapped_lowercase_dictamen_declarando_remains_spoken() -> None:
    turn = make_turn(
        "Sr. Biella Calvet.- elaboramos un\n"
        "dictamen declarando la emergencia sanitaria mientras revisabamos "
        "el proyecto, el informe, el anexo y la ley. "
        f"{repeated_body(body_words=260)}"
    )

    assert_fully_spoken(turn)


def test_chair_project_description_before_packet_remains_spoken() -> None:
    turn = make_turn(
        "Sr. Presidente.- Corresponde considerar los dictamenes recaidos en el\n"
        "proyecto de ley por el que se crea el Programa Nacional, contenido en el "
        "Orden del Dia N 186.\n" + intro_packet()
    )

    result = classify_speaker_turn_content(turn)
    speech_text = "".join(span.text for span in result.spans if span.include_in_speech)

    assert result.documentary_boundary is not None
    assert "proyecto de ley por el que se crea el Programa Nacional" in speech_text
    assert "Orden del Dia N 186." in speech_text
    assert result.spans[1].text.startswith("I\nDictamen de mayoria")
    assert_exact_segment_reconstruction(turn, result)


def test_named_vote_sentence_before_packet_remains_spoken() -> None:
    turn = make_turn(
        "Sr. Acuna.- Nuestro bloque va a votar por la negativa el\n"
        "proyecto de declaracion contenido en el expediente mencionado y en el "
        "Orden del Dia N 45.\n" + intro_packet()
    )

    result = classify_speaker_turn_content(turn)
    speech_text = "".join(span.text for span in result.spans if span.include_in_speech)

    assert result.documentary_boundary is not None
    assert "va a votar por la negativa el\nproyecto de declaracion" in speech_text
    assert "Orden del Dia N 45." in speech_text
    assert result.spans[1].text.startswith("I\nDictamen de mayoria")
    assert_exact_segment_reconstruction(turn, result)


def test_long_uppercase_title_rewinds_to_first_title_line() -> None:
    title = "\n".join(f"TITULO FORMAL EXTENSO LINEA {index}" for index in range(1, 13))
    turn = make_turn(
        "Sr. Presidente.- Corresponde considerar el asunto.\n"
        f"{title}\n"
        "PROYECTO DE LEY\n"
        "HONORABLE CAMARA:\n"
        f"{repeated_body(body_words=220)}"
    )

    result = classify_speaker_turn_content(turn)

    assert result.documentary_boundary is not None
    assert result.spans[0].text.strip() == "Corresponde considerar el asunto."
    assert result.spans[1].text.startswith("TITULO FORMAL EXTENSO LINEA 1")
    assert "PROYECTO DE LEY" in result.spans[1].text
    assert_exact_segment_reconstruction(turn, result)


def test_ordinary_legislator_speech_mentioning_project_remains_spoken() -> None:
    body = " ".join("argumento" for _ in range(260))
    turn = make_turn(
        "Sr. Alpha.- Este proyecto de ley debe analizarse junto con "
        f"cada articulo, anexo e informe. {body}"
    )

    result = classify_speaker_turn_content(turn)

    assert result.documentary_boundary is None
    assert [span.content_kind for span in result.spans] == [TurnContentKind.SPOKEN_TEXT]
    assert result.documentary_word_count == 0


def test_long_executive_official_address_with_formal_words_remains_spoken() -> None:
    turn = make_turn(
        "Sr. Jefe de Gabinete de Ministros.- "
        "Informo sobre leyes, proyectos, articulos y reportes.\n"
        "PROYECTO DE LEY\n"
        "INFORME\n"
        "ANEXO\n"
        f"{repeated_body(body_words=260)}"
    )

    result = classify_speaker_turn_content(turn)

    assert result.documentary_boundary is None
    assert [span.content_kind for span in result.spans] == [TurnContentKind.SPOKEN_TEXT]
    assert result.documentary_word_count == 0


def test_presidential_opening_address_remains_spoken() -> None:
    turn = make_turn(
        "Sr. Presidente de la Naci\u00f3n.- "
        "Vengo a inaugurar el periodo de sesiones ordinarias.\n"
        "PROYECTO DE LEY\n"
        "INFORME\n"
        f"{repeated_body(body_words=260)}"
    )

    result = classify_speaker_turn_content(turn)

    assert result.documentary_boundary is None
    assert [span.content_kind for span in result.spans] == [TurnContentKind.SPOKEN_TEXT]
    assert result.documentary_word_count == 0


def test_named_speaker_long_prefix_rejects_generic_formal_bundle() -> None:
    prefix = " ".join(f"palabra{index}" for index in range(251))
    turn = make_turn("Sr. Alpha.- " + prefix + "\n" + formal_legislative_packet())

    result = classify_speaker_turn_content(turn)

    assert result.documentary_boundary is None
    assert [span.content_kind for span in result.spans] == [TurnContentKind.SPOKEN_TEXT]


def test_splits_multiblock_turn_losslessly() -> None:
    first_text = "Sr. Presidente. – Introducción de la Presidencia."
    second_text = formal_order_document(body_words=250)

    turn = make_turn(
        first_text,
        second_text,
    )
    result = classify_speaker_turn_content(turn)

    assert result.documentary_boundary is not None

    spans_by_block = {}

    for span in result.spans:
        spans_by_block.setdefault(
            span.block_reference,
            [],
        ).append(span)

    for segment in turn.segments:
        spans = spans_by_block[segment.block_reference]
        reconstructed = "".join(span.text for span in spans)

        assert reconstructed == (segment.text)
        assert spans[0].start == (segment.start)
        assert spans[-1].end == (segment.end)

    assert_exact_segment_reconstruction(turn, result)


def test_unattributed_text_is_preserved_but_excluded() -> None:
    result = parse_speaker_turns(
        [
            SpeakerTurnInputBlock(
                page_number=1,
                reading_order=1,
                structural_zone=("proceedings"),
                content_role="transcript",
                include_in_discourse=True,
                text=("Texto sin marcador."),
            )
        ]
    )

    turn = result.turns[0]
    content = classify_speaker_turn_content(turn)

    assert len(content.spans) == 1
    assert content.spans[0].content_kind == (TurnContentKind.UNATTRIBUTED_TEXT)
    assert content.spans[0].include_in_speech is False


def test_splits_inline_parenthetical_without_losing_speech() -> None:
    turn = make_turn("Sr. Alpha. – Primera oración. (Aplausos.) Segunda oración.")

    result = classify_speaker_turn_content(turn)

    assert [span.content_kind for span in result.spans] == [
        TurnContentKind.SPOKEN_TEXT,
        TurnContentKind.STAGE_DIRECTION,
        TurnContentKind.SPOKEN_TEXT,
    ]
    assert result.spans[1].text == "(Aplausos.)"
    assert "Segunda oración." in (result.spans[2].text)


def test_detects_multiline_parenthetical_stage_direction() -> None:
    turn = make_turn(
        "Sr. Alpha. – "
        "Texto previo. "
        "(Aplausos. Varios señores diputados\n"
        "hablan a la vez.) "
        "Texto posterior."
    )

    result = classify_speaker_turn_content(turn)

    stage_spans = [
        span for span in result.spans if (span.content_kind == TurnContentKind.STAGE_DIRECTION)
    ]

    assert len(stage_spans) == 1
    assert "\n" in stage_spans[0].text
    assert "Texto posterior." in "".join(
        span.text for span in result.spans if span.include_in_speech
    )


def test_detects_numbered_vease_note_and_rejects_spoken_ver() -> None:
    turn = make_turn(
        "Sr. Alpha. – "
        "Hay que ver cómo resolverlo.\n"
        "1. Véase el texto de la sanción "
        "en el Apéndi-\n"
        "ce. (Pág. 10.)\n"
        "Continúa la intervención."
    )

    result = classify_speaker_turn_content(turn)

    editorial_spans = [
        span for span in result.spans if (span.content_kind == TurnContentKind.EDITORIAL_NOTE)
    ]

    assert len(editorial_spans) == 1
    assert editorial_spans[0].text.startswith("1. Véase")
    assert "Apéndi-\nce" in (editorial_spans[0].text)

    spoken_text = "".join(span.text for span in result.spans if span.include_in_speech)

    assert "Hay que ver cómo resolverlo." in spoken_text
    assert "Continúa la intervención." in spoken_text


def test_detects_wrapped_presidency_direction() -> None:
    turn = make_turn(
        "Sr. Alpha. – Texto previo.\n"
        "–Ocupa la Presidencia la señora vicepre-\n"
        "sidenta de la Honorable Cámara.\n"
        "Texto posterior."
    )

    result = classify_speaker_turn_content(turn)

    stage_spans = [
        span for span in result.spans if (span.content_kind == TurnContentKind.STAGE_DIRECTION)
    ]

    assert len(stage_spans) == 1
    assert stage_spans[0].text == (
        "–Ocupa la Presidencia la señora vicepre-\nsidenta de la Honorable Cámara."
    )


def test_detects_standalone_manifestations_and_murmurs() -> None:
    turn = make_turn(
        "Sr. Alpha. – Texto inicial.\nMANIFESTACIONES\n- Murmullos en el recinto.\nTexto final."
    )

    result = classify_speaker_turn_content(turn)

    stage_spans = [
        span.text for span in result.spans if (span.content_kind == TurnContentKind.STAGE_DIRECTION)
    ]

    assert stage_spans == [
        "MANIFESTACIONES",
        "- Murmullos en el recinto.",
    ]


def test_documentary_insert_is_not_refined_as_stage_or_editorial() -> None:
    document = formal_order_document(body_words=250) + "\n1. Véase el Apéndice." + "\n(Aplausos.)"
    turn = make_turn("Sr. Presidente. – Se considera el asunto.\n" + document)

    result = classify_speaker_turn_content(turn)

    documentary_spans = [
        span for span in result.spans if (span.content_kind == TurnContentKind.DOCUMENTARY_INSERT)
    ]

    assert documentary_spans
    assert all(
        span.content_kind == TurnContentKind.DOCUMENTARY_INSERT for span in documentary_spans
    )
    assert not any(
        span.content_kind
        in {
            TurnContentKind.STAGE_DIRECTION,
            TurnContentKind.EDITORIAL_NOTE,
        }
        for span in result.spans
    )


def test_extends_hyphenated_stage_direction_across_segments() -> None:
    turn = make_turn(
        (
            "Sr. Leito. – Texto previo.\n"
            "–Ocupa la Presidencia el señor presidente\n"
            "de la Honorable Cámara, doctor Sergio To-"
        ),
        ("más Massa.\nTexto posterior."),
    )

    result = classify_speaker_turn_content(turn)

    stage_spans = [
        span for span in result.spans if (span.content_kind == TurnContentKind.STAGE_DIRECTION)
    ]

    assert len(stage_spans) == 2
    assert stage_spans[0].text.endswith("Sergio To-")
    assert stage_spans[1].text == "más Massa."
    assert stage_spans[1].classification_method == ("cross_segment_hyphenated_stage_direction")
    assert stage_spans[0].block_reference != stage_spans[1].block_reference

    spoken_text = "".join(span.text for span in result.spans if span.include_in_speech)

    assert "Texto previo." in spoken_text
    assert "Texto posterior." in spoken_text
    assert "más Massa." not in spoken_text

    reconstructed_segments = []

    for segment in turn.segments:
        segment_spans = [
            span
            for span in result.spans
            if (
                span.page_number == segment.page_number
                and span.reading_order == segment.reading_order
                and span.start >= segment.start
                and span.end <= segment.end
            )
        ]
        segment_spans.sort(
            key=lambda span: (
                span.start,
                span.end,
            )
        )

        assert segment_spans
        assert segment_spans[0].start == segment.start
        assert segment_spans[-1].end == segment.end

        for previous, current in zip(
            segment_spans,
            segment_spans[1:],
            strict=False,
        ):
            assert previous.end == current.start

        reconstructed_segment = "".join(span.text for span in segment_spans)

        assert reconstructed_segment == segment.text

        reconstructed_segments.append(reconstructed_segment)

    assert "\n".join(reconstructed_segments) == turn.text


def test_detects_complex_applause_parenthetical() -> None:
    turn = make_turn(
        "Sr. Alpha. – Texto previo. "
        "(Aplausos. Varios señores diputados "
        "rodean y felicitan al orador.)"
    )

    result = classify_speaker_turn_content(turn)

    stage_spans = [
        span for span in result.spans if (span.content_kind == TurnContentKind.STAGE_DIRECTION)
    ]

    assert len(stage_spans) == 1
    assert stage_spans[0].text == (
        "(Aplausos. Varios señores diputados rodean y felicitan al orador.)"
    )


def test_extends_open_presidency_direction_across_segments() -> None:
    turn = make_turn(
        ("Sr. Alpha. – Texto previo.\n- Ocupa la Presidencia el señor"),
        ("presidente de la Honorable Cámara, doctor Martín Alexis Menem.\nTexto posterior."),
    )

    result = classify_speaker_turn_content(turn)

    stage_spans = [
        span for span in result.spans if (span.content_kind == TurnContentKind.STAGE_DIRECTION)
    ]

    assert len(stage_spans) == 2
    assert stage_spans[0].text == ("- Ocupa la Presidencia el señor")
    assert stage_spans[1].text == ("presidente de la Honorable Cámara, doctor Martín Alexis Menem.")
    assert stage_spans[1].classification_method == ("cross_segment_open_stage_direction")

    spoken_text = "".join(span.text for span in result.spans if span.include_in_speech)

    assert "Texto previo." in spoken_text
    assert "Texto posterior." in spoken_text
    assert "Martín Alexis Menem" not in (spoken_text)


def test_extends_outside_microphone_direction_across_segments() -> None:
    turn = make_turn(
        ("Sr. Presidente. – Texto previo.\n- La señora diputada Tepp hace"),
        (
            "uso de la palabra fuera de\n"
            "micrófono, por lo que no se\n"
            "alcanzan\n"
            "a\n"
            "percibir\n"
            "sus\n"
            "manifestaciones."
        ),
    )

    result = classify_speaker_turn_content(turn)

    stage_spans = [
        span for span in result.spans if (span.content_kind == TurnContentKind.STAGE_DIRECTION)
    ]

    assert len(stage_spans) == 2
    assert stage_spans[0].text == ("- La señora diputada Tepp hace")
    assert stage_spans[1].text.startswith("uso de la palabra fuera de")
    assert stage_spans[1].text.endswith("manifestaciones.")

    assert not any(span.text == "manifestaciones." for span in stage_spans)


def test_detects_multiline_outside_microphone_direction() -> None:
    turn = make_turn(
        "Sr. Alpha. – Texto previo.\n"
        "- -La\n"
        "señora\n"
        "diputada\n"
        "Olmos\n"
        "realiza manifestaciones fuera\n"
        "de\n"
        "micrófono\n"
        "que\n"
        "no\n"
        "se\n"
        "alcanzan a percibir.\n"
        "Texto posterior."
    )

    result = classify_speaker_turn_content(turn)

    stage_spans = [
        span for span in result.spans if (span.content_kind == TurnContentKind.STAGE_DIRECTION)
    ]

    assert len(stage_spans) == 1
    assert stage_spans[0].text.startswith("- -La\nseñora")
    assert stage_spans[0].text.endswith("alcanzan a percibir.")

    spoken_text = "".join(span.text for span in result.spans if span.include_in_speech)

    assert "Texto previo." in spoken_text
    assert "Texto posterior." in spoken_text
    assert "diputada\nOlmos" not in (spoken_text)
