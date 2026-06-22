from argentine_deputies_discursive_distance.speaker import (
    MarkerPosition,
    MarkerSeparatorKind,
    SpeakerLabelFamily,
    find_speaker_markers,
)


def test_finds_standard_marker_with_exact_offsets() -> None:
    text = "Sr. Presidente (Menem). – Queda abierta la sesión."

    markers = find_speaker_markers(text)

    assert len(markers) == 1
    marker = markers[0]

    assert marker.start == 0
    assert marker.end == (text.index("–") + 1)
    assert marker.raw_marker == ("Sr. Presidente (Menem). –")
    assert marker.raw_label == ("Presidente (Menem)")
    assert marker.normalized_label == ("PRESIDENTE (MENEM)")
    assert marker.family == (SpeakerLabelFamily.CHAIR)
    assert marker.position == (MarkerPosition.BLOCK_START)


def test_uses_terminal_explicit_marker_after_preamble() -> None:
    text = (
        "Sr. Presidente (Fellner). – "
        "Tiene la palabra el señor diputado "
        "por Buenos Aires.\n"
        "Sr. Atanasof. – "
        "Señor presidente: comenzaré."
    )

    markers = find_speaker_markers(text)

    assert [marker.normalized_label for marker in markers] == [
        "PRESIDENTE (FELLNER)",
        "ATANASOF",
    ]

    assert markers[1].raw_marker == ("Sr. Atanasof. –")


def test_finds_marker_with_multiline_title_and_label() -> None:
    text = "Texto anterior.\nSr.\nPresidente\n(Menem).- Continúe."

    markers = find_speaker_markers(text)

    assert len(markers) == 1
    marker = markers[0]

    assert marker.raw_marker == ("Sr.\nPresidente\n(Menem).-")
    assert marker.raw_label == ("Presidente (Menem)")
    assert marker.normalized_label == ("PRESIDENTE (MENEM)")
    assert marker.is_multiline is True
    assert marker.position == (MarkerPosition.EMBEDDED)


def test_finds_multiline_person_name() -> None:
    text = "Sr.\nCorrea\nLlano.- Señor presidente."

    markers = find_speaker_markers(text)

    assert len(markers) == 1
    assert markers[0].normalized_label == "CORREA LLANO"
    assert markers[0].is_multiline is True


def test_accepts_audited_title_without_period() -> None:
    text = "Sr Yoma. – Señor presidente: comenzaré."

    markers = find_speaker_markers(text)

    assert len(markers) == 1
    marker = markers[0]

    assert marker.raw_title == "Sr"
    assert marker.normalized_title == ("SR.")
    assert marker.normalized_label == ("YOMA")
    assert marker.detection_confidence == 0.95
    assert "title_without_period" in marker.detection_method


def test_accepts_audited_dash_only_separator() -> None:
    text = "Sra. Carrizo (A. C.) – O sea que este es el error."

    markers = find_speaker_markers(text)

    assert len(markers) == 1
    marker = markers[0]

    assert marker.normalized_label == ("CARRIZO (A. C.)")
    assert marker.separator_kind == (MarkerSeparatorKind.DASH_ONLY)
    assert marker.detection_confidence == 0.95


def test_rejects_narrative_colon_and_false_honorific_prefixes() -> None:
    text = (
        "El señor diputado dijo: esto es "
        "importante. Sri Lanka aparece en "
        "la lista. Expreso Tigre Iguazú SRL "
        "presentó una acción."
    )

    assert find_speaker_markers(text) == ()


def test_classifies_secretary_and_executive_roles() -> None:
    text = (
        "Sr. Secretario (Pagán).- "
        "Se registraron los votos.\n"
        "Sr. Jefe de Gabinete de Ministros. – "
        "Responderé las preguntas."
    )

    markers = find_speaker_markers(text)

    assert len(markers) == 2
    assert markers[0].family == (SpeakerLabelFamily.CHAMBER_SECRETARY)
    assert markers[1].family == (SpeakerLabelFamily.EXECUTIVE_OFFICIAL)
