"""Tests for the reference-data validator."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.validate_reference_data import (
    ALIASES_FILE,
    BLOC_ALIGNMENT_FILE,
    BLOC_MEMBERSHIP_FILE,
    LEGISLATORS_FILE,
    REQUIRED_COLUMNS,
    SOURCES_FILE,
    ReferenceDataError,
    validate_reference_data,
)

LEGISLATOR_ROW = {
    "legislator_id": "leg-0001",
    "canonical_name": "Juana Pérez",
    "given_names": "Juana",
    "surname": "Pérez",
    "province": "Buenos Aires",
    "chamber": "deputies",
    "valid_from": "2009-12-10",
    "valid_to": "",
    "source_id": "src-roster",
    "review_status": "reviewed_confident",
    "notes": "",
}

SOURCE_ROW = {
    "source_id": "src-roster",
    "source_type": "official_chamber_record",
    "title": "Official roster",
    "publisher": "HCDN",
    "url": "https://example.test/roster",
    "retrieved_at": "2026-01-01",
    "coverage_start": "2009-12-10",
    "coverage_end": "",
    "local_snapshot": "",
    "notes": "",
}

ALIAS_ROW = {
    "alias_raw": "PEREZ",
    "alias_normalized": "PEREZ",
    "legislator_id": "leg-0001",
    "valid_from": "2009-12-10",
    "valid_to": "",
    "alias_type": "official_name",
    "confidence": "high",
    "review_status": "reviewed_confident",
    "source_id": "src-roster",
    "notes": "",
}

MEMBERSHIP_ROW = {
    "legislator_id": "leg-0001",
    "bloc_name_raw": "Frente Ejemplo",
    "bloc_name_normalized": "FRENTE EJEMPLO",
    "valid_from": "2009-12-10",
    "valid_to": "2011-12-09",
    "source_id": "src-roster",
    "confidence": "high",
    "review_status": "reviewed_confident",
    "notes": "",
}

ALIGNMENT_ROW = {
    "bloc_name_normalized": "FRENTE EJEMPLO",
    "valid_from": "2009-12-10",
    "valid_to": "2011-12-09",
    "alignment": "opposition_core",
    "reason": "Declared opposition bloc for the full interval.",
    "source_id": "src-roster",
    "review_status": "reviewed_confident",
    "notes": "",
}


def _write_csv(path: Path, *, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_reference_dir(
    tmp_path: Path,
    *,
    legislators: list[dict[str, str]] | None = None,
    aliases: list[dict[str, str]] | None = None,
    bloc_membership: list[dict[str, str]] | None = None,
    bloc_alignment: list[dict[str, str]] | None = None,
    sources: list[dict[str, str]] | None = None,
) -> Path:
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()

    _write_csv(
        reference_dir / LEGISLATORS_FILE,
        columns=REQUIRED_COLUMNS[LEGISLATORS_FILE],
        rows=legislators if legislators is not None else [LEGISLATOR_ROW],
    )
    _write_csv(
        reference_dir / ALIASES_FILE,
        columns=REQUIRED_COLUMNS[ALIASES_FILE],
        rows=aliases if aliases is not None else [ALIAS_ROW],
    )
    _write_csv(
        reference_dir / BLOC_MEMBERSHIP_FILE,
        columns=REQUIRED_COLUMNS[BLOC_MEMBERSHIP_FILE],
        rows=bloc_membership if bloc_membership is not None else [MEMBERSHIP_ROW],
    )
    _write_csv(
        reference_dir / BLOC_ALIGNMENT_FILE,
        columns=REQUIRED_COLUMNS[BLOC_ALIGNMENT_FILE],
        rows=bloc_alignment if bloc_alignment is not None else [ALIGNMENT_ROW],
    )
    _write_csv(
        reference_dir / SOURCES_FILE,
        columns=REQUIRED_COLUMNS[SOURCES_FILE],
        rows=sources if sources is not None else [SOURCE_ROW],
    )

    return reference_dir


def test_valid_reference_data_has_no_errors(tmp_path: Path) -> None:
    reference_dir = _write_reference_dir(tmp_path)

    report = validate_reference_data(reference_dir)

    assert report.errors == []
    assert report.is_valid


def test_empty_tables_are_valid(tmp_path: Path) -> None:
    reference_dir = _write_reference_dir(
        tmp_path,
        legislators=[],
        aliases=[],
        bloc_membership=[],
        bloc_alignment=[],
        sources=[],
    )

    report = validate_reference_data(reference_dir)

    assert report.is_valid


def test_missing_file_raises(tmp_path: Path) -> None:
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()

    with pytest.raises(ReferenceDataError):
        validate_reference_data(reference_dir)


def test_required_field_empty_is_an_error(tmp_path: Path) -> None:
    broken_legislator = dict(LEGISLATOR_ROW)
    broken_legislator["surname"] = ""
    reference_dir = _write_reference_dir(tmp_path, legislators=[broken_legislator])

    report = validate_reference_data(reference_dir)

    assert any("surname" in error for error in report.errors)


def test_open_ended_valid_to_is_not_an_error(tmp_path: Path) -> None:
    reference_dir = _write_reference_dir(tmp_path)

    report = validate_reference_data(reference_dir)

    assert report.is_valid


def test_invalid_iso_date_is_an_error(tmp_path: Path) -> None:
    broken_legislator = dict(LEGISLATOR_ROW)
    broken_legislator["valid_from"] = "10-12-2009"
    reference_dir = _write_reference_dir(tmp_path, legislators=[broken_legislator])

    report = validate_reference_data(reference_dir)

    assert any("valid_from" in error for error in report.errors)


def test_valid_from_after_valid_to_is_an_error(tmp_path: Path) -> None:
    broken_membership = dict(MEMBERSHIP_ROW)
    broken_membership["valid_from"] = "2012-01-01"
    broken_membership["valid_to"] = "2011-01-01"
    reference_dir = _write_reference_dir(tmp_path, bloc_membership=[broken_membership])

    report = validate_reference_data(reference_dir)

    assert any("is after" in error for error in report.errors)


def test_unknown_enum_value_is_an_error(tmp_path: Path) -> None:
    broken_alignment = dict(ALIGNMENT_ROW)
    broken_alignment["alignment"] = "secretly_aligned"
    reference_dir = _write_reference_dir(tmp_path, bloc_alignment=[broken_alignment])

    report = validate_reference_data(reference_dir)

    assert any("alignment" in error for error in report.errors)


def test_unknown_source_id_is_an_error(tmp_path: Path) -> None:
    broken_legislator = dict(LEGISLATOR_ROW)
    broken_legislator["source_id"] = "src-missing"
    reference_dir = _write_reference_dir(tmp_path, legislators=[broken_legislator])

    report = validate_reference_data(reference_dir)

    assert any("src-missing" in error for error in report.errors)


def test_unknown_legislator_id_is_an_error(tmp_path: Path) -> None:
    broken_alias = dict(ALIAS_ROW)
    broken_alias["legislator_id"] = "leg-missing"
    reference_dir = _write_reference_dir(tmp_path, aliases=[broken_alias])

    report = validate_reference_data(reference_dir)

    assert any("leg-missing" in error for error in report.errors)


def test_exact_duplicate_rows_are_an_error(tmp_path: Path) -> None:
    reference_dir = _write_reference_dir(tmp_path, aliases=[ALIAS_ROW, dict(ALIAS_ROW)])

    report = validate_reference_data(reference_dir)

    assert any("duplicate" in error for error in report.errors)


def test_ambiguous_alignment_cannot_be_reviewed_confident(tmp_path: Path) -> None:
    broken_alignment = dict(ALIGNMENT_ROW)
    broken_alignment["alignment"] = "ambiguous_independent"
    broken_alignment["review_status"] = "reviewed_confident"
    reference_dir = _write_reference_dir(tmp_path, bloc_alignment=[broken_alignment])

    report = validate_reference_data(reference_dir)

    assert any("ambiguous_independent" in error for error in report.errors)


def test_overlapping_membership_intervals_are_a_warning(tmp_path: Path) -> None:
    first = dict(MEMBERSHIP_ROW)
    second = dict(MEMBERSHIP_ROW)
    second["valid_from"] = "2011-01-01"
    second["valid_to"] = "2012-12-09"
    second["bloc_name_normalized"] = "OTRO BLOQUE"
    second["bloc_name_raw"] = "Otro Bloque"
    reference_dir = _write_reference_dir(tmp_path, bloc_membership=[first, second])

    report = validate_reference_data(reference_dir)

    assert report.is_valid
    assert any("overlapping" in warning for warning in report.warnings)


def test_overlapping_alignment_intervals_with_different_alignment_is_an_error(
    tmp_path: Path,
) -> None:
    first = dict(ALIGNMENT_ROW)
    second = dict(ALIGNMENT_ROW)
    second["valid_from"] = "2011-01-01"
    second["valid_to"] = "2012-12-09"
    second["alignment"] = "government_core"
    reference_dir = _write_reference_dir(tmp_path, bloc_alignment=[first, second])

    report = validate_reference_data(reference_dir)

    assert any("overlapping alignment" in error for error in report.errors)


def test_overlapping_alignment_intervals_with_same_alignment_is_a_warning(
    tmp_path: Path,
) -> None:
    first = dict(ALIGNMENT_ROW)
    second = dict(ALIGNMENT_ROW)
    second["valid_from"] = "2011-01-01"
    second["valid_to"] = "2012-12-09"
    reference_dir = _write_reference_dir(tmp_path, bloc_alignment=[first, second])

    report = validate_reference_data(reference_dir)

    assert report.is_valid
    assert any("overlapping alignment" in warning for warning in report.warnings)


def test_missing_source_id_for_open_ended_intervals_still_checked(tmp_path: Path) -> None:
    broken_source = dict(SOURCE_ROW)
    broken_source["source_id"] = ""
    reference_dir = _write_reference_dir(tmp_path, sources=[broken_source])

    report = validate_reference_data(reference_dir)

    assert not report.is_valid
