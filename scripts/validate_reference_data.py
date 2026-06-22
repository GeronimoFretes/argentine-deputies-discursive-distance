"""Deterministic validation of the political metadata reference tables.

See ``data/reference/README.md`` for the schemas and controlled vocabularies this
script enforces, and ``docs/POLITICAL_METADATA_METHODOLOGY.md`` for the reasoning
behind the rules.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

DEFAULT_REFERENCE_DIR = Path("data/reference")

LEGISLATORS_FILE = "legislators.csv"
ALIASES_FILE = "legislator_aliases.csv"
BLOC_MEMBERSHIP_FILE = "bloc_membership.csv"
BLOC_ALIGNMENT_FILE = "bloc_alignment.csv"
SOURCES_FILE = "sources.csv"

REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    LEGISLATORS_FILE: (
        "legislator_id",
        "canonical_name",
        "given_names",
        "surname",
        "province",
        "chamber",
        "valid_from",
        "valid_to",
        "source_id",
        "review_status",
        "notes",
    ),
    ALIASES_FILE: (
        "alias_raw",
        "alias_normalized",
        "legislator_id",
        "valid_from",
        "valid_to",
        "alias_type",
        "confidence",
        "review_status",
        "source_id",
        "notes",
    ),
    BLOC_MEMBERSHIP_FILE: (
        "legislator_id",
        "bloc_name_raw",
        "bloc_name_normalized",
        "valid_from",
        "valid_to",
        "source_id",
        "confidence",
        "review_status",
        "notes",
    ),
    BLOC_ALIGNMENT_FILE: (
        "bloc_name_normalized",
        "valid_from",
        "valid_to",
        "alignment",
        "reason",
        "source_id",
        "review_status",
        "notes",
    ),
    SOURCES_FILE: (
        "source_id",
        "source_type",
        "title",
        "publisher",
        "url",
        "retrieved_at",
        "coverage_start",
        "coverage_end",
        "local_snapshot",
        "notes",
    ),
}

REQUIRED_NON_EMPTY: dict[str, tuple[str, ...]] = {
    LEGISLATORS_FILE: ("legislator_id", "canonical_name", "surname", "valid_from", "source_id"),
    ALIASES_FILE: ("alias_raw", "alias_normalized", "legislator_id", "valid_from", "source_id"),
    BLOC_MEMBERSHIP_FILE: (
        "legislator_id",
        "bloc_name_raw",
        "bloc_name_normalized",
        "valid_from",
        "source_id",
    ),
    BLOC_ALIGNMENT_FILE: ("bloc_name_normalized", "valid_from", "alignment", "source_id"),
    SOURCES_FILE: ("source_id", "source_type", "title"),
}

DATE_COLUMNS: dict[str, tuple[str, ...]] = {
    LEGISLATORS_FILE: ("valid_from", "valid_to"),
    ALIASES_FILE: ("valid_from", "valid_to"),
    BLOC_MEMBERSHIP_FILE: ("valid_from", "valid_to"),
    BLOC_ALIGNMENT_FILE: ("valid_from", "valid_to"),
    SOURCES_FILE: ("coverage_start", "coverage_end"),
}

OPEN_ENDED_DATE_COLUMNS: dict[str, tuple[str, ...]] = {
    LEGISLATORS_FILE: ("valid_to",),
    ALIASES_FILE: ("valid_to",),
    BLOC_MEMBERSHIP_FILE: ("valid_to",),
    BLOC_ALIGNMENT_FILE: ("valid_to",),
    SOURCES_FILE: ("coverage_start", "coverage_end"),
}

INTERVAL_COLUMNS: dict[str, tuple[str, str]] = {
    LEGISLATORS_FILE: ("valid_from", "valid_to"),
    ALIASES_FILE: ("valid_from", "valid_to"),
    BLOC_MEMBERSHIP_FILE: ("valid_from", "valid_to"),
    BLOC_ALIGNMENT_FILE: ("valid_from", "valid_to"),
}

REVIEW_STATUS_VALUES = frozenset(
    {
        "pending_research",
        "reviewed_confident",
        "reviewed_uncertain",
        "conflicting_sources",
        "needs_manual_decision",
    }
)

CONFIDENCE_VALUES = frozenset({"high", "medium", "low"})

ALIAS_TYPE_VALUES = frozenset(
    {
        "official_name",
        "transcript_surname",
        "transcript_full_name",
        "initials_variant",
        "accent_variant",
        "compound_surname_variant",
        "manual_exception",
    }
)

ALIGNMENT_VALUES = frozenset(
    {
        "government_core",
        "opposition_core",
        "ambiguous_independent",
        "excluded",
    }
)

SOURCE_TYPE_VALUES = frozenset(
    {
        "official_chamber_record",
        "official_government_record",
        "secondary_academic",
        "secondary_journalistic",
        "archival_snapshot",
        "other",
    }
)

ENUM_COLUMNS: dict[str, dict[str, frozenset[str]]] = {
    LEGISLATORS_FILE: {"review_status": REVIEW_STATUS_VALUES, "chamber": frozenset({"deputies"})},
    ALIASES_FILE: {
        "alias_type": ALIAS_TYPE_VALUES,
        "confidence": CONFIDENCE_VALUES,
        "review_status": REVIEW_STATUS_VALUES,
    },
    BLOC_MEMBERSHIP_FILE: {
        "confidence": CONFIDENCE_VALUES,
        "review_status": REVIEW_STATUS_VALUES,
    },
    BLOC_ALIGNMENT_FILE: {
        "alignment": ALIGNMENT_VALUES,
        "review_status": REVIEW_STATUS_VALUES,
    },
    SOURCES_FILE: {"source_type": SOURCE_TYPE_VALUES},
}

LEGISLATOR_ID_REFERENCE_COLUMNS: tuple[str, ...] = (ALIASES_FILE, BLOC_MEMBERSHIP_FILE)
SOURCE_ID_REFERENCE_FILES: tuple[str, ...] = (
    LEGISLATORS_FILE,
    ALIASES_FILE,
    BLOC_MEMBERSHIP_FILE,
    BLOC_ALIGNMENT_FILE,
)


class ReferenceDataError(RuntimeError):
    """Raised when reference data cannot be validated structurally."""


@dataclass(slots=True)
class ValidationReport:
    """Structural errors and review warnings found across the reference tables."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Return whether no structural error was found."""
        return not self.errors


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _check_columns(
    *,
    file_name: str,
    rows: list[dict[str, str]],
    report: ValidationReport,
) -> None:
    expected = REQUIRED_COLUMNS[file_name]

    if not rows:
        return

    actual = tuple(rows[0].keys())

    if actual != expected:
        report.errors.append(
            f"{file_name}: header columns {actual} do not match required columns {expected}."
        )


def _check_required_non_empty(
    *,
    file_name: str,
    rows: list[dict[str, str]],
    report: ValidationReport,
) -> None:
    for row_index, row in enumerate(rows, start=2):
        for column in REQUIRED_NON_EMPTY.get(file_name, ()):
            if not row.get(column, "").strip():
                report.errors.append(
                    f"{file_name}:{row_index}: required field '{column}' is empty."
                )


def _parse_date(
    *,
    file_name: str,
    row_index: int,
    column: str,
    value: str,
    allow_empty: bool,
    report: ValidationReport,
) -> date | None:
    if not value.strip():
        if allow_empty:
            return None

        report.errors.append(f"{file_name}:{row_index}: '{column}' must not be empty.")
        return None

    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        report.errors.append(
            f"{file_name}:{row_index}: '{column}' value '{value}' is not a valid ISO date."
        )
        return None


def _check_dates(
    *,
    file_name: str,
    rows: list[dict[str, str]],
    report: ValidationReport,
) -> None:
    open_ended = OPEN_ENDED_DATE_COLUMNS.get(file_name, ())

    for row_index, row in enumerate(rows, start=2):
        for column in DATE_COLUMNS.get(file_name, ()):
            _parse_date(
                file_name=file_name,
                row_index=row_index,
                column=column,
                value=row.get(column, ""),
                allow_empty=column in open_ended,
                report=report,
            )


def _check_intervals(
    *,
    file_name: str,
    rows: list[dict[str, str]],
    report: ValidationReport,
) -> None:
    start_column, end_column = INTERVAL_COLUMNS.get(file_name, (None, None))

    if start_column is None or end_column is None:
        return

    for row_index, row in enumerate(rows, start=2):
        start_value = row.get(start_column, "").strip()
        end_value = row.get(end_column, "").strip()

        if not start_value or not end_value:
            continue

        try:
            start_date = date.fromisoformat(start_value)
            end_date = date.fromisoformat(end_value)
        except ValueError:
            continue

        if start_date > end_date:
            report.errors.append(
                f"{file_name}:{row_index}: '{start_column}' ({start_value}) is after "
                f"'{end_column}' ({end_value})."
            )


def _check_enums(
    *,
    file_name: str,
    rows: list[dict[str, str]],
    report: ValidationReport,
) -> None:
    for row_index, row in enumerate(rows, start=2):
        for column, allowed in ENUM_COLUMNS.get(file_name, {}).items():
            value = row.get(column, "").strip()

            if not value:
                continue

            if value not in allowed:
                report.errors.append(
                    f"{file_name}:{row_index}: '{column}' value '{value}' is not one of "
                    f"{sorted(allowed)}."
                )


def _check_ambiguous_alignment_not_confident(
    *,
    rows: list[dict[str, str]],
    report: ValidationReport,
) -> None:
    for row_index, row in enumerate(rows, start=2):
        if (
            row.get("alignment", "").strip() == "ambiguous_independent"
            and row.get("review_status", "").strip() == "reviewed_confident"
        ):
            report.errors.append(
                f"{BLOC_ALIGNMENT_FILE}:{row_index}: alignment 'ambiguous_independent' must not "
                "use review_status 'reviewed_confident'."
            )


def _check_duplicate_rows(
    *,
    file_name: str,
    rows: list[dict[str, str]],
    report: ValidationReport,
) -> None:
    seen: dict[tuple[str, ...], int] = {}

    for row_index, row in enumerate(rows, start=2):
        key = tuple(row.values())

        if key in seen:
            report.errors.append(
                f"{file_name}:{row_index}: exact duplicate of row at line {seen[key]}."
            )
        else:
            seen[key] = row_index


def _check_source_references(
    *,
    file_name: str,
    rows: list[dict[str, str]],
    known_source_ids: set[str],
    report: ValidationReport,
) -> None:
    for row_index, row in enumerate(rows, start=2):
        source_id = row.get("source_id", "").strip()

        if source_id and source_id not in known_source_ids:
            report.errors.append(
                f"{file_name}:{row_index}: source_id '{source_id}' is not defined in "
                f"{SOURCES_FILE}."
            )


def _check_legislator_references(
    *,
    file_name: str,
    rows: list[dict[str, str]],
    known_legislator_ids: set[str],
    report: ValidationReport,
) -> None:
    for row_index, row in enumerate(rows, start=2):
        legislator_id = row.get("legislator_id", "").strip()

        if legislator_id and legislator_id not in known_legislator_ids:
            report.errors.append(
                f"{file_name}:{row_index}: legislator_id '{legislator_id}' is not defined in "
                f"{LEGISLATORS_FILE}."
            )


def _parse_interval_bounds(row: dict[str, str]) -> tuple[date, date | None] | None:
    start_value = row.get("valid_from", "").strip()
    end_value = row.get("valid_to", "").strip()

    if not start_value:
        return None

    try:
        start_date = date.fromisoformat(start_value)
    except ValueError:
        return None

    if not end_value:
        return start_date, None

    try:
        end_date = date.fromisoformat(end_value)
    except ValueError:
        return None

    return start_date, end_date


def _intervals_overlap(
    *,
    first: tuple[date, date | None],
    second: tuple[date, date | None],
) -> bool:
    first_start, first_end = first
    second_start, second_end = second

    first_end_value = first_end or date.max
    second_end_value = second_end or date.max

    return first_start <= second_end_value and second_start <= first_end_value


def _check_overlapping_membership(
    *,
    rows: list[dict[str, str]],
    report: ValidationReport,
) -> None:
    by_legislator: dict[str, list[tuple[int, tuple[date, date | None]]]] = {}

    for row_index, row in enumerate(rows, start=2):
        legislator_id = row.get("legislator_id", "").strip()
        bounds = _parse_interval_bounds(row)

        if not legislator_id or bounds is None:
            continue

        by_legislator.setdefault(legislator_id, []).append((row_index, bounds))

    for legislator_id, intervals in by_legislator.items():
        for first_position in range(len(intervals)):
            for second_position in range(first_position + 1, len(intervals)):
                first_row_index, first_bounds = intervals[first_position]
                second_row_index, second_bounds = intervals[second_position]

                if _intervals_overlap(first=first_bounds, second=second_bounds):
                    report.warnings.append(
                        f"{BLOC_MEMBERSHIP_FILE}: legislator_id '{legislator_id}' has overlapping "
                        f"membership intervals at lines {first_row_index} and {second_row_index}; "
                        "requires manual review."
                    )


def _check_overlapping_alignment(
    *,
    rows: list[dict[str, str]],
    report: ValidationReport,
) -> None:
    by_bloc: dict[str, list[tuple[int, tuple[date, date | None], str]]] = {}

    for row_index, row in enumerate(rows, start=2):
        bloc_name = row.get("bloc_name_normalized", "").strip()
        bounds = _parse_interval_bounds(row)
        alignment = row.get("alignment", "").strip()

        if not bloc_name or bounds is None:
            continue

        by_bloc.setdefault(bloc_name, []).append((row_index, bounds, alignment))

    for bloc_name, intervals in by_bloc.items():
        for first_position in range(len(intervals)):
            for second_position in range(first_position + 1, len(intervals)):
                first_row_index, first_bounds, first_alignment = intervals[first_position]
                second_row_index, second_bounds, second_alignment = intervals[second_position]

                if _intervals_overlap(first=first_bounds, second=second_bounds):
                    severity = (
                        report.errors if first_alignment != second_alignment else report.warnings
                    )
                    severity.append(
                        f"{BLOC_ALIGNMENT_FILE}: bloc '{bloc_name}' has overlapping alignment "
                        f"intervals at lines {first_row_index} and {second_row_index}."
                    )


def _load_table(reference_dir: Path, file_name: str) -> list[dict[str, str]]:
    path = reference_dir / file_name

    if not path.is_file():
        raise ReferenceDataError(f"Missing required reference file: {path}")

    return _read_rows(path)


def validate_reference_data(reference_dir: Path) -> ValidationReport:
    """Validate every reference table and return a structural/review report."""
    report = ValidationReport()

    tables: dict[str, list[dict[str, str]]] = {
        file_name: _load_table(reference_dir, file_name) for file_name in REQUIRED_COLUMNS
    }

    for file_name, rows in tables.items():
        _check_columns(file_name=file_name, rows=rows, report=report)
        _check_required_non_empty(file_name=file_name, rows=rows, report=report)
        _check_dates(file_name=file_name, rows=rows, report=report)
        _check_intervals(file_name=file_name, rows=rows, report=report)
        _check_enums(file_name=file_name, rows=rows, report=report)
        _check_duplicate_rows(file_name=file_name, rows=rows, report=report)

    _check_ambiguous_alignment_not_confident(rows=tables[BLOC_ALIGNMENT_FILE], report=report)

    known_source_ids = {
        row["source_id"].strip() for row in tables[SOURCES_FILE] if row.get("source_id", "").strip()
    }
    known_legislator_ids = {
        row["legislator_id"].strip()
        for row in tables[LEGISLATORS_FILE]
        if row.get("legislator_id", "").strip()
    }

    for file_name in SOURCE_ID_REFERENCE_FILES:
        _check_source_references(
            file_name=file_name,
            rows=tables[file_name],
            known_source_ids=known_source_ids,
            report=report,
        )

    for file_name in LEGISLATOR_ID_REFERENCE_COLUMNS:
        _check_legislator_references(
            file_name=file_name,
            rows=tables[file_name],
            known_legislator_ids=known_legislator_ids,
            report=report,
        )

    _check_overlapping_membership(rows=tables[BLOC_MEMBERSHIP_FILE], report=report)
    _check_overlapping_alignment(rows=tables[BLOC_ALIGNMENT_FILE], report=report)

    return report


def _print_report(report: ValidationReport) -> None:
    for error in report.errors:
        print(f"ERROR: {error}", file=sys.stderr)

    for warning in report.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    print(
        f"{len(report.errors)} error(s), {len(report.warnings)} warning(s).",
        file=sys.stderr,
    )


def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DEFAULT_REFERENCE_DIR,
        help="Directory containing the reference CSV files.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    """Run the validator as a script and return a process exit code."""
    args = _parse_args(argv)

    try:
        report = validate_reference_data(args.reference_dir)
    except ReferenceDataError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    _print_report(report)

    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
