import json
from pathlib import Path
from typing import Any

import pytest

from argentine_deputies_discursive_distance.structure_batch import (
    StructureBatchError,
    run_structure_batch,
)
from argentine_deputies_discursive_distance.structure_pipeline import (
    StructurePipelineError,
)


def write_pdf_summary(
    *,
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.write_text(
        json.dumps(
            {
                "batch_version": "1",
                "records": records,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def fake_structure_result(
    *,
    output_root: Path,
    source_record_id: str,
    reused: bool,
) -> dict[str, Any]:
    output_directory = output_root / source_record_id

    return {
        "reused": reused,
        "boundaries": {
            "start_anchor": {"reference": "p2:b1"},
            "end_anchor": {"reference": "p9:b4"},
            "post_start_anchor": {"reference": "p9:b5"},
            "end_method": ("explicit_closing"),
        },
        "outputs": {
            "structural_blocks_path": str(output_directory / "structural_blocks.jsonl"),
            "structural_pages_path": str(output_directory / "structural_pages.jsonl"),
        },
        "statistics": {
            "page_count": 9,
            "block_count": 100,
            "included_block_count": 80,
            "excluded_block_count": 20,
            "included_word_count": 5000,
            "structural_zone_counts": {
                "front_matter": 10,
                "proceedings": 85,
                "post_proceedings": 5,
            },
            "content_role_counts": {
                "transcript": 80,
                "procedural": 10,
                "other": 10,
            },
            "mixed_structural_zone_page_numbers": [
                2,
                9,
            ],
            "mixed_content_role_page_numbers": [
                2,
                9,
            ],
            "minimum_classification_confidence": (0.8),
        },
    }


def test_run_structure_batch_preserves_source_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_summary_path = tmp_path / "pdf_summary.json"
    output_root = tmp_path / "structure"
    summary_path = tmp_path / "structure_summary.json"

    write_pdf_summary(
        path=pdf_summary_path,
        records=[
            {
                "label": "second",
                "source_record_id": "record-2",
                "document_path": ("record-2.json"),
                "session_date": ("2020-01-02"),
            },
            {
                "label": "first",
                "source_record_id": "record-1",
                "document_path": ("record-1.json"),
                "session_date": ("2020-01-01"),
            },
        ],
    )

    calls = []

    def fake_segment_document(
        *,
        document_path: Path,
        output_root: Path,
        force: bool = False,
    ) -> dict[str, Any]:
        source_record_id = document_path.stem
        calls.append(
            (
                source_record_id,
                output_root,
                force,
            )
        )

        return fake_structure_result(
            output_root=output_root,
            source_record_id=(source_record_id),
            reused=(source_record_id == "record-2"),
        )

    monkeypatch.setattr(
        ("argentine_deputies_discursive_distance.structure_batch.segment_document"),
        fake_segment_document,
    )

    summary = run_structure_batch(
        pdf_summary_path=(pdf_summary_path),
        output_root=output_root,
        summary_path=summary_path,
        force=True,
    )

    assert [record["label"] for record in summary["records"]] == [
        "second",
        "first",
    ]
    assert summary["segmentation_reused_count"] == 1
    assert calls == [
        (
            "record-2",
            output_root,
            True,
        ),
        (
            "record-1",
            output_root,
            True,
        ),
    ]

    persisted = json.loads(summary_path.read_text(encoding="utf-8"))
    assert persisted == summary


def test_run_structure_batch_rejects_duplicate_record_ids(
    tmp_path: Path,
) -> None:
    pdf_summary_path = tmp_path / "pdf_summary.json"

    write_pdf_summary(
        path=pdf_summary_path,
        records=[
            {
                "label": "one",
                "source_record_id": ("duplicate"),
                "document_path": ("one.json"),
            },
            {
                "label": "two",
                "source_record_id": ("duplicate"),
                "document_path": ("two.json"),
            },
        ],
    )

    with pytest.raises(
        StructureBatchError,
        match=("Duplicate PDF batch source_record_id"),
    ):
        run_structure_batch(
            pdf_summary_path=(pdf_summary_path),
            output_root=(tmp_path / "structure"),
            summary_path=(tmp_path / "summary.json"),
        )


def test_run_structure_batch_wraps_pipeline_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_summary_path = tmp_path / "pdf_summary.json"

    write_pdf_summary(
        path=pdf_summary_path,
        records=[
            {
                "label": "pilot",
                "source_record_id": ("record-1"),
                "document_path": ("record-1.json"),
            }
        ],
    )

    def fail_segment_document(
        *,
        document_path: Path,
        output_root: Path,
        force: bool = False,
    ) -> dict[str, Any]:
        del document_path
        del output_root
        del force

        raise StructurePipelineError("Synthetic failure.")

    monkeypatch.setattr(
        ("argentine_deputies_discursive_distance.structure_batch.segment_document"),
        fail_segment_document,
    )

    with pytest.raises(
        StructureBatchError,
        match=("Structural segmentation failed for pilot"),
    ):
        run_structure_batch(
            pdf_summary_path=(pdf_summary_path),
            output_root=(tmp_path / "structure"),
            summary_path=(tmp_path / "summary.json"),
        )
