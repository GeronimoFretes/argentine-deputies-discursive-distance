import json
from pathlib import Path
from typing import Any

from argentine_deputies_discursive_distance.pdf_pipeline import (
    sha256_file,
)
from argentine_deputies_discursive_distance.structure_pipeline import (
    segment_document,
)


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for record in records:
            output_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            output_file.write("\n")


def read_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def build_extraction_bundle(
    tmp_path: Path,
    *,
    source_record_id: str = "record-1",
) -> tuple[
    Path,
    Path,
    Path,
]:
    extraction_directory = tmp_path / "extraction" / source_record_id
    extraction_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    pages_path = extraction_directory / "pages.jsonl"
    blocks_path = extraction_directory / "blocks.jsonl"
    document_path = extraction_directory / "document.json"

    pages = [
        {
            "source_record_id": (source_record_id),
            "page_number": 1,
            "width": 500,
            "height": 700,
            "rotation": 0,
            "layout": "single_column",
            "image_block_count": 0,
            "text_block_count": 1,
            "character_count": 7,
            "word_count": 1,
            "text": "SUMARIO",
        },
        {
            "source_record_id": (source_record_id),
            "page_number": 2,
            "width": 500,
            "height": 700,
            "rotation": 0,
            "layout": "single_column",
            "image_block_count": 0,
            "text_block_count": 4,
            "character_count": 180,
            "word_count": 30,
            "text": (
                "Sr. Presidente. – "
                "Queda abierta la sesión.\n"
                "Sr. Diputado. – Intervención.\n"
                "Queda levantada la sesión.\n"
                "–Es la hora 18."
            ),
        },
    ]

    blocks = [
        {
            "source_record_id": (source_record_id),
            "page_number": 1,
            "reading_order": 1,
            "region": "body_full",
            "raw_block_number": 0,
            "x0": 50,
            "y0": 100,
            "x1": 450,
            "y1": 150,
            "text": "SUMARIO",
        },
        {
            "source_record_id": (source_record_id),
            "page_number": 2,
            "reading_order": 1,
            "region": "body_full",
            "raw_block_number": 0,
            "x0": 50,
            "y0": 100,
            "x1": 450,
            "y1": 180,
            "text": ("Sr. Presidente. – Queda abierta la sesión."),
        },
        {
            "source_record_id": (source_record_id),
            "page_number": 2,
            "reading_order": 2,
            "region": "body_full",
            "raw_block_number": 1,
            "x0": 50,
            "y0": 190,
            "x1": 450,
            "y1": 300,
            "text": ("Sr. Diputado. – Intervención sustantiva."),
        },
        {
            "source_record_id": (source_record_id),
            "page_number": 2,
            "reading_order": 3,
            "region": "body_full",
            "raw_block_number": 2,
            "x0": 50,
            "y0": 310,
            "x1": 450,
            "y1": 380,
            "text": ("Queda levantada la sesión."),
        },
        {
            "source_record_id": (source_record_id),
            "page_number": 2,
            "reading_order": 4,
            "region": "body_full",
            "raw_block_number": 3,
            "x0": 50,
            "y0": 390,
            "x1": 450,
            "y1": 430,
            "text": "–Es la hora 18.",
        },
    ]

    write_jsonl(
        pages_path,
        pages,
    )
    write_jsonl(
        blocks_path,
        blocks,
    )

    document = {
        "extractor_version": "1",
        "source_record_id": (source_record_id),
        "pdf": {
            "path": "unused.pdf",
            "sha256": "pdf-sha",
            "size_bytes": 100,
            "page_count": 2,
            "metadata": {},
        },
        "extraction": {
            "pages_path": str(pages_path),
            "pages_sha256": (sha256_file(pages_path)),
            "pages_size_bytes": (pages_path.stat().st_size),
            "blocks_path": str(blocks_path),
            "blocks_sha256": (sha256_file(blocks_path)),
            "blocks_size_bytes": (blocks_path.stat().st_size),
            "total_text_blocks": 5,
            "total_characters": 187,
            "total_words": 31,
            "layout_counts": {"single_column": 2},
            "empty_page_numbers": [],
            "low_text_page_numbers": [],
        },
    }

    document_path.write_text(
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return (
        document_path,
        pages_path,
        blocks_path,
    )


def test_segment_document_writes_block_and_page_outputs(
    tmp_path: Path,
) -> None:
    (
        document_path,
        _,
        _,
    ) = build_extraction_bundle(tmp_path)

    result = segment_document(
        document_path=document_path,
        output_root=(tmp_path / "structure"),
    )

    assert result["reused"] is False
    assert result["boundaries"]["start_anchor"]["reference"] == "p2:b1"
    assert result["boundaries"]["end_anchor"]["reference"] == "p2:b3"
    assert result["boundaries"]["post_start_anchor"]["reference"] == "p2:b4"

    structural_blocks_path = Path(result["outputs"]["structural_blocks_path"])
    structural_pages_path = Path(result["outputs"]["structural_pages_path"])

    blocks = read_jsonl(structural_blocks_path)
    pages = read_jsonl(structural_pages_path)

    blocks_by_reference = {
        (f"p{block['page_number']}:b{block['reading_order']}"): block for block in blocks
    }
    pages_by_number = {int(page["page_number"]): page for page in pages}

    assert blocks_by_reference["p2:b2"]["include_in_discourse"] is True
    assert blocks_by_reference["p2:b2"]["structural_zone"] == "proceedings"
    assert blocks_by_reference["p2:b4"]["structural_zone"] == "post_proceedings"
    assert blocks_by_reference["p2:b4"]["include_in_discourse"] is False
    assert pages_by_number[2]["mixed_structural_zone"] is True
    assert pages_by_number[2]["included_block_count"] == 1


def test_segment_document_reuses_valid_outputs(
    tmp_path: Path,
) -> None:
    (
        document_path,
        _,
        _,
    ) = build_extraction_bundle(tmp_path)
    output_root = tmp_path / "structure"

    first = segment_document(
        document_path=document_path,
        output_root=output_root,
    )

    structural_blocks_path = Path(first["outputs"]["structural_blocks_path"])
    structural_pages_path = Path(first["outputs"]["structural_pages_path"])

    before_hashes = {
        "blocks": sha256_file(structural_blocks_path),
        "pages": sha256_file(structural_pages_path),
    }

    second = segment_document(
        document_path=document_path,
        output_root=output_root,
    )

    assert second["reused"] is True
    assert before_hashes == {
        "blocks": sha256_file(structural_blocks_path),
        "pages": sha256_file(structural_pages_path),
    }


def test_segment_document_rebuilds_corrupted_output(
    tmp_path: Path,
) -> None:
    (
        document_path,
        _,
        _,
    ) = build_extraction_bundle(tmp_path)
    output_root = tmp_path / "structure"

    first = segment_document(
        document_path=document_path,
        output_root=output_root,
    )

    structural_blocks_path = Path(first["outputs"]["structural_blocks_path"])
    structural_blocks_path.write_text(
        "corrupted\n",
        encoding="utf-8",
    )

    second = segment_document(
        document_path=document_path,
        output_root=output_root,
    )

    assert second["reused"] is False

    rebuilt = read_jsonl(structural_blocks_path)

    assert len(rebuilt) == 5
    assert rebuilt[2]["include_in_discourse"] is True


def test_segment_document_invalidates_cache_when_source_changes(
    tmp_path: Path,
) -> None:
    (
        document_path,
        _,
        blocks_path,
    ) = build_extraction_bundle(tmp_path)
    output_root = tmp_path / "structure"

    first = segment_document(
        document_path=document_path,
        output_root=output_root,
    )

    source_blocks = read_jsonl(blocks_path)
    source_blocks.append(
        {
            "source_record_id": ("record-1"),
            "page_number": 2,
            "reading_order": 5,
            "region": "body_full",
            "raw_block_number": 4,
            "x0": 50,
            "y0": 440,
            "x1": 450,
            "y1": 480,
            "text": "Final metadata.",
        }
    )
    write_jsonl(
        blocks_path,
        source_blocks,
    )

    document = json.loads(document_path.read_text(encoding="utf-8"))
    document["extraction"]["blocks_sha256"] = sha256_file(blocks_path)
    document["extraction"]["blocks_size_bytes"] = blocks_path.stat().st_size
    document["extraction"]["total_text_blocks"] = 6

    document_path.write_text(
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    second = segment_document(
        document_path=document_path,
        output_root=output_root,
    )

    assert first["reused"] is False
    assert second["reused"] is False
    assert second["statistics"]["block_count"] == 6
