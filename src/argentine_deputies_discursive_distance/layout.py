"""Coordinate-aware ordering of text blocks extracted from PDF pages."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class PageLayout(StrEnum):
    """Broad reading-layout classification for one PDF page."""

    SINGLE_COLUMN = "single_column"
    TWO_COLUMN = "two_column"
    MIXED = "mixed"


class BlockRegion(StrEnum):
    """Geometric region assigned to an extracted text block."""

    HEADER = "header"
    BODY_LEFT = "body_left"
    BODY_RIGHT = "body_right"
    BODY_FULL = "body_full"
    FOOTER = "footer"


@dataclass(frozen=True, slots=True)
class RawTextBlock:
    """One text block extracted from a PDF page before ordering."""

    page_number: int
    raw_block_number: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def width(self) -> float:
        """Return the block width."""
        return self.x1 - self.x0

    @property
    def center_x(self) -> float:
        """Return the horizontal center coordinate."""
        return (self.x0 + self.x1) / 2

    @property
    def center_y(self) -> float:
        """Return the vertical center coordinate."""
        return (self.y0 + self.y1) / 2

    @property
    def character_count(self) -> int:
        """Return the number of text characters."""
        return len(self.text)


@dataclass(frozen=True, slots=True)
class OrderedTextBlock:
    """A text block with an assigned region and reading position."""

    block: RawTextBlock
    reading_order: int
    region: BlockRegion


@dataclass(frozen=True, slots=True)
class OrderedPage:
    """Ordered text blocks and layout diagnostics for one page."""

    page_number: int
    width: float
    height: float
    layout: PageLayout
    image_block_count: int
    blocks: tuple[OrderedTextBlock, ...]

    @property
    def text(self) -> str:
        """Return page text in inferred reading order."""
        return "\n".join(ordered.block.text for ordered in self.blocks)


def normalize_extracted_text(value: str) -> str:
    """Normalize Unicode and horizontal whitespace while retaining lines."""
    normalized = unicodedata.normalize("NFKC", value)

    lines = []

    for raw_line in normalized.splitlines():
        line = re.sub(r"[ \t\f\v]+", " ", raw_line).strip()

        if line:
            lines.append(line)

    return "\n".join(lines)


def parse_pymupdf_blocks(
    *,
    raw_blocks: Sequence[Sequence[Any]],
    page_number: int,
) -> tuple[list[RawTextBlock], int]:
    """Convert PyMuPDF block tuples into typed text blocks."""
    text_blocks: list[RawTextBlock] = []
    image_block_count = 0

    for raw_block in raw_blocks:
        if len(raw_block) < 7:
            continue

        block_type = int(raw_block[6])

        if block_type == 1:
            image_block_count += 1
            continue

        if block_type != 0:
            continue

        text = normalize_extracted_text(str(raw_block[4]))

        if not text:
            continue

        text_blocks.append(
            RawTextBlock(
                page_number=page_number,
                raw_block_number=int(raw_block[5]),
                x0=float(raw_block[0]),
                y0=float(raw_block[1]),
                x1=float(raw_block[2]),
                y1=float(raw_block[3]),
                text=text,
            )
        )

    return text_blocks, image_block_count


def _vertical_key(block: RawTextBlock) -> tuple[float, float, int]:
    return (
        block.y0,
        block.x0,
        block.raw_block_number,
    )


def _region_for_block(
    *,
    block: RawTextBlock,
    page_width: float,
    page_height: float,
    header_ratio: float,
    footer_ratio: float,
) -> BlockRegion:
    if block.y1 <= page_height * header_ratio:
        return BlockRegion.HEADER

    if block.y0 >= page_height * footer_ratio:
        return BlockRegion.FOOTER

    page_center = page_width / 2
    crosses_center = block.x0 < page_center < block.x1
    center_offset = abs(block.center_x - page_center)

    is_wide_spanning_block = crosses_center and block.width >= page_width * 0.35
    is_centered_section_heading = (
        center_offset <= page_width * 0.03 and block.width >= page_width * 0.08
    )

    if is_wide_spanning_block or is_centered_section_heading:
        return BlockRegion.BODY_FULL

    if block.center_x < page_center:
        return BlockRegion.BODY_LEFT

    return BlockRegion.BODY_RIGHT


def classify_page_layout(
    *,
    blocks: Sequence[RawTextBlock],
    page_width: float,
    page_height: float,
    header_ratio: float = 0.09,
    footer_ratio: float = 0.93,
) -> PageLayout:
    """Classify a page using the geometry and text mass of body blocks."""
    body_blocks = [
        block
        for block in blocks
        if _region_for_block(
            block=block,
            page_width=page_width,
            page_height=page_height,
            header_ratio=header_ratio,
            footer_ratio=footer_ratio,
        )
        not in {
            BlockRegion.HEADER,
            BlockRegion.FOOTER,
        }
    ]

    left_blocks = []
    right_blocks = []
    full_width_blocks = []

    for block in body_blocks:
        region = _region_for_block(
            block=block,
            page_width=page_width,
            page_height=page_height,
            header_ratio=header_ratio,
            footer_ratio=footer_ratio,
        )

        if region == BlockRegion.BODY_LEFT:
            left_blocks.append(block)
        elif region == BlockRegion.BODY_RIGHT:
            right_blocks.append(block)
        else:
            full_width_blocks.append(block)

    left_characters = sum(block.character_count for block in left_blocks)
    right_characters = sum(block.character_count for block in right_blocks)

    has_two_substantive_sides = left_characters >= 50 and right_characters >= 50

    if not has_two_substantive_sides:
        return PageLayout.SINGLE_COLUMN

    if full_width_blocks:
        return PageLayout.MIXED

    return PageLayout.TWO_COLUMN


def _order_side_segment(
    *,
    blocks: Sequence[RawTextBlock],
    page_width: float,
) -> list[RawTextBlock]:
    page_center = page_width / 2

    left = sorted(
        (block for block in blocks if block.center_x < page_center),
        key=_vertical_key,
    )
    right = sorted(
        (block for block in blocks if block.center_x >= page_center),
        key=_vertical_key,
    )

    return [*left, *right]


def order_page_blocks(
    *,
    blocks: Sequence[RawTextBlock],
    page_width: float,
    page_height: float,
    header_ratio: float = 0.09,
    footer_ratio: float = 0.93,
) -> OrderedPage:
    """Infer reading order while respecting multi-column layouts."""
    if not blocks:
        page_number = 1
    else:
        page_numbers = {block.page_number for block in blocks}

        if len(page_numbers) != 1:
            raise ValueError("All blocks must belong to the same page.")

        page_number = next(iter(page_numbers))

    regions = {
        block: _region_for_block(
            block=block,
            page_width=page_width,
            page_height=page_height,
            header_ratio=header_ratio,
            footer_ratio=footer_ratio,
        )
        for block in blocks
    }

    headers = sorted(
        (block for block in blocks if regions[block] == BlockRegion.HEADER),
        key=_vertical_key,
    )
    footers = sorted(
        (block for block in blocks if regions[block] == BlockRegion.FOOTER),
        key=_vertical_key,
    )
    body = [
        block
        for block in blocks
        if regions[block]
        not in {
            BlockRegion.HEADER,
            BlockRegion.FOOTER,
        }
    ]

    layout = classify_page_layout(
        blocks=blocks,
        page_width=page_width,
        page_height=page_height,
        header_ratio=header_ratio,
        footer_ratio=footer_ratio,
    )

    ordered_body: list[RawTextBlock]

    if layout == PageLayout.SINGLE_COLUMN:
        ordered_body = sorted(
            body,
            key=_vertical_key,
        )
    else:
        full_width = sorted(
            (block for block in body if regions[block] == BlockRegion.BODY_FULL),
            key=_vertical_key,
        )
        side_blocks = [
            block
            for block in body
            if regions[block]
            in {
                BlockRegion.BODY_LEFT,
                BlockRegion.BODY_RIGHT,
            }
        ]

        ordered_body = []
        remaining = list(side_blocks)
        previous_separator_y = float("-inf")

        for separator in full_width:
            separator_y = separator.center_y

            segment = [
                block for block in remaining if previous_separator_y < block.center_y < separator_y
            ]

            ordered_body.extend(
                _order_side_segment(
                    blocks=segment,
                    page_width=page_width,
                )
            )
            ordered_body.append(separator)

            segment_set = set(segment)
            remaining = [block for block in remaining if block not in segment_set]
            previous_separator_y = separator_y

        ordered_body.extend(
            _order_side_segment(
                blocks=remaining,
                page_width=page_width,
            )
        )

    complete_order = [
        *headers,
        *ordered_body,
        *footers,
    ]

    ordered_blocks = tuple(
        OrderedTextBlock(
            block=block,
            reading_order=index,
            region=regions[block],
        )
        for index, block in enumerate(
            complete_order,
            start=1,
        )
    )

    return OrderedPage(
        page_number=page_number,
        width=page_width,
        height=page_height,
        layout=layout,
        image_block_count=0,
        blocks=ordered_blocks,
    )


def extract_ordered_page(
    *,
    raw_blocks: Sequence[Sequence[Any]],
    page_number: int,
    page_width: float,
    page_height: float,
) -> OrderedPage:
    """Convert PyMuPDF output directly into an ordered page."""
    text_blocks, image_block_count = parse_pymupdf_blocks(
        raw_blocks=raw_blocks,
        page_number=page_number,
    )

    ordered_page = order_page_blocks(
        blocks=text_blocks,
        page_width=page_width,
        page_height=page_height,
    )

    return OrderedPage(
        page_number=ordered_page.page_number,
        width=ordered_page.width,
        height=ordered_page.height,
        layout=ordered_page.layout,
        image_block_count=image_block_count,
        blocks=ordered_page.blocks,
    )
