from argentine_deputies_discursive_distance.layout import (
    BlockRegion,
    PageLayout,
    RawTextBlock,
    classify_page_layout,
    normalize_extracted_text,
    order_page_blocks,
)


def make_block(
    *,
    number: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    text: str,
) -> RawTextBlock:
    return RawTextBlock(
        page_number=1,
        raw_block_number=number,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        text=text,
    )


def test_normalize_extracted_text_preserves_line_boundaries() -> None:
    value = " First   line \n\n Second\tline "

    assert normalize_extracted_text(value) == ("First line\nSecond line")


def test_two_column_page_reads_complete_left_column_first() -> None:
    blocks = [
        make_block(
            number=0,
            x0=36,
            y0=40,
            x1=460,
            y1=52,
            text="Page header",
        ),
        make_block(
            number=1,
            x0=36,
            y0=58,
            x1=243,
            y1=570,
            text=("Left column upper contains a sufficiently long parliamentary paragraph."),
        ),
        make_block(
            number=2,
            x0=52,
            y0=592,
            x1=224,
            y1=603,
            text="Left column heading",
        ),
        make_block(
            number=3,
            x0=36,
            y0=609,
            x1=243,
            y1=656,
            text=(
                "Left column lower continues the substantive intervention "
                "before the right column begins."
            ),
        ),
        make_block(
            number=4,
            x0=252,
            y0=58,
            x1=459,
            y1=617,
            text=("Right column body contains another sufficiently long parliamentary paragraph."),
        ),
        make_block(
            number=5,
            x0=297,
            y0=620,
            x1=408,
            y1=631,
            text="Closing time",
        ),
    ]

    ordered = order_page_blocks(
        blocks=blocks,
        page_width=495,
        page_height=700,
    )

    assert ordered.layout == PageLayout.TWO_COLUMN
    assert [item.block.text for item in ordered.blocks] == [
        "Page header",
        ("Left column upper contains a sufficiently long parliamentary paragraph."),
        "Left column heading",
        (
            "Left column lower continues the substantive intervention "
            "before the right column begins."
        ),
        ("Right column body contains another sufficiently long parliamentary paragraph."),
        "Closing time",
    ]


def test_single_column_page_uses_vertical_order() -> None:
    blocks = [
        make_block(
            number=2,
            x0=70,
            y0=300,
            x1=430,
            y1=350,
            text="Second paragraph",
        ),
        make_block(
            number=1,
            x0=70,
            y0=100,
            x1=430,
            y1=180,
            text="First paragraph",
        ),
    ]

    ordered = order_page_blocks(
        blocks=blocks,
        page_width=500,
        page_height=700,
    )

    assert ordered.layout == PageLayout.SINGLE_COLUMN
    assert [item.block.text for item in ordered.blocks] == [
        "First paragraph",
        "Second paragraph",
    ]


def test_full_width_separator_divides_column_zones() -> None:
    blocks = [
        make_block(
            number=1,
            x0=30,
            y0=100,
            x1=235,
            y1=180,
            text=("Top left contains enough substantive text to represent a genuine left column."),
        ),
        make_block(
            number=2,
            x0=265,
            y0=100,
            x1=470,
            y1=180,
            text=(
                "Top right contains enough substantive text to represent a genuine right column."
            ),
        ),
        make_block(
            number=3,
            x0=100,
            y0=220,
            x1=400,
            y1=250,
            text="Full-width section heading",
        ),
        make_block(
            number=4,
            x0=30,
            y0=280,
            x1=235,
            y1=360,
            text=("Bottom left contains enough substantive text after the full-width separator."),
        ),
        make_block(
            number=5,
            x0=265,
            y0=280,
            x1=470,
            y1=360,
            text=("Bottom right contains enough substantive text after the full-width separator."),
        ),
    ]

    ordered = order_page_blocks(
        blocks=blocks,
        page_width=500,
        page_height=700,
    )

    assert ordered.layout == PageLayout.MIXED
    assert [item.block.text for item in ordered.blocks] == [
        ("Top left contains enough substantive text to represent a genuine left column."),
        ("Top right contains enough substantive text to represent a genuine right column."),
        "Full-width section heading",
        ("Bottom left contains enough substantive text after the full-width separator."),
        ("Bottom right contains enough substantive text after the full-width separator."),
    ]


def test_header_and_footer_regions_are_preserved() -> None:
    blocks = [
        make_block(
            number=1,
            x0=20,
            y0=20,
            x1=480,
            y1=40,
            text="Header",
        ),
        make_block(
            number=2,
            x0=70,
            y0=100,
            x1=430,
            y1=500,
            text="Body",
        ),
        make_block(
            number=3,
            x0=400,
            y0=670,
            x1=480,
            y1=690,
            text="Footer",
        ),
    ]

    ordered = order_page_blocks(
        blocks=blocks,
        page_width=500,
        page_height=700,
    )

    assert [item.region for item in ordered.blocks] == [
        BlockRegion.HEADER,
        BlockRegion.BODY_FULL,
        BlockRegion.FOOTER,
    ]


def test_layout_requires_text_on_both_sides() -> None:
    blocks = [
        make_block(
            number=1,
            x0=30,
            y0=100,
            x1=230,
            y1=400,
            text="A substantial left-column paragraph " * 5,
        ),
        make_block(
            number=2,
            x0=270,
            y0=100,
            x1=470,
            y1=120,
            text="Tiny",
        ),
    ]

    layout = classify_page_layout(
        blocks=blocks,
        page_width=500,
        page_height=700,
    )

    assert layout == PageLayout.SINGLE_COLUMN


def test_narrow_centered_heading_separates_column_zones() -> None:
    blocks = [
        make_block(
            number=1,
            x0=30,
            y0=100,
            x1=235,
            y1=180,
            text=(
                "Upper left contains enough substantive text to represent "
                "the first column before the centered heading."
            ),
        ),
        make_block(
            number=2,
            x0=265,
            y0=100,
            x1=470,
            y1=180,
            text=(
                "Upper right contains enough substantive text to represent "
                "the second column before the centered heading."
            ),
        ),
        make_block(
            number=3,
            x0=225,
            y0=220,
            x1=275,
            y1=245,
            text="APPENDIX",
        ),
        make_block(
            number=4,
            x0=30,
            y0=280,
            x1=235,
            y1=360,
            text=(
                "Lower left contains enough substantive text after the centered section heading."
            ),
        ),
        make_block(
            number=5,
            x0=265,
            y0=280,
            x1=470,
            y1=360,
            text=(
                "Lower right contains enough substantive text after the centered section heading."
            ),
        ),
    ]

    ordered = order_page_blocks(
        blocks=blocks,
        page_width=500,
        page_height=700,
    )

    assert ordered.layout == PageLayout.MIXED
    assert [item.block.text for item in ordered.blocks] == [
        (
            "Upper left contains enough substantive text to represent "
            "the first column before the centered heading."
        ),
        (
            "Upper right contains enough substantive text to represent "
            "the second column before the centered heading."
        ),
        "APPENDIX",
        ("Lower left contains enough substantive text after the centered section heading."),
        ("Lower right contains enough substantive text after the centered section heading."),
    ]
