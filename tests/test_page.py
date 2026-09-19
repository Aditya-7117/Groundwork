"""Rebuilding a Wikipedia page from Natural Questions' token stream.

Natural Questions gives each page as a list of tokens: words and HTML tags, each with the byte
positions it occupied in the original HTML. Spacing is recovered from those positions, so tokens
that sat next to each other ("Ireland" then ".") join without a space.

The helper below writes token streams compactly: tokens are separated by spaces, and a leading "~"
means the token touched the previous one with no whitespace between them.
"""

from collections.abc import Collection

import pytest

from groundwork.page import BOILERPLATE_SECTIONS, BuiltPage, TokenStream, build_page

LEAD = ""


def tokens(spec: str) -> TokenStream:
    """Turn a compact spec into a token stream."""
    words, is_html, starts, ends = [], [], [], []
    position = 0
    for index, item in enumerate(spec.split()):
        glued = item.startswith("~")
        token = item.removeprefix("~")
        if index and not glued:
            position += 1
        words.append(token)
        is_html.append(token.startswith("<"))
        starts.append(position)
        position += len(token.encode("utf-8"))
        ends.append(position)
    return TokenStream(tokens=words, is_html=is_html, start_bytes=starts, end_bytes=ends)


def page(spec: str, drop_sections: Collection[str] = BOILERPLATE_SECTIONS) -> BuiltPage:
    return build_page(
        page_id="p1",
        title="Dublin",
        url="https://en.wikipedia.org/wiki/Dublin",
        stream=tokens(spec),
        drop_sections=drop_sections,
    )


SIMPLE = "<P> Dublin is in Ireland ~. </P> <H2> History </H2> <P> Founded in 841 ~. </P>"


class TestBlocks:
    def test_paragraphs_and_headings_become_blocks(self) -> None:
        built = page(SIMPLE)
        assert [(b.kind, b.section, b.text) for b in built.page.blocks] == [
            ("paragraph", LEAD, "Dublin is in Ireland."),
            ("heading", "History", "History"),
            ("paragraph", "History", "Founded in 841."),
        ]

    def test_block_offsets_point_into_the_page_text(self) -> None:
        built = page(SIMPLE)
        for block in built.page.blocks:
            assert built.page.text[block.start : block.end] == block.text

    def test_punctuation_joins_without_a_space(self) -> None:
        assert "Ireland." in page(SIMPLE).page.text

    def test_list_items_are_their_own_blocks(self) -> None:
        built = page("<Ul> <Li> First item </Li> <Li> Second item </Li> </Ul>")
        assert [(b.kind, b.text) for b in built.page.blocks] == [
            ("list_item", "First item"),
            ("list_item", "Second item"),
        ]

    def test_pages_without_any_text_have_no_blocks(self) -> None:
        assert page("<P> </P>").page.blocks == ()


class TestTables:
    TABLE = (
        "<Table> <Caption> Population </Caption> "
        "<Tr> <Th> Year </Th> <Th> People </Th> </Tr> "
        "<Tr> <Td> 1841 </Td> <Td> 232,726 </Td> </Tr> "
        "<Tr> <Td> 1901 </Td> <Td> 290,638 </Td> </Tr> </Table>"
    )

    def test_a_table_is_one_block_with_its_rows_kept(self) -> None:
        built = page(self.TABLE)
        [block] = built.page.blocks
        assert block.kind == "table"
        assert block.caption == "Population"
        assert block.header == ("Year", "People")
        assert block.rows == (("1841", "232,726"), ("1901", "290,638"))

    def test_rows_are_serialised_with_their_column_names(self) -> None:
        # Decision 40: every value carries its column, so "1841" cannot be read as a population.
        [block] = page(self.TABLE).page.blocks
        assert block.text == "Year: 1841 | People: 232,726\nYear: 1901 | People: 290,638"

    def test_a_table_without_a_header_row_keeps_bare_values(self) -> None:
        built = page("<Table> <Tr> <Td> a </Td> <Td> b </Td> </Tr> </Table>")
        [block] = built.page.blocks
        assert block.header == ()
        assert block.text == "a | b"

    def test_nested_tables_do_not_start_a_second_block(self) -> None:
        built = page(
            "<Table> <Tr> <Td> outer </Td> </Tr> "
            "<Table> <Tr> <Td> inner </Td> </Tr> </Table> </Table>"
        )
        assert len(built.page.blocks) == 1
        assert built.page.blocks[0].rows == (("outer",), ("inner",))


class TestSections:
    def test_boilerplate_sections_are_dropped(self) -> None:
        built = page("<P> Body ~. </P> <H2> References </H2> <P> A citation </P>")
        assert [b.text for b in built.page.blocks] == ["Body."]
        assert "References" in BOILERPLATE_SECTIONS

    def test_dropping_can_be_turned_off(self) -> None:
        built = page("<P> Body ~. </P> <H2> References </H2> <P> A citation </P>", drop_sections=())
        assert [b.text for b in built.page.blocks] == ["Body.", "References", "A citation"]

    def test_subsection_headings_become_the_current_section(self) -> None:
        built = page("<H2> History </H2> <H3> Medieval </H3> <P> Founded ~. </P>")
        assert built.page.blocks[-1].section == "Medieval"


class TestAnswerSpans:
    def test_a_token_range_maps_to_characters_in_the_page_text(self) -> None:
        built = page(SIMPLE)
        # Tokens 9-14 are "<P> Founded in 841 . </P>", the second paragraph.
        span = built.char_span(9, 15)
        assert span is not None
        assert built.page.text[span[0] : span[1]] == "Founded in 841."

    def test_a_span_inside_a_dropped_section_has_no_mapping(self) -> None:
        built = page("<P> Body ~. </P> <H2> References </H2> <P> A citation </P>")
        assert built.char_span(8, 11) is None

    @pytest.mark.parametrize(("start", "end"), [(-1, 3), (0, 99), (5, 2)])
    def test_invalid_token_ranges_are_rejected(self, start: int, end: int) -> None:
        with pytest.raises(ValueError, match="token range"):
            page(SIMPLE).char_span(start, end)


class TestTagsWithAttributes:
    """Real tags carry attributes, so the token is `<Td colspan="6">`, not `<Td>`."""

    def stream(self, items: list[str]) -> TokenStream:
        starts, ends, position = [], [], 0
        for index, token in enumerate(items):
            if index:
                position += 1
            starts.append(position)
            position += len(token.encode("utf-8"))
            ends.append(position)
        return TokenStream(
            tokens=items,
            is_html=[t.startswith("<") for t in items],
            start_bytes=starts,
            end_bytes=ends,
        )

    def build(self, items: list[str]) -> BuiltPage:
        return build_page(page_id="p1", title="T", url="u", stream=self.stream(items))

    def test_cells_with_attributes_are_read(self) -> None:
        built = self.build(
            [
                '<Table class="wikitable">',
                "<Tr>",
                '<Td colspan="6">',
                "Benny",
                "dies",
                "</Td>",
                "</Tr>",
                "</Table>",
            ]
        )
        assert [b.text for b in built.page.blocks] == ["Benny dies"]
        span = built.char_span(3, 5)
        assert span is not None
        assert built.page.text[span[0] : span[1]] == "Benny dies"

    def test_paragraphs_and_headings_with_attributes_are_read(self) -> None:
        built = self.build(['<H2 id="s">', "History", "</H2>", '<P class="x">', "Text", "</P>"])
        assert [(b.kind, b.section, b.text) for b in built.page.blocks] == [
            ("heading", "History", "History"),
            ("paragraph", "History", "Text"),
        ]

    def test_a_table_of_only_header_cells_is_kept_as_data(self) -> None:
        # Some Wikipedia tables mark every cell as a header. Treating them as a header row with
        # no data would drop the table, and with it any answer inside it.
        built = self.build(
            [
                "<Table>",
                "<Tr>",
                "<Th>",
                "Alpha",
                "</Th>",
                "<Th>",
                "Beta",
                "</Th>",
                "</Tr>",
                "</Table>",
            ]
        )
        assert [b.text for b in built.page.blocks] == ["Alpha | Beta"]
        span = built.char_span(3, 4)
        assert span is not None
        assert built.page.text[span[0] : span[1]] == "Alpha"


def test_content_tokens_counts_words_not_tags() -> None:
    built = page(SIMPLE)
    assert built.content_tokens == 10  # 5 words + 1 heading + 4 words, excluding every tag
    assert sum(1 for span in built.token_spans if span is not None) == built.content_tokens
