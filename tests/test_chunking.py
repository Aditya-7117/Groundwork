"""The three chunking strategies, and how tables are chunked.

Sizes here are deliberately tiny (a handful of words) so every expected chunk can be read at a
glance. The real runs use about 150 words with 30 words of overlap.
"""

import pytest

from groundwork.chunking import ChunkingSettings, chunk_page, overlapping_chunks
from groundwork.page import Block, Page


def page_of(*blocks: Block, title: str = "Dublin") -> Page:
    text, placed, position = "", [], 0
    for block in blocks:
        if text:
            text += "\n\n"
            position += 2
        placed.append(
            Block(
                kind=block.kind,
                section=block.section,
                text=block.text,
                start=position,
                end=position + len(block.text),
                caption=block.caption,
                header=block.header,
                rows=block.rows,
            )
        )
        text += block.text
        position += len(block.text)
    return Page(page_id="p1", title=title, url="u", text=text, blocks=tuple(placed))


def prose(text: str, section: str = "") -> Block:
    return Block(kind="paragraph", section=section, text=text, start=0, end=len(text))


def table(
    rows: tuple[tuple[str, ...], ...], header: tuple[str, ...], section: str = "", caption: str = ""
) -> Block:
    text = "\n".join(
        " | ".join(
            f"{header[i]}: {value}" if i < len(header) else value for i, value in enumerate(row)
        )
        for row in rows
    )
    return Block(
        kind="table",
        section=section,
        text=text,
        start=0,
        end=len(text),
        caption=caption,
        header=header,
        rows=rows,
    )


SMALL = ChunkingSettings(strategy="fixed_words", size=6, overlap=2)


class TestPrefix:
    def test_every_chunk_names_its_page_and_section(self) -> None:
        page = page_of(prose("alpha beta gamma delta", section="History"))
        [chunk] = chunk_page(page, SMALL)
        assert chunk.text == "Dublin > History\nalpha beta gamma delta"

    def test_the_opening_of_a_page_has_no_section(self) -> None:
        [chunk] = chunk_page(page_of(prose("alpha beta")), SMALL)
        assert chunk.text == "Dublin\nalpha beta"

    def test_the_prefix_is_not_counted_against_the_chunk_size(self) -> None:
        # Otherwise a long title would shrink every chunk on that page.
        page = page_of(prose("one two three four five six"))
        [chunk] = chunk_page(page, SMALL)
        assert chunk.text.endswith("one two three four five six")


class TestFixedWords:
    def test_windows_ignore_paragraph_boundaries(self) -> None:
        # Twelve words, size 6 and overlap 2, so each window starts 4 words after the last:
        # words 1-6, then 5-10, then 9-12. Paragraph ends are ignored, which is the point.
        page = page_of(prose("a1 a2 a3 a4"), prose("b1 b2 b3 b4"), prose("c1 c2 c3 c4"))
        bodies = [chunk.text.split("\n", 1)[1] for chunk in chunk_page(page, SMALL)]
        assert bodies == [
            "a1 a2 a3 a4\n\nb1 b2",
            "b1 b2 b3 b4\n\nc1 c2",
            "c1 c2 c3 c4",
        ]

    def test_chunk_offsets_point_at_the_page_text(self) -> None:
        page = page_of(prose("a1 a2 a3 a4"), prose("b1 b2 b3 b4"))
        for chunk in chunk_page(page, SMALL):
            assert page.text[chunk.start : chunk.end] == chunk.text.split("\n", 1)[1]

    def test_chunk_ids_are_unique_and_name_their_page(self) -> None:
        page = page_of(prose("a1 a2 a3 a4"), prose("b1 b2 b3 b4"))
        ids = [chunk.chunk_id for chunk in chunk_page(page, SMALL)]
        assert len(set(ids)) == len(ids)
        assert all(chunk_id.startswith("p1#") for chunk_id in ids)


class TestSentenceAware:
    # Four sentences of three words. Size 7 fits two sentences per chunk; overlap 3 repeats the
    # last whole sentence.
    SETTINGS = ChunkingSettings(strategy="sentence_aware", size=7, overlap=3)

    def test_chunks_start_and_end_on_sentences(self) -> None:
        page = page_of(prose("One two three. Four five six. Seven eight nine. Ten eleven twelve."))
        bodies = [chunk.text.split("\n", 1)[1] for chunk in chunk_page(page, self.SETTINGS)]
        for body in bodies:
            assert body.endswith(".")
        assert bodies[0].startswith("One two three.")

    def test_consecutive_chunks_repeat_whole_sentences(self) -> None:
        page = page_of(prose("One two three. Four five six. Seven eight nine. Ten eleven twelve."))
        bodies = [chunk.text.split("\n", 1)[1] for chunk in chunk_page(page, self.SETTINGS)]
        assert bodies == [
            "One two three. Four five six.",
            "Four five six. Seven eight nine.",
            "Seven eight nine. Ten eleven twelve.",
        ]

    def test_abbreviations_do_not_end_a_sentence(self) -> None:
        page = page_of(prose("He met Dr. Smith in Washington. She left."))
        [chunk] = chunk_page(page, ChunkingSettings(strategy="sentence_aware", size=40, overlap=0))
        assert "Dr. Smith" in chunk.text


class TestSectionAware:
    SETTINGS = ChunkingSettings(strategy="section_aware", size=20, overlap=4)

    def test_chunks_never_cross_a_section(self) -> None:
        page = page_of(
            prose("History one. History two.", section="History"),
            prose("Economy one. Economy two.", section="Economy"),
        )
        chunks = chunk_page(page, self.SETTINGS)
        assert len(chunks) == 2
        assert "History" in chunks[0].text
        assert "Economy" not in chunks[0].text
        assert "Economy" in chunks[1].text
        assert "History" not in chunks[1].text


class TestTables:
    ROWS = (("1841", "232,726"), ("1901", "290,638"), ("1951", "522,183"))
    HEADER = ("Year", "People")

    def test_a_table_chunk_repeats_the_header_and_never_splits_a_row(self) -> None:
        page = page_of(table(self.ROWS, self.HEADER, section="Population", caption="By year"))
        chunks = chunk_page(page, ChunkingSettings(strategy="fixed_words", size=8, overlap=0))
        assert len(chunks) > 1
        for chunk in chunks:
            body = chunk.text.split("\n", 1)[1]
            assert chunk.text.startswith("Dublin > Population > By year")
            for line in body.splitlines():
                assert line.count("|") == 1  # whole rows only

    def test_a_small_table_stays_in_one_chunk(self) -> None:
        page = page_of(table(self.ROWS[:1], self.HEADER))
        [chunk] = chunk_page(page, ChunkingSettings(strategy="fixed_words", size=50, overlap=0))
        assert chunk.text.endswith("Year: 1841 | People: 232,726")
        assert chunk.kind == "table"

    def test_tables_stay_separate_from_prose(self) -> None:
        page = page_of(prose("alpha beta"), table(self.ROWS[:1], self.HEADER), prose("gamma delta"))
        kinds = [
            chunk.kind
            for chunk in chunk_page(
                page, ChunkingSettings(strategy="fixed_words", size=50, overlap=0)
            )
        ]
        assert kinds == ["prose", "table", "prose"]

    def test_the_ablation_flattens_tables_into_prose(self) -> None:
        # Decision 40's comparison: no column names, no caption, chunked like ordinary text.
        page = page_of(table(self.ROWS[:1], self.HEADER))
        settings = ChunkingSettings(strategy="fixed_words", size=50, overlap=0, flatten_tables=True)
        [chunk] = chunk_page(page, settings)
        assert chunk.kind == "prose"
        assert "Year:" not in chunk.text
        assert chunk.text.endswith("1841 232,726")


class TestRelevance:
    def test_chunks_overlapping_the_answer_span_are_relevant(self) -> None:
        page = page_of(prose("a1 a2 a3 a4"), prose("b1 b2 b3 b4"))
        chunks = chunk_page(page, SMALL)
        answer = page.text.index("b1"), page.text.index("b1") + len("b1 b2 b3 b4")
        relevant = overlapping_chunks(chunks, answer[0], answer[1])
        assert relevant
        for chunk_id in relevant:
            chunk = next(c for c in chunks if c.chunk_id == chunk_id)
            assert chunk.start < answer[1]
            assert answer[0] < chunk.end

    def test_a_span_outside_every_chunk_has_no_relevant_chunk(self) -> None:
        chunks = chunk_page(page_of(prose("a1 a2")), SMALL)
        assert overlapping_chunks(chunks, 10_000, 10_010) == ()


class TestSettings:
    @pytest.mark.parametrize(
        ("size", "overlap"),
        [(0, 0), (10, 10), (10, -1)],
        ids=["size-zero", "overlap-equals-size", "negative"],
    )
    def test_invalid_sizes_are_rejected(self, size: int, overlap: int) -> None:
        with pytest.raises(ValueError, match=r"size|overlap"):
            ChunkingSettings(strategy="fixed_words", size=size, overlap=overlap)
