import re
from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st

from groundwork.chunking import Chunk, chunk_documents, chunk_fixed_words
from groundwork.config import ChunkingConfig
from groundwork.corpus import Document

# Character offsets in "a b c d e f g": a=0 b=2 c=4 d=6 e=8 f=10 g=12, each one character long.
SEVEN_WORDS = "a b c d e f g"


class TestRejectsInvalidParameters:
    def test_size_below_one(self) -> None:
        with pytest.raises(ValueError, match="size must be at least 1"):
            chunk_fixed_words("d1", SEVEN_WORDS, size=0, overlap=0)

    def test_overlap_equal_to_size_would_never_advance(self) -> None:
        with pytest.raises(ValueError, match="overlap must be at least 0 and less than size"):
            chunk_fixed_words("d1", SEVEN_WORDS, size=3, overlap=3)

    def test_negative_overlap(self) -> None:
        with pytest.raises(ValueError, match="overlap must be at least 0 and less than size"):
            chunk_fixed_words("d1", SEVEN_WORDS, size=3, overlap=-1)


class TestFixedWords:
    def test_overlapping_windows(self) -> None:
        # size 3, overlap 1, so each window starts 2 words after the previous one:
        # words 0-2 "a b c", words 2-4 "c d e", words 4-6 "e f g". The last window reaches the
        # final word, so chunking stops there.
        assert chunk_fixed_words("d1", SEVEN_WORDS, size=3, overlap=1) == [
            Chunk(chunk_id="d1#0", doc_id="d1", start=0, end=5, text="a b c"),
            Chunk(chunk_id="d1#1", doc_id="d1", start=4, end=9, text="c d e"),
            Chunk(chunk_id="d1#2", doc_id="d1", start=8, end=13, text="e f g"),
        ]

    def test_final_window_may_be_shorter(self) -> None:
        # Eight words: windows 0-2, 2-4, 4-6, then 6-7 holds only "g h".
        chunks = chunk_fixed_words("d1", "a b c d e f g h", size=3, overlap=1)
        assert [chunk.text for chunk in chunks] == ["a b c", "c d e", "e f g", "g h"]

    def test_no_overlap(self) -> None:
        chunks = chunk_fixed_words("d1", SEVEN_WORDS, size=3, overlap=0)
        assert [chunk.text for chunk in chunks] == ["a b c", "d e f", "g"]

    def test_short_document_is_one_chunk_with_original_spacing(self) -> None:
        text = "Title\n\nBody  text."
        assert chunk_fixed_words("d1", text, size=200, overlap=50) == [
            Chunk(chunk_id="d1#0", doc_id="d1", start=0, end=len(text), text=text)
        ]

    def test_leading_and_trailing_whitespace_is_not_part_of_a_chunk(self) -> None:
        chunks = chunk_fixed_words("d1", "  one two  ", size=5, overlap=0)
        assert [(chunk.start, chunk.end, chunk.text) for chunk in chunks] == [(2, 9, "one two")]

    @pytest.mark.parametrize("text", ["", "   \n\t "])
    def test_text_without_words_gives_no_chunks(self, text: str) -> None:
        assert chunk_fixed_words("d1", text, size=3, overlap=1) == []


def test_chunk_documents_uses_title_and_body_in_document_order() -> None:
    documents = [
        Document(doc_id="b", title="Two", text="words here"),
        Document(doc_id="a", title="", text="one"),
    ]
    config = ChunkingConfig(strategy="fixed_words", size=2, overlap=0)
    assert [(chunk.chunk_id, chunk.text) for chunk in chunk_documents(documents, config)] == [
        ("b#0", "Two\n\nwords"),
        ("b#1", "here"),
        ("a#0", "one"),
    ]


words = st.lists(st.text(alphabet="abcxyz.,", min_size=1, max_size=4), min_size=1, max_size=40)
separators = st.sampled_from([" ", "  ", "\n", "\n\n", "\t"])


@st.composite
def chunking_cases(draw: st.DrawFn) -> tuple[str, int, int]:
    tokens = draw(words)
    text = tokens[0] + "".join(draw(separators) + token for token in tokens[1:])
    size = draw(st.integers(min_value=1, max_value=10))
    overlap = draw(st.integers(min_value=0, max_value=size - 1))
    return text, size, overlap


def _word_spans(text: str) -> list[tuple[int, int]]:
    return [match.span() for match in re.finditer(r"\S+", text)]


@given(chunking_cases())
def test_every_word_is_in_some_chunk(case: tuple[str, int, int]) -> None:
    text, size, overlap = case
    chunks = chunk_fixed_words("d", text, size=size, overlap=overlap)
    for start, end in _word_spans(text):
        assert any(chunk.start <= start and end <= chunk.end for chunk in chunks)


@given(chunking_cases())
def test_chunks_respect_size_and_share_exactly_the_overlap(case: tuple[str, int, int]) -> None:
    text, size, overlap = case
    chunks = chunk_fixed_words("d", text, size=size, overlap=overlap)
    spans = _word_spans(text)
    covered = [
        {i for i, (start, end) in enumerate(spans) if chunk.start <= start and end <= chunk.end}
        for chunk in chunks
    ]
    assert all(1 <= len(words_in_chunk) <= size for words_in_chunk in covered)
    for previous, current in pairwise(covered):
        assert len(previous & current) == overlap


@given(chunking_cases())
def test_chunk_text_is_the_exact_slice_of_the_source(case: tuple[str, int, int]) -> None:
    text, size, overlap = case
    for chunk in chunk_fixed_words("d", text, size=size, overlap=overlap):
        assert chunk.text == text[chunk.start : chunk.end]
