"""The fast retrieval path must agree exactly with the reference implementations.

The pure-Python BM25 index and rank_pages_by_best_chunk are pinned by hand-worked tests. The
vectorised versions are only trusted because these tests prove they return the same scores and
the same order, including how ties are broken.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from groundwork.bm25 import BM25Index, SparseBM25
from groundwork.chunking import Chunk
from groundwork.ranking import rank_pages_by_best_chunk
from groundwork.retrieval import ChunkTable, rank_from_scores

VOCABULARY = ["tea", "river", "ring", "rings", "black", "green", "shannon", "gas", "giant", "the"]


def chunk(page: str, position: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{page}#{position}",
        page_id=page,
        text=text,
        start=0,
        end=len(text),
        kind="prose",
    )


@st.composite
def corpora(draw: st.DrawFn) -> tuple[list[Chunk], str]:
    pages = draw(st.integers(min_value=1, max_value=5))
    chunks = []
    for page in range(pages):
        for position in range(draw(st.integers(min_value=1, max_value=4))):
            words = draw(st.lists(st.sampled_from(VOCABULARY), min_size=1, max_size=8))
            chunks.append(chunk(f"p{page}", position, " ".join(words)))
    query = " ".join(draw(st.lists(st.sampled_from(VOCABULARY), min_size=1, max_size=4)))
    return chunks, query


class TestSparseBM25:
    @settings(max_examples=200)
    @given(case=corpora(), stem=st.booleans())
    def test_scores_match_the_reference_exactly(
        self, *, case: tuple[list[Chunk], str], stem: bool
    ) -> None:
        chunks, query = case
        reference = {
            scored.chunk.chunk_id: scored.score
            for scored in BM25Index(chunks, k1=0.9, b=0.4, stem=stem).search(query)
        }
        scores = SparseBM25(chunks, k1=0.9, b=0.4, stem=stem).scores(query)
        fast = {chunks[i].chunk_id: float(scores[i]) for i in np.flatnonzero(scores)}
        assert fast.keys() == reference.keys()
        for chunk_id, value in reference.items():
            assert fast[chunk_id] == pytest.approx(value, rel=1e-12, abs=1e-12)

    def test_a_query_sharing_no_terms_scores_every_chunk_zero(self) -> None:
        chunks = [chunk("p0", 0, "green tea"), chunk("p1", 0, "black tea")]
        assert not SparseBM25(chunks, k1=0.9, b=0.4).scores("coffee").any()

    def test_rejects_an_index_without_tokens(self) -> None:
        with pytest.raises(ValueError, match="at least one token"):
            SparseBM25([chunk("p0", 0, "... ---")], k1=0.9, b=0.4)


class TestRankFromScores:
    @settings(max_examples=200)
    @given(case=corpora(), depth=st.integers(min_value=1, max_value=12))
    def test_rankings_match_the_reference_including_ties(
        self, case: tuple[list[Chunk], str], depth: int
    ) -> None:
        chunks, query = case
        reference = BM25Index(chunks, k1=0.9, b=0.4).search(query)
        expected_chunks = [scored.chunk.chunk_id for scored in reference[:depth]]
        expected_pages = [p.page_id for p in rank_pages_by_best_chunk(reference, depth=depth)]

        table = ChunkTable(chunks)
        scores = SparseBM25(chunks, k1=0.9, b=0.4).scores(query)
        result = rank_from_scores(table, scores, depth=depth, positive_only=True)
        assert [scored.chunk.chunk_id for scored in result.chunks] == expected_chunks
        assert [page.page_id for page in result.pages] == expected_pages

    def test_equal_scores_are_ordered_by_chunk_and_page_id(self) -> None:
        chunks = [chunk("p2", 0, "x"), chunk("p1", 0, "x"), chunk("p1", 1, "x")]
        result = rank_from_scores(ChunkTable(chunks), np.ones(3), depth=5, positive_only=False)
        assert [s.chunk.chunk_id for s in result.chunks] == ["p1#0", "p1#1", "p2#0"]
        assert [p.page_id for p in result.pages] == ["p1", "p2"]

    def test_positive_only_leaves_out_chunks_that_share_no_term(self) -> None:
        chunks = [chunk("p0", 0, "x"), chunk("p1", 0, "y")]
        result = rank_from_scores(
            ChunkTable(chunks), np.array([0.0, 2.0]), depth=5, positive_only=True
        )
        assert [s.chunk.chunk_id for s in result.chunks] == ["p1#0"]
