"""Reranking re-orders the head of a ranking and leaves the rest alone."""

import numpy as np
import pytest

from fakes import WordCountScorer
from groundwork.chunking import Chunk
from groundwork.rerank import RerankError, get_reranker, rerank
from groundwork.retrieval import ChunkTable, Retrieval, rank_from_scores


def chunk(page: str, position: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{page}#{position}",
        page_id=page,
        text=text,
        start=0,
        end=len(text),
        kind="prose",
    )


# First-stage order is a, b, c, d, e: the scores below descend.
CHUNKS = [
    chunk("a", 0, "tea"),
    chunk("b", 0, "gas giant"),
    chunk("c", 0, "tea tea"),
    chunk("d", 0, "river"),
    chunk("e", 0, "tea tea tea"),
]
FIRST_STAGE = rank_from_scores(
    ChunkTable(CHUNKS), np.array([5.0, 4.0, 3.0, 2.0, 1.0]), depth=5, positive_only=True
)


def ids(retrieval: Retrieval) -> list[str]:
    return [item.chunk.chunk_id for item in retrieval.chunks]


class TestRerank:
    def test_the_head_is_reordered_and_the_tail_keeps_its_place(self) -> None:
        # Within the top 4, "tea" counts are a:1, b:0, c:2, d:0. The chunk at rank 5 has the most
        # "tea" of all, but it lies below the rerank depth, so it stays last.
        result = rerank(FIRST_STAGE, "q", WordCountScorer("tea"), depth=4)
        assert ids(result) == ["c#0", "a#0", "b#0", "d#0", "e#0"]

    def test_equal_scores_keep_first_stage_order(self) -> None:
        # b and d both score 0; b ranked above d before, so it still does.
        result = rerank(FIRST_STAGE, "q", WordCountScorer("tea"), depth=4)
        assert ids(result).index("b#0") < ids(result).index("d#0")

    def test_the_scorer_sees_only_the_head(self) -> None:
        scorer = WordCountScorer("tea")
        rerank(FIRST_STAGE, "q", scorer, depth=2)
        assert scorer.passages_scored == [2]

    def test_pages_follow_the_new_chunk_order(self) -> None:
        result = rerank(FIRST_STAGE, "q", WordCountScorer("giant"), depth=5)
        assert result.pages[0].page_id == "b"

    def test_depth_beyond_the_ranking_reranks_everything(self) -> None:
        result = rerank(FIRST_STAGE, "q", WordCountScorer("tea"), depth=50)
        assert ids(result)[0] == "e#0"

    def test_an_empty_ranking_is_returned_unchanged(self) -> None:
        empty = Retrieval(chunks=(), pages=())
        assert rerank(empty, "q", WordCountScorer("tea"), depth=5) == empty

    def test_depth_below_one_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="depth must be at least 1"):
            rerank(FIRST_STAGE, "q", WordCountScorer("tea"), depth=0)


def test_an_unknown_reranker_names_the_known_ones() -> None:
    with pytest.raises(RerankError, match="known: bge-reranker-v2-m3"):
        get_reranker("colbert")
