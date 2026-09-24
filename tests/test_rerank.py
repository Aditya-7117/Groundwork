"""Reranking re-orders the head of a ranking and leaves the rest alone."""

from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from fakes import WordCountScorer
from groundwork.chunking import Chunk
from groundwork.rerank import RerankError, ReusingScorer, get_reranker, rerank
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


class Ticking:
    """A clock that advances one second per scored passage, via the scorer it wraps."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class TimedScorer(WordCountScorer):
    def __init__(self, word: str, clock: Ticking) -> None:
        super().__init__(word)
        self.clock = clock

    def score(self, question: str, passages: Sequence[str]) -> NDArray[np.float64]:
        self.clock.now += len(passages)
        return super().score(question, passages)


class TestReusingScorer:
    def test_each_pair_is_scored_once(self) -> None:
        inner = WordCountScorer("tea")
        reusing = ReusingScorer(inner)
        first = reusing.score("q", ["tea", "tea tea", "river"])
        second = reusing.score("q", ["tea tea", "gas", "tea"])
        assert first.tolist() == [1.0, 2.0, 0.0]
        assert second.tolist() == [2.0, 0.0, 1.0]
        assert inner.passages_scored == [3, 1]

    def test_the_same_passage_for_another_question_is_scored_again(self) -> None:
        inner = WordCountScorer("tea")
        reusing = ReusingScorer(inner)
        reusing.score("q1", ["tea"])
        reusing.score("q2", ["tea"])
        assert inner.passages_scored == [1, 1]

    def test_reused_scores_are_charged_what_they_first_cost(self) -> None:
        clock = Ticking()
        reusing = ReusingScorer(TimedScorer("tea", clock), clock=clock)
        reusing.score("q", ["a", "b", "c", "d"])
        reusing.score("q", ["a", "b"])
        # Four passages took four seconds; the two reused ones are charged one second each.
        assert reusing.fresh_seconds == 4.0
        assert reusing.charged_seconds == 6.0

    def test_reranking_with_reuse_gives_the_same_order(self) -> None:
        plain = rerank(FIRST_STAGE, "q", WordCountScorer("tea"), depth=4)
        reusing = ReusingScorer(WordCountScorer("tea"))
        rerank(FIRST_STAGE, "q", reusing, depth=4)
        assert ids(rerank(FIRST_STAGE, "q", reusing, depth=4)) == ids(plain)
