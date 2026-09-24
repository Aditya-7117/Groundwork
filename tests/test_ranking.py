import pytest

from groundwork.bm25 import ScoredChunk
from groundwork.chunking import Chunk
from groundwork.ranking import RankedPage, rank_pages_by_best_chunk


def _scored(chunk_id: str, score: float) -> ScoredChunk:
    page_id = chunk_id.split("#", maxsplit=1)[0]
    chunk = Chunk(chunk_id=chunk_id, page_id=page_id, text="x", start=0, end=1, kind="prose")
    return ScoredChunk(chunk=chunk, score=score)


def test_rejects_depth_below_one() -> None:
    with pytest.raises(ValueError, match="depth must be at least 1"):
        rank_pages_by_best_chunk([], depth=0)


def test_page_score_is_its_best_chunk() -> None:
    # p1's chunks score 1.0 and 3.0, so p1 scores 3.0 and outranks p2 at 2.0.
    scored = [_scored("p1#0", 1.0), _scored("p2#0", 2.0), _scored("p1#1", 3.0)]
    assert rank_pages_by_best_chunk(scored, depth=10) == [
        RankedPage(page_id="p1", score=3.0, best_chunk_id="p1#1"),
        RankedPage(page_id="p2", score=2.0, best_chunk_id="p2#0"),
    ]


def test_depth_counts_pages_not_chunks() -> None:
    scored = [_scored("p1#0", 5.0), _scored("p1#1", 4.0), _scored("p2#0", 3.0)]
    assert [ranked.page_id for ranked in rank_pages_by_best_chunk(scored, depth=2)] == [
        "p1",
        "p2",
    ]


def test_depth_truncates() -> None:
    scored = [_scored("p1#0", 3.0), _scored("p2#0", 2.0), _scored("p3#0", 1.0)]
    assert [ranked.page_id for ranked in rank_pages_by_best_chunk(scored, depth=2)] == [
        "p1",
        "p2",
    ]


def test_ties_are_broken_by_page_id() -> None:
    scored = [_scored("p2#0", 1.0), _scored("p1#0", 1.0)]
    assert [ranked.page_id for ranked in rank_pages_by_best_chunk(scored, depth=5)] == [
        "p1",
        "p2",
    ]
