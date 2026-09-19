import pytest

from groundwork.bm25 import ScoredChunk
from groundwork.chunking import Chunk
from groundwork.ranking import RankedDocument, rank_documents_by_best_chunk


def _scored(chunk_id: str, score: float) -> ScoredChunk:
    doc_id = chunk_id.split("#", maxsplit=1)[0]
    chunk = Chunk(chunk_id=chunk_id, doc_id=doc_id, start=0, end=1, text="x")
    return ScoredChunk(chunk=chunk, score=score)


def test_rejects_depth_below_one() -> None:
    with pytest.raises(ValueError, match="depth must be at least 1"):
        rank_documents_by_best_chunk([], depth=0)


def test_document_score_is_its_best_chunk() -> None:
    # d1's chunks score 1.0 and 3.0, so d1 scores 3.0 and outranks d2 at 2.0.
    scored = [_scored("d1#0", 1.0), _scored("d2#0", 2.0), _scored("d1#1", 3.0)]
    assert rank_documents_by_best_chunk(scored, depth=10) == [
        RankedDocument(doc_id="d1", score=3.0, best_chunk_id="d1#1"),
        RankedDocument(doc_id="d2", score=2.0, best_chunk_id="d2#0"),
    ]


def test_depth_counts_documents_not_chunks() -> None:
    scored = [_scored("d1#0", 5.0), _scored("d1#1", 4.0), _scored("d2#0", 3.0)]
    assert [ranked.doc_id for ranked in rank_documents_by_best_chunk(scored, depth=2)] == [
        "d1",
        "d2",
    ]


def test_depth_truncates() -> None:
    scored = [_scored("d1#0", 3.0), _scored("d2#0", 2.0), _scored("d3#0", 1.0)]
    assert [ranked.doc_id for ranked in rank_documents_by_best_chunk(scored, depth=2)] == [
        "d1",
        "d2",
    ]


def test_ties_are_broken_by_document_id() -> None:
    scored = [_scored("d2#0", 1.0), _scored("d1#0", 1.0)]
    assert [ranked.doc_id for ranked in rank_documents_by_best_chunk(scored, depth=5)] == [
        "d1",
        "d2",
    ]
