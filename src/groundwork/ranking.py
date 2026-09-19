"""Turning a ranking of chunks into a ranking of documents.

Relevance judgements are made per document, but retrieval scores chunks. Each document is scored
by its best chunk, an approach known as MaxP in the passage-retrieval literature. The reasoning
is in docs/decisions/0005-document-ranking-from-chunks.md.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from groundwork.bm25 import ScoredChunk


@dataclass(frozen=True, slots=True, kw_only=True)
class RankedDocument:
    """A document's position-determining score, and the chunk that earned it."""

    doc_id: str
    score: float
    best_chunk_id: str


def rank_documents_by_best_chunk(
    scored_chunks: Iterable[ScoredChunk], *, depth: int
) -> list[RankedDocument]:
    """Rank documents by their highest-scoring chunk and keep the top `depth`.

    Ties between documents are broken by document id, and ties between chunks of one document by
    chunk id, so the result is fully deterministic.

    Raises:
        ValueError: If depth is below 1.
    """
    if depth < 1:
        raise ValueError(f"depth must be at least 1, got {depth}")
    best: dict[str, ScoredChunk] = {}
    for scored in scored_chunks:
        current = best.get(scored.chunk.doc_id)
        if (
            current is None
            or scored.score > current.score
            or (scored.score == current.score and scored.chunk.chunk_id < current.chunk.chunk_id)
        ):
            best[scored.chunk.doc_id] = scored
    ranked = sorted(best.values(), key=lambda scored: (-scored.score, scored.chunk.doc_id))
    return [
        RankedDocument(
            doc_id=scored.chunk.doc_id, score=scored.score, best_chunk_id=scored.chunk.chunk_id
        )
        for scored in ranked[:depth]
    ]
