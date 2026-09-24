"""Turning a ranking of chunks into a ranking of pages.

Relevance judgements are made per page, but retrieval scores chunks. Each page is scored by its
best chunk, an approach known as MaxP in the passage-retrieval literature. The reasoning is in
docs/decisions/0005-document-ranking-from-chunks.md.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from groundwork.bm25 import ScoredChunk


@dataclass(frozen=True, slots=True, kw_only=True)
class RankedPage:
    """A page's position-determining score, and the chunk that earned it."""

    page_id: str
    score: float
    best_chunk_id: str


def rank_pages_by_best_chunk(
    scored_chunks: Iterable[ScoredChunk], *, depth: int
) -> list[RankedPage]:
    """Rank pages by their highest-scoring chunk and keep the top `depth`.

    Ties between pages are broken by page id, and ties between chunks of one page by chunk id, so
    the result is fully deterministic.

    Raises:
        ValueError: If depth is below 1.
    """
    if depth < 1:
        raise ValueError(f"depth must be at least 1, got {depth}")
    best: dict[str, ScoredChunk] = {}
    for scored in scored_chunks:
        current = best.get(scored.chunk.page_id)
        if (
            current is None
            or scored.score > current.score
            or (scored.score == current.score and scored.chunk.chunk_id < current.chunk.chunk_id)
        ):
            best[scored.chunk.page_id] = scored
    ranked = sorted(best.values(), key=lambda scored: (-scored.score, scored.chunk.page_id))
    return [
        RankedPage(
            page_id=scored.chunk.page_id, score=scored.score, best_chunk_id=scored.chunk.chunk_id
        )
        for scored in ranked[:depth]
    ]
