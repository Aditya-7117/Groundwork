"""Turning a score for every chunk into ranked chunks and ranked pages.

Every first-stage retriever in the grid produces the same thing: one score per chunk. BM25 gives
zero to chunks that share no word with the question; embedding models score every chunk. This
module ranks those scores the same way for all of them, with the same deterministic tie rules as
the reference ranking code, so no retriever gains or loses from how its ties happen to fall.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from groundwork.bm25 import ScoredChunk
from groundwork.chunking import Chunk
from groundwork.ranking import RankedPage


class ChunkTable:
    """The chunks being searched, with the orderings needed to break ties quickly."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        """Index chunks by position, page and sorted id."""
        self.chunks = tuple(chunks)
        count = len(self.chunks)
        by_id = sorted(range(count), key=lambda position: self.chunks[position].chunk_id)
        self.id_rank = np.empty(count, dtype=np.int64)
        self.id_rank[by_id] = np.arange(count, dtype=np.int64)
        page_ids = sorted({chunk.page_id for chunk in self.chunks})
        page_position = {page_id: position for position, page_id in enumerate(page_ids)}
        self.page_ids = tuple(page_ids)
        self.page_index = np.fromiter(
            (page_position[chunk.page_id] for chunk in self.chunks), dtype=np.int64, count=count
        )

    def __len__(self) -> int:
        return len(self.chunks)


@dataclass(frozen=True, slots=True, kw_only=True)
class Retrieval:
    """One question's retrieved chunks and pages, best first."""

    chunks: tuple[ScoredChunk, ...]
    pages: tuple[RankedPage, ...]


def rank_from_scores(
    table: ChunkTable, scores: NDArray[np.float64], *, depth: int, positive_only: bool
) -> Retrieval:
    """Rank chunks by score, and pages by their best chunk, keeping the top depth of each.

    Equal scores are ordered by chunk id, and pages by page id, exactly as in BM25Index and
    rank_pages_by_best_chunk.

    Args:
        table: The chunks the scores refer to, in the same order.
        scores: One score per chunk.
        depth: How many chunks, and how many pages, to keep.
        positive_only: Leave out chunks scoring 0 or less, as keyword retrieval does.

    Raises:
        ValueError: If depth is below 1 or the scores do not match the table.
    """
    if depth < 1:
        raise ValueError(f"depth must be at least 1, got {depth}")
    if scores.shape != (len(table),):
        raise ValueError(f"expected {len(table)} scores, got shape {scores.shape}")
    candidates = np.flatnonzero(scores > 0) if positive_only else np.arange(len(table))
    order = candidates[np.lexsort((table.id_rank[candidates], -scores[candidates]))]

    ranked_chunks = tuple(
        ScoredChunk(chunk=table.chunks[position], score=float(scores[position]))
        for position in order[:depth]
    )
    # A page's best chunk is its first appearance in the chunk order, and pages appear in the
    # order of their best chunks. Because chunk ids begin with the page id, ties between pages
    # fall in page-id order, matching rank_pages_by_best_chunk.
    _, first_seen = np.unique(table.page_index[order], return_index=True)
    ranked_pages = tuple(
        RankedPage(
            page_id=table.chunks[order[position]].page_id,
            score=float(scores[order[position]]),
            best_chunk_id=table.chunks[order[position]].chunk_id,
        )
        for position in np.sort(first_seen)[:depth]
    )
    return Retrieval(chunks=ranked_chunks, pages=ranked_pages)
