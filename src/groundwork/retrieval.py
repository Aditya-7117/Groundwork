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
        self.position = {chunk.chunk_id: index for index, chunk in enumerate(self.chunks)}
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


def fuse_reciprocal_rank(
    table: ChunkTable,
    score_vectors: Sequence[NDArray[np.float64]],
    *,
    k: int,
    candidates: int,
    positive_only: Sequence[bool],
) -> NDArray[np.float64]:
    """Merge several rankings with reciprocal rank fusion.

    Each ranking contributes 1 / (k + rank) to every chunk in its top `candidates`. Ranks, not
    raw scores, are combined, so a keyword score and a cosine similarity never need to be put on
    the same scale. k = 60 is the value from the method's original paper (Cormack, Clarke and
    Buettcher, 2009).

    Returns:
        A fused score per chunk; chunks outside every top list score 0.

    Raises:
        ValueError: If k or candidates is below 1, or the inputs disagree in length.
    """
    if k < 1 or candidates < 1:
        raise ValueError(f"k and candidates must be at least 1, got {k} and {candidates}")
    if len(score_vectors) != len(positive_only):
        raise ValueError("score_vectors and positive_only must have the same length")
    fused = np.zeros(len(table), dtype=np.float64)
    for scores, positive in zip(score_vectors, positive_only, strict=True):
        ranked = rank_from_scores(table, scores, depth=candidates, positive_only=positive)
        for rank, item in enumerate(ranked.chunks, start=1):
            fused[table.position[item.chunk.chunk_id]] += 1.0 / (k + rank)
    return fused
