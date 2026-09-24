"""Second-stage reranking: re-order the top first-stage results with a slower, sharper model.

A first-stage retriever scores the question and each chunk separately, which is what makes it
fast enough to search everything. A cross-encoder reads the question and a chunk together, so it
can judge whether that chunk actually answers that question, but it is far too slow to run on
every chunk. So it re-orders only the top of the first-stage ranking (the top 50, decision 47).
Chunks below that depth keep their first-stage order after the reranked ones.
"""

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from groundwork.bm25 import ScoredChunk
from groundwork.ranking import RankedPage
from groundwork.retrieval import Retrieval

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class RerankerModel:
    """A registered cross-encoder, pinned to one published revision."""

    key: str
    name: str
    revision: str
    max_length: int


RERANKERS: dict[str, RerankerModel] = {
    model.key: model
    for model in (
        RerankerModel(
            key="bge-reranker-v2-m3",
            name="BAAI/bge-reranker-v2-m3",
            revision="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
            max_length=512,
        ),
    )
}


class RerankError(ValueError):
    """A reranker is unknown."""


class Scorer(Protocol):
    """Anything that scores (question, passage) pairs; higher means more relevant.

    Attributes:
        device: Where the model runs, recorded with every result.
        precision: The numeric precision it runs in, recorded with every result.
    """

    device: str
    precision: str

    def score(self, question: str, passages: Sequence[str]) -> NDArray[np.float64]:
        """Return one relevance score per passage."""
        ...


def get_reranker(key: str) -> RerankerModel:
    """Return a registered reranker.

    Raises:
        RerankError: If no reranker has this key.
    """
    try:
        return RERANKERS[key]
    except KeyError as error:
        raise RerankError(
            f"unknown reranker {key!r}; known: {', '.join(sorted(RERANKERS))}"
        ) from error


class CrossEncoderScorer:
    """A real cross-encoder on the Apple GPU in half precision (decision 46)."""

    def __init__(
        self,
        model: RerankerModel,
        *,
        cache_dir: Path,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        """Load the pinned weights; on the Apple GPU if present, unless a device is given."""
        import torch  # noqa: PLC0415 -- heavy import, only paid when reranking is used
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.float16 if device == "mps" else torch.float32
        self._model = CrossEncoder(
            model.name,
            revision=model.revision,
            device=device,
            max_length=model.max_length,
            cache_folder=str(cache_dir),
            model_kwargs={"torch_dtype": dtype},
        )
        self._batch_size = batch_size
        self.precision = "float16" if dtype == torch.float16 else "float32"
        self.device = device

    def score(self, question: str, passages: Sequence[str]) -> NDArray[np.float64]:
        """Score each passage against the question."""
        pairs = [(question, passage) for passage in passages]
        scores = self._model.predict(pairs, batch_size=self._batch_size, show_progress_bar=False)
        return np.asarray(scores, dtype=np.float64)


class ReusingScorer:
    """Scores each (question, passage) pair once and reuses the score afterwards (decision 64).

    Setups over the same chunker share most of their top 50, so across a grid the same pair would
    otherwise be scored many times. Reuse must not make a later setup look cheaper than it is, so
    every returned score is charged the time it took when it was first computed: the time is
    split evenly across the pairs of the batch that produced it.

    Attributes:
        fresh_seconds: Time actually spent scoring, in total.
        charged_seconds: What the returned scores cost when computed, in total. Equal to
            fresh_seconds when nothing was reused.
    """

    def __init__(self, scorer: Scorer, *, clock: Callable[[], float] = time.perf_counter) -> None:
        """Wrap a scorer."""
        self._scorer = scorer
        self._clock = clock
        self._scores: dict[tuple[str, str], tuple[float, float]] = {}
        self.device = scorer.device
        self.precision = scorer.precision
        self.fresh_seconds = 0.0
        self.charged_seconds = 0.0

    def score(self, question: str, passages: Sequence[str]) -> NDArray[np.float64]:
        """Return one score per passage, scoring only pairs not seen before."""
        missing = [p for p in dict.fromkeys(passages) if (question, p) not in self._scores]
        if missing:
            started = self._clock()
            values = self._scorer.score(question, missing)
            spent = self._clock() - started
            self.fresh_seconds += spent
            for passage, value in zip(missing, values, strict=True):
                self._scores[(question, passage)] = (float(value), spent / len(missing))
        pairs = [self._scores[(question, passage)] for passage in passages]
        self.charged_seconds += sum(cost for _, cost in pairs)
        return np.array([value for value, _ in pairs], dtype=np.float64)


def rerank(retrieval: Retrieval, question: str, scorer: Scorer, *, depth: int) -> Retrieval:
    """Re-order the top depth chunks by the scorer, keep the rest in first-stage order.

    Pages are ranked by their best chunk in the new order. Equal reranker scores keep their
    first-stage order, so the result is deterministic.

    Raises:
        ValueError: If depth is below 1.
    """
    if depth < 1:
        raise ValueError(f"depth must be at least 1, got {depth}")
    head, tail = retrieval.chunks[:depth], retrieval.chunks[depth:]
    if not head:
        return retrieval
    scores = scorer.score(question, [item.chunk.text for item in head])
    # A stable sort on the negated scores keeps first-stage order among exact ties.
    order = np.argsort(-scores, kind="stable")
    reranked = tuple(ScoredChunk(chunk=head[i].chunk, score=float(scores[i])) for i in order)
    chunks = reranked + tail
    pages: list[RankedPage] = []
    seen: set[str] = set()
    for item in chunks:
        if item.chunk.page_id not in seen:
            seen.add(item.chunk.page_id)
            pages.append(
                RankedPage(
                    page_id=item.chunk.page_id, score=item.score, best_chunk_id=item.chunk.chunk_id
                )
            )
    limit = len(retrieval.pages) if retrieval.pages else len(pages)
    return Retrieval(chunks=chunks, pages=tuple(pages[:limit]))
