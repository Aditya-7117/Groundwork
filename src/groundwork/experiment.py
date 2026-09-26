"""Running one experiment: chunk, index, retrieve, and score every question.

Everything except wall-clock timing is a pure function of the config and the evaluation set. The
only randomness is the optional question sample, drawn from a random.Random seeded by the config,
so the same inputs always give the same rankings and metrics.

Every metric is reported twice: once at passage level, where a chunk counts when it overlaps the
annotated answer span, and once at page level. Results are also broken down by answer type, so
paragraph, table and list questions can be compared.
"""

import logging
import math
import random
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from groundwork.bm25 import SparseBM25
from groundwork.chunking import Chunk, ChunkingSettings, chunk_pages
from groundwork.config import ExperimentConfig
from groundwork.embeddings import (
    DenseIndex,
    EmbeddingModel,
    Encoder,
    SentenceTransformerEncoder,
    VectorCache,
    chunk_embeddings,
    get_model,
)
from groundwork.evaluation import EvalQuestion, EvaluationSet, chunk_relevance
from groundwork.metrics import (
    Judgements,
    mean_over_queries,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from groundwork.rerank import (
    CrossEncoderScorer,
    RerankerModel,
    ReusingScorer,
    Scorer,
    get_reranker,
    rerank,
)
from groundwork.retrieval import ChunkTable, Retrieval, fuse_reciprocal_rank, rank_from_scores

logger = logging.getLogger(__name__)

type Clock = Callable[[], float]
type Searcher = Callable[[str], Retrieval]

_METRICS = (
    ("precision", precision_at_k),
    ("recall", recall_at_k),
    ("ndcg", ndcg_at_k),
    ("rr", reciprocal_rank_at_k),
)


class ModelProvider(Protocol):
    """Where the neural models come from. Tests pass small fakes; real runs load the weights."""

    def vector_cache(self, encoder: Encoder) -> VectorCache:
        """Where an encoder's chunk vectors are cached, and whether missing ones may be encoded."""
        ...

    def encoder(self, model: EmbeddingModel) -> Encoder:
        """Return an encoder for an embedding model."""
        ...

    def scorer(self, model: RerankerModel) -> Scorer:
        """Return a scorer for a reranker."""
        ...


class LocalModels:
    """Real models, loaded on first use and kept for the rest of the process."""

    def __init__(
        self,
        *,
        cache_dir: Path,
        weights_dir: Path,
        device: str | None = None,
        stored_vectors: str | None = None,
    ) -> None:
        """Remember where vectors and weights are cached, and where models run (None: best).

        Args:
            cache_dir: Where chunk vectors are cached.
            weights_dir: Where model weights are cached.
            device: Where models run; None picks the Apple GPU when there is one.
            stored_vectors: Read chunk vectors computed in this precision, and never encode
                chunks: for live search, whose machine can be far slower than the evaluation's.
        """
        self._cache_dir = cache_dir
        self._weights_dir = weights_dir
        self._device = device
        self._stored_vectors = stored_vectors
        self._encoders: dict[str, Encoder] = {}
        self._scorers: dict[str, ReusingScorer] = {}

    def vector_cache(self, encoder: Encoder) -> VectorCache:
        """The evaluation's own vectors when live, otherwise this encoder's, encoding as needed."""
        if self._stored_vectors is not None:
            return VectorCache(
                directory=self._cache_dir, precision=self._stored_vectors, encode_missing=False
            )
        return VectorCache(directory=self._cache_dir, precision=encoder.precision)

    def encoder(self, model: EmbeddingModel) -> Encoder:
        """Load an embedding model once."""
        if model.key not in self._encoders:
            self._encoders[model.key] = SentenceTransformerEncoder(
                model, cache_dir=self._weights_dir, device=self._device
            )
        return self._encoders[model.key]

    def scorer(self, model: RerankerModel) -> Scorer:
        """Load a reranker once, reusing its scores across every run in this process."""
        if model.key not in self._scorers:
            self._scorers[model.key] = ReusingScorer(
                CrossEncoderScorer(model, cache_dir=self._weights_dir, device=self._device)
            )
        return self._scorers[model.key]


@dataclass(frozen=True, slots=True, kw_only=True)
class FirstStage:
    """A built first-stage retriever.

    Attributes:
        search: Returns the ranking for one question.
        stage_seconds: What building it took, per stage.
        models: The neural models it uses, with revisions, device and precision.
    """

    search: Searcher
    stage_seconds: dict[str, float]
    models: dict[str, object]


@dataclass(frozen=True, slots=True, kw_only=True)
class QuestionResult:
    """One question's rankings and metrics.

    Attributes:
        question_id: The question.
        answer_type: Whether its answer sits in a paragraph, a table or a list.
        chunk_ranking: Retrieved chunk ids, best first.
        page_ranking: Retrieved page ids, best first, each scored by its best chunk.
        metrics: Metric name to value, prefixed "passage." or "page.".
    """

    question_id: str
    answer_type: str
    chunk_ranking: tuple[str, ...]
    page_ranking: tuple[str, ...]
    metrics: Mapping[str, float]


@dataclass(frozen=True, slots=True, kw_only=True)
class ExperimentResult:
    """Everything a run measured.

    Attributes:
        questions: Per-question results, ordered by question id.
        aggregate: Each metric averaged over questions, with rr reported as mrr.
        by_answer_type: The same averages within each answer type.
        chunk_count: Chunks indexed.
        excluded_question_ids: Questions left out because no chunk covers their answer.
        stage_seconds: Wall-clock seconds per pipeline stage. "embedding" is the time the chunk
            vectors took to compute, even when this run read them from the cache.
        retrieval_latency_ms: Per-question first-stage latency: mean, p50, p95 and max.
        rerank_latency_ms: Per-question reranking latency, empty when there is no reranker. A
            score reused from an earlier setup counts at what it cost when first computed.
        models: The neural models used, their pinned revisions, device and precision.
    """

    questions: tuple[QuestionResult, ...]
    aggregate: Mapping[str, float]
    by_answer_type: Mapping[str, Mapping[str, float]]
    chunk_count: int
    excluded_question_ids: tuple[str, ...]
    stage_seconds: Mapping[str, float]
    retrieval_latency_ms: Mapping[str, float]
    rerank_latency_ms: Mapping[str, float] = field(default_factory=dict)
    models: Mapping[str, object] = field(default_factory=dict)


def run_experiment(
    config: ExperimentConfig,
    evaluation_set: EvaluationSet,
    *,
    clock: Clock = time.perf_counter,
    models: ModelProvider | None = None,
) -> ExperimentResult:
    """Run one configuration over an evaluation set and score it.

    Args:
        config: The experiment definition.
        evaluation_set: Pages and questions to score against.
        clock: Monotonic clock in seconds, used only for timings.
        models: Where neural models come from; required for dense, hybrid and reranking.

    Raises:
        ValueError: If the config asks for more questions than the set can evaluate, or needs
            models and none were provided.
    """
    stage_seconds: dict[str, float] = {}
    settings = ChunkingSettings(
        strategy=config.chunking.strategy,
        size=config.chunking.size,
        overlap=config.chunking.overlap,
        flatten_tables=config.chunking.flatten_tables,
    )

    started = clock()
    chunks = chunk_pages(evaluation_set.pages, settings)
    chunks_by_page: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        chunks_by_page.setdefault(chunk.page_id, []).append(chunk)
    stage_seconds["chunking"] = clock() - started

    table = ChunkTable(chunks)
    first_stage = build_first_stage(config, chunks, table, models, clock)
    stage_seconds.update(first_stage.stage_seconds)
    model_record = first_stage.models

    questions, excluded, judgements = _select(config, evaluation_set, chunks_by_page)

    retrievals: dict[str, Retrieval] = {}
    latencies_ms: list[float] = []
    for question in questions:
        started = clock()
        retrievals[question.question_id] = first_stage.search(question.text)
        latencies_ms.append((clock() - started) * 1000)
    stage_seconds["retrieval"] = sum(latencies_ms) / 1000

    rerank_ms: list[float] = []
    if config.rerank is not None:
        if models is None:
            raise ValueError("reranking needs a model provider")
        reranker = get_reranker(config.rerank.model)
        provided = models.scorer(reranker)
        scorer = provided if isinstance(provided, ReusingScorer) else ReusingScorer(provided)
        model_record["reranker"] = {
            "name": reranker.name,
            "revision": reranker.revision,
            "max_length": reranker.max_length,
            "depth": config.rerank.depth,
            "device": scorer.device,
            "precision": scorer.precision,
        }
        for question in questions:
            fresh, charged = scorer.fresh_seconds, scorer.charged_seconds
            started = clock()
            retrievals[question.question_id] = rerank(
                retrievals[question.question_id],
                question.text,
                scorer,
                depth=config.rerank.depth,
            )
            # A reused score is charged what it cost when computed, so reuse never flatters
            # a setup's latency.
            reused = (scorer.charged_seconds - charged) - (scorer.fresh_seconds - fresh)
            rerank_ms.append((clock() - started + reused) * 1000)
        stage_seconds["reranking"] = sum(rerank_ms) / 1000

    started = clock()
    results = tuple(
        _question_result(
            question,
            retrievals[question.question_id],
            judgements[question.question_id],
            config.evaluation.cutoffs,
        )
        for question in questions
    )
    aggregate = _average(results)
    by_answer_type = {
        answer_type: _average(tuple(r for r in results if r.answer_type == answer_type))
        for answer_type in sorted({result.answer_type for result in results})
    }
    stage_seconds["evaluation"] = clock() - started

    logger.info(
        "experiment finished",
        extra={
            "experiment": config.name,
            "questions": len(results),
            "chunks": len(chunks),
            "excluded": len(excluded),
        },
    )
    return ExperimentResult(
        questions=results,
        aggregate=aggregate,
        by_answer_type=by_answer_type,
        chunk_count=len(chunks),
        excluded_question_ids=excluded,
        stage_seconds=stage_seconds,
        retrieval_latency_ms=_summarise(latencies_ms),
        rerank_latency_ms=_summarise(rerank_ms) if rerank_ms else {},
        models=model_record,
    )


def build_first_stage(
    config: ExperimentConfig,
    chunks: Sequence[Chunk],
    table: ChunkTable,
    models: ModelProvider | None,
    clock: Clock,
) -> FirstStage:
    """Build the first-stage searcher the config describes.

    The match over methods is exhaustive, so adding a method without handling it here is a type
    error rather than a config that silently runs something else.

    Raises:
        ValueError: If the method needs models and none were provided.
    """
    retrieval = config.retrieval
    stage_seconds: dict[str, float] = {}
    record: dict[str, object] = {}

    bm25: SparseBM25 | None = None
    if retrieval.bm25 is not None:
        started = clock()
        bm25 = SparseBM25(
            chunks, k1=retrieval.bm25.k1, b=retrieval.bm25.b, stem=retrieval.bm25.stem
        )
        stage_seconds["indexing"] = clock() - started

    dense: DenseIndex | None = None
    if retrieval.dense is not None:
        if models is None:
            raise ValueError("dense retrieval needs a model provider")
        embedding = get_model(retrieval.dense.model)
        encoder = models.encoder(embedding)
        vectors, stage_seconds["embedding"] = chunk_embeddings(
            chunks, embedding, encoder, cache=models.vector_cache(encoder)
        )
        dense = DenseIndex(vectors, embedding, encoder)
        record["embedding"] = {
            "name": embedding.name,
            "revision": embedding.revision,
            "query_instruction": embedding.query_instruction,
            "max_seq_length": embedding.max_seq_length,
            "device": encoder.device,
            "precision": encoder.precision,
        }

    fusion = retrieval.fusion
    if fusion is not None:
        record["fusion"] = {"method": "reciprocal rank fusion", "k": fusion.k}

    def keyword(query: str) -> NDArray[np.float64]:
        if bm25 is None:
            raise ValueError("keyword retrieval was not configured")
        return bm25.scores(query)

    def semantic(query: str) -> NDArray[np.float64]:
        if dense is None:
            raise ValueError("dense retrieval was not configured")
        return dense.scores(query)

    def scores(query: str) -> tuple[NDArray[np.float64], bool]:
        """Score every chunk, and say whether a zero score means "not retrieved".

        BM25 gives zero to a chunk sharing no word with the question, and such a chunk is not
        retrieved. A cosine similarity can be zero or negative and still rank, so dense keeps
        every chunk. Fused scores are positive exactly for chunks some retriever returned.
        """
        match retrieval.method:
            case "bm25":
                return keyword(query), True
            case "dense":
                return semantic(query), False
            case "hybrid":
                if fusion is None:
                    raise ValueError("hybrid retrieval needs [retrieval.fusion]")
                fused = fuse_reciprocal_rank(
                    table,
                    [keyword(query), semantic(query)],
                    k=fusion.k,
                    candidates=fusion.candidates,
                    positive_only=[True, False],
                )
                return fused, True

    def search(query: str) -> Retrieval:
        vector, positive_only = scores(query)
        return rank_from_scores(table, vector, depth=retrieval.depth, positive_only=positive_only)

    return FirstStage(search=search, stage_seconds=stage_seconds, models=record)


def _select(
    config: ExperimentConfig,
    evaluation_set: EvaluationSet,
    chunks_by_page: Mapping[str, Sequence[Chunk]],
) -> tuple[tuple[EvalQuestion, ...], tuple[str, ...], dict[str, Judgements]]:
    """Choose the questions to evaluate, and work out which chunks answer each one.

    Raises:
        ValueError: If the sample asks for more questions than are evaluable.
    """
    evaluable: list[EvalQuestion] = []
    excluded: list[str] = []
    judgements: dict[str, Judgements] = {}
    for question in sorted(evaluation_set.questions, key=lambda item: item.question_id):
        relevance = chunk_relevance(question, chunks_by_page)
        if relevance and any(grade > 0 for grade in question.page_relevance.values()):
            evaluable.append(question)
            judgements[question.question_id] = relevance
        else:
            excluded.append(question.question_id)
    if excluded:
        logger.warning(
            "questions excluded: no chunk covers the answer",
            extra={"count": len(excluded)},
        )
    if not evaluable:
        raise ValueError("the evaluation set has no question whose answer any chunk covers")

    limit = config.corpus.query_limit
    if limit is None:
        return tuple(evaluable), tuple(excluded), judgements
    if limit > len(evaluable):
        raise ValueError(
            f"query_limit {limit} exceeds the {len(evaluable)} evaluable questions in the corpus"
        )
    # A seeded, reproducible sample is the point here; S311 concerns cryptographic randomness.
    sample = random.Random(config.seed).sample(evaluable, limit)  # noqa: S311
    chosen = tuple(sorted(sample, key=lambda item: item.question_id))
    return chosen, tuple(excluded), judgements


def _question_result(
    question: EvalQuestion,
    retrieval: Retrieval,
    chunk_judgements: Judgements,
    cutoffs: Sequence[int],
) -> QuestionResult:
    chunk_ranking = tuple(item.chunk.chunk_id for item in retrieval.chunks)
    page_ranking = tuple(page.page_id for page in retrieval.pages)
    metrics: dict[str, float] = {}
    for level, ranking, judgements in (
        ("passage", chunk_ranking, chunk_judgements),
        ("page", page_ranking, dict(question.page_relevance)),
    ):
        for name, metric in _METRICS:
            for k in cutoffs:
                metrics[f"{level}.{name}@{k}"] = metric(ranking, judgements, k)
    return QuestionResult(
        question_id=question.question_id,
        answer_type=question.answer_type,
        chunk_ranking=chunk_ranking,
        page_ranking=page_ranking,
        metrics=metrics,
    )


def _average(results: Sequence[QuestionResult]) -> dict[str, float]:
    if not results:
        return {}
    return {
        name.replace(".rr@", ".mrr@"): mean_over_queries(result.metrics[name] for result in results)
        for name in results[0].metrics
    }


def _summarise(latencies_ms: Sequence[float]) -> dict[str, float]:
    ordered = sorted(latencies_ms)
    if not ordered:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    # Nearest-rank percentile: defined for any sample size, including a single question.
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    return {
        "mean": statistics.fmean(ordered),
        "p50": statistics.median(ordered),
        "p95": p95,
        "max": ordered[-1],
    }
