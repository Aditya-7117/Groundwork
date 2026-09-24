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
from dataclasses import dataclass

from groundwork.bm25 import BM25Index
from groundwork.chunking import Chunk, ChunkingSettings, chunk_pages
from groundwork.config import ExperimentConfig, RetrievalConfig
from groundwork.evaluation import EvalQuestion, EvaluationSet, chunk_relevance
from groundwork.metrics import (
    Judgements,
    mean_over_queries,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)
from groundwork.ranking import rank_pages_by_best_chunk

logger = logging.getLogger(__name__)

type Clock = Callable[[], float]

_METRICS = (
    ("precision", precision_at_k),
    ("recall", recall_at_k),
    ("ndcg", ndcg_at_k),
    ("rr", reciprocal_rank_at_k),
)


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
        stage_seconds: Wall-clock seconds per pipeline stage.
        retrieval_latency_ms: Per-question retrieval latency: mean, p50, p95 and max.
    """

    questions: tuple[QuestionResult, ...]
    aggregate: Mapping[str, float]
    by_answer_type: Mapping[str, Mapping[str, float]]
    chunk_count: int
    excluded_question_ids: tuple[str, ...]
    stage_seconds: Mapping[str, float]
    retrieval_latency_ms: Mapping[str, float]


def run_experiment(
    config: ExperimentConfig, evaluation_set: EvaluationSet, *, clock: Clock = time.perf_counter
) -> ExperimentResult:
    """Run one configuration over an evaluation set and score it.

    Raises:
        ValueError: If the config asks for more questions than the set can evaluate.
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

    started = clock()
    index = _build_index(chunks, config.retrieval)
    stage_seconds["indexing"] = clock() - started

    questions, excluded, judgements = _select(config, evaluation_set, chunks_by_page)

    rankings: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    latencies_ms: list[float] = []
    depth = config.retrieval.depth
    for question in questions:
        started = clock()
        scored = index.search(question.text)
        chunk_ranking = tuple(item.chunk.chunk_id for item in scored[:depth])
        page_ranking = tuple(page.page_id for page in rank_pages_by_best_chunk(scored, depth=depth))
        latencies_ms.append((clock() - started) * 1000)
        rankings[question.question_id] = (chunk_ranking, page_ranking)
    stage_seconds["retrieval"] = sum(latencies_ms) / 1000

    started = clock()
    results = tuple(
        QuestionResult(
            question_id=question.question_id,
            answer_type=question.answer_type,
            chunk_ranking=rankings[question.question_id][0],
            page_ranking=rankings[question.question_id][1],
            metrics=_score(
                rankings[question.question_id],
                judgements[question.question_id],
                dict(question.page_relevance),
                config.evaluation.cutoffs,
            ),
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
    )


def _build_index(chunks: Sequence[Chunk], retrieval: RetrievalConfig) -> BM25Index:
    # An exhaustive match: adding a retrieval method to the config without handling it here is a
    # type error, rather than a config that silently runs BM25.
    match retrieval.method:
        case "bm25":
            return BM25Index(chunks, k1=retrieval.k1, b=retrieval.b, stem=retrieval.stem)


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


def _score(
    rankings: tuple[tuple[str, ...], tuple[str, ...]],
    chunk_judgements: Judgements,
    page_judgements: Judgements,
    cutoffs: Sequence[int],
) -> dict[str, float]:
    chunk_ranking, page_ranking = rankings
    scores: dict[str, float] = {}
    for level, ranking, judgements in (
        ("passage", chunk_ranking, chunk_judgements),
        ("page", page_ranking, page_judgements),
    ):
        for name, metric in _METRICS:
            for k in cutoffs:
                scores[f"{level}.{name}@{k}"] = metric(ranking, judgements, k)
    return scores


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
