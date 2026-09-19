"""Running one experiment: chunk, index, retrieve, and score every evaluable query.

The run is a pure function of the config and the corpus. The only randomness is the optional
query sample, which draws from a random.Random seeded by the config, so the same inputs always
give the same rankings and metrics. Wall-clock timings are the one part of a result that varies
between runs.
"""

import logging
import math
import random
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from groundwork.bm25 import BM25Index
from groundwork.chunking import chunk_documents
from groundwork.config import ExperimentConfig
from groundwork.corpus import Corpus
from groundwork.metrics import mean_over_queries, ndcg_at_k, recall_at_k, reciprocal_rank_at_k
from groundwork.ranking import RankedDocument, rank_documents_by_best_chunk

logger = logging.getLogger(__name__)

type Clock = Callable[[], float]


@dataclass(frozen=True, slots=True, kw_only=True)
class QueryResult:
    """One query's ranking and its metrics.

    Attributes:
        query_id: The query.
        ranking: Retrieved documents, best first, at most the configured depth.
        metrics: recall@k, ndcg@k and rr@k (reciprocal rank) for each configured cutoff.
    """

    query_id: str
    ranking: tuple[RankedDocument, ...]
    metrics: Mapping[str, float]


@dataclass(frozen=True, slots=True, kw_only=True)
class ExperimentResult:
    """Everything a run measured.

    Attributes:
        queries: Per-query results, ordered by query id.
        aggregate: Each metric averaged over queries. Mean reciprocal rank is reported as mrr@k.
        excluded_query_ids: Judged queries with no document graded above zero. The metrics are
            undefined for them, so they are left out and listed here instead.
        chunk_count: Number of chunks indexed.
        stage_seconds: Wall-clock seconds spent in each pipeline stage.
        retrieval_latency_ms: Per-query retrieval latency summary: mean, p50, p95 and max.
    """

    queries: tuple[QueryResult, ...]
    aggregate: Mapping[str, float]
    excluded_query_ids: tuple[str, ...]
    chunk_count: int
    stage_seconds: Mapping[str, float]
    retrieval_latency_ms: Mapping[str, float]


def run_experiment(
    config: ExperimentConfig, corpus: Corpus, *, clock: Clock = time.perf_counter
) -> ExperimentResult:
    """Run one configuration over a corpus and score it.

    Args:
        config: The experiment definition.
        corpus: Documents, queries and judgements to evaluate against.
        clock: Monotonic clock in seconds, used only for timings.

    Raises:
        ValueError: If the config asks for more queries than the corpus can evaluate.
    """
    query_ids, excluded = _select_queries(config, corpus)
    stage_seconds: dict[str, float] = {}

    started = clock()
    chunks = chunk_documents(corpus.documents.values(), config.chunking)
    stage_seconds["chunking"] = clock() - started

    started = clock()
    index = BM25Index(chunks, k1=config.retrieval.k1, b=config.retrieval.b)
    stage_seconds["indexing"] = clock() - started

    rankings: dict[str, tuple[RankedDocument, ...]] = {}
    latencies_ms: list[float] = []
    for query_id in query_ids:
        started = clock()
        ranked = rank_documents_by_best_chunk(
            index.search(corpus.queries[query_id].text), depth=config.retrieval.depth
        )
        latencies_ms.append((clock() - started) * 1000)
        rankings[query_id] = tuple(ranked)
    stage_seconds["retrieval"] = sum(latencies_ms) / 1000

    started = clock()
    queries = tuple(
        QueryResult(
            query_id=query_id,
            ranking=rankings[query_id],
            metrics=_score_query(
                [ranked.doc_id for ranked in rankings[query_id]],
                corpus.judgements[query_id],
                config.evaluation.cutoffs,
            ),
        )
        for query_id in query_ids
    )
    aggregate = {
        name.replace("rr@", "mrr@"): mean_over_queries(query.metrics[name] for query in queries)
        for name in queries[0].metrics
    }
    stage_seconds["evaluation"] = clock() - started

    logger.info(
        "experiment finished",
        extra={"experiment": config.name, "queries": len(queries), "chunks": len(chunks)},
    )
    return ExperimentResult(
        queries=queries,
        aggregate=aggregate,
        excluded_query_ids=excluded,
        chunk_count=len(chunks),
        stage_seconds=stage_seconds,
        retrieval_latency_ms=_summarise(latencies_ms),
    )


def _select_queries(config: ExperimentConfig, corpus: Corpus) -> tuple[list[str], tuple[str, ...]]:
    """Return the query ids to evaluate, and the judged ids excluded for having no relevant doc."""
    evaluable = sorted(
        query_id
        for query_id, judgements in corpus.judgements.items()
        if any(grade > 0 for grade in judgements.values())
    )
    excluded = tuple(sorted(set(corpus.judgements) - set(evaluable)))
    if excluded:
        logger.warning(
            "queries excluded: no document graded above zero",
            extra={"count": len(excluded), "query_ids": list(excluded)},
        )
    if not evaluable:
        raise ValueError("the corpus has no query with a relevant document to evaluate")

    limit = config.corpus.query_limit
    if limit is None:
        return evaluable, excluded
    if limit > len(evaluable):
        raise ValueError(
            f"query_limit {limit} exceeds the {len(evaluable)} evaluable queries in the corpus"
        )
    # A seeded, reproducible sample is the point here; S311 concerns cryptographic randomness.
    sample = random.Random(config.seed).sample(evaluable, limit)  # noqa: S311
    return sorted(sample), excluded


def _score_query(
    ranking: Sequence[str], judgements: Mapping[str, int], cutoffs: Sequence[int]
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, metric in (
        ("recall", recall_at_k),
        ("ndcg", ndcg_at_k),
        ("rr", reciprocal_rank_at_k),
    ):
        for k in cutoffs:
            metrics[f"{name}@{k}"] = metric(ranking, judgements, k)
    return metrics


def _summarise(latencies_ms: Sequence[float]) -> dict[str, float]:
    ordered = sorted(latencies_ms)
    # Nearest-rank percentile: defined for any sample size, including a single query.
    p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
    return {
        "mean": statistics.fmean(ordered),
        "p50": statistics.median(ordered),
        "p95": p95,
        "max": ordered[-1],
    }
