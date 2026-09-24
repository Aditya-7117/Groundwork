"""Retrieval metrics for a single query: precision@k, recall@k, nDCG@k and reciprocal rank at k.

The conventions match trec_eval, the reference implementation that published retrieval baselines
are computed with, so numbers from this module are comparable with those baselines. The reasoning
is recorded in docs/decisions/0002-metric-conventions.md.

- Relevance judgements map a document id to an integer grade. A document is relevant when its
  grade is greater than zero. A document with no judgement counts as grade zero.
- nDCG uses linear gain (the grade itself) with a 1 / log2(rank + 1) discount. The ideal ranking
  is built from every judged document for the query, whether or not it was retrieved.
- Each metric scores one query. Averaging over queries is a separate step, so the caller decides
  explicitly which queries take part.
"""

import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from statistics import fmean

type Judgements = Mapping[str, int]
type Metric = Callable[[Sequence[str], Judgements, int], float]


def recall_at_k(ranking: Sequence[str], judgements: Judgements, k: int) -> float:
    """Return the fraction of relevant documents that appear in the top k of the ranking.

    Args:
        ranking: Document ids in rank order, best first.
        judgements: Relevance grade per judged document id for this query.
        k: Number of top-ranked positions to consider.

    Returns:
        A value in [0, 1].

    Raises:
        ValueError: If the input is malformed or the query has no relevant document.
    """
    relevant = _relevant_ids(ranking, judgements, k)
    found = sum(1 for doc_id in ranking[:k] if doc_id in relevant)
    return found / len(relevant)


def ndcg_at_k(ranking: Sequence[str], judgements: Judgements, k: int) -> float:
    """Return normalised discounted cumulative gain over the top k of the ranking.

    DCG sums each retrieved document's grade divided by log2(rank + 1), so a relevant document
    counts for less the further down it sits. Dividing by the DCG of the best possible ranking
    puts the result in [0, 1].

    Args:
        ranking: Document ids in rank order, best first.
        judgements: Relevance grade per judged document id for this query.
        k: Number of top-ranked positions to consider.

    Returns:
        A value in [0, 1].

    Raises:
        ValueError: If the input is malformed or the query has no relevant document.
    """
    _relevant_ids(ranking, judgements, k)
    gains = [judgements.get(doc_id, 0) for doc_id in ranking[:k]]
    ideal_gains = sorted((grade for grade in judgements.values() if grade > 0), reverse=True)[:k]
    return _dcg(gains) / _dcg(ideal_gains)


def reciprocal_rank_at_k(ranking: Sequence[str], judgements: Judgements, k: int) -> float:
    """Return 1 / rank of the first relevant document in the top k, or 0 if there is none.

    Averaged over queries with mean_over_queries, this gives mean reciprocal rank (MRR@k).

    Args:
        ranking: Document ids in rank order, best first.
        judgements: Relevance grade per judged document id for this query.
        k: Number of top-ranked positions to consider.

    Returns:
        A value in [0, 1].

    Raises:
        ValueError: If the input is malformed or the query has no relevant document.
    """
    relevant = _relevant_ids(ranking, judgements, k)
    for rank, doc_id in enumerate(ranking[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def precision_at_k(ranking: Sequence[str], judgements: Judgements, k: int) -> float:
    """Return the fraction of the top k retrieved items that are relevant.

    The denominator is the number of items actually retrieved, min(k, len(ranking)), rather than
    k itself, so a ranking shorter than k is not penalised for positions it never had a chance to
    fill.

    Args:
        ranking: Document ids in rank order, best first.
        judgements: Relevance grade per judged document id for this query.
        k: Number of top-ranked positions to consider.

    Returns:
        A value in [0, 1].

    Raises:
        ValueError: If the input is malformed or the query has no relevant document.
    """
    relevant = _relevant_ids(ranking, judgements, k)
    if not ranking:
        return 0.0
    retrieved = ranking[:k]
    found = sum(1 for doc_id in retrieved if doc_id in relevant)
    return found / len(retrieved)


def mean_over_queries(scores: Iterable[float]) -> float:
    """Return the arithmetic mean of per-query scores, each query weighted equally.

    Raises:
        ValueError: If there are no scores, since an average of nothing would read as zero.
    """
    values = list(scores)
    if not values:
        raise ValueError("cannot average a metric over zero queries; need at least one query")
    return fmean(values)


def _relevant_ids(ranking: Sequence[str], judgements: Judgements, k: int) -> frozenset[str]:
    """Validate one query's input and return the ids of its relevant documents."""
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    duplicates = sorted(doc_id for doc_id, count in Counter(ranking).items() if count > 1)
    if duplicates:
        raise ValueError(f"ranking contains duplicate document ids: {', '.join(duplicates)}")
    negative = sorted(doc_id for doc_id, grade in judgements.items() if grade < 0)
    if negative:
        raise ValueError(f"negative relevance grade for document ids: {', '.join(negative)}")
    relevant = frozenset(doc_id for doc_id, grade in judgements.items() if grade > 0)
    if not relevant:
        raise ValueError("judgements contain no relevant document, so the metric is undefined")
    return relevant


def _dcg(gains: Iterable[int]) -> float:
    # fsum is exactly rounded and independent of summation order, so a perfect ranking scores
    # exactly 1.0 however ties among equal grades happen to be ordered.
    return math.fsum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
