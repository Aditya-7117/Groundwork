"""Properties every correct implementation of the metrics must satisfy, on generated inputs.

The hand-worked tests pin exact values on a few inputs. These check invariants across thousands of
random rankings and judgements, which catches mistakes that only show up on shapes nobody thought
to work out by hand.
"""

from hypothesis import given
from hypothesis import strategies as st

from groundwork.metrics import (
    Judgements,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)

DOC_IDS = [f"d{i}" for i in range(12)]

rankings = st.lists(st.sampled_from(DOC_IDS), unique=True, max_size=len(DOC_IDS))
judgement_sets = st.dictionaries(
    keys=st.sampled_from(DOC_IDS), values=st.integers(min_value=0, max_value=3), min_size=1
).filter(lambda judgements: any(grade > 0 for grade in judgements.values()))
cutoffs = st.integers(min_value=1, max_value=len(DOC_IDS) + 3)


def _ideal_ranking(judgements: Judgements) -> list[str]:
    relevant = [doc_id for doc_id, grade in judgements.items() if grade > 0]
    return sorted(relevant, key=lambda doc_id: (-judgements[doc_id], doc_id))


@given(ranking=rankings, judgements=judgement_sets, k=cutoffs)
def test_every_metric_lies_between_zero_and_one(
    ranking: list[str], judgements: Judgements, k: int
) -> None:
    for metric in (recall_at_k, ndcg_at_k, reciprocal_rank_at_k, precision_at_k):
        assert 0.0 <= metric(ranking, judgements, k) <= 1.0


@given(ranking=rankings, judgements=judgement_sets, k=cutoffs)
def test_recall_never_decreases_as_k_grows(
    ranking: list[str], judgements: Judgements, k: int
) -> None:
    assert recall_at_k(ranking, judgements, k) <= recall_at_k(ranking, judgements, k + 1)


@given(judgements=judgement_sets, k=cutoffs)
def test_ideal_ranking_scores_one_on_ndcg(judgements: Judgements, k: int) -> None:
    assert ndcg_at_k(_ideal_ranking(judgements), judgements, k) == 1.0


@given(judgements=judgement_sets)
def test_ideal_ranking_has_full_recall_at_number_of_relevant(judgements: Judgements) -> None:
    ideal = _ideal_ranking(judgements)
    assert recall_at_k(ideal, judgements, len(ideal)) == 1.0


@given(ranking=rankings, judgements=judgement_sets, k=cutoffs)
def test_reciprocal_rank_is_one_over_first_relevant_position(
    ranking: list[str], judgements: Judgements, k: int
) -> None:
    score = reciprocal_rank_at_k(ranking, judgements, k)
    positions = [
        rank for rank, doc_id in enumerate(ranking[:k], start=1) if judgements.get(doc_id, 0) > 0
    ]
    expected = 1.0 / positions[0] if positions else 0.0
    assert score == expected


@given(ranking=rankings, judgements=judgement_sets, k=cutoffs)
def test_documents_below_the_cutoff_do_not_change_the_score(
    ranking: list[str], judgements: Judgements, k: int
) -> None:
    truncated = ranking[:k]
    for metric in (recall_at_k, ndcg_at_k, reciprocal_rank_at_k):
        assert metric(ranking, judgements, k) == metric(truncated, judgements, k)


@given(ranking=rankings, judgements=judgement_sets, k=cutoffs)
def test_promoting_a_relevant_document_never_lowers_ndcg(
    ranking: list[str], judgements: Judgements, k: int
) -> None:
    # Swapping a document with the one directly above it, when the lower one has the higher
    # grade, moves gain to a less discounted position, so nDCG cannot fall.
    for position in range(1, len(ranking)):
        above, below = ranking[position - 1], ranking[position]
        if judgements.get(below, 0) > judgements.get(above, 0):
            promoted = list(ranking)
            promoted[position - 1], promoted[position] = below, above
            assert ndcg_at_k(promoted, judgements, k) >= ndcg_at_k(ranking, judgements, k)
