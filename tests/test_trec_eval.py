"""The metric code agrees with trec_eval, the reference implementation, on random rankings.

trec_eval is the tool the information retrieval community checks numbers with. Hand-worked tests
pin the conventions; this test shows those conventions are trec_eval's own, on inputs nobody
chose: graded judgements, relevant documents never retrieved, rankings shorter than the cutoff.
Scores fall strictly with rank, so trec_eval's own tie-breaking never comes into play.
"""

import pytest
import pytrec_eval
from hypothesis import given, settings
from hypothesis import strategies as st

from groundwork.metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank_at_k

CUTOFFS = (1, 5, 10)
DOCUMENTS = [f"d{i:02d}" for i in range(30)]


@st.composite
def queries(draw: st.DrawFn) -> tuple[list[str], dict[str, int]]:
    ranking = draw(st.lists(st.sampled_from(DOCUMENTS), min_size=1, max_size=15, unique=True))
    judged = draw(st.lists(st.sampled_from(DOCUMENTS), min_size=1, max_size=10, unique=True))
    grades = {doc: draw(st.integers(min_value=0, max_value=3)) for doc in judged}
    # Every evaluated question has at least one relevant document; the pipeline excludes the rest.
    grades[judged[0]] = draw(st.integers(min_value=1, max_value=3))
    return ranking, grades


@settings(max_examples=300)
@given(query=queries())
def test_every_metric_matches_trec_eval(query: tuple[list[str], dict[str, int]]) -> None:
    ranking, grades = query
    measures = {f"P_{k}" for k in CUTOFFS} | {f"recall_{k}" for k in CUTOFFS}
    measures |= {f"ndcg_cut_{k}" for k in CUTOFFS} | {"recip_rank"}
    evaluator = pytrec_eval.RelevanceEvaluator({"q": grades}, measures)
    run = {"q": {doc: float(len(ranking) - rank) for rank, doc in enumerate(ranking)}}
    reference = evaluator.evaluate(run)["q"]

    for k in CUTOFFS:
        assert precision_at_k(ranking, grades, k) == pytest.approx(reference[f"P_{k}"], abs=1e-12)
        assert recall_at_k(ranking, grades, k) == pytest.approx(reference[f"recall_{k}"], abs=1e-12)
        assert ndcg_at_k(ranking, grades, k) == pytest.approx(reference[f"ndcg_cut_{k}"], abs=1e-12)
    # trec_eval's reciprocal rank has no cutoff; at a cutoff past the ranking's end ours equals it.
    assert reciprocal_rank_at_k(ranking, grades, len(DOCUMENTS)) == pytest.approx(
        reference["recip_rank"], abs=1e-12
    )
