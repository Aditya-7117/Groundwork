"""Hand-worked checks for the retrieval metrics.

Every expected value below was worked out by hand before the implementation existed, and the
arithmetic is written out next to it. The tests compare against those literal numbers rather than
recomputing the formula, because a test that recomputes the formula shares any mistake in it.

Discount table used throughout, 1 / log2(rank + 1), to six decimal places:

    rank 1: 1 / log2(2) = 1 / 1        = 1.000000
    rank 2: 1 / log2(3) = 1 / 1.584963 = 0.630930
    rank 3: 1 / log2(4) = 1 / 2        = 0.500000
    rank 4: 1 / log2(5) = 1 / 2.321928 = 0.430677
    rank 5: 1 / log2(6) = 1 / 2.584963 = 0.386853
    rank 6: 1 / log2(7) = 1 / 2.807355 = 0.356207
"""

import pytest

from groundwork.metrics import (
    Metric,
    mean_over_queries,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)

TOLERANCE = 1e-6

# Binary example. Five documents retrieved; three relevant documents exist, d2, d4 and d9, and d9
# was never retrieved.
#
#     rank:      1   2   3   4   5
#     document:  d1  d2  d3  d4  d5
#     relevant:  -   yes -   yes -
BINARY_RANKING = ["d1", "d2", "d3", "d4", "d5"]
BINARY_JUDGEMENTS = {"d2": 1, "d4": 1, "d9": 1}

# Graded example from the worked example on Wikipedia's "Discounted cumulative gain" article.
# Six documents retrieved with grades 3, 2, 3, 0, 1, 2. Two more documents, d7 (grade 3) and
# d8 (grade 2), are judged but were not retrieved, so the ideal ordering is 3, 3, 3, 2, 2, 2.
GRADED_RANKING = ["d1", "d2", "d3", "d4", "d5", "d6"]
GRADED_JUDGEMENTS = {"d1": 3, "d2": 2, "d3": 3, "d4": 0, "d5": 1, "d6": 2, "d7": 3, "d8": 2}


class TestInvalidInput:
    """A metric that quietly accepts malformed input produces a plausible wrong number."""

    @pytest.mark.parametrize(
        "metric", [recall_at_k, ndcg_at_k, reciprocal_rank_at_k, precision_at_k]
    )
    @pytest.mark.parametrize("k", [0, -1])
    def test_rejects_k_below_one(self, metric: Metric, k: int) -> None:
        with pytest.raises(ValueError, match="k must be at least 1"):
            metric(BINARY_RANKING, BINARY_JUDGEMENTS, k)

    @pytest.mark.parametrize(
        "metric", [recall_at_k, ndcg_at_k, reciprocal_rank_at_k, precision_at_k]
    )
    def test_rejects_duplicate_documents_in_ranking(self, metric: Metric) -> None:
        # A duplicate would let one relevant document count twice towards recall and DCG.
        with pytest.raises(ValueError, match="duplicate document ids: d2"):
            metric(["d1", "d2", "d2"], BINARY_JUDGEMENTS, 3)

    @pytest.mark.parametrize(
        "metric", [recall_at_k, ndcg_at_k, reciprocal_rank_at_k, precision_at_k]
    )
    def test_rejects_query_with_no_relevant_document(self, metric: Metric) -> None:
        # Recall divides by the number of relevant documents and nDCG by the ideal DCG. Both are
        # zero here, so the metric is undefined and the caller must exclude the query explicitly.
        with pytest.raises(ValueError, match="no relevant document"):
            metric(BINARY_RANKING, {"d1": 0, "d2": 0}, 5)

    @pytest.mark.parametrize(
        "metric", [recall_at_k, ndcg_at_k, reciprocal_rank_at_k, precision_at_k]
    )
    def test_rejects_negative_grades(self, metric: Metric) -> None:
        with pytest.raises(ValueError, match="negative relevance grade"):
            metric(BINARY_RANKING, {"d2": 1, "d3": -1}, 5)

    def test_mean_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError, match="at least one query"):
            mean_over_queries([])


class TestRecallAtK:
    def test_nothing_relevant_in_top_one(self) -> None:
        # Top 1 is {d1}. No relevant document, so 0 / 3.
        assert recall_at_k(BINARY_RANKING, BINARY_JUDGEMENTS, 1) == pytest.approx(0.0)

    def test_one_of_three_in_top_three(self) -> None:
        # Top 3 is {d1, d2, d3}. Only d2 is relevant, so 1 / 3.
        assert recall_at_k(BINARY_RANKING, BINARY_JUDGEMENTS, 3) == pytest.approx(
            0.333333, abs=TOLERANCE
        )

    def test_two_of_three_in_top_five(self) -> None:
        # Top 5 holds d2 and d4. d9 was never retrieved, so 2 / 3.
        assert recall_at_k(BINARY_RANKING, BINARY_JUDGEMENTS, 5) == pytest.approx(
            0.666667, abs=TOLERANCE
        )

    def test_k_beyond_ranking_length_counts_missing_positions_as_misses(self) -> None:
        # Only five documents were retrieved, so the top 10 is the same five: still 2 / 3.
        assert recall_at_k(BINARY_RANKING, BINARY_JUDGEMENTS, 10) == pytest.approx(
            0.666667, abs=TOLERANCE
        )

    def test_grade_zero_is_not_relevant(self) -> None:
        # d4 is judged but graded 0, so the relevant set is {d1, d2, d3, d5, d6, d7, d8}: seven
        # documents. The top 6 holds d1, d2, d3, d5 and d6, so 5 / 7.
        assert recall_at_k(GRADED_RANKING, GRADED_JUDGEMENTS, 6) == pytest.approx(
            0.714286, abs=TOLERANCE
        )

    def test_empty_ranking_scores_zero(self) -> None:
        assert recall_at_k([], BINARY_JUDGEMENTS, 5) == pytest.approx(0.0)


class TestNdcgAtK:
    def test_binary_at_five(self) -> None:
        # DCG@5  = 1 x 0.630930 (d2, rank 2) + 1 x 0.430677 (d4, rank 4) = 1.061607
        # IDCG@5 = three relevant documents placed at ranks 1 to 3
        #        = 1.000000 + 0.630930 + 0.500000                           = 2.130930
        # nDCG@5 = 1.061607 / 2.130930                                      = 0.498189
        assert ndcg_at_k(BINARY_RANKING, BINARY_JUDGEMENTS, 5) == pytest.approx(
            0.498189, abs=TOLERANCE
        )

    def test_binary_at_three_keeps_ideal_over_all_relevant(self) -> None:
        # DCG@3  = 0.630930 (d2 at rank 2)
        # IDCG@3 = 1.000000 + 0.630930 + 0.500000 = 2.130930. d9 counts towards the ideal even
        #          though it was never retrieved; leaving it out would inflate the score.
        # nDCG@3 = 0.630930 / 2.130930 = 0.296082
        assert ndcg_at_k(BINARY_RANKING, BINARY_JUDGEMENTS, 3) == pytest.approx(
            0.296082, abs=TOLERANCE
        )

    def test_nothing_relevant_in_top_one(self) -> None:
        assert ndcg_at_k(BINARY_RANKING, BINARY_JUDGEMENTS, 1) == pytest.approx(0.0)

    def test_graded_wikipedia_example(self) -> None:
        #   rank   discount   actual grade   actual gain   ideal grade   ideal gain
        #   1      1.000000   3              3.000000      3             3.000000
        #   2      0.630930   2              1.261860      3             1.892790
        #   3      0.500000   3              1.500000      3             1.500000
        #   4      0.430677   0              0.000000      2             0.861354
        #   5      0.386853   1              0.386853      2             0.773706
        #   6      0.356207   2              0.712414      2             0.712414
        #   sum                              6.861127                    8.740264
        #
        # The ideal sum is 8.740262 before the terms are rounded to six places.
        # nDCG@6 = 6.861127 / 8.740262 = 0.785002, which Wikipedia rounds to 0.785.
        #
        # This example also pins the gain function. With exponential gain (2^grade - 1) the
        # answer would differ, so a switch of convention cannot pass silently.
        assert ndcg_at_k(GRADED_RANKING, GRADED_JUDGEMENTS, 6) == pytest.approx(
            0.785002, abs=TOLERANCE
        )

    def test_unjudged_document_counts_as_grade_zero(self) -> None:
        # dx is not in the judgements at all. Ranking [dx, d2]:
        # DCG@2  = 0 x 1.000000 + 1 x 0.630930 = 0.630930
        # IDCG@2 = three relevant documents, top two positions: 1.000000 + 0.630930 = 1.630930
        # nDCG@2 = 0.630930 / 1.630930 = 0.386853
        assert ndcg_at_k(["dx", "d2"], BINARY_JUDGEMENTS, 2) == pytest.approx(
            0.386853, abs=TOLERANCE
        )


class TestReciprocalRankAtK:
    def test_first_relevant_at_rank_two(self) -> None:
        # d2 is the first relevant document, at rank 2, so 1 / 2.
        assert reciprocal_rank_at_k(BINARY_RANKING, BINARY_JUDGEMENTS, 10) == pytest.approx(0.5)

    def test_first_relevant_beyond_cutoff_scores_zero(self) -> None:
        # The cutoff is rank 1 and d1 is not relevant.
        assert reciprocal_rank_at_k(BINARY_RANKING, BINARY_JUDGEMENTS, 1) == pytest.approx(0.0)

    def test_grade_zero_does_not_count_as_first_relevant(self) -> None:
        # Ranking [d4, d1] with d4 judged grade 0 and d1 grade 3. The first relevant document
        # is d1 at rank 2, so 1 / 2.
        assert reciprocal_rank_at_k(["d4", "d1"], GRADED_JUDGEMENTS, 2) == pytest.approx(0.5)


class TestPrecisionAtK:
    def test_nothing_relevant_in_top_one(self) -> None:
        # Top 1 is {d1}. d1 is not relevant, so 0 / 1.
        assert precision_at_k(BINARY_RANKING, BINARY_JUDGEMENTS, 1) == pytest.approx(0.0)

    def test_one_of_three_in_top_three(self) -> None:
        # Top 3 is {d1, d2, d3}. Only d2 is relevant, so 1 / 3.
        assert precision_at_k(BINARY_RANKING, BINARY_JUDGEMENTS, 3) == pytest.approx(
            0.333333, abs=TOLERANCE
        )

    def test_two_of_five_in_top_five(self) -> None:
        # Top 5 is every document retrieved. d2 and d4 are relevant, so 2 / 5.
        assert precision_at_k(BINARY_RANKING, BINARY_JUDGEMENTS, 5) == pytest.approx(0.4)

    def test_k_beyond_ranking_length_counts_empty_positions_as_misses(self) -> None:
        # Only five documents were retrieved. Positions 6 to 10 are empty and count as misses,
        # as in trec_eval, so 2 / 10.
        assert precision_at_k(BINARY_RANKING, BINARY_JUDGEMENTS, 10) == pytest.approx(0.2)

    def test_grade_zero_is_not_relevant(self) -> None:
        # d4 is judged but graded 0. The top 6 holds d1, d2, d3, d5 and d6 as relevant and d4 as
        # not, so 5 / 6.
        assert precision_at_k(GRADED_RANKING, GRADED_JUDGEMENTS, 6) == pytest.approx(
            0.833333, abs=TOLERANCE
        )

    def test_empty_ranking_scores_zero(self) -> None:
        assert precision_at_k([], BINARY_JUDGEMENTS, 5) == pytest.approx(0.0)


class TestMeanOverQueries:
    def test_mean_reciprocal_rank(self) -> None:
        # Three queries whose first relevant documents sit at ranks 1, 3 and nowhere:
        # MRR = (1/1 + 1/3 + 0) / 3 = 1.333333 / 3 = 0.444444
        assert mean_over_queries([1.0, 1 / 3, 0.0]) == pytest.approx(0.444444, abs=TOLERANCE)
