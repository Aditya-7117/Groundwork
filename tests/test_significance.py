"""Significance tests against cases small enough to work out by hand."""

import itertools

import numpy as np
import pytest

from groundwork.significance import bootstrap_interval, compare, holm, randomisation_test


class TestRandomisationTest:
    def test_matches_the_exact_answer_from_every_sign_flip(self) -> None:
        # Differences 1, 2, 3 have mean 2. Of the 8 ways to flip their signs, only all-plus and
        # all-minus reach an absolute mean of 2, so the exact p-value is 2/8 = 0.25.
        differences = [1.0, 2.0, 3.0]
        exact = np.mean(
            [
                abs(np.mean(np.multiply(signs, differences))) >= 2 - 1e-12
                for signs in itertools.product((-1, 1), repeat=3)
            ]
        )
        assert exact == 0.25
        p = randomisation_test(differences, [0.0, 0.0, 0.0], trials=20_000, seed=1)
        assert p == pytest.approx(0.25, abs=0.01)

    def test_identical_setups_are_not_significantly_different(self) -> None:
        scores = [0.0, 1.0, 1.0, 0.0, 0.5]
        assert randomisation_test(scores, scores, trials=1_000, seed=1) == 1.0

    def test_a_consistent_large_difference_is_significant(self) -> None:
        # 200 questions, every one better by 1: only 2 of 2^200 flips reach that mean.
        p = randomisation_test([1.0] * 200, [0.0] * 200, trials=1_000, seed=1)
        assert p == pytest.approx(1 / 1_001)

    def test_the_sign_of_the_difference_does_not_matter(self) -> None:
        a, b = [0.9, 0.4, 0.7, 0.2], [0.1, 0.5, 0.3, 0.2]
        assert randomisation_test(a, b, trials=2_000, seed=4) == randomisation_test(
            b, a, trials=2_000, seed=4
        )

    def test_the_same_seed_gives_the_same_p_value(self) -> None:
        a, b = [0.3, 0.8, 0.1, 0.9], [0.4, 0.2, 0.2, 0.7]
        assert randomisation_test(a, b, seed=7) == randomisation_test(a, b, seed=7)

    @pytest.mark.parametrize(
        ("a", "b", "message"),
        [([1.0], [1.0, 2.0], "differ in length"), ([], [], "empty")],
    )
    def test_rejects_bad_samples(self, a: list[float], b: list[float], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            randomisation_test(a, b, seed=1)


class TestBootstrapInterval:
    def test_a_constant_sample_has_a_zero_width_interval(self) -> None:
        assert bootstrap_interval([0.5] * 50, trials=500, seed=1) == (0.5, 0.5)

    def test_the_interval_contains_the_sample_mean(self) -> None:
        values = np.random.default_rng(3).random(300).tolist()
        low, high = bootstrap_interval(values, trials=2_000, seed=1)
        assert low < float(np.mean(values)) < high

    def test_the_width_matches_the_standard_error(self) -> None:
        # For a large sample the 95% interval is close to mean +/- 1.96 standard errors.
        values = np.random.default_rng(5).random(2_000)
        low, high = bootstrap_interval(values.tolist(), trials=4_000, seed=2)
        standard_error = values.std(ddof=1) / np.sqrt(values.size)
        assert high - low == pytest.approx(2 * 1.96 * standard_error, rel=0.1)

    @pytest.mark.parametrize(
        ("values", "level", "message"),
        [([], 0.95, "empty"), ([1.0], 1.0, "between 0 and 1")],
    )
    def test_rejects_bad_input(self, values: list[float], level: float, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            bootstrap_interval(values, seed=1, level=level)


class TestHolm:
    def test_hand_worked_adjustment(self) -> None:
        # Sorted: 0.01, 0.03, 0.04. Multiply by 3, 2, 1: 0.03, 0.06, 0.04. Adjusted values may
        # not fall as p grows, so the last becomes 0.06.
        adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03})
        assert adjusted == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})

    def test_adjusted_values_are_capped_at_one(self) -> None:
        assert holm({"a": 0.6, "b": 0.7}) == pytest.approx({"a": 1.0, "b": 1.0})

    def test_one_comparison_is_left_unchanged(self) -> None:
        assert holm({"only": 0.02}) == {"only": 0.02}

    def test_rejects_a_p_value_outside_zero_to_one(self) -> None:
        with pytest.raises(ValueError, match="outside 0 to 1"):
            holm({"a": 1.5})


PER_QUESTION = {
    "strong": {"q1": {"r": 1.0}, "q2": {"r": 1.0}, "q3": {"r": 1.0}, "q4": {"r": 0.0}},
    "weak": {"q1": {"r": 0.0}, "q2": {"r": 1.0}, "q3": {"r": 0.0}, "q4": {"r": 0.0}},
    "other": {"q2": {"r": 1.0}, "q3": {"r": 1.0}, "q9": {"r": 0.0}},
}


class TestCompare:
    def test_reports_means_difference_interval_and_both_p_values(self) -> None:
        [result] = compare(PER_QUESTION, [("strong", "weak")], "r", trials=2_000, seed=1)
        assert (result.questions, result.mean_a, result.mean_b) == (4, 0.75, 0.25)
        assert result.difference == pytest.approx(0.5)
        assert result.ci_low <= 0.5 <= result.ci_high
        # Differences 1, 0, 1, 0: flips reach |mean| 0.5 only when both 1s share a sign, 1/2.
        assert result.p_value == pytest.approx(0.5, abs=0.03)
        assert result.p_holm == result.p_value

    def test_only_shared_questions_are_compared(self) -> None:
        [result] = compare(PER_QUESTION, [("strong", "other")], "r", trials=100, seed=1)
        assert result.questions == 2

    def test_correction_runs_across_the_whole_family(self) -> None:
        results = compare(
            PER_QUESTION, [("strong", "weak"), ("weak", "other")], "r", trials=500, seed=1
        )
        assert all(r.p_holm >= r.p_value for r in results)

    def test_rejects_an_unknown_setup(self) -> None:
        with pytest.raises(ValueError, match="unknown setup 'missing'"):
            compare(PER_QUESTION, [("strong", "missing")], "r", seed=1)
