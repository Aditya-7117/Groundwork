"""Agreement statistics against textbook examples."""

import pytest

from groundwork.agreement import agreement, cohens_kappa, roc_auc


def labels(counts: dict[tuple[str, str], int]) -> tuple[list[str], list[str]]:
    first: list[str] = []
    second: list[str] = []
    for (a, b), count in counts.items():
        first += [a] * count
        second += [b] * count
    return first, second


class TestCohensKappa:
    def test_the_textbook_example(self) -> None:
        # 50 items: both yes 20, yes/no 5, no/yes 10, both no 15. Observed agreement 0.7;
        # chance agreement 0.5 * 0.6 + 0.5 * 0.4 = 0.5; kappa = (0.7 - 0.5) / 0.5 = 0.4.
        first, second = labels(
            {("yes", "yes"): 20, ("yes", "no"): 5, ("no", "yes"): 10, ("no", "no"): 15}
        )
        assert cohens_kappa(first, second) == pytest.approx(0.4)

    def test_perfect_agreement_is_one(self) -> None:
        assert cohens_kappa(["a", "b", "c"], ["a", "b", "c"]) == pytest.approx(1.0)

    def test_agreement_no_better_than_chance_is_zero(self) -> None:
        first, second = labels({("a", "a"): 1, ("a", "b"): 1, ("b", "a"): 1, ("b", "b"): 1})
        assert cohens_kappa(first, second) == pytest.approx(0.0)

    def test_a_dominant_label_does_not_flatter_the_raters(self) -> None:
        # 90% raw agreement, but only because both raters nearly always say "supported".
        first, second = labels({("s", "s"): 90, ("s", "n"): 5, ("n", "s"): 5})
        assert cohens_kappa(first, second) < 0

    def test_one_shared_label_throughout_is_undefined(self) -> None:
        with pytest.raises(ValueError, match="undefined"):
            cohens_kappa(["s", "s"], ["s", "s"])

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="different numbers of items"):
            cohens_kappa(["a"], ["a", "b"])


def test_agreement_reports_the_confusion_table_and_an_interval() -> None:
    first, second = labels(
        {("yes", "yes"): 20, ("yes", "no"): 5, ("no", "yes"): 10, ("no", "no"): 15}
    )
    result = agreement(first, second, seed=1, trials=500)
    assert result.items == 50
    assert result.observed == pytest.approx(0.7)
    assert result.confusion == {"yes": {"yes": 20, "no": 5}, "no": {"yes": 10, "no": 15}}
    assert result.kappa_low < 0.4 < result.kappa_high


class TestRocAuc:
    def test_hand_worked_pairs(self) -> None:
        # Positives 0.9 and 0.3, negatives 0.8 and 0.1: three of four pairs are ordered right.
        assert roc_auc([0.9, 0.8, 0.3, 0.1], [True, False, True, False]) == pytest.approx(0.75)

    def test_ties_count_half(self) -> None:
        assert roc_auc([0.5, 0.5], [True, False]) == pytest.approx(0.5)

    def test_needs_both_classes(self) -> None:
        with pytest.raises(ValueError, match="at least one positive and one negative"):
            roc_auc([0.1, 0.2], [True, True])
