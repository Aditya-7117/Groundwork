"""How far two raters agree: the measure behind "the judge gets judged".

Raw agreement flatters raters when one label dominates: if 90% of answers are supported, two
raters who both always say "supported" agree 90% of the time while judging nothing. Cohen's kappa
corrects for this. It compares observed agreement p_o with the agreement p_e expected if each
rater labelled at random with their own label frequencies:

    kappa = (p_o - p_e) / (1 - p_e)

1 is perfect agreement, 0 is no better than chance, and negative is worse than chance.

A baseline that gives a score rather than a label is also measured without any threshold, by the
area under the ROC curve: the chance that a randomly chosen positive answer scores above a
randomly chosen negative one (0.5 is guessing, 1.0 is perfect separation).
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True, kw_only=True)
class Agreement:
    """Agreement between two raters over the same items.

    Attributes:
        items: How many items both rated.
        observed: Share of items with the same label.
        kappa: Cohen's kappa.
        kappa_low: Lower end of the 95% bootstrap interval for kappa.
        kappa_high: Upper end of that interval.
        confusion: Counts, first rater's label then second rater's label.
    """

    items: int
    observed: float
    kappa: float
    kappa_low: float
    kappa_high: float
    confusion: dict[str, dict[str, int]]


def cohens_kappa(first: Sequence[str], second: Sequence[str]) -> float:
    """Cohen's kappa for two raters' labels of the same items.

    Raises:
        ValueError: If the inputs are empty or differ in length, or both raters used a single
            identical label throughout, where kappa is undefined (0 / 0).
    """
    if len(first) != len(second):
        raise ValueError(f"raters labelled different numbers of items: {len(first)}, {len(second)}")
    if not first:
        raise ValueError("cannot measure agreement over no items")
    count = len(first)
    observed = sum(a == b for a, b in zip(first, second, strict=True)) / count
    labels = set(first) | set(second)
    expected = sum((first.count(label) / count) * (second.count(label) / count) for label in labels)
    if expected == 1:
        raise ValueError("kappa is undefined when both raters use one and the same label")
    return (observed - expected) / (1 - expected)


def agreement(
    first: Sequence[str], second: Sequence[str], *, seed: int, trials: int = 2_000
) -> Agreement:
    """Kappa with a bootstrap interval, observed agreement and the confusion table.

    Raises:
        ValueError: As cohens_kappa.
    """
    kappa = cohens_kappa(first, second)
    low, high = _bootstrap(first, second, cohens_kappa, seed=seed, trials=trials)
    confusion: dict[str, dict[str, int]] = {}
    for a, b in zip(first, second, strict=True):
        confusion.setdefault(a, {}).setdefault(b, 0)
        confusion[a][b] += 1
    return Agreement(
        items=len(first),
        observed=sum(a == b for a, b in zip(first, second, strict=True)) / len(first),
        kappa=kappa,
        kappa_low=low,
        kappa_high=high,
        confusion=confusion,
    )


def roc_auc(scores: Sequence[float], positive: Sequence[bool]) -> float:
    """Area under the ROC curve, by counting correctly ordered positive-negative pairs.

    Ties count half, which is the Mann-Whitney U statistic divided by the number of pairs.

    Raises:
        ValueError: If the inputs differ in length or either class is empty.
    """
    if len(scores) != len(positive):
        raise ValueError("scores and labels differ in length")
    values = np.asarray(scores, dtype=np.float64)
    is_positive = np.asarray(positive, dtype=bool)
    positives, negatives = values[is_positive], values[~is_positive]
    if positives.size == 0 or negatives.size == 0:
        raise ValueError("AUC needs at least one positive and one negative item")
    greater = (positives[:, None] > negatives[None, :]).sum()
    ties = (positives[:, None] == negatives[None, :]).sum()
    return float((greater + 0.5 * ties) / (positives.size * negatives.size))


def _bootstrap(
    first: Sequence[str],
    second: Sequence[str],
    statistic: Callable[[Sequence[str], Sequence[str]], float],
    *,
    seed: int,
    trials: int,
) -> tuple[float, float]:
    """Percentile interval of a two-rater statistic, resampling items with replacement.

    A resample in which kappa is undefined is skipped; if every one is, the interval collapses to
    the point estimate.
    """
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(trials):
        rows = rng.integers(0, len(first), size=len(first))
        try:
            values.append(statistic([first[i] for i in rows], [second[i] for i in rows]))
        except ValueError:
            continue
    if not values:
        point = statistic(first, second)
        return point, point
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)
