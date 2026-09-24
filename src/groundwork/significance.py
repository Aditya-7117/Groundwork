"""Is a difference between two setups real, or could it be luck in which questions were asked?

Every setup answers the same questions, so comparisons are paired: the unit is the per-question
difference in a metric. Three tools, all standard in information retrieval evaluation:

- The paired randomisation test asks how often a difference at least this large would appear if
  the two setups were interchangeable. Under that assumption each question's difference is equally
  likely to have either sign, so the test flips signs at random many times and counts. It assumes
  no particular distribution, which matters because recall@k per question is only ever 0 or 1.
- The bootstrap confidence interval resamples questions with replacement and reports the middle
  95% of the resampled mean differences: a range for the true difference, not just a verdict.
- Holm's correction adjusts p-values when many comparisons are made at once. Thirty setups give
  plenty of chances for one comparison to look significant by luck; Holm controls the chance of
  even one false claim across the whole family, and is never weaker than Bonferroni.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

TRIALS = 10_000
"""Resamples per test. With 10,000 the smallest reportable p-value is about 0.0001."""

_BATCH = 500


@dataclass(frozen=True, slots=True, kw_only=True)
class Comparison:
    """Setup a against setup b on one metric, over the questions both evaluated.

    Attributes:
        a: First setup's name.
        b: Second setup's name.
        metric: Metric compared, for example "passage.ndcg@10".
        questions: Questions in the comparison.
        mean_a: a's mean over those questions.
        mean_b: b's mean over those questions.
        difference: mean_a minus mean_b.
        ci_low: Lower end of the 95% bootstrap interval for the difference.
        ci_high: Upper end of that interval.
        p_value: Two-sided paired randomisation test.
        p_holm: p_value after Holm's correction across the whole family of comparisons.
    """

    a: str
    b: str
    metric: str
    questions: int
    mean_a: float
    mean_b: float
    difference: float
    ci_low: float
    ci_high: float
    p_value: float
    p_holm: float


def randomisation_test(
    a: Sequence[float], b: Sequence[float], *, trials: int = TRIALS, seed: int
) -> float:
    """Two-sided p-value for the mean paired difference, by random sign flips.

    The count includes the observed arrangement itself, so the p-value is never exactly zero: with
    finitely many trials, "never seen" only bounds the probability.

    Raises:
        ValueError: If the inputs are empty or differ in length, or trials is below 1.
    """
    differences = _differences(a, b)
    if trials < 1:
        raise ValueError(f"trials must be at least 1, got {trials}")
    observed = abs(float(differences.mean()))
    # Floating-point sums of the same numbers in another order can differ in the last bits.
    threshold = observed - 1e-12
    rng = np.random.default_rng(seed)
    at_least = 0
    for start in range(0, trials, _BATCH):
        signs = rng.choice((-1.0, 1.0), size=(min(_BATCH, trials - start), len(differences)))
        means = np.abs(signs @ differences) / len(differences)
        at_least += int(np.count_nonzero(means >= threshold))
    return (at_least + 1) / (trials + 1)


def bootstrap_interval(
    values: Sequence[float],
    *,
    trials: int = TRIALS,
    seed: int,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of values.

    Raises:
        ValueError: If values is empty, trials is below 1, or level is not between 0 and 1.
    """
    sample = np.asarray(values, dtype=np.float64)
    if sample.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if trials < 1:
        raise ValueError(f"trials must be at least 1, got {trials}")
    if not 0 < level < 1:
        raise ValueError(f"level must lie between 0 and 1, got {level}")
    rng = np.random.default_rng(seed)
    means: list[NDArray[np.float64]] = []
    for start in range(0, trials, _BATCH):
        rows = rng.integers(0, sample.size, size=(min(_BATCH, trials - start), sample.size))
        means.append(sample[rows].mean(axis=1))
    tail = (1 - level) / 2
    low, high = np.quantile(np.concatenate(means), [tail, 1 - tail])
    return float(low), float(high)


def holm(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return Holm-adjusted p-values.

    Sort ascending, multiply the i-th smallest (counting from 0) by (m - i), keep the running
    maximum so adjusted values never decrease, and cap at 1.

    Raises:
        ValueError: If a p-value lies outside 0 to 1.
    """
    for key, value in p_values.items():
        if not 0 <= value <= 1:
            raise ValueError(f"p-value for {key!r} is {value}, outside 0 to 1")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * value))
        adjusted[key] = running
    return adjusted


def compare(
    per_question: Mapping[str, Mapping[str, Mapping[str, float]]],
    pairs: Sequence[tuple[str, str]],
    metric: str,
    *,
    trials: int = TRIALS,
    seed: int,
) -> tuple[Comparison, ...]:
    """Compare each pair of setups on one metric, correcting across all the pairs given.

    Args:
        per_question: Setup name, then question id, then metric name, to value.
        pairs: The family of comparisons, as (a, b) setup names.
        metric: The metric to compare.
        trials: Resamples per test and per interval.
        seed: Seeds every test and interval, so a report reproduces exactly.

    Raises:
        ValueError: If a pair names an unknown setup, or two setups share no question.
    """
    rows: list[tuple[str, str, list[str], list[float], list[float]]] = []
    for a, b in pairs:
        for name in (a, b):
            if name not in per_question:
                raise ValueError(f"unknown setup {name!r}")
        shared = sorted(per_question[a].keys() & per_question[b].keys())
        if not shared:
            raise ValueError(f"{a} and {b} share no evaluated question")
        rows.append(
            (
                a,
                b,
                shared,
                [per_question[a][q][metric] for q in shared],
                [per_question[b][q][metric] for q in shared],
            )
        )
    raw = {
        f"{a}|{b}": randomisation_test(values_a, values_b, trials=trials, seed=seed)
        for a, b, _, values_a, values_b in rows
    }
    adjusted = holm(raw)
    comparisons = []
    for a, b, shared, values_a, values_b in rows:
        differences = np.asarray(values_a) - np.asarray(values_b)
        low, high = bootstrap_interval(differences.tolist(), trials=trials, seed=seed)
        comparisons.append(
            Comparison(
                a=a,
                b=b,
                metric=metric,
                questions=len(shared),
                mean_a=float(np.mean(values_a)),
                mean_b=float(np.mean(values_b)),
                difference=float(differences.mean()),
                ci_low=low,
                ci_high=high,
                p_value=raw[f"{a}|{b}"],
                p_holm=adjusted[f"{a}|{b}"],
            )
        )
    return tuple(comparisons)


def _differences(a: Sequence[float], b: Sequence[float]) -> NDArray[np.float64]:
    if len(a) != len(b):
        raise ValueError(f"paired samples differ in length: {len(a)} and {len(b)}")
    if not a:
        raise ValueError("cannot test empty samples")
    return np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
