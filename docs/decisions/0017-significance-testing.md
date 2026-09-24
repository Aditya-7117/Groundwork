# 0017. Significance testing

Date: 2026-09-24. Status: accepted.

## Context

Thirty setups over the same questions invite reading a lucky difference as a real one.

## Options

- Paired randomisation test, paired t-test, or Wilcoxon signed-rank test.
- Every pair of setups (435 comparisons), each against BM25, or families chosen to match the
  claims made.

## Decision

- A paired randomisation test (10,000 sign flips) for p-values, which assumes no distribution;
  recall per question is only ever 0 or 1.
- A percentile bootstrap (10,000 resamples) for a 95% interval on each difference.
- Holm's correction within each family.
- Two families, each on passage nDCG@10 and recall@10: the winner against each other setup, and
  each first stage with against without the reranker.

## Consequences

- Every claim in the README is backed by a test in one of the two families.
- With 10,000 resamples the smallest reportable p-value is about 0.0001.
