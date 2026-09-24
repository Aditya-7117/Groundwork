# 0021. Regression gate and trec_eval check

Date: 2026-09-24. Status: accepted.

## Context

A change that silently lowers retrieval quality, or a metric that drifts from the standard
definition, would invalidate every conclusion. CI must catch both on every push, without the
multi-gigabyte corpus.

## Options

- Download the corpus in CI; commit a small slice; or skip the gate.
- Tolerance: zero, a round number, or measured noise.
- Metric check: hand-worked tests only, or also the reference implementation.

## Decision

- A committed golden slice: 100 questions stratified by answer type with a fixed seed, their
  pages and 200 random distractor pages, copied line for line from the built corpus (Wikipedia
  text, CC BY-SA 3.0, credited in the slice). It rebuilds byte for byte.
- The gate re-runs BM25 and MiniLM on CPU and fails if passage recall@10 or nDCG@10 falls below
  the committed baseline by more than its tolerance. BM25's tolerance is zero, since repeated runs
  are identical; MiniLM's is set from the difference measured on the first CI run.
- Every metric is checked against trec_eval, through its pytrec_eval binding, on random rankings.
  That check found precision@k dividing by the items returned rather than by k; it now follows
  trec_eval.

## Consequences

- Moving the baseline takes a deliberate `--record` and a commit that shows the new values.
- The slice adds about 7 MB to the repository.
