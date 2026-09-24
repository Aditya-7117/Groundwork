# 0012. BM25 with stemming, vectorised

Date: 2026-09-24. Status: accepted. Extends 0006.

## Context

The pure-Python BM25 of 0006 took 574 ms per question at the median over the full corpus, which
would misreport retrieval latency and slow the grid. Anserini's published BM25 baseline also
stems words, which the first version did not.

## Options

- Keep the pure-Python index; adopt a BM25 library with its own scoring variant; or vectorise the
  same arithmetic with a sparse matrix.
- With or without Porter stemming.

## Decision

- Porter stemming, Lucene's IDF, k1 = 0.9 and b = 0.4, Anserini's published defaults, not tuned
  on the test questions.
- A scipy sparse-matrix implementation for the runs, with the pure-Python version kept as the
  reference. Property tests on random corpora require identical scores and identical order,
  ties included.

## Consequences

- On the full corpus both versions gave identical per-question metrics for all 3,220 questions and
  byte-identical ranking files; retrieval time fell from 2,673 s to 143 s.
- The reference stays in the codebase only to check the fast version.
