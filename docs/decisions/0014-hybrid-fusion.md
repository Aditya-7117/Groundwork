# 0014. Hybrid retrieval by reciprocal rank fusion

Date: 2026-09-24. Status: accepted.

## Context

Keyword and embedding retrieval fail differently, so combining them may beat either. Their
scores are on unrelated scales.

## Options

- A weighted sum of normalised scores, which needs a tuned weight.
- Reciprocal rank fusion: each ranking gives every chunk 1 / (k + rank), summed.

## Decision

Reciprocal rank fusion with k = 60, the value from the method's paper (Cormack, Clarke and
Buettcher, 2009), over the top 100 of each ranking, the full retrieval depth. A chunk BM25 did not
retrieve (score zero, no shared word) gets nothing from BM25.

## Consequences

- No weight is tuned on the test questions.
- Measured on fixed chunks with MiniLM, fusion raised nDCG@10 and MRR@10 but lowered recall@10
  slightly; the report shows all three.
