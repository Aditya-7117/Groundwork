# 0002. Retrieval metric conventions

Date: 2026-09-19. Status: accepted.

## Context

recall@k, nDCG@k and MRR each have variants that give different numbers from the same ranking.
If the variant is not fixed and written down, a result cannot be compared with anything, including
the published baselines this project will cite.

## Options and decisions

| Question | Options | Decision | Why |
|---|---|---|---|
| Which grade counts as relevant? | grade ≥ 1; grade ≥ 2 on graded sets | grade ≥ 1 | Matches trec_eval's default relevance level. |
| Documents nobody judged | grade 0; exclude from the ranking | grade 0 | Standard practice. Excluding them would reward a retriever for surfacing unjudged documents. |
| nDCG gain | linear (the grade); exponential (2^grade − 1) | linear | trec_eval and the BEIR benchmark use linear gain, so the numbers stay comparable. On binary judgements the two are identical anyway. |
| nDCG ideal ranking | all judged documents; only the retrieved ones | all judged documents | Building the ideal from retrieved documents alone lets a retriever that misses relevant documents score 1.0. |
| MRR depth | whole ranking; cut off at k | cut off at k (MRR@k) | Depth is then an explicit parameter, and the score does not change silently when retrieval depth changes. |
| Queries with no relevant document | score 0; skip; raise | raise, and the caller excludes the query explicitly | Recall and nDCG divide by zero there. Silently scoring 0 would drag averages down for a reason that has nothing to do with retrieval quality. |
| Averaging | per query (macro); pooled over all documents (micro) | per query | Every query counts equally, as in trec_eval and BEIR. |

The metric functions also reject a ranking with duplicate document ids, because a duplicate lets
one relevant document count twice.

## Consequences

- Graded numbers from this project are not comparable with papers that use exponential gain.
  Every write-up that reports nDCG on a graded corpus must say that linear gain was used.
- The test suite pins each choice. The Wikipedia graded example fails under exponential gain, and
  the unretrieved relevant document d9 fails an ideal ranking built only from retrieved documents.
