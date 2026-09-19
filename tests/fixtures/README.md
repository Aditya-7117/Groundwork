# Test fixtures

`tiny-beir/` is a six-document corpus in the BEIR file layout, written for this repository's
tests. It is small enough to reason about by hand:

| Query | Relevant documents | What it exercises |
|---|---|---|
| q1 `bitter green tea` | d1 (grade 1) | A straightforward lexical match. |
| q2 `longest river Ireland` | d4 (grade 2), d3 (grade 1) | Graded judgements with more than one relevant document. |
| q3 `ringed gas giant` | d5 (grade 1) | Vocabulary mismatch: the relevant document says "ring", not "ringed", so a lexical retriever misses it and ranks d6 instead. |
| q4 | none | A query with no judgements in the split, which must be left out of evaluation. |
