# 0006. BM25 as the lexical baseline, implemented in the standard library

Date: 2026-09-19. Status: accepted.

## Context

The first end-to-end run needs a retriever. The embedding models and rerankers that the grid will
compare are still undecided, and they bring heavy dependencies and cost. A lexical baseline is
also something the finished comparison needs anyway: without one, a dense retriever's number has
nothing to be measured against.

## Options

- **In-house BM25**, about 60 lines of standard-library Python.
- **`rank_bm25`**, a small third-party package.
- **Pyserini (Anserini)**, the reference BM25 behind the BEIR baseline. It needs a Java runtime.

## Decision

In-house BM25, Lucene variant (the IDF never goes negative), with every step pinned by hand-worked
unit tests. It adds no dependency, fits in one file someone can read in five minutes, and its
behaviour on ties and repeated query terms is specified and tested rather than inherited.

Tokenisation is lowercase word splitting with no stemming and no stopword list. The parameters
are set in the config. The provisional config uses k1 = 0.9 and b = 0.4, the Anserini defaults the
BEIR paper used.

## Consequences

- Scores will not match Anserini exactly. Anserini stems words (Porter stemmer), removes
  stopwords and indexes title and body as separate fields, so this BM25 is expected to land near,
  not on, the published SciFact figure. How far off it lands is itself a check: a large gap would
  point to a bug.
- Scoring is pure Python and single-threaded. That is fine for corpora of tens of thousands of
  chunks, and would need replacing for millions.
