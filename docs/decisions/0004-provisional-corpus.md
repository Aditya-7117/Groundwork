# 0004. Provisional corpus for pipeline plumbing

Date: 2026-09-19. Status: accepted as provisional. The project's real corpus has not been chosen.

## Context

The pipeline has to run end to end before the real corpus and question set are chosen. Plumbing
needs a small, public corpus with known relevance judgements. It is placeholder infrastructure,
not the corpus the project's conclusions will rest on.

## Options

- **BEIR SciFact:** 5,183 scientific abstracts, 300 test claims, binary judgements, 2.8 MB.
- **BEIR NFCorpus:** 3,633 medical documents, 323 test queries, graded judgements.
- **A hand-written corpus:** fully controlled, but proves nothing about real text.

## Decision

BEIR SciFact, fetched from the BEIR distribution and pinned by SHA-256. It is the smallest
standard BEIR set, its licence terms are checked (claims CC BY 4.0, abstracts ODC-By 1.0, per the
`allenai/scifact` repository), and the BEIR paper (Thakur et al., 2021, Table 2) publishes a BM25
nDCG@10 of 0.665 for it. That published number is an external sanity check on this pipeline's
BM25 and metric code together.

The archive's MD5 matches the one in the BEIR README (`5f7d1de60b170fc8027bb7898e2efca1`), and its
SHA-256 is pinned in `src/groundwork/sources.py`. The corpus is downloaded on first use and never
committed.

## Consequences

- Every config and results artefact that uses it carries `provisional` in its name or metadata,
  so its numbers cannot be mistaken for project results.
- SciFact judgements are binary, so the graded branch of nDCG is exercised only by the unit tests
  and the fixture corpus until a graded corpus is chosen.
- Adding the real corpus means one new entry in `SOURCES`, plus a loader if it is not in the BEIR
  layout.
