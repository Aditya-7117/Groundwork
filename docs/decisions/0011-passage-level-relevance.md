# 0011. Passage-level relevance

Date: 2026-09-24. Status: accepted. Supersedes 0005 as the primary measure; 0005's best-chunk
rule still ranks pages for the page-level numbers.

## Context

Natural Questions marks the exact span of each answer in its page. The generator receives
chunks, not pages, so the question that matters is whether the chunks retrieved contain the
answer.

## Options

- **Page level:** a page is relevant if it holds the answer, and it is ranked by its best chunk.
- **Passage level:** a chunk is relevant if it overlaps the marked answer span.
- **Both,** with one declared primary.

## Decision

Both, with passage level primary. Every metric is computed at both levels and broken down by
answer type (paragraph, table, list). Questions whose answer no chunk covers are excluded and
listed in the artefact, never scored as zero.

## Consequences

- Page-level numbers flatter the system: for BM25 over fixed chunks, page recall@10 is 0.942 and
  passage recall@10 0.468. Reporting both makes the gap visible.
- Relevance depends on the chunking, which is intended, since chunking is part of what the grid
  compares.
