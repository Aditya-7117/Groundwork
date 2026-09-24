# 0010. Chunking strategies and tables

Date: 2026-09-20. Status: accepted.

## Context

Retrieval and generation both work on chunks, so how pages are cut changes every number. The
grid compares chunking strategies, and table questions are about a fifth of the questions, so
tables need a deliberate treatment rather than whatever prose chunking does to them.

## Options

- **Sizes:** tokens of a particular model, or words. About 150 words with 30 words of overlap
  fits the smaller embedding model's input limit.
- **Boundaries:** cut blindly; pack whole sentences; pack whole sentences without crossing a
  section heading.
- **Tables:** flatten to prose like any other text; or keep them as their own chunks with their
  structure.

## Decision

- Three strategies over about 150 words with about 30 words of overlap: `fixed_words` cuts blindly
  (the naive baseline), `sentence_aware` packs whole sentences found by NLTK's Punkt model and
  overlaps by whole sentences, `section_aware` does the same without crossing a heading.
- Every chunk starts with its page title and section ("Title > Section"); tables add their caption.
- Tables are separate chunks in every strategy. Each row is written "Column: value | Column:
  value", the header is repeated in every chunk of a table, merged cells are expanded, and large
  tables are split between rows, never within one.
- An ablation switch, `flatten_tables`, treats tables as loose prose, so the winning setup can be
  re-run without the table handling and the difference measured on table questions.

## Consequences

- Chunk counts differ by strategy (fixed 341,126; sentence 403,120; section 425,991), so larger
  indexes cost more to embed and search; the artefact records the count.
- The Punkt model is downloaded once and its digest recorded with every run.
- The benefit of the table layout is a measured claim only once the ablation has run.
