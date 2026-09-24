# 0018. Writing answers

Date: 2026-09-24. Status: accepted. Builds on 0008.

## Context

Stage two needs answers written from each setup's retrieved passages under identical conditions,
so that only retrieval differs between setups.

## Options

- **Passages given:** the top 3, 5 or 10 chunks.
- **Style:** a short phrase; one sentence; one sentence with citations.
- **Declining:** allowed or not.

## Decision

The top five chunks, one short sentence from the passages only, or exactly "I don't know" when
they lack the answer. Thinking off, temperature 0, fixed seed. Each answer record keeps the
question and the passage texts. Before writing, the setup's chunks are rebuilt and their count
checked against the retrieval run.

## Consequences

- Five passages line answer quality up with recall@5.
- Declined answers are counted apart and never judged as grounded; for correctness they count as
  not correct.
- Generation time depends on memory pressure from other applications, so timings are compared
  only between answers generated under the same conditions.
