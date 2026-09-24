# 0015. Cross-encoder reranking

Date: 2026-09-24. Status: accepted.

## Context

A cross-encoder reads the question and a chunk together, so it judges fit better than either
first-stage retriever, but it is far too slow to score every chunk.

## Options

- **Model:** BAAI/bge-reranker-v2-m3.
- **Depth:** 20, 50 or 100 first-stage results.
- **Reuse:** score every setup's pairs afresh, or score each (question, chunk text) pair once.

## Decision

bge-reranker-v2-m3, pinned, in half precision, re-ordering the top 50; ties keep first-stage
order and chunks below 50 keep theirs. Each pair is scored once per process and reused across
setups, charged at the time it first took.

## Consequences

- Depth 20 would cap how far down a relevant chunk can be rescued; 100 would double the cost.
- Reranking latency is reported apart from first-stage latency and never flattered by reuse.
