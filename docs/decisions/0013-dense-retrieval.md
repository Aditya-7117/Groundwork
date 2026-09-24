# 0013. Dense retrieval

Date: 2026-09-24. Status: accepted.

## Context

Embedding models retrieve by meaning rather than shared words. The grid compares a small, fast
model with a larger, newer one, and needs their vectors for three chunkings, about 1.2 million
chunks in all.

## Options

- **Models:** all-MiniLM-L6-v2 (22M parameters) against Qwen3-Embedding-0.6B.
- **Search:** exact comparison with every chunk, or an approximate nearest-neighbour index.
- **Precision:** full or half.
- **Cache:** keyed on the whole chunk list, or on each distinct chunk text.

## Decision

- Both models, each pinned to one published revision and used as its authors recommend: Qwen3
  queries carry the model card's instruction line; MiniLM's do not.
- Exact search, so no approximate index adds errors of its own to the comparison.
- Half precision on the Apple GPU: measured 2.3 times faster, with 99.8% of nearest neighbours
  unchanged. Full precision on a CPU.
- Vectors cached per distinct chunk text, in shards, so identical text is encoded once across
  chunkers and ablations, and an interrupted run resumes. A shard's encoding time is split over
  its texts and charged to every chunk that uses one.

## Consequences

- Every run records each model's name, revision, device and precision.
- The reported embedding time is what encoding the corpus cost, even when a run reused vectors.
