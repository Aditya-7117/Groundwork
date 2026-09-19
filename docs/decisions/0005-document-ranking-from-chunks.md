# 0005. Ranking documents from chunk scores

Date: 2026-09-19. Status: accepted as provisional. Revisit once the real corpus is chosen, since
its judgements may be at passage level rather than document level.

## Context

Retrieval scores chunks, but BEIR-style relevance judgements are per document. Metrics need a
document ranking, so chunk scores have to be turned into document scores. The choice changes
every number the harness reports, so it has to be explicit.

## Options

- **Best chunk (MaxP):** a document scores as its highest-scoring chunk.
- **Sum of chunk scores:** rewards long documents simply for having more chunks.
- **First chunk only (FirstP):** ignores relevant material past the opening.
- **Score chunks directly against passage-level judgements:** needs a corpus that has them.

## Decision

Best chunk. It is the standard aggregation in the passage-retrieval literature (Dai and Callan,
2019, call it MaxP). It does not favour long documents, and the best chunk is also the passage
the generation stage would receive, so retrieval and generation are evaluated on the same unit.

Aggregation runs over every scored chunk, not just the top of the chunk ranking. That way `depth`
counts documents exactly, however many chunks each document has.

## Consequences

- Chunking parameters affect document scores. That is intended: comparing chunking strategies is
  part of the experiment grid.
- If the real corpus has passage-level judgements, this decision is replaced rather than extended.
