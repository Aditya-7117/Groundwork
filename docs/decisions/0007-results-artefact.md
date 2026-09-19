# 0007. Results artefact format

Date: 2026-09-19. Status: accepted.

## Context

Someone else must be able to reproduce any published number. That needs a record, next to every
result, of exactly what produced it, and a way to check the metric computation independently of
this project's own code.

## Options

- **Metrics only**, in a CSV or a single JSON file.
- **Metrics plus provenance** in JSON: config, code version, corpus checksum, environment.
- **Provenance plus the raw rankings**, in a format that standard evaluation tools read.
- **An experiment tracker** such as MLflow, which brings a server and a dependency.

## Decision

A directory per run holding `result.json` (provenance and metrics) and `run.trec` (rankings in the
TREC run format).

- **Named `<UTC start>-<first 12 hex characters of the config digest>`** under
  `results/<experiment>/`, and never overwritten. Two runs of one config sit side by side, and the
  name alone says which experiment definition produced them.
- **Provenance recorded:** the full parsed config and its digest; the package version and git
  commit, plus whether the working tree had uncommitted changes, because a result from a dirty tree
  cannot be rebuilt from its commit; the corpus URL, SHA-256 and licence; Python version,
  operating system, CPU model, core count and memory; and the start and finish time in UTC.
- **`schema_version`** is incremented whenever a field changes meaning or moves, so old artefacts
  stay readable.
- **The TREC run file** lets `trec_eval`, the reference implementation, re-score the rankings. If
  its numbers disagree with `result.json`, the metric code is wrong.
- **Written through a temporary directory** and renamed into place, so a crash never leaves an
  artefact that looks complete.

`results/` is not tracked by git. A result gets committed on purpose, when it is published, rather
than every development run landing in the history.

## Consequences

- Timings are the only part of an artefact that differs between two runs of the same config on the
  same code, corpus and machine.
- Scores in the run file are written at full precision. When scores tie exactly, trec_eval orders
  the tied documents by its own rule, which can differ from this project's tie-break by document
  id.
