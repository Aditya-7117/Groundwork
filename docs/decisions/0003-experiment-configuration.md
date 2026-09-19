# 0003. Experiment configuration format

Date: 2026-09-19. Status: accepted.

## Context

Every experiment has to be defined by a file in `configs/`, not by edits to code, so that any
published number can be traced back to exactly one definition and rerun.

## Options

- **File format:** TOML, read by the standard library's `tomllib`; YAML, which needs PyYAML; or
  JSON, which has no comments.
- **Validation:** hand-written checks over frozen dataclasses, or a pydantic model.
- **Defaults:** fill in missing values, or require every value.

## Decision

- **TOML.** It needs no third-party parser, allows comments explaining why a value was chosen, and
  is the same format as `pyproject.toml`.
- **Frozen dataclasses with explicit validation**, which keeps the runtime dependency set empty
  for now. If pydantic enters the project for another reason, config validation should move to it
  to avoid keeping two validation styles.
- **No defaults, and unknown keys are errors.** A misspelt key such as `overlab` would otherwise be
  ignored, and the run would quietly use a value nobody chose.
- **Experiment identity is a SHA-256 digest of the parsed values**, not of the file bytes. Editing a
  comment does not change the experiment; changing any value does.
- **Machine-specific paths are not in the config.** Where the corpus is cached and where results
  are written are command-line options, so the same experiment has the same digest on every
  machine.

## Consequences

- Adding a config field means updating the dataclass, the parser and the tests together. That is
  more code than a pydantic model, and it is all in one module.
- The test suite loads every committed config, so an invalid config cannot be merged.
