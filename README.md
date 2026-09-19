# Groundwork

Groundwork measures which retrieval-augmented generation configuration answers questions correctly,
how quickly, and at what cost. Every experiment is defined by a file in `configs/`, run with a
fixed seed, and written to a results artefact that records the exact configuration, code version
and hardware that produced it.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```sh
uv sync
```

## Checks

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```
