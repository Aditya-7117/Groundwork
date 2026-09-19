# 0001. Python toolchain

Date: 2026-09-19. Status: accepted.

## Context

The harness has to run from a clean machine using the README alone, and its numbers are only as
credible as the code that produces them. That needs a pinned environment, strict static checking
and one lint and format tool.

## Options

- **Environment and lockfile:** uv, Poetry, or pip with pip-tools.
- **Type checker:** mypy or pyright, either in strict mode.
- **Python version:** 3.12 or 3.14, both installed locally.

## Decision

- **uv** manages the virtual environment and writes `uv.lock`. It resolves and installs in seconds,
  reads standard `pyproject.toml` metadata, and supports dependency groups, so development tools
  stay out of the runtime dependency list. Poetry uses its own metadata conventions. pip-tools
  needs a second tool to create the environment.
- **mypy in strict mode** over `src/` and `tests/`. It is the reference implementation of Python
  typing and runs as an ordinary Python package inside the locked environment. pyright would need
  Node.js on the machine.
- **Python 3.12** as the floor and the development version. The later work needs embedding models
  and rerankers, and 3.12 is the version that machine-learning wheels support most widely.
- **ruff** for both linting and formatting. The rule set in `pyproject.toml` includes checks that
  enforce the hygiene rules mechanically: no `print` (`T20`), no commented-out code (`ERA`), no
  blind `except` (`BLE`), no naive datetimes (`DTZ`), plus docstrings on public code (`D`).

## Consequences

- Contributors need uv installed. `.venv` is still a standard virtual environment, so everything
  in it also runs without uv.
- Moving to Python 3.14 later means changing `requires-python`, `.python-version` and the ruff and
  mypy targets together.
