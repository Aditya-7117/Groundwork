"""Experiment configuration: one TOML file in configs/ defines one experiment completely.

Validation is strict. Unknown keys, missing keys, wrong types and inconsistent values fail at load
time with the file and key named. There are no defaults, because a typo that silently fell back
to a default would produce a result from an experiment nobody asked for.
"""

import hashlib
import json
import math
import re
import tomllib
from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

type ChunkingStrategy = Literal["fixed_words"]
type RetrievalMethod = Literal["bm25"]

CHUNKING_STRATEGIES: tuple[ChunkingStrategy, ...] = ("fixed_words",)
RETRIEVAL_METHODS: tuple[RetrievalMethod, ...] = ("bm25",)

_NAME_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


class ConfigError(ValueError):
    """An experiment config file is missing, malformed or inconsistent."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CorpusConfig:
    """Which corpus to evaluate on, and which of its queries.

    Attributes:
        name: Registered corpus id, for example "beir-scifact".
        split: Which set of relevance judgements to evaluate against, for example "test".
        query_limit: Evaluate a seeded random sample of this many queries, or all when None.
    """

    name: str
    split: str
    query_limit: int | None


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkingConfig:
    """How documents are split into the passages that retrieval searches over.

    Attributes:
        strategy: Chunking strategy id.
        size: Words per chunk.
        overlap: Words shared between consecutive chunks of the same document.
    """

    strategy: ChunkingStrategy
    size: int
    overlap: int


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievalConfig:
    """How chunks are scored against a query.

    Attributes:
        method: Retrieval method id.
        depth: Number of documents returned per query.
        k1: BM25 term-frequency saturation. Higher values let repeated terms keep adding score.
        b: BM25 length normalisation, from 0 (none) to 1 (full).
    """

    method: RetrievalMethod
    depth: int
    k1: float
    b: float


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationConfig:
    """Which rank cutoffs the metrics are reported at.

    Attributes:
        cutoffs: Values of k for recall@k, nDCG@k and MRR@k.
    """

    cutoffs: tuple[int, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class ExperimentConfig:
    """A complete, validated experiment definition.

    Attributes:
        name: Lowercase slug that names the results directory.
        description: What the experiment is for, in plain words.
        seed: Seed for every source of randomness in the run.
        corpus: Corpus and query selection.
        chunking: Chunking strategy and parameters.
        retrieval: Retrieval method and parameters.
        evaluation: Metric cutoffs.
    """

    name: str
    description: str
    seed: int
    corpus: CorpusConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    evaluation: EvaluationConfig


def load_config(path: Path) -> ExperimentConfig:
    """Read and validate an experiment config file.

    Raises:
        ConfigError: If the file is missing, is not valid TOML, or fails validation.
    """
    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except FileNotFoundError as error:
        raise ConfigError(f"{path}: config file not found") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path}: not valid TOML: {error}") from error
    return _parse(_Table(raw, source=path, section=None))


def config_digest(config: ExperimentConfig) -> str:
    """Return a SHA-256 hex digest that identifies the experiment a config defines.

    The digest covers the parsed values, not the file bytes, so comments, whitespace and key
    order do not change it, and any change to a value does.
    """
    canonical = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse(root: "_Table") -> ExperimentConfig:
    root.reject_unknown(
        {"name", "description", "seed", "corpus", "chunking", "retrieval", "evaluation"}
    )
    name = root.string("name")
    if not _NAME_PATTERN.fullmatch(name):
        raise root.error(f"name must be lowercase letters, digits and single hyphens, got {name!r}")

    corpus = root.table("corpus")
    corpus.reject_unknown({"name", "split", "query_limit"})

    chunking = root.table("chunking")
    chunking.reject_unknown({"strategy", "size", "overlap"})
    size = chunking.integer("size", minimum=1)
    overlap = chunking.integer("overlap", minimum=0)
    if overlap >= size:
        raise chunking.error(f"overlap must be less than size, got overlap {overlap}, size {size}")

    retrieval = root.table("retrieval")
    retrieval.reject_unknown({"method", "depth", "k1", "b"})
    depth = retrieval.integer("depth", minimum=1)
    k1 = retrieval.number("k1")
    if k1 < 0:
        raise retrieval.error(f"k1 must be at least 0, got {k1}")
    b = retrieval.number("b")
    if not 0 <= b <= 1:
        raise retrieval.error(f"b must be between 0 and 1, got {b}")

    evaluation = root.table("evaluation")
    evaluation.reject_unknown({"cutoffs"})
    cutoffs = evaluation.integers("cutoffs", minimum=1)
    if len(set(cutoffs)) != len(cutoffs):
        raise evaluation.error(f"cutoffs must not repeat, got {list(cutoffs)}")
    if max(cutoffs) > depth:
        raise evaluation.error(f"cutoff {max(cutoffs)} exceeds retrieval depth {depth}")

    return ExperimentConfig(
        name=name,
        description=root.string("description"),
        seed=root.integer("seed", minimum=0),
        corpus=CorpusConfig(
            name=corpus.string("name"),
            split=corpus.string("split"),
            query_limit=corpus.optional_integer("query_limit", minimum=1),
        ),
        chunking=ChunkingConfig(
            strategy=chunking.choice("strategy", CHUNKING_STRATEGIES),
            size=size,
            overlap=overlap,
        ),
        retrieval=RetrievalConfig(
            method=retrieval.choice("method", RETRIEVAL_METHODS), depth=depth, k1=k1, b=b
        ),
        evaluation=EvaluationConfig(cutoffs=cutoffs),
    )


class _Table:
    """One TOML table, with typed accessors whose errors name the file, section and key."""

    def __init__(self, data: Mapping[str, object], *, source: Path, section: str | None) -> None:
        self._data = data
        self._source = source
        self._section = section

    def error(self, message: str) -> ConfigError:
        location = f"[{self._section}]." if self._section else ""
        return ConfigError(f"{self._source}: {location}{message}")

    def reject_unknown(self, allowed: Collection[str]) -> None:
        unknown = sorted(set(self._data) - set(allowed))
        if unknown:
            raise self._table_error(f"has unknown key: {', '.join(unknown)}")

    def table(self, key: str) -> "_Table":
        value = self._require(key)
        if not isinstance(value, dict):
            raise self.error(f"{key} must be a table")
        return _Table(value, source=self._source, section=key)

    def string(self, key: str) -> str:
        value = self._require(key)
        if not isinstance(value, str) or not value.strip():
            raise self.error(f"{key} must be a non-empty string")
        return value

    def choice[T: str](self, key: str, options: tuple[T, ...]) -> T:
        value = self.string(key)
        for option in options:
            if value == option:
                return option
        raise self.error(f"{key} must be one of: {', '.join(options)}; got {value!r}")

    def integer(self, key: str, *, minimum: int) -> int:
        return self._check_integer(key, self._require(key), minimum)

    def optional_integer(self, key: str, *, minimum: int) -> int | None:
        if key not in self._data:
            return None
        return self._check_integer(key, self._data[key], minimum)

    def number(self, key: str) -> float:
        value = self._require(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise self.error(f"{key} must be a number")
        if not math.isfinite(value):
            raise self.error(f"{key} must be finite")
        return float(value)

    def integers(self, key: str, *, minimum: int) -> tuple[int, ...]:
        value = self._require(key)
        if not isinstance(value, list) or not value:
            raise self.error(f"{key} must be a non-empty list of integers")
        return tuple(self._check_integer(key, item, minimum) for item in value)

    def _require(self, key: str) -> object:
        if key not in self._data:
            raise self._table_error(f"is missing key: {key}")
        return self._data[key]

    def _check_integer(self, key: str, value: object, minimum: int) -> int:
        # bool is a subclass of int, so it has to be excluded explicitly.
        if isinstance(value, bool) or not isinstance(value, int):
            raise self.error(f"{key} must be an integer")
        if value < minimum:
            raise self.error(f"{key} must be at least {minimum}, got {value}")
        return value

    def _table_error(self, message: str) -> ConfigError:
        label = f"[{self._section}]" if self._section else "config"
        return ConfigError(f"{self._source}: {label} {message}")
