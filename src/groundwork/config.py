"""Experiment configuration: one TOML file in configs/ defines one experiment completely.

Validation is strict: unknown keys, missing keys, wrong types and inconsistent values all fail at
load time with the file and the offending field named. There are no defaults for anything that
changes a result, because a typo that quietly fell back to a default would produce a number from
an experiment nobody asked for.
"""

import hashlib
import json
import re
import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from groundwork.chunking import ChunkStrategy

_NAME_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


class ConfigError(ValueError):
    """An experiment config file is missing, malformed or inconsistent."""


class _Section(BaseModel):
    """Base for every config section: no unknown keys, no silent type changes, immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class CorpusConfig(_Section):
    """Which corpus to evaluate on, and which of its questions.

    Attributes:
        name: Registered corpus id, for example "natural-questions".
        split: Which judgements to evaluate against.
        query_limit: Evaluate a seeded random sample of this many questions, or all when unset.
    """

    name: str
    split: str
    query_limit: int | None = Field(default=None, ge=1)


class ChunkingConfig(_Section):
    """How pages are split into the passages retrieval searches.

    Attributes:
        strategy: Chunking strategy id.
        size: Word budget per chunk, excluding the title and section prefix.
        overlap: Words repeated from the previous chunk.
        flatten_tables: Ablation: treat tables as loose text instead of keeping their rows.
    """

    strategy: ChunkStrategy
    size: int = Field(ge=1)
    overlap: int = Field(ge=0)
    flatten_tables: bool = False

    @model_validator(mode="after")
    def _overlap_below_size(self) -> Self:
        if self.overlap >= self.size:
            raise ValueError(f"overlap must be less than size, got {self.overlap} and {self.size}")
        return self


class BM25Config(_Section):
    """Keyword retrieval settings.

    Attributes:
        k1: Term-frequency saturation.
        b: Length normalisation, 0 to 1.
        stem: Reduce words to their stems, as Anserini's BM25 does.
    """

    k1: float = Field(ge=0)
    b: float = Field(ge=0, le=1)
    stem: bool


class DenseConfig(_Section):
    """Embedding retrieval settings.

    Attributes:
        model: Registered embedding model id, for example "Qwen3-Embedding-0.6B".
    """

    model: str


class FusionConfig(_Section):
    """How hybrid retrieval merges the keyword and embedding rankings.

    Attributes:
        k: Reciprocal rank fusion constant; each ranking adds 1 / (k + rank).
        candidates: How deep into each ranking the fusion looks.
    """

    k: int = Field(ge=1)
    candidates: int = Field(ge=1)


class RetrievalConfig(_Section):
    """How chunks are found for a question.

    Attributes:
        method: bm25 (keywords), dense (embeddings) or hybrid (both, fused).
        depth: Number of chunks, and of pages, returned per question.
        bm25: Keyword settings, required for bm25 and hybrid.
        dense: Embedding settings, required for dense and hybrid.
        fusion: Fusion settings, required for hybrid.
    """

    method: Literal["bm25", "dense", "hybrid"]
    depth: int = Field(ge=1)
    bm25: BM25Config | None = None
    dense: DenseConfig | None = None
    fusion: FusionConfig | None = None

    @model_validator(mode="after")
    def _sections_match_the_method(self) -> Self:
        needed = {
            "bm25": {"bm25"},
            "dense": {"dense"},
            "hybrid": {"bm25", "dense", "fusion"},
        }[self.method]
        present = {name for name in ("bm25", "dense", "fusion") if getattr(self, name) is not None}
        if missing := needed - present:
            raise ValueError(f"method {self.method} needs [retrieval.{', '.join(sorted(missing))}]")
        if extra := present - needed:
            raise ValueError(
                f"method {self.method} does not use [retrieval.{', '.join(sorted(extra))}]; "
                "remove it so the config says exactly what runs"
            )
        return self


class RerankConfig(_Section):
    """Second-stage re-ordering of the first-stage results.

    Attributes:
        model: Registered reranker id, for example "bge-reranker-v2-m3".
        depth: How many first-stage chunks are re-ordered.
    """

    model: str
    depth: int = Field(ge=1)


class EvaluationConfig(_Section):
    """Which rank cutoffs the metrics are reported at.

    Attributes:
        cutoffs: Values of k for precision@k, recall@k, nDCG@k and MRR@k.
    """

    cutoffs: tuple[int, ...]

    @field_validator("cutoffs", mode="before")
    @classmethod
    def _accept_a_toml_list(cls, value: object) -> object:
        """TOML arrays arrive as lists; the values inside are still checked strictly."""
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _cutoffs_are_sane(self) -> Self:
        if not self.cutoffs:
            raise ValueError("cutoffs must not be empty")
        if any(cutoff < 1 for cutoff in self.cutoffs):
            raise ValueError(f"every cutoff must be at least 1, got {list(self.cutoffs)}")
        if len(set(self.cutoffs)) != len(self.cutoffs):
            raise ValueError(f"cutoffs must not repeat, got {list(self.cutoffs)}")
        return self


class ExperimentConfig(_Section):
    """A complete, validated experiment definition.

    Attributes:
        name: Lowercase slug naming the results directory.
        description: What the experiment is for, in plain words.
        seed: Seed for every source of randomness in the run.
        corpus: Corpus and question selection.
        chunking: Chunking strategy and parameters.
        retrieval: Retrieval method and parameters.
        rerank: Optional second-stage reranker.
        evaluation: Metric cutoffs.
    """

    name: str
    description: str
    seed: int = Field(ge=0)
    corpus: CorpusConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    rerank: RerankConfig | None = None
    evaluation: EvaluationConfig

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if not _NAME_PATTERN.fullmatch(self.name):
            raise ValueError(
                f"name must be lowercase letters, digits and single hyphens, got {self.name!r}"
            )
        if not self.description.strip():
            raise ValueError("description must not be empty")
        deepest = max(self.evaluation.cutoffs)
        if deepest > self.retrieval.depth:
            raise ValueError(
                f"cutoff {deepest} exceeds retrieval depth {self.retrieval.depth}: the metric "
                "would report a number that could not have been measured"
            )
        if self.rerank is not None and self.rerank.depth > self.retrieval.depth:
            raise ValueError(
                f"rerank depth {self.rerank.depth} exceeds retrieval depth {self.retrieval.depth}"
            )
        return self


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
    try:
        return ExperimentConfig.model_validate(raw)
    except ValidationError as error:
        raise ConfigError(f"{path}: {_explain(error)}") from error


def config_digest(config: ExperimentConfig) -> str:
    """Return a SHA-256 hex digest identifying the experiment a config defines.

    The digest covers the parsed values, not the file bytes, so comments, whitespace and key order
    do not change it, and any change to a value does.
    """
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _explain(error: ValidationError) -> str:
    """Turn pydantic's report into one line per problem, naming the field."""
    problems = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "config"
        problems.append(f"[{location}] {item['msg']}")
    return "; ".join(problems)
