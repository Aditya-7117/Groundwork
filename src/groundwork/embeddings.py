"""Embedding models: turning chunks and questions into vectors, and searching them exactly.

An embedding model maps a text to a vector so that texts with similar meaning point in similar
directions. Retrieval then ranks every chunk by the dot product of its vector with the question's
(the vectors are normalised, so this is cosine similarity). The search is exact: every chunk is
compared, so no approximate index adds its own errors to the comparison (decision 18).

Encoding 341,000 chunks takes hours for the larger model, so chunk vectors are cached on disk in
shards. An interrupted run resumes from the last finished shard. The cache key covers the model,
its pinned revision, the numeric precision and the exact chunk texts, so a cached file can never
be reused for different inputs.
"""

import hashlib
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from groundwork.chunking import Chunk

logger = logging.getLogger(__name__)

_SHARD_SIZE = 20_000


@dataclass(frozen=True, slots=True, kw_only=True)
class EmbeddingModel:
    """A registered embedding model, pinned to one published revision.

    Attributes:
        key: Id used in configs.
        name: Hugging Face repository.
        revision: Commit of that repository, so the weights cannot change underneath a result.
        query_instruction: Text put before every question, as the model's authors recommend.
        max_seq_length: Longest input, in the model's own tokens, before text is cut off.
        dimensions: Length of each vector.
    """

    key: str
    name: str
    revision: str
    query_instruction: str
    max_seq_length: int
    dimensions: int


EMBEDDING_MODELS: dict[str, EmbeddingModel] = {
    model.key: model
    for model in (
        EmbeddingModel(
            key="all-MiniLM-L6-v2",
            name="sentence-transformers/all-MiniLM-L6-v2",
            revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            query_instruction="",
            max_seq_length=256,
            dimensions=384,
        ),
        EmbeddingModel(
            key="Qwen3-Embedding-0.6B",
            name="Qwen/Qwen3-Embedding-0.6B",
            revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
            # The format from the model card: an instruction line, then the query (decision 48).
            query_instruction=(
                "Instruct: Given a web search query, retrieve relevant passages that answer the "
                "query\nQuery:"
            ),
            max_seq_length=512,
            dimensions=1024,
        ),
    )
}


class EmbeddingError(ValueError):
    """An embedding model is unknown, or a cache does not match what it should hold."""


class Encoder(Protocol):
    """Anything that turns texts into normalised vectors, one row per text.

    Attributes:
        device: Where the model runs, recorded with every result.
        precision: The numeric precision it runs in; part of the vector cache key.
    """

    device: str
    precision: str

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Return a (len(texts), dimensions) array of unit-length vectors."""
        ...


def get_model(key: str) -> EmbeddingModel:
    """Return a registered model.

    Raises:
        EmbeddingError: If no model has this key.
    """
    try:
        return EMBEDDING_MODELS[key]
    except KeyError as error:
        known = ", ".join(sorted(EMBEDDING_MODELS))
        raise EmbeddingError(f"unknown embedding model {key!r}; known: {known}") from error


class SentenceTransformerEncoder:
    """A real model, run on the Apple GPU in half precision (decision 46)."""

    def __init__(self, model: EmbeddingModel, *, cache_dir: Path, batch_size: int = 32) -> None:
        """Load the pinned model weights."""
        import torch  # noqa: PLC0415 -- heavy import, only paid when a model is actually used
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if device == "mps" else torch.float32
        self._model = SentenceTransformer(
            model.name,
            revision=model.revision,
            device=device,
            cache_folder=str(cache_dir),
            model_kwargs={"torch_dtype": dtype},
        )
        self._model.max_seq_length = model.max_seq_length
        self._batch_size = batch_size
        self.precision = "float16" if dtype == torch.float16 else "float32"
        self.device = device

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Encode texts into unit-length float32 vectors."""
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def chunk_embeddings(
    chunks: Sequence[Chunk],
    model: EmbeddingModel,
    encoder: Encoder,
    *,
    cache_dir: Path,
    precision: str,
) -> tuple[NDArray[np.float16], float]:
    """Return one vector per chunk, from the cache where possible, and the encoding time.

    Vectors are stored in half precision; searching converts them back to float32. The time is
    the total spent encoding, recorded per shard when it was written, so a run served from the
    cache still reports what embedding the corpus actually cost.

    Raises:
        EmbeddingError: If a cached shard has the wrong shape.
    """
    digest = hashlib.sha256()
    for part in (model.name, model.revision, precision):
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    for chunk in chunks:
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\0")
    key = digest.hexdigest()[:16]
    directory = cache_dir / f"{model.key}-{key}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").write_text(
        json.dumps(
            {
                "model": model.name,
                "revision": model.revision,
                "precision": precision,
                "chunks": len(chunks),
                "shard_size": _SHARD_SIZE,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    shards: list[NDArray[np.float16]] = []
    encode_seconds = 0.0
    started = time.perf_counter()
    for shard, first in enumerate(range(0, len(chunks), _SHARD_SIZE)):
        texts = [chunk.text for chunk in chunks[first : first + _SHARD_SIZE]]
        path = directory / f"shard-{shard:04d}.npy"
        if path.is_file():
            vectors = np.load(path)
            if vectors.shape != (len(texts), model.dimensions):
                raise EmbeddingError(
                    f"{path} holds shape {vectors.shape}, expected {(len(texts), model.dimensions)}"
                )
            timing = path.with_suffix(".seconds")
            encode_seconds += float(timing.read_text()) if timing.is_file() else 0.0
        else:
            shard_started = time.perf_counter()
            vectors = encoder.encode(texts).astype(np.float16)
            seconds = time.perf_counter() - shard_started
            encode_seconds += seconds
            partial = path.with_suffix(".partial.npy")
            np.save(partial, vectors)
            path.with_suffix(".seconds").write_text(f"{seconds}\n", encoding="utf-8")
            partial.rename(path)
            done = first + len(texts)
            elapsed = time.perf_counter() - started
            logger.info(
                "embedding shard written",
                extra={
                    "model": model.key,
                    "done": done,
                    "total": len(chunks),
                    "minutes_elapsed": round(elapsed / 60, 1),
                },
            )
        shards.append(vectors)
    vectors_all = np.concatenate(shards) if shards else np.zeros((0, model.dimensions), np.float16)
    return vectors_all, encode_seconds


class DenseIndex:
    """Exact search over chunk vectors."""

    def __init__(
        self, vectors: NDArray[np.float16], model: EmbeddingModel, encoder: Encoder
    ) -> None:
        """Hold the chunk vectors in float32 for searching."""
        self._vectors = vectors.astype(np.float32)
        self._model = model
        self._encoder = encoder

    def scores(self, query: str) -> NDArray[np.float64]:
        """Return every chunk's cosine similarity to the question."""
        [vector] = self._encoder.encode([f"{self._model.query_instruction}{query}"])
        scores: NDArray[np.float64] = np.asarray(self._vectors @ vector, dtype=np.float64)
        return scores
