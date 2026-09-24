"""Embedding models: turning chunks and questions into vectors, and searching them exactly.

An embedding model maps a text to a vector so that texts with similar meaning point in similar
directions. Retrieval then ranks every chunk by the dot product of its vector with the question's
(the vectors are normalised, so this is cosine similarity). The search is exact: every chunk is
compared, so no approximate index adds its own errors to the comparison (decision 18).

Encoding the corpus takes hours for the larger model, so vectors are cached on disk, one per
distinct text (decision 69). Identical text, such as a table chunked the same way by every chunker,
is encoded once and shared; an ablation that changes only the table chunks re-encodes only those.
The cache is kept apart per model, pinned revision and numeric precision, and each vector is found
by a digest of its exact text, so it can never be served for different input. Vectors are written
in shards as they are encoded, so an interrupted run resumes from the last finished shard.
"""

import hashlib
import json
import logging
import re
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
_SHARD_FILE = re.compile(r"shard-(\d{4})\.npy")


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
    """A real model: half precision on the Apple GPU (decision 46), full precision on a CPU."""

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        cache_dir: Path,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        """Load the pinned weights; on the Apple GPU if present, unless a device is given."""
        import torch  # noqa: PLC0415 -- heavy import, only paid when a model is actually used
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
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


class VectorStore:
    """One vector per distinct text, for one model at one precision, kept on disk in shards.

    Each shard is three files: shard-NNNN.keys (one text digest per line, in row order),
    shard-NNNN.seconds (how long the shard took to encode) and shard-NNNN.npy (the vectors).
    The .npy file is renamed into place last, so a shard without it never existed.
    """

    def __init__(self, directory: Path, dimensions: int) -> None:
        """Open a store, reading every finished shard's keys.

        Raises:
            EmbeddingError: If a finished shard is missing its keys or has the wrong shape.
        """
        self._directory = directory
        self._dimensions = dimensions
        self._rows: dict[str, tuple[int, int]] = {}
        self._cost: dict[int, float] = {}
        directory.mkdir(parents=True, exist_ok=True)
        for path in sorted(directory.glob("shard-*.npy")):
            match = _SHARD_FILE.fullmatch(path.name)
            if match is None:
                continue
            shard = int(match.group(1))
            keys_path = path.with_suffix(".keys")
            if not keys_path.is_file():
                raise EmbeddingError(f"{path} has no keys file")
            keys = keys_path.read_text(encoding="utf-8").split()
            vectors = np.load(path, mmap_mode="r")
            if vectors.shape != (len(keys), dimensions):
                raise EmbeddingError(
                    f"{path} holds shape {vectors.shape}, expected {(len(keys), dimensions)}"
                )
            seconds = float(path.with_suffix(".seconds").read_text(encoding="utf-8"))
            self._cost[shard] = seconds / len(keys) if keys else 0.0
            for row, key in enumerate(keys):
                self._rows[key] = (shard, row)

    def __contains__(self, key: str) -> bool:
        """Whether the store holds a vector for this text digest."""
        return key in self._rows

    def add(self, keys: Sequence[str], vectors: NDArray[np.float16], seconds: float) -> None:
        """Append a shard of vectors and the time they took to encode."""
        shard = max(self._cost, default=-1) + 1
        stem = self._directory / f"shard-{shard:04d}"
        stem.with_suffix(".keys").write_text("\n".join(keys) + "\n", encoding="utf-8")
        stem.with_suffix(".seconds").write_text(f"{seconds}\n", encoding="utf-8")
        partial = stem.with_suffix(".partial.npy")
        np.save(partial, vectors)
        partial.rename(stem.with_suffix(".npy"))
        self._cost[shard] = seconds / len(keys)
        for row, key in enumerate(keys):
            self._rows[key] = (shard, row)

    def gather(self, keys: Sequence[str]) -> tuple[NDArray[np.float16], float]:
        """Return the vectors for keys in order, and what they cost when first encoded."""
        located = np.array([self._rows[key] for key in keys], dtype=np.int64).reshape(-1, 2)
        out = np.empty((len(keys), self._dimensions), dtype=np.float16)
        for shard in np.unique(located[:, 0]).tolist():
            selected = located[:, 0] == shard
            vectors = np.load(self._directory / f"shard-{shard:04d}.npy", mmap_mode="r")
            out[selected] = vectors[located[selected, 1]]
        charged = sum(self._cost[shard] for shard in located[:, 0].tolist())
        return out, charged


def chunk_embeddings(
    chunks: Sequence[Chunk],
    model: EmbeddingModel,
    encoder: Encoder,
    *,
    cache_dir: Path,
    precision: str,
) -> tuple[NDArray[np.float16], float]:
    """Return one vector per chunk, encoding only texts the cache lacks, and the encoding cost.

    Vectors are stored in half precision; searching converts them back to float32. The cost is
    what encoding these chunks took when each vector was first computed, charged per chunk, so a
    run served from the cache, or sharing vectors with another chunker, still reports what
    embedding this corpus actually costs.

    Raises:
        EmbeddingError: If a cached shard is damaged or the encoder returns the wrong shape.
    """
    identity = hashlib.sha256()
    for part in (model.name, model.revision, precision):
        identity.update(part.encode("utf-8"))
        identity.update(b"\0")
    directory = cache_dir / f"{model.key}-{identity.hexdigest()[:16]}"
    store = VectorStore(directory, model.dimensions)
    (directory / "meta.json").write_text(
        json.dumps(
            {"model": model.name, "revision": model.revision, "precision": precision}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )

    keys = [_text_key(chunk.text) for chunk in chunks]
    todo: dict[str, str] = {}
    for key, chunk in zip(keys, chunks, strict=True):
        if key not in store and key not in todo:
            todo[key] = chunk.text
    pending = list(todo.items())
    started = time.perf_counter()
    for first in range(0, len(pending), _SHARD_SIZE):
        batch = pending[first : first + _SHARD_SIZE]
        batch_started = time.perf_counter()
        vectors = encoder.encode([text for _, text in batch]).astype(np.float16)
        seconds = time.perf_counter() - batch_started
        if vectors.shape != (len(batch), model.dimensions):
            raise EmbeddingError(
                f"encoder returned shape {vectors.shape}, expected {(len(batch), model.dimensions)}"
            )
        store.add([key for key, _ in batch], vectors, seconds)
        logger.info(
            "embedding shard written",
            extra={
                "model": model.key,
                "done": first + len(batch),
                "to_encode": len(pending),
                "chunks": len(chunks),
                "minutes_elapsed": round((time.perf_counter() - started) / 60, 1),
            },
        )
    return store.gather(keys)


def _text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


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
