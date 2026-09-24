"""Small stand-ins for the neural models, so no test downloads weights or needs a GPU.

The fake encoder hashes each word into one dimension of the vector, so texts that share words
point in similar directions. Dense retrieval then behaves like a crude keyword search, which a
test can reason about by hand. The fake scorer counts one chosen word, so a test can predict
exactly how reranking re-orders a list.
"""

import re
import zlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from groundwork.embeddings import EmbeddingModel
from groundwork.rerank import RerankerModel

_WORD = re.compile(r"[a-z0-9]+")


class HashingEncoder:
    """Bag-of-words vectors: each word adds one to a dimension chosen by its CRC-32."""

    device = "cpu"
    precision = "float32"

    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.calls.append(list(texts))
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in _WORD.findall(text.lower()):
                vectors[row, zlib.crc32(word.encode("utf-8")) % self.dimensions] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return (vectors / np.where(norms == 0, 1.0, norms)).astype(np.float32)


class WordCountScorer:
    """Scores each passage by how often it contains one word."""

    device = "cpu"
    precision = "float32"

    def __init__(self, word: str) -> None:
        self.word = word
        self.passages_scored: list[int] = []

    def score(self, question: str, passages: Sequence[str]) -> NDArray[np.float64]:
        del question
        self.passages_scored.append(len(passages))
        return np.array(
            [_WORD.findall(passage.lower()).count(self.word) for passage in passages],
            dtype=np.float64,
        )


class FakeModels:
    """A model provider handing out the fakes above, one per model, like the real one."""

    def __init__(self, cache_dir: Path, *, rerank_word: str = "jupiter") -> None:
        self._cache_dir = cache_dir
        self.encoders: dict[str, HashingEncoder] = {}
        self.reranker = WordCountScorer(rerank_word)

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def encoder(self, model: EmbeddingModel) -> HashingEncoder:
        return self.encoders.setdefault(model.key, HashingEncoder(model.dimensions))

    def scorer(self, model: RerankerModel) -> WordCountScorer:
        del model
        return self.reranker
