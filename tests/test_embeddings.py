"""The chunk-vector cache and exact dense search.

Encoding the full corpus with the larger model takes hours, so the cache has to be right: reused
only for exactly the same inputs, resumable after an interruption, and honest about what the
encoding cost.
"""

import time
from collections.abc import Sequence
from itertools import chain, repeat
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from fakes import HashingEncoder
from groundwork import embeddings
from groundwork.chunking import Chunk
from groundwork.embeddings import (
    DenseIndex,
    EmbeddingError,
    chunk_embeddings,
    get_model,
)

MINILM = get_model("all-MiniLM-L6-v2")
QWEN = get_model("Qwen3-Embedding-0.6B")


def chunk(position: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"p#{position}", page_id="p", text=text, start=0, end=len(text), kind="prose"
    )


CHUNKS = [
    chunk(0, "green tea is bitter"),
    chunk(1, "the river shannon"),
    chunk(2, "jupiter is a gas giant"),
    chunk(3, "black tea"),
    chunk(4, "the longest river"),
]


class FailsAfter(HashingEncoder):
    """Encodes a set number of batches, then fails as an interrupted run would."""

    def __init__(self, dimensions: int, batches: int) -> None:
        super().__init__(dimensions)
        self.batches = batches

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if len(self.calls) == self.batches:
            raise RuntimeError("interrupted")
        return super().encode(texts)


class TestChunkEmbeddings:
    def test_vectors_are_computed_once_then_read_from_the_cache(self, tmp_path: Path) -> None:
        first_encoder = HashingEncoder(MINILM.dimensions)
        first, first_seconds = chunk_embeddings(
            CHUNKS, MINILM, first_encoder, cache_dir=tmp_path, precision="float32"
        )
        second_encoder = HashingEncoder(MINILM.dimensions)
        second, second_seconds = chunk_embeddings(
            CHUNKS, MINILM, second_encoder, cache_dir=tmp_path, precision="float32"
        )
        assert first_encoder.calls
        assert not second_encoder.calls
        np.testing.assert_array_equal(first, second)
        assert first.shape == (len(CHUNKS), MINILM.dimensions)
        assert first.dtype == np.float16

        # A cache hit still reports what encoding cost when it happened, never zero.
        assert second_seconds == pytest.approx(first_seconds)

    def test_only_a_changed_chunk_text_is_encoded(self, tmp_path: Path) -> None:
        chunk_embeddings(
            CHUNKS, MINILM, HashingEncoder(MINILM.dimensions), cache_dir=tmp_path, precision="f"
        )
        changed = [*CHUNKS[:-1], chunk(4, "the shortest river")]
        encoder = HashingEncoder(MINILM.dimensions)
        chunk_embeddings(changed, MINILM, encoder, cache_dir=tmp_path, precision="f")
        assert encoder.calls == [["the shortest river"]]

    def test_identical_text_is_encoded_once_and_charged_per_chunk(self, tmp_path: Path) -> None:
        # Another chunker producing the same text, such as a table chunk, reuses its vector.
        twice = [chunk(0, "black tea"), chunk(1, "black tea")]
        encoder = HashingEncoder(MINILM.dimensions)
        vectors, _ = chunk_embeddings(twice, MINILM, encoder, cache_dir=tmp_path, precision="f")
        assert encoder.calls == [["black tea"]]
        np.testing.assert_array_equal(vectors[0], vectors[1])

    def test_the_charge_splits_a_shards_time_across_its_texts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Start, shard start, shard end, log line; then a steady clock for the cache hit.
        clock = chain([0.0, 100.0, 104.0, 110.0], repeat(200.0))
        monkeypatch.setattr(time, "perf_counter", lambda: next(clock))
        _, seconds = chunk_embeddings(
            CHUNKS[:2],
            MINILM,
            HashingEncoder(MINILM.dimensions),
            cache_dir=tmp_path,
            precision="f",
        )
        # One shard of two texts took 104 - 100 = 4 seconds: 2 seconds per chunk, 4 in all.
        assert seconds == pytest.approx(4.0)
        _, reused = chunk_embeddings(
            [CHUNKS[0]],
            MINILM,
            HashingEncoder(MINILM.dimensions),
            cache_dir=tmp_path,
            precision="f",
        )
        assert reused == pytest.approx(2.0)

    def test_a_different_precision_is_encoded_afresh(self, tmp_path: Path) -> None:
        chunk_embeddings(
            CHUNKS, MINILM, HashingEncoder(MINILM.dimensions), cache_dir=tmp_path, precision="a"
        )
        encoder = HashingEncoder(MINILM.dimensions)
        chunk_embeddings(CHUNKS, MINILM, encoder, cache_dir=tmp_path, precision="b")
        assert encoder.calls

    def test_an_interrupted_run_resumes_from_its_last_finished_shard(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(embeddings, "_SHARD_SIZE", 2)
        with pytest.raises(RuntimeError, match="interrupted"):
            chunk_embeddings(
                CHUNKS,
                MINILM,
                FailsAfter(MINILM.dimensions, batches=2),
                cache_dir=tmp_path,
                precision="f",
            )
        resumed = HashingEncoder(MINILM.dimensions)
        vectors, _ = chunk_embeddings(CHUNKS, MINILM, resumed, cache_dir=tmp_path, precision="f")
        assert resumed.calls == [[CHUNKS[4].text]]
        fresh, _ = chunk_embeddings(
            CHUNKS,
            MINILM,
            HashingEncoder(MINILM.dimensions),
            cache_dir=tmp_path / "fresh",
            precision="f",
        )
        np.testing.assert_array_equal(vectors, fresh)

    def test_a_cached_shard_of_the_wrong_shape_is_rejected(self, tmp_path: Path) -> None:
        chunk_embeddings(
            CHUNKS, MINILM, HashingEncoder(MINILM.dimensions), cache_dir=tmp_path, precision="f"
        )
        [shard] = tmp_path.glob("*/shard-0000.npy")
        np.save(shard, np.zeros((2, 3), dtype=np.float16))
        with pytest.raises(EmbeddingError, match="holds shape"):
            chunk_embeddings(
                CHUNKS,
                MINILM,
                HashingEncoder(MINILM.dimensions),
                cache_dir=tmp_path,
                precision="f",
            )


class TestDenseIndex:
    def test_the_chunk_sharing_the_most_words_scores_highest(self, tmp_path: Path) -> None:
        encoder = HashingEncoder(MINILM.dimensions)
        vectors, _ = chunk_embeddings(CHUNKS, MINILM, encoder, cache_dir=tmp_path, precision="f")
        scores = DenseIndex(vectors, MINILM, encoder).scores("which gas giant is jupiter")
        assert int(np.argmax(scores)) == 2
        assert scores.dtype == np.float64

    def test_scores_are_cosine_similarities(self, tmp_path: Path) -> None:
        encoder = HashingEncoder(MINILM.dimensions)
        vectors, _ = chunk_embeddings(CHUNKS, MINILM, encoder, cache_dir=tmp_path, precision="f")
        scores = DenseIndex(vectors, MINILM, encoder).scores("black tea")
        # Identical text gives similarity 1, up to the half-precision storage of chunk vectors.
        assert scores[3] == pytest.approx(1.0, abs=1e-3)
        assert scores.max() <= 1.0 + 1e-3

    def test_the_query_instruction_is_put_before_the_question(self, tmp_path: Path) -> None:
        # Qwen3-Embedding is trained to see an instruction before each query, and not before
        # passages; leaving it out costs retrieval quality, per the model card.
        encoder = HashingEncoder(QWEN.dimensions)
        vectors, _ = chunk_embeddings(CHUNKS, QWEN, encoder, cache_dir=tmp_path, precision="f")
        DenseIndex(vectors, QWEN, encoder).scores("longest river")
        assert encoder.calls[0] == [c.text for c in CHUNKS]
        assert encoder.calls[-1] == [f"{QWEN.query_instruction}longest river"]


def test_an_unknown_model_names_the_known_ones() -> None:
    with pytest.raises(EmbeddingError, match=r"known: Qwen3-Embedding-0\.6B, all-MiniLM-L6-v2"):
        get_model("bert")
