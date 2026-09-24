"""Hand-worked checks for BM25 scoring.

Three chunks, lengths in tokens:

    c0 "tea tea leaves"  length 3
    c1 "black tea"       length 2
    c2 "river water"     length 2

N = 3 chunks, average length = 7 / 3 = 2.333333.

IDF, Lucene variant, ln(1 + (N - df + 0.5) / (df + 0.5)):
    "tea"   appears in 2 chunks: ln(1 + 1.5 / 2.5) = ln(1.6)      = 0.470004
    "black" appears in 1 chunk:  ln(1 + 2.5 / 1.5) = ln(2.666667) = 0.980829
    "river" appears in 1 chunk:  the same, 0.980829

Term-frequency part, tf x (k1 + 1) / (tf + k1 x (1 - b + b x length / average)), k1 = 1.2, b = 0.75:
    c0, tf 2: normaliser 0.25 + 0.75 x 3 / 2.333333 = 1.214286
              2 x 2.2 / (2 + 1.2 x 1.214286) = 4.4 / 3.457143 = 1.272727
    c1, tf 1: normaliser 0.25 + 0.75 x 2 / 2.333333 = 0.892857
              1 x 2.2 / (1 + 1.2 x 0.892857) = 2.2 / 2.071429 = 1.062069
"""

import pytest

from groundwork.bm25 import BM25Index, tokenize
from groundwork.chunking import Chunk

TOLERANCE = 1e-6


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, page_id=chunk_id, text=text, start=0, end=len(text), kind="prose"
    )


CHUNKS = [_chunk("c0", "tea tea leaves"), _chunk("c1", "black tea"), _chunk("c2", "river water")]


def _ranked(index: BM25Index, query: str) -> tuple[list[str], list[float]]:
    results = index.search(query)
    return [scored.chunk.chunk_id for scored in results], [scored.score for scored in results]


class TestRejectsInvalidIndex:
    def test_no_chunks(self) -> None:
        with pytest.raises(ValueError, match="at least one token"):
            BM25Index([], k1=1.2, b=0.75)

    def test_negative_k1(self) -> None:
        with pytest.raises(ValueError, match="k1 must be at least 0"):
            BM25Index(CHUNKS, k1=-0.1, b=0.75)

    @pytest.mark.parametrize("b", [-0.1, 1.1])
    def test_b_outside_zero_to_one(self, b: float) -> None:
        with pytest.raises(ValueError, match="b must be between 0 and 1"):
            BM25Index(CHUNKS, k1=1.2, b=b)

    def test_chunks_without_tokens(self) -> None:
        # Average length would be zero, and length normalisation divides by it.
        with pytest.raises(ValueError, match="at least one token"):
            BM25Index([_chunk("c0", "... ---")], k1=1.2, b=0.75)


class TestTokenize:
    def test_lowercases_and_splits_on_non_word_characters(self) -> None:
        assert tokenize("Green-tea, BITTER; 2 cups!") == ["green", "tea", "bitter", "2", "cups"]

    def test_stem_defaults_to_off(self) -> None:
        assert tokenize("rings") == ["rings"]

    def test_stem_true_reduces_inflected_forms_to_a_common_root(self) -> None:
        assert tokenize("rings", stem=True) == tokenize("ring", stem=True) == ["ring"]


class TestScores:
    def test_single_term(self) -> None:
        # c0: 0.470004 x 1.272727 = 0.598186
        # c1: 0.470004 x 1.062069 = 0.499176
        # c2 has no query term and is not returned.
        ids, scores = _ranked(BM25Index(CHUNKS, k1=1.2, b=0.75), "tea")
        assert ids == ["c0", "c1"]
        assert scores == pytest.approx([0.598186, 0.499176], abs=TOLERANCE)

    def test_ties_are_broken_by_chunk_id(self) -> None:
        # c1 matches "black" and c2 matches "river". Both terms have IDF 0.980829 and both chunks
        # have length 2 and tf 1, so both score 0.980829 x 1.062069 = 1.041708. The tie is broken
        # by chunk id so that the order does not depend on hashing or insertion order.
        ids, scores = _ranked(BM25Index(CHUNKS, k1=1.2, b=0.75), "river black")
        assert ids == ["c1", "c2"]
        assert scores == pytest.approx([1.041708, 1.041708], abs=TOLERANCE)

    def test_repeated_query_term_counts_each_time(self) -> None:
        # As in Anserini's bag-of-words query, a term that appears twice in the query contributes
        # twice: c0 scores 2 x 0.5981864 = 1.196373, using the single-term score before rounding.
        ids, scores = _ranked(BM25Index(CHUNKS, k1=1.2, b=0.75), "tea tea")
        assert ids[0] == "c0"
        assert scores[0] == pytest.approx(1.196373, abs=TOLERANCE)

    def test_without_length_normalisation(self) -> None:
        # b = 0 makes the normaliser 1 for every chunk:
        # c0: 0.470004 x (2 x 2.2 / (2 + 1.2)) = 0.470004 x 1.375 = 0.646255
        # c1: 0.470004 x (1 x 2.2 / (1 + 1.2)) = 0.470004 x 1     = 0.470004
        ids, scores = _ranked(BM25Index(CHUNKS, k1=1.2, b=0.0), "tea")
        assert ids == ["c0", "c1"]
        assert scores == pytest.approx([0.646255, 0.470004], abs=TOLERANCE)

    @pytest.mark.parametrize("query", ["coffee", "", "?!"])
    def test_query_with_no_indexed_term_returns_nothing(self, query: str) -> None:
        assert BM25Index(CHUNKS, k1=1.2, b=0.75).search(query) == []


class TestStemming:
    def test_query_variant_matches_indexed_root_only_when_stemming_is_on(self) -> None:
        # The chunk holds "ring" and the query holds "rings". A match here proves the query is
        # stemmed too, not only the index: if only the index were stemmed, "rings" would remain
        # its own term on the query side and this would still miss.
        index = BM25Index([_chunk("c0", "the one ring")], k1=1.2, b=0.75, stem=True)
        assert [scored.chunk.chunk_id for scored in index.search("rings")] == ["c0"]

    def test_query_variant_does_not_match_without_stemming(self) -> None:
        index = BM25Index([_chunk("c0", "the one ring")], k1=1.2, b=0.75, stem=False)
        assert index.search("rings") == []

    def test_indexed_variant_matches_query_root_when_stemming_is_on(self) -> None:
        # Reversed: the chunk holds the inflected form and the query holds the root, so a match
        # here proves the indexed chunks are stemmed as well as the query.
        index = BM25Index([_chunk("c0", "many rings were forged")], k1=1.2, b=0.75, stem=True)
        assert [scored.chunk.chunk_id for scored in index.search("ring")] == ["c0"]
