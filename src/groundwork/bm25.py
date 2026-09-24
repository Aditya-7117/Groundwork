"""BM25 lexical retrieval over chunks, implemented with the standard library.

Scoring follows the Lucene variant of BM25 that Anserini uses:

    score(query, chunk) = sum over query terms t of
        idf(t) x tf x (k1 + 1) / (tf + k1 x (1 - b + b x length / average_length))

    idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))

tf is how often t occurs in the chunk, df is how many chunks contain t, N is the number of chunks,
and lengths are counted in tokens. This IDF never goes negative, unlike the original Robertson
form. A term repeated in the query contributes once per repetition.

Tokenisation is deliberately plain: lowercase, then split on anything that is not a letter, digit
or underscore. There is no stopword list. Stemming is optional and off by default; when it is on,
each token is reduced with NLTK's Porter stemmer, so "ring" and "rings" become the same term.
"""

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from nltk.stem import PorterStemmer

from groundwork.chunking import Chunk

_TOKEN = re.compile(r"\w+")
_STEMMER = PorterStemmer()
_STEM_CACHE: dict[str, str] = {}


def _stem(word: str) -> str:
    """Return word's Porter stem, from a module-level cache since words repeat constantly."""
    cached = _STEM_CACHE.get(word)
    if cached is None:
        cached = _STEMMER.stem(word)
        _STEM_CACHE[word] = cached
    return cached


def tokenize(text: str, *, stem: bool = False) -> list[str]:
    """Lowercase text and split it into word tokens.

    Args:
        text: The text to tokenise.
        stem: Whether to reduce each token to its Porter stem, so that inflected forms like
            "rings" collapse onto their root, "ring".

    Returns:
        The tokens, lowercased and in order.
    """
    words = _TOKEN.findall(text.lower())
    return [_stem(word) for word in words] if stem else words


@dataclass(frozen=True, slots=True, kw_only=True)
class ScoredChunk:
    """A chunk and its retrieval score for one query."""

    chunk: Chunk
    score: float


class BM25Index:
    """An in-memory inverted index that scores chunks against a query with BM25."""

    def __init__(self, chunks: Sequence[Chunk], *, k1: float, b: float, stem: bool = False) -> None:
        """Index chunks for BM25 scoring.

        Args:
            chunks: The chunks to index.
            k1: Term-frequency saturation, at least 0.
            b: Length normalisation, from 0 to 1.
            stem: Whether to reduce tokens to their Porter stem before indexing. Queries are
                stemmed the same way by search, so this must match how the chunks were written.

        Raises:
            ValueError: If a parameter is out of range or the chunks contain no tokens.
        """
        if k1 < 0:
            raise ValueError(f"k1 must be at least 0, got {k1}")
        if not 0 <= b <= 1:
            raise ValueError(f"b must be between 0 and 1, got {b}")

        self._chunks = list(chunks)
        self._k1 = k1
        self._stem = stem
        self._postings: dict[str, list[tuple[int, int]]] = {}
        lengths: list[int] = []
        for position, chunk in enumerate(self._chunks):
            counts = Counter(tokenize(chunk.text, stem=stem))
            lengths.append(counts.total())
            for term, frequency in counts.items():
                self._postings.setdefault(term, []).append((position, frequency))

        total_tokens = sum(lengths)
        if total_tokens == 0:
            raise ValueError("BM25 needs at least one token across the indexed chunks")
        average_length = total_tokens / len(lengths)
        self._normalisers = [k1 * (1 - b + b * length / average_length) for length in lengths]
        chunk_count = len(self._chunks)
        self._idf = {
            term: math.log(1 + (chunk_count - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self._postings.items()
        }

    def search(self, query: str) -> list[ScoredChunk]:
        """Return every chunk that shares at least one term with the query, best first.

        Equal scores are ordered by chunk id, so the ranking never depends on hash or insertion
        order.
        """
        scores: dict[int, float] = {}
        for term in tokenize(query, stem=self._stem):
            if term not in self._postings:
                continue
            idf = self._idf[term]
            for position, frequency in self._postings[term]:
                gain = idf * frequency * (self._k1 + 1) / (frequency + self._normalisers[position])
                scores[position] = scores.get(position, 0.0) + gain
        ranked = sorted(scores.items(), key=lambda item: (-item[1], self._chunks[item[0]].chunk_id))
        return [
            ScoredChunk(chunk=self._chunks[position], score=score) for position, score in ranked
        ]
