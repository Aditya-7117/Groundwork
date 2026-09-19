"""Splitting documents into the chunks that retrieval searches over.

A chunk keeps the character span it came from, so a later stage can cite exactly which part of
which document an answer rests on.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from groundwork.config import ChunkingConfig
from groundwork.corpus import Document

_WORD = re.compile(r"\S+")


@dataclass(frozen=True, slots=True, kw_only=True)
class Chunk:
    """A contiguous span of one document.

    Attributes:
        chunk_id: "<doc_id>#<position>", unique within one chunking of a corpus.
        doc_id: The document the chunk came from.
        start: Offset of the first character in the document's indexable text.
        end: Offset one past the last character.
        text: The exact characters between start and end.
    """

    chunk_id: str
    doc_id: str
    start: int
    end: int
    text: str


def chunk_documents(documents: Iterable[Document], config: ChunkingConfig) -> list[Chunk]:
    """Chunk every document with the configured strategy, in the order given."""
    match config.strategy:
        case "fixed_words":
            return [
                chunk
                for document in documents
                for chunk in chunk_fixed_words(
                    document.doc_id,
                    document.indexable_text,
                    size=config.size,
                    overlap=config.overlap,
                )
            ]


def chunk_fixed_words(doc_id: str, text: str, *, size: int, overlap: int) -> list[Chunk]:
    """Split text into windows of `size` words, each sharing `overlap` words with the previous.

    A word is a run of non-whitespace characters. Windows start every size - overlap words and
    stop at the first window that reaches the final word, so the last chunk may be shorter.
    Text with no words gives no chunks.

    Raises:
        ValueError: If size is below 1, or overlap is negative or not less than size.
    """
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}")
    if not 0 <= overlap < size:
        raise ValueError(f"overlap must be at least 0 and less than size, got {overlap}")

    spans = [match.span() for match in _WORD.finditer(text)]
    chunks: list[Chunk] = []
    for position, first in enumerate(range(0, len(spans), size - overlap)):
        last = min(first + size, len(spans)) - 1
        start, end = spans[first][0], spans[last][1]
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}#{position}",
                doc_id=doc_id,
                start=start,
                end=end,
                text=text[start:end],
            )
        )
        if last == len(spans) - 1:
            break
    return chunks
