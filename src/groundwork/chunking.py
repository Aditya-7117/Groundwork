"""Splitting pages into the chunks that retrieval searches over.

Three strategies are compared in the grid:

- fixed_words: pack prose into windows of a fixed number of words, cutting wherever the count
  runs out. This is the naive baseline the other two have to beat.
- sentence_aware: pack whole sentences up to the same budget, so no chunk starts or ends
  mid-sentence.
- section_aware: the same, but a chunk never crosses a section heading.

Two rules hold for every strategy, so the comparison isolates the strategy itself:

- Each chunk begins with its page title and section, because a passage lifted out of a page loses
  what it is about. The prefix does not count against the word budget.
- Tables are chunked separately, never mixed into prose: each chunk repeats the table's header and
  contains whole rows. Setting flatten_tables turns that off, which is the ablation that measures
  what the table handling is worth (decision 40).
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from groundwork.page import Block, Page
from groundwork.sentences import sentence_spans

type ChunkStrategy = Literal["fixed_words", "sentence_aware", "section_aware"]
type ChunkKind = Literal["prose", "table"]

_WORD = re.compile(r"\S+")


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkingSettings:
    """How a page is split.

    Attributes:
        strategy: Which chunking strategy to use.
        size: Word budget per chunk, excluding the title and section prefix.
        overlap: Words a chunk repeats from the one before it.
        flatten_tables: Ablation: treat tables as loose text instead of keeping their rows.
    """

    strategy: ChunkStrategy
    size: int
    overlap: int
    flatten_tables: bool = False

    def __post_init__(self) -> None:
        """Check the budget makes sense.

        Raises:
            ValueError: If size is below 1, or overlap is negative or not less than size.
        """
        if self.size < 1:
            raise ValueError(f"size must be at least 1, got {self.size}")
        if not 0 <= self.overlap < self.size:
            raise ValueError(
                f"overlap must be at least 0 and less than size, "
                f"got {self.overlap} with size {self.size}"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class Chunk:
    """A passage of one page, as retrieval sees it.

    Attributes:
        chunk_id: "<page_id>#<position>".
        page_id: The page it came from.
        text: The prefix line, then the passage.
        start: Where the passage starts in the page text.
        end: Where it ends.
        kind: Whether it came from prose or from a table.
    """

    chunk_id: str
    page_id: str
    text: str
    start: int
    end: int
    kind: ChunkKind


def chunk_pages(pages: Iterable[Page], settings: ChunkingSettings) -> list[Chunk]:
    """Chunk every page, in order."""
    return [chunk for page in pages for chunk in chunk_page(page, settings)]


def chunk_page(page: Page, settings: ChunkingSettings) -> list[Chunk]:
    """Split one page into chunks."""
    pieces: list[tuple[str, int, int, ChunkKind]] = []
    prose_run: list[Block] = []
    for block in page.blocks:
        if block.kind == "table" and not settings.flatten_tables:
            pieces.extend(_flush_prose(prose_run, page, settings))
            prose_run = []
            pieces.extend(_table_pieces(block, settings))
        else:
            prose_run.append(block)
    pieces.extend(_flush_prose(prose_run, page, settings))

    return [
        Chunk(
            chunk_id=f"{page.page_id}#{position}",
            page_id=page.page_id,
            text=f"{_prefix(page.title, section)}\n{body}",
            start=start,
            end=end,
            kind=kind,
        )
        for position, (body, start, end, kind, section) in enumerate(_with_sections(pieces, page))
    ]


def overlapping_chunks(chunks: Sequence[Chunk], start: int, end: int) -> tuple[str, ...]:
    """Return the ids of chunks covering any part of [start, end), the relevance rule."""
    return tuple(chunk.chunk_id for chunk in chunks if chunk.start < end and start < chunk.end)


def _prefix(title: str, section: str) -> str:
    return f"{title} > {section}" if section else title


def _with_sections(
    pieces: Sequence[tuple[str, int, int, ChunkKind]], page: Page
) -> list[tuple[str, int, int, ChunkKind, str]]:
    """Attach each piece's section, taken from the block its text starts in."""
    resolved: list[tuple[str, int, int, ChunkKind, str]] = []
    for body, start, end, kind in pieces:
        section = ""
        caption = ""
        for block in page.blocks:
            if block.start <= start < block.end or (block.start <= start and end <= block.end):
                section, caption = block.section, block.caption
                break
        if kind == "table" and caption:
            section = f"{section} > {caption}" if section else caption
        resolved.append((body, start, end, kind, section))
    return resolved


def _flush_prose(
    blocks: Sequence[Block], page: Page, settings: ChunkingSettings
) -> list[tuple[str, int, int, ChunkKind]]:
    """Chunk a run of consecutive prose blocks."""
    if not blocks:
        return []
    rendered = [(_rendered(block, settings), block) for block in blocks]
    source = "\n\n".join(text for text, _ in rendered)
    spans = _block_spans(rendered)
    if settings.strategy == "fixed_words":
        windows = _word_windows(source, settings)
    else:
        windows = _sentence_windows(
            source,
            settings,
            by_section=settings.strategy == "section_aware",
            blocks=rendered,
        )
    return [
        (source[a:b], *_to_page(a, b, spans), "prose") for a, b in windows if source[a:b].strip()
    ]


def _rendered(block: Block, settings: ChunkingSettings) -> str:
    """A block's text as it will be chunked: flattened when the ablation asks for it."""
    if block.kind == "table" and settings.flatten_tables:
        return " ".join(value for row in block.rows for value in row)
    return block.text


def _block_spans(rendered: Sequence[tuple[str, Block]]) -> list[tuple[int, int, Block, bool]]:
    """Where each block sits in the joined source, and whether its text was left unchanged."""
    spans: list[tuple[int, int, Block, bool]] = []
    position = 0
    for text, block in rendered:
        spans.append((position, position + len(text), block, text == block.text))
        position += len(text) + 2
    return spans


def _to_page(
    start: int, end: int, spans: Sequence[tuple[int, int, Block, bool]]
) -> tuple[int, int]:
    """Translate a span of the joined source back to page character offsets."""
    page_start: int | None = None
    page_end: int | None = None
    for span_start, span_end, block, exact in spans:
        if page_start is None and span_start <= start <= span_end:
            page_start = block.start + (start - span_start) if exact else block.start
        if span_start <= end <= span_end:
            page_end = block.start + (end - span_start) if exact else block.end
    return (
        page_start if page_start is not None else spans[0][2].start,
        page_end if page_end is not None else spans[-1][2].end,
    )


def _word_windows(source: str, settings: ChunkingSettings) -> list[tuple[int, int]]:
    """Fixed-size windows over the words of source, ignoring every boundary."""
    words = [match.span() for match in _WORD.finditer(source)]
    if not words:
        return []
    stride = settings.size - settings.overlap
    windows: list[tuple[int, int]] = []
    for first in range(0, len(words), stride):
        last = min(first + settings.size, len(words)) - 1
        windows.append((words[first][0], words[last][1]))
        if last == len(words) - 1:
            break
    return windows


def _sentence_windows(
    source: str,
    settings: ChunkingSettings,
    *,
    by_section: bool,
    blocks: Sequence[tuple[str, Block]],
) -> list[tuple[int, int]]:
    """Windows of whole sentences, optionally never crossing a section."""
    groups = _section_groups(source, blocks) if by_section else [(0, len(source))]
    windows: list[tuple[int, int]] = []
    for group_start, group_end in groups:
        segment = source[group_start:group_end]
        sentences = [
            (group_start + a, group_start + b)
            for a, b in sentence_spans(segment)
            if segment[a:b].strip()
        ]
        windows.extend(_pack_sentences(source, sentences, settings))
    return windows


def _section_groups(source: str, blocks: Sequence[tuple[str, Block]]) -> list[tuple[int, int]]:
    """Split the joined source into one range per section."""
    if not blocks:
        return []
    groups: list[tuple[int, int]] = []
    position = start = 0
    current = blocks[0][1].section
    for text, block in blocks:
        if block.section != current:
            groups.append((start, position - 2))
            start, current = position, block.section
        position += len(text) + 2
    groups.append((start, len(source)))
    return groups


def _pack_sentences(
    source: str, sentences: Sequence[tuple[int, int]], settings: ChunkingSettings
) -> list[tuple[int, int]]:
    """Pack whole sentences up to the word budget, repeating sentences to make the overlap."""
    if not sentences:
        return []
    lengths = [len(_WORD.findall(source[start:end])) for start, end in sentences]
    windows: list[tuple[int, int]] = []
    index = 0
    while index < len(sentences):
        words = 0
        last = index
        while last < len(sentences) and (words == 0 or words + lengths[last] <= settings.size):
            words += lengths[last]
            last += 1
        last = max(last, index + 1)
        windows.append((sentences[index][0], sentences[last - 1][1]))
        if last >= len(sentences):
            break
        repeated, step = 0, last
        while step > index + 1 and repeated < settings.overlap:
            step -= 1
            repeated += lengths[step]
        index = step
    return windows


def _table_pieces(
    block: Block, settings: ChunkingSettings
) -> list[tuple[str, int, int, ChunkKind]]:
    """Split a table into chunks of whole rows, each repeating the header."""
    lines: list[str] = []
    offsets: list[tuple[int, int]] = []
    position = 0
    for line in block.text.split("\n"):
        lines.append(line)
        offsets.append((position, position + len(line)))
        position += len(line) + 1

    pieces: list[tuple[str, int, int, ChunkKind]] = []
    current: list[int] = []
    words = 0
    for index, line in enumerate(lines):
        line_words = len(_WORD.findall(line))
        if current and words + line_words > settings.size:
            pieces.append(_table_piece(block, lines, offsets, current))
            current, words = [], 0
        current.append(index)
        words += line_words
    if current:
        pieces.append(_table_piece(block, lines, offsets, current))
    return pieces


def _table_piece(
    block: Block,
    lines: Sequence[str],
    offsets: Sequence[tuple[int, int]],
    indices: Sequence[int],
) -> tuple[str, int, int, ChunkKind]:
    body = "\n".join(lines[index] for index in indices)
    start = block.start + offsets[indices[0]][0]
    end = block.start + offsets[indices[-1]][1]
    return (body, start, end, "table")
