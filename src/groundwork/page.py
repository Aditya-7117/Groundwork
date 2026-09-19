"""Rebuilding a Wikipedia page from Natural Questions' token stream.

Natural Questions gives each page as a list of tokens: words and HTML tags, each carrying the byte
positions it occupied in the original HTML. Working from those tokens, rather than parsing the raw
HTML, keeps the mapping to the annotated answer spans exact, because the annotations are token
positions too.

A page becomes an ordered list of blocks: headings, paragraphs, list items and tables. Tables keep
their rows, so a chunker can serialise them with their column names instead of flattening them into
a line of loose values. Each block records where it sits in the page text, and every token records
which characters it produced, so an annotated answer span can be turned into a character range.
"""

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import Literal

type BlockKind = Literal["heading", "paragraph", "list_item", "table"]

BOILERPLATE_SECTIONS = frozenset(
    {
        "References",
        "References and notes",
        "Notes",
        "Notes and references",
        "Footnotes",
        "Citations",
        "External links",
        "External link",
        "See also",
        "Further reading",
        "Bibliography",
        "Sources",
        "Works cited",
    }
)
"""Sections that hold citations and link lists rather than article content."""

_HEADING_TAGS = {"H1", "H2", "H3", "H4", "H5", "H6"}
_TEXT_BLOCK_TAGS: dict[str, BlockKind] = {
    "P": "paragraph",
    "LI": "list_item",
    "DD": "list_item",
    "DT": "list_item",
}
_CELL_TAGS = {"TD", "TH"}


def tag_name(token: str) -> tuple[str, bool]:
    """Return a tag's upper-case name and whether it closes an element.

    Tags in the source carry attributes, so the token is `<Td colspan="6">` rather than `<Td>`.
    """
    inner = token.removeprefix("<").removesuffix(">").strip()
    closing = inner.startswith("/")
    name = inner.lstrip("/").split(maxsplit=1)[0] if inner.lstrip("/").strip() else ""
    return name.upper(), closing


@dataclass(frozen=True, slots=True)
class TokenStream:
    """Natural Questions' tokens for one page, with the byte positions they occupied.

    Attributes:
        tokens: Words and HTML tags, in document order.
        is_html: Whether each token is a tag.
        start_bytes: Byte offset of each token in the original HTML.
        end_bytes: Byte offset one past each token.
    """

    tokens: Sequence[str]
    is_html: Sequence[bool]
    start_bytes: Sequence[int]
    end_bytes: Sequence[int]

    def __post_init__(self) -> None:
        """Check the four arrays describe the same tokens.

        Raises:
            ValueError: If their lengths differ.
        """
        lengths = {len(self.tokens), len(self.is_html), len(self.start_bytes), len(self.end_bytes)}
        if len(lengths) != 1:
            raise ValueError(
                f"token arrays must have the same length, got lengths {sorted(lengths)}"
            )

    def __len__(self) -> int:
        return len(self.tokens)


@dataclass(frozen=True, slots=True, kw_only=True)
class Block:
    """One element of a page.

    Attributes:
        kind: What the element is.
        section: The nearest preceding section heading, empty for the page's opening.
        text: The block's text as it appears in the page text.
        start: Character offset of the text within the page text.
        end: Character offset one past its end.
        caption: A table's caption, empty otherwise.
        header: A table's column names, empty when it has no header row.
        rows: A table's rows, each a tuple of cell values.
    """

    kind: BlockKind
    section: str
    text: str
    start: int
    end: int
    caption: str = ""
    header: tuple[str, ...] = ()
    rows: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class Page:
    """A page rebuilt from its token stream."""

    page_id: str
    title: str
    url: str
    text: str
    blocks: tuple[Block, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class BuiltPage:
    """A page together with the mapping from its source tokens to characters.

    Attributes:
        page: The rebuilt page.
        token_spans: Per source token, the characters it produced, or None when the token was
            dropped (a tag, or content inside a dropped section).
    """

    page: Page
    token_spans: tuple[tuple[int, int] | None, ...]

    def char_span(self, start_token: int, end_token: int) -> tuple[int, int] | None:
        """Return the character range covered by tokens [start_token, end_token).

        Returns:
            The range, or None when every token in it was dropped.

        Raises:
            ValueError: If the range is not within the page's tokens.
        """
        if not 0 <= start_token < end_token <= len(self.token_spans):
            raise ValueError(
                f"token range [{start_token}, {end_token}) is outside the page's "
                f"{len(self.token_spans)} tokens"
            )
        spans = [span for span in self.token_spans[start_token:end_token] if span is not None]
        if not spans:
            return None
        return min(span[0] for span in spans), max(span[1] for span in spans)


@dataclass(slots=True)
class _Pending:
    """A block being assembled, with the tokens that produced each piece of its text."""

    kind: BlockKind
    section: str
    text: str = ""
    token_ranges: dict[int, tuple[int, int]] = field(default_factory=dict)
    caption: str = ""
    header: tuple[str, ...] = ()
    rows: list[tuple[str, ...]] = field(default_factory=list)
    raw_rows: list[tuple[bool, tuple[str, ...], list[dict[int, tuple[int, int]]]]] = field(
        default_factory=list
    )

    def add_word(self, index: int, word: str, *, glued: bool) -> None:
        """Append a word, separated by a space unless it touched the previous token."""
        if self.text and not glued:
            self.text += " "
        start = len(self.text)
        self.text += word
        self.token_ranges[index] = (start, len(self.text))


def build_page(
    *,
    page_id: str,
    title: str,
    url: str,
    stream: TokenStream,
    drop_sections: Collection[str] = BOILERPLATE_SECTIONS,
) -> BuiltPage:
    """Rebuild a page from its token stream.

    Args:
        page_id: Identifier for the page.
        title: Article title.
        url: Article URL.
        stream: The page's tokens and their byte positions.
        drop_sections: Section headings whose content is left out.
    """
    builder = _PageBuilder({section.casefold() for section in drop_sections})
    for index, token in enumerate(stream.tokens):
        glued = index > 0 and stream.start_bytes[index] == stream.end_bytes[index - 1]
        if stream.is_html[index]:
            builder.tag(token)
        else:
            builder.word(index, token, glued=glued)
    builder.finish()
    return builder.result(page_id=page_id, title=title, url=url, token_count=len(stream))


class _PageBuilder:
    """Walks the token stream and assembles blocks."""

    def __init__(self, dropped_sections: Collection[str]) -> None:
        self._dropped = dropped_sections
        self._section = ""
        self._pending: _Pending | None = None
        self._heading: _Pending | None = None
        self._table: _Pending | None = None
        self._cell: _Pending | None = None
        self._cell_is_header = False
        self._row: list[tuple[str, _Pending]] = []
        self._table_depth = 0
        self._blocks: list[tuple[Block, dict[int, tuple[int, int]]]] = []

    def tag(self, token: str) -> None:
        """Handle one HTML tag token."""
        name, closing = tag_name(token)
        if name in _HEADING_TAGS and not closing:
            self.finish()
            self._heading = _Pending(kind="heading", section=self._section)
        elif name in _HEADING_TAGS and closing:
            self._close_heading(name)
        elif name == "TABLE" and not closing:
            self._table_depth += 1
            if self._table_depth == 1:
                self.finish()
                self._table = _Pending(kind="table", section=self._section)
        elif name == "TABLE" and closing:
            self._table_depth = max(0, self._table_depth - 1)
            if self._table_depth == 0:
                self._close_table()
        elif name in _CELL_TAGS and not closing and self._table is not None:
            self._cell = _Pending(kind="paragraph", section=self._section)
            self._cell_is_header = name == "TH"
        elif name in _CELL_TAGS and closing and self._cell is not None:
            self._row.append(("header" if self._cell_is_header else "data", self._cell))
            self._cell = None
        elif name == "TR" and closing and self._table is not None:
            self._close_row()
        elif name == "CAPTION" and not closing and self._table is not None:
            self._cell = _Pending(kind="paragraph", section=self._section)
            self._cell_is_header = False
        elif name == "CAPTION" and closing and self._table is not None and self._cell is not None:
            self._table.caption = self._cell.text
            self._cell = None
        elif name in _TEXT_BLOCK_TAGS and not closing:
            self.finish()
            self._pending = _Pending(kind=_TEXT_BLOCK_TAGS[name], section=self._section)

    def word(self, index: int, word: str, *, glued: bool) -> None:
        """Handle one word token."""
        target = self._cell or self._heading or self._pending
        if target is None:
            return
        target.add_word(index, word, glued=glued)

    def finish(self) -> None:
        """Close whatever text block is open."""
        if self._pending is not None:
            self._emit(self._pending)
            self._pending = None

    def _close_heading(self, name: str) -> None:
        heading = self._heading
        self._heading = None
        if heading is None:
            return
        # An H1 is the article title, so it ends any section rather than starting one.
        self._section = "" if name == "H1" else heading.text
        heading.section = self._section
        self._emit(heading)

    def _close_row(self) -> None:
        if self._table is None or not self._row:
            return
        all_header = {kind for kind, _ in self._row} == {"header"}
        values = tuple(cell.text for _, cell in self._row)
        maps = [dict(cell.token_ranges) for _, cell in self._row]
        self._table.raw_rows.append((all_header, values, maps))
        self._row = []

    def _close_table(self) -> None:
        table = self._table
        self._table = None
        if table is None:
            return
        self._table, saved = table, None  # _close_row needs the table while flushing the last row
        self._close_row()
        self._table = saved
        if not table.raw_rows:
            return

        # The opening row is the header only when it is all header cells and data rows follow.
        # A table made entirely of header cells is read as data, so nothing is lost.
        first_all_header, first_values, first_maps = table.raw_rows[0]
        if first_all_header and len(table.raw_rows) > 1:
            table.header = first_values
            header_maps = first_maps
            data_rows = table.raw_rows[1:]
        else:
            header_maps = []
            data_rows = table.raw_rows
        table.rows = [values for _, values, _ in data_rows]

        lines: list[str] = []
        token_ranges: dict[int, tuple[int, int]] = {}
        for _, row, maps in data_rows:
            offset = sum(len(line) + 1 for line in lines) if lines else 0
            parts: list[str] = []
            for position, value in enumerate(row):
                label = table.header[position] if position < len(table.header) else ""
                piece = f"{label}: {value}" if label else value
                start = offset + sum(len(p) + 3 for p in parts) + (len(label) + 2 if label else 0)
                if position < len(maps):
                    for token, (local_start, local_end) in maps[position].items():
                        token_ranges[token] = (start + local_start, start + local_end)
                parts.append(piece)
            lines.append(" | ".join(parts))
        table.text = "\n".join(lines)
        # Column names appear once per row, so header tokens point at the table as a whole.
        for mapping in header_maps:
            for token in mapping:
                token_ranges[token] = (0, len(table.text))
        table.token_ranges = token_ranges
        self._emit(table)

    def _emit(self, pending: _Pending) -> None:
        if not pending.text.strip() or pending.section.casefold() in self._dropped:
            return
        start = sum(len(block.text) + 2 for block, _ in self._blocks)
        block = Block(
            kind=pending.kind,
            section=pending.section,
            text=pending.text,
            start=start,
            end=start + len(pending.text),
            caption=pending.caption,
            header=pending.header,
            rows=tuple(pending.rows),
        )
        shifted = {
            token: (start + span[0], start + span[1])
            for token, span in pending.token_ranges.items()
        }
        self._blocks.append((block, shifted))

    def result(self, *, page_id: str, title: str, url: str, token_count: int) -> BuiltPage:
        """Assemble the page and its token mapping.

        The mapping covers every source token, so a span inside a dropped section resolves to
        "no mapping" rather than falling outside the page.
        """
        blocks = tuple(block for block, _ in self._blocks)
        spans: dict[int, tuple[int, int]] = {}
        for _, mapping in self._blocks:
            spans.update(mapping)
        token_spans = tuple(spans.get(index) for index in range(token_count))
        return BuiltPage(
            page=Page(
                page_id=page_id,
                title=title,
                url=url,
                text="\n\n".join(block.text for block in blocks),
                blocks=blocks,
            ),
            token_spans=token_spans,
        )
