"""The corpus data model and a loader for the BEIR file layout.

A BEIR-format corpus directory holds three things:

- corpus.jsonl: one document per line, {"_id", "title", "text"}.
- queries.jsonl: one query per line, {"_id", "text"}.
- qrels/<split>.tsv: relevance judgements ("qrels" in retrieval jargon), one per line as
  query-id, corpus-id, integer grade, under a header row.

Loading is strict. Every line is validated and every error names its file and line, because a
silently skipped row changes the question set that every metric is averaged over.
"""

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from groundwork.metrics import Judgements

_QRELS_HEADER = ["query-id", "corpus-id", "score"]


class CorpusError(ValueError):
    """A corpus file is missing, malformed or inconsistent with the rest of the corpus."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Document:
    """A retrievable document."""

    doc_id: str
    title: str
    text: str

    @property
    def indexable_text(self) -> str:
        """Title and body joined, which is the text that gets chunked and indexed."""
        return f"{self.title}\n\n{self.text}" if self.title else self.text


@dataclass(frozen=True, slots=True, kw_only=True)
class Query:
    """A question with relevance judgements."""

    query_id: str
    text: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Corpus:
    """Documents, the queries judged in one split, and their relevance judgements.

    Attributes:
        documents: Every document, by id.
        queries: Only the queries that have at least one judgement in the loaded split.
        judgements: Query id to that query's relevance grades by document id.
    """

    documents: Mapping[str, Document]
    queries: Mapping[str, Query]
    judgements: Mapping[str, Judgements]


def load_beir(directory: Path, *, split: str) -> Corpus:
    """Load a BEIR-format corpus directory, keeping the queries judged in one split.

    Raises:
        CorpusError: If a file is missing or any line fails validation.
    """
    documents: dict[str, Document] = {}
    for location, record in _read_jsonl(directory / "corpus.jsonl"):
        doc_id = _string_field(record, "_id", location)
        if doc_id in documents:
            raise CorpusError(f"{location}: duplicate id {doc_id}")
        documents[doc_id] = Document(
            doc_id=doc_id,
            title=_string_field(record, "title", location, allow_empty=True),
            text=_string_field(record, "text", location),
        )

    all_queries: dict[str, Query] = {}
    for location, record in _read_jsonl(directory / "queries.jsonl"):
        query_id = _string_field(record, "_id", location)
        if query_id in all_queries:
            raise CorpusError(f"{location}: duplicate id {query_id}")
        all_queries[query_id] = Query(
            query_id=query_id, text=_string_field(record, "text", location)
        )

    judgements = _read_qrels(directory / "qrels" / f"{split}.tsv", all_queries, documents)
    queries = {query_id: all_queries[query_id] for query_id in judgements}
    return Corpus(documents=documents, queries=queries, judgements=judgements)


def _read_jsonl(path: Path) -> Iterator[tuple[str, Mapping[str, object]]]:
    for location, line in _read_lines(path):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise CorpusError(f"{location}: not valid JSON: {error}") from error
        if not isinstance(record, dict):
            raise CorpusError(f"{location}: expected a JSON object")
        yield location, record


def _read_qrels(
    path: Path, queries: Mapping[str, Query], documents: Mapping[str, Document]
) -> dict[str, dict[str, int]]:
    judgements: dict[str, dict[str, int]] = {}
    lines = _read_lines(path)
    header = next(lines, None)
    if header is None or header[1].split("\t") != _QRELS_HEADER:
        raise CorpusError(f"{path}:1: expected header {'\t'.join(_QRELS_HEADER)!r}")
    for location, line in lines:
        fields = line.split("\t")
        if len(fields) != len(_QRELS_HEADER):
            raise CorpusError(f"{location}: expected 3 tab-separated fields, got {len(fields)}")
        query_id, doc_id, raw_grade = fields
        try:
            grade = int(raw_grade)
        except ValueError as error:
            raise CorpusError(f"{location}: grade must be an integer, got {raw_grade!r}") from error
        if query_id not in queries:
            raise CorpusError(f"{location}: unknown query id {query_id}")
        if doc_id not in documents:
            raise CorpusError(f"{location}: unknown document id {doc_id}")
        query_judgements = judgements.setdefault(query_id, {})
        if doc_id in query_judgements:
            raise CorpusError(f"{location}: repeats the judgement for {query_id}, {doc_id}")
        query_judgements[doc_id] = grade
    return judgements


def _read_lines(path: Path) -> Iterator[tuple[str, str]]:
    """Yield (file:line, text) for each non-blank line."""
    try:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                stripped = line.rstrip("\r\n")
                if stripped.strip():
                    yield f"{path}:{number}", stripped
    except FileNotFoundError as error:
        raise CorpusError(f"{path}: file not found") from error


def _string_field(
    record: Mapping[str, object], key: str, location: str, *, allow_empty: bool = False
) -> str:
    value = record.get(key)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise CorpusError(f"{location}: {key} must be {qualifier}")
    return value
