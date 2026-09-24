"""The evaluation set: the pages retrieval searches, and the questions it is scored on.

Relevance is judged at two levels, and both are reported:

- **Passage level**, the primary one: a chunk counts as relevant when it overlaps the annotated
  answer span. This is what a generator actually receives, so it is the number that matters.
- **Page level**: the page holding the answer. Easier, and reported alongside so the two can be
  compared.

A corpus that carries only page-level judgements, such as a BEIR set, has no answer spans. Every
chunk of a relevant page is then treated as relevant, which is the closest equivalent.
"""

import gzip
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from groundwork.chunking import Chunk, overlapping_chunks
from groundwork.corpus import Corpus
from groundwork.page import Block, BlockKind, Page

logger = logging.getLogger(__name__)


class EvaluationSetError(ValueError):
    """A built corpus is missing or malformed."""


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalQuestion:
    """A question, and where its answer lives.

    Attributes:
        question_id: Identifier used in results and run files.
        text: The question as asked.
        page_relevance: Relevant page ids and their grades.
        spans: Answer spans as (page id, start, end). Empty when the corpus judges pages only.
        short_answers: The reference answers, for the correctness check.
        answer_type: Whether the answer sits in a paragraph, a table or a list.
    """

    question_id: str
    text: str
    page_relevance: Mapping[str, int]
    spans: tuple[tuple[str, int, int], ...]
    short_answers: tuple[str, ...]
    answer_type: str


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationSet:
    """Everything a run is scored against."""

    name: str
    pages: tuple[Page, ...]
    questions: tuple[EvalQuestion, ...]
    meta: Mapping[str, object]


def load_built_corpus(directory: Path) -> EvaluationSet:
    """Load a corpus built by the Natural Questions loader.

    Raises:
        EvaluationSetError: If the directory does not hold a built corpus.
    """
    pages_path = directory / "pages.jsonl.gz"
    questions_path = directory / "questions.jsonl.gz"
    meta_path = directory / "meta.json"
    for path in (pages_path, questions_path, meta_path):
        if not path.is_file():
            raise EvaluationSetError(
                f"{path} not found; build the corpus first with 'groundwork build-corpus'"
            )

    pages = tuple(
        record.to_page() for record in _parse(_PageRecord, _read_jsonl(pages_path), pages_path)
    )
    questions = tuple(
        record.to_question()
        for record in _parse(_QuestionRecord, _read_jsonl(questions_path), questions_path)
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    logger.info(
        "evaluation set loaded",
        extra={"pages": len(pages), "questions": len(questions), "path": str(directory)},
    )
    return EvaluationSet(name=directory.parent.name, pages=pages, questions=questions, meta=meta)


def from_beir(corpus: Corpus, name: str) -> EvaluationSet:
    """Adapt a BEIR-style corpus, which judges whole documents rather than passages.

    Each document becomes a one-block page, and every chunk of a relevant document counts as
    relevant, since there are no answer spans to narrow it down.
    """
    pages = tuple(
        Page(
            page_id=document.doc_id,
            title=document.title,
            url="",
            text=document.text,
            blocks=(
                Block(
                    kind="paragraph",
                    section="",
                    text=document.text,
                    start=0,
                    end=len(document.text),
                ),
            ),
        )
        for document in corpus.documents.values()
    )
    questions = tuple(
        EvalQuestion(
            question_id=query_id,
            text=corpus.queries[query_id].text,
            page_relevance=dict(judgements),
            spans=(),
            short_answers=(),
            answer_type="unknown",
        )
        for query_id, judgements in sorted(corpus.judgements.items())
    )
    return EvaluationSet(name=name, pages=pages, questions=questions, meta={"format": "beir"})


def chunk_relevance(
    question: EvalQuestion, chunks_by_page: Mapping[str, Sequence[Chunk]]
) -> dict[str, int]:
    """Return the relevant chunk ids for one question, with their grades.

    A chunk is relevant when it overlaps an answer span. Without spans, every chunk of a relevant
    page inherits that page's grade.
    """
    relevance: dict[str, int] = {}
    if question.spans:
        for page_id, start, end in question.spans:
            grade = question.page_relevance.get(page_id, 1)
            for chunk_id in overlapping_chunks(chunks_by_page.get(page_id, ()), start, end):
                relevance[chunk_id] = max(relevance.get(chunk_id, 0), grade)
        return relevance
    for page_id, grade in question.page_relevance.items():
        for chunk in chunks_by_page.get(page_id, ()):
            relevance[chunk.chunk_id] = grade
    return relevance


def _read_jsonl(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [line for line in handle if line.strip()]


class _BlockRecord(BaseModel):
    """One block as written by the corpus builder."""

    model_config = ConfigDict(extra="forbid")

    kind: BlockKind
    section: str
    text: str
    start: int
    end: int
    caption: str = ""
    header: tuple[str, ...] = ()
    rows: tuple[tuple[str, ...], ...] = ()

    def to_block(self) -> Block:
        """Convert to the page model."""
        return Block(**self.model_dump())


class _PageRecord(BaseModel):
    """One page as written by the corpus builder."""

    model_config = ConfigDict(extra="forbid")

    page_id: str
    title: str
    url: str
    text: str
    blocks: tuple[_BlockRecord, ...]

    def to_page(self) -> Page:
        """Convert to the page model."""
        return Page(
            page_id=self.page_id,
            title=self.title,
            url=self.url,
            text=self.text,
            blocks=tuple(block.to_block() for block in self.blocks),
        )


class _QuestionRecord(BaseModel):
    """One question as written by the corpus builder."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    question: str
    page_id: str
    answer_start: int
    answer_end: int
    short_answers: tuple[str, ...] = ()
    answer_type: str = "unknown"

    def to_question(self) -> EvalQuestion:
        """Convert to the evaluation model."""
        return EvalQuestion(
            question_id=self.question_id,
            text=self.question,
            page_relevance={self.page_id: 1},
            spans=((self.page_id, self.answer_start, self.answer_end),),
            short_answers=self.short_answers,
            answer_type=self.answer_type,
        )


def _parse[T: BaseModel](model: type[T], lines: Sequence[str], path: Path) -> list[T]:
    """Validate every line, naming the file and line number when one is malformed.

    Raises:
        EvaluationSetError: If a line does not match the model.
    """
    parsed = []
    for number, line in enumerate(lines, start=1):
        try:
            parsed.append(model.model_validate_json(line))
        except ValidationError as error:
            raise EvaluationSetError(f"{path}:{number}: {error}") from error
    return parsed
