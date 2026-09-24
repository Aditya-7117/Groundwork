"""Building the evaluation corpus from Google's Natural Questions.

The dataset ships one row per question: the question text, the whole Wikipedia page it was asked
about, and five annotators' answers. An answer has two parts, the paragraph, table or list that
contains it (the "long answer") and the exact words inside it (the "short answer").

Two rules shape the corpus, both decided before building (decisions 28 and 29):

- A question is used only when at least two of the five annotators agreed on both a long and a
  short answer. That leaves the questions with a definite, checkable answer.
- An article can appear more than once, as different revisions. Keeping several revisions would
  let a retriever find the right paragraph in the wrong revision and be scored as a miss, so only
  the fullest revision of each article is kept, and answers from other revisions are relocated
  into it by matching their text. A question whose paragraph no longer exists there is dropped
  and counted, never silently.
"""

import gzip
import hashlib
import json
import logging
from collections.abc import Collection, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow.parquet as pq

from groundwork.download import Downloader, SourceError, download_https
from groundwork.page import BOILERPLATE_SECTIONS, BuiltPage, TokenStream, build_page, tag_name

logger = logging.getLogger(__name__)

type AnswerType = Literal["paragraph", "table", "list"]
# A row as the dataset file hands it over: nested dictionaries and lists. It is turned into typed
# records (_Row, _Answer, Question) before anything else in the codebase sees it.
type DatasetRow = dict[str, Any]

_TABLE_TAGS = {"TABLE", "TR", "TD", "TH"}
_LIST_TAGS = {"UL", "OL", "LI", "DL", "DD", "DT"}
_COLUMNS = ["id", "document", "question", "annotations"]
_BATCH_ROWS = 64
_MINIMUM_AGREEMENT = 2


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceFile:
    """One published dataset file, pinned by digest."""

    url: str
    sha256: str
    size: int

    @property
    def name(self) -> str:
        """The file's name on disk."""
        return self.url.rsplit("/", 1)[-1]


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetSource:
    """A dataset pinned to one published revision."""

    name: str
    dataset: str
    revision: str
    split: str
    licence: str
    files: tuple[SourceFile, ...]


def _hf_file(name: str, sha256: str, size: int, revision: str) -> SourceFile:
    return SourceFile(
        url=(
            "https://huggingface.co/datasets/google-research-datasets/natural_questions/"
            f"resolve/{revision}/default/{name}"
        ),
        sha256=sha256,
        size=size,
    )


_REVISION = "e8103d566bef4154c2c12b17c6095ec5275840cc"

NATURAL_QUESTIONS = DatasetSource(
    name="natural-questions",
    dataset="google-research-datasets/natural_questions",
    revision=_REVISION,
    split="validation",
    licence="CC BY-SA 3.0 (dataset card, checked 2026-09-19)",
    files=tuple(
        _hf_file(name, sha256, size, _REVISION)
        for name, sha256, size in [
            (
                "validation-00000-of-00007.parquet",
                "6566a5b9ac5dbe569f77cd7e565908a4194b538353d5dcd6243d7ea15e8e29fa",
                193231423,
            ),
            (
                "validation-00001-of-00007.parquet",
                "b919537b9fb8bd4a2cfa3dd4e80d330efd699bae29f0b5d82dd52107949c71bb",
                185358345,
            ),
            (
                "validation-00002-of-00007.parquet",
                "5563dcb73c204dc6e35cf74db9c6f66357eb7e546e4f5cbe4bbb6f625574f8db",
                188512970,
            ),
            (
                "validation-00003-of-00007.parquet",
                "6eef9059390bd60620d304d666851e696420b1910f2121d31ebd8f9812d7cdd7",
                190204207,
            ),
            (
                "validation-00004-of-00007.parquet",
                "e103d41d9821dda8edd5d2f1ee709953a1720d4aaeac08d0c3debfc598b429fb",
                195713724,
            ),
            (
                "validation-00005-of-00007.parquet",
                "bcfb9bb030f08a999043f1cc68480e48ae7daccc813a0766a1db1ca00e35ac32",
                190264679,
            ),
            (
                "validation-00006-of-00007.parquet",
                "4c82530fa32d129db257f12b1ffdd9f4db208083b701e4937ec5e0221d47e465",
                195203006,
            ),
        ]
    ),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Question:
    """A question and the characters of its page that answer it."""

    question_id: str
    question: str
    page_id: str
    answer_start: int
    answer_end: int
    short_answers: tuple[str, ...]
    answer_type: AnswerType


@dataclass(frozen=True, slots=True, kw_only=True)
class BuildReport:
    """What a build produced, and what it left out.

    Attributes:
        pages: Pages written.
        questions: Questions written.
        words: Words across the written pages.
        source_tokens: Words (not tags) in the source pages that were kept.
        mapped_tokens: How many of those reached the built pages. The difference is content in
            dropped sections, plus a small remainder in malformed markup.
        without_agreed_answer: Rows where fewer than two annotators agreed on an answer.
        relocated: Questions whose answer was matched into a different revision of the article.
        not_relocatable: Questions dropped because their answer is absent from the kept revision.
        dropped_span: Questions dropped because their answer lay in a section left out.
        question_ids_dropped: The dropped question ids, so nothing disappears silently.
    """

    pages: int
    questions: int
    words: int
    source_tokens: int
    mapped_tokens: int
    without_agreed_answer: int
    relocated: int
    not_relocatable: int
    dropped_span: int
    question_ids_dropped: tuple[str, ...]


def fetch(
    source: DatasetSource, data_dir: Path, *, download: Downloader | None = None
) -> list[Path]:
    """Download the dataset's files, verifying each against its pinned digest.

    Args:
        source: The pinned dataset.
        data_dir: Cache directory; files land in data_dir / source.name.
        download: Function that saves a URL to a path. Defaults to an HTTPS download.

    Raises:
        SourceError: If a downloaded file does not match its digest.
    """
    target = data_dir / source.name
    target.mkdir(parents=True, exist_ok=True)
    paths = []
    for file in source.files:
        path = target / file.name
        if path.is_file() and path.stat().st_size == file.size and _sha256(path) == file.sha256:
            logger.debug("dataset file already present", extra={"file": file.name})
        else:
            logger.info("downloading dataset file", extra={"file": file.name, "url": file.url})
            (download or download_https)(file.url, path)
            actual = _sha256(path)
            if actual != file.sha256:
                path.unlink(missing_ok=True)
                raise SourceError(
                    f"SHA-256 mismatch for {file.name}: expected {file.sha256}, got {actual}"
                )
        paths.append(path)
    return paths


def build_corpus(
    paths: Sequence[Path],
    out_dir: Path,
    *,
    drop_sections: Collection[str] = BOILERPLATE_SECTIONS,
) -> BuildReport:
    """Build the corpus and write it to out_dir as gzipped JSON Lines.

    Writes pages.jsonl.gz, questions.jsonl.gz and meta.json.

    Args:
        paths: The dataset files to read.
        out_dir: Directory to write the built corpus into.
        drop_sections: Section headings whose content is left out of every page.
    """
    rows = list(_scan(paths))
    without_agreed_answer = sum(1 for row in rows if row.question is None)

    # One page per article: the revision with the most words wins.
    best: dict[str, _Row] = {}
    for row in rows:
        current = best.get(row.title)
        if current is None or row.words > current.words:
            best[row.title] = row
    kept_keys = {row.key for row in best.values()}
    multi_revision = {title for title in best if sum(1 for row in rows if row.title == title) > 1}

    out_dir.mkdir(parents=True, exist_ok=True)
    kept_text: dict[str, tuple[str, str]] = {}
    questions: list[Question] = []
    pending: list[tuple[_Row, str, tuple[str, ...], AnswerType]] = []
    pages = words = dropped_span = source_tokens = mapped_tokens = 0

    with gzip.open(out_dir / "pages.jsonl.gz", "wt", encoding="utf-8") as page_file:
        for row, built in _pages(paths, rows, kept_keys | {r.key for r in rows if r.question}):
            keep = row.key in kept_keys
            if keep:
                page_file.write(json.dumps(_page_json(built), ensure_ascii=False) + "\n")
                pages += 1
                words += len(built.page.text.split())
                source_tokens += built.content_tokens
                mapped_tokens += sum(1 for span in built.token_spans if span is not None)
                if row.title in multi_revision:
                    kept_text[row.title] = (built.page.page_id, built.page.text)
            answer = row.question
            if answer is None:
                continue
            span = built.char_span(answer.long_start, answer.long_end)
            if span is None:
                dropped_span += 1
                logger.debug("answer lies in a dropped section", extra={"question": answer.qid})
                continue
            passage = built.page.text[span[0] : span[1]]
            shorts = tuple(
                built.page.text[found[0] : found[1]]
                for found in (
                    built.char_span(start, end)
                    for start, end in zip(answer.short_starts, answer.short_ends, strict=True)
                )
                if found is not None
            )
            if keep:
                questions.append(
                    Question(
                        question_id=answer.qid,
                        question=answer.text,
                        page_id=built.page.page_id,
                        answer_start=span[0],
                        answer_end=span[1],
                        short_answers=shorts,
                        answer_type=answer.answer_type,
                    )
                )
            else:
                pending.append((row, passage, shorts, answer.answer_type))

    relocated, dropped_ids = _relocate(pending, kept_text, questions)
    questions.sort(key=lambda question: question.question_id)
    with gzip.open(out_dir / "questions.jsonl.gz", "wt", encoding="utf-8") as question_file:
        for question in questions:
            question_file.write(json.dumps(asdict(question), ensure_ascii=False) + "\n")

    report = BuildReport(
        pages=pages,
        questions=len(questions),
        words=words,
        source_tokens=source_tokens,
        mapped_tokens=mapped_tokens,
        without_agreed_answer=without_agreed_answer,
        relocated=relocated,
        not_relocatable=len(dropped_ids),
        dropped_span=dropped_span,
        question_ids_dropped=dropped_ids,
    )
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "source": {
                    "dataset": NATURAL_QUESTIONS.dataset,
                    "revision": NATURAL_QUESTIONS.revision,
                    "split": NATURAL_QUESTIONS.split,
                    "licence": NATURAL_QUESTIONS.licence,
                    "files": [{"url": f.url, "sha256": f.sha256} for f in NATURAL_QUESTIONS.files],
                },
                "dropped_sections": sorted(drop_sections),
                "report": asdict(report),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info(
        "corpus built",
        extra={k: v for k, v in asdict(report).items() if k != "question_ids_dropped"},
    )
    return report


def _relocate(
    pending: Sequence[tuple["_Row", str, tuple[str, ...], AnswerType]],
    kept_text: dict[str, tuple[str, str]],
    questions: list[Question],
) -> tuple[int, tuple[str, ...]]:
    """Move answers from other revisions into the kept revision by matching their text."""
    relocated = 0
    dropped: list[str] = []
    for row, passage, shorts, answer_type in pending:
        answer = row.question
        if answer is None:
            continue
        target = kept_text.get(row.title)
        found = target[1].find(passage) if target else -1
        if target is None or found < 0:
            dropped.append(answer.qid)
            continue
        questions.append(
            Question(
                question_id=answer.qid,
                question=answer.text,
                page_id=target[0],
                answer_start=found,
                answer_end=found + len(passage),
                short_answers=shorts,
                answer_type=answer_type,
            )
        )
        relocated += 1
    return relocated, tuple(sorted(dropped))


@dataclass(frozen=True, slots=True, kw_only=True)
class _Answer:
    qid: str
    text: str
    long_start: int
    long_end: int
    short_starts: tuple[int, ...]
    short_ends: tuple[int, ...]
    answer_type: AnswerType


@dataclass(frozen=True, slots=True, kw_only=True)
class _Row:
    key: tuple[str, int]
    title: str
    url: str
    words: int
    question: _Answer | None


def _scan(paths: Sequence[Path]) -> Iterator[_Row]:
    """First pass: one lightweight record per dataset row."""
    for path in paths:
        for index, row in _rows(path):
            tokens = row["document"]["tokens"]
            yield _Row(
                key=(path.name, index),
                title=row["document"]["title"],
                url=row["document"]["url"],
                words=sum(1 for flag in tokens["is_html"] if not flag),
                question=_answer_of(row, tokens["token"]),
            )


def _pages(
    paths: Sequence[Path], rows: Sequence[_Row], needed: Collection[tuple[str, int]]
) -> Iterator[tuple[_Row, BuiltPage]]:
    """Second pass: rebuild the pages that are kept or carry a question."""
    by_key = {row.key: row for row in rows}
    for path in paths:
        for index, raw in _rows(path):
            key = (path.name, index)
            if key not in needed:
                continue
            row = by_key[key]
            tokens = raw["document"]["tokens"]
            yield (
                row,
                build_page(
                    page_id=_page_id(row.url),
                    title=row.title,
                    url=row.url,
                    stream=TokenStream(
                        tokens=tokens["token"],
                        is_html=tokens["is_html"],
                        start_bytes=tokens["start_byte"],
                        end_bytes=tokens["end_byte"],
                    ),
                ),
            )


def _rows(path: Path) -> Iterator[tuple[int, DatasetRow]]:
    index = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=_BATCH_ROWS, columns=_COLUMNS):
        for row in batch.to_pylist():
            yield index, row
            index += 1


def _answer_of(row: DatasetRow, tokens: Sequence[str]) -> _Answer | None:
    """Return the answer two annotators agreed on, or None when there is none."""
    annotations = row["annotations"]
    longs = [a for a in annotations["long_answer"] if a["start_token"] != -1]
    shorts = [a for a in annotations["short_answers"] if a["text"]]
    if len(longs) < _MINIMUM_AGREEMENT or len(shorts) < _MINIMUM_AGREEMENT:
        return None
    paired = [
        (long_answer, short)
        for long_answer, short in zip(
            annotations["long_answer"], annotations["short_answers"], strict=True
        )
        if long_answer["start_token"] != -1 and short["text"]
    ]
    if not paired:
        return None
    long_answer, short = paired[0]
    return _Answer(
        qid=row["id"],
        text=row["question"]["text"],
        long_start=long_answer["start_token"],
        long_end=long_answer["end_token"],
        short_starts=tuple(short["start_token"]),
        short_ends=tuple(short["end_token"]),
        answer_type=_answer_type(tokens[long_answer["start_token"]]),
    )


def _answer_type(token: str) -> AnswerType:
    name, _ = tag_name(token)
    if name in _TABLE_TAGS:
        return "table"
    if name in _LIST_TAGS:
        return "list"
    return "paragraph"


def _page_json(built: BuiltPage) -> dict[str, object]:
    page = built.page
    return {
        "page_id": page.page_id,
        "title": page.title,
        "url": page.url,
        "text": page.text,
        "blocks": [asdict(block) for block in page.blocks],
    }


def _page_id(url: str) -> str:
    return "nq-" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
