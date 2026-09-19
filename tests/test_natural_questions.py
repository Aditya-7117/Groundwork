"""Building the Natural Questions corpus from the published dataset files.

The fixtures below write small Parquet files in the same shape as the real ones, so the loader is
exercised end to end without the network and without the 1.3 GB download.
"""

import gzip
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from groundwork.natural_questions import NATURAL_QUESTIONS, build_corpus

NO_ANSWER = {
    "start_token": -1,
    "end_token": -1,
    "candidate_index": -1,
    "start_byte": -1,
    "end_byte": -1,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a gzipped JSON Lines file."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def tokens_of(spec: str) -> dict[str, list[Any]]:
    """Turn "<P> Dublin is ~. </P>" into the token columns the dataset carries."""
    token, is_html, start_byte, end_byte = [], [], [], []
    position = 0
    for index, item in enumerate(spec.split()):
        glued = item.startswith("~")
        text = item.removeprefix("~")
        if index and not glued:
            position += 1
        token.append(text)
        is_html.append(text.startswith("<"))
        start_byte.append(position)
        position += len(text.encode("utf-8"))
        end_byte.append(position)
    return {"token": token, "is_html": is_html, "start_byte": start_byte, "end_byte": end_byte}


def example(
    *,
    example_id: str,
    title: str,
    url: str,
    spec: str,
    question: str,
    long_answer: tuple[int, int] | None,
    short_answer: tuple[int, int] | None,
    annotators: int = 5,
) -> dict[str, object]:
    """One dataset row: a question, its page, and five annotators' answers."""
    long_answers: list[dict[str, Any]] = []
    short_answers: list[dict[str, Any]] = []
    for index in range(annotators):
        answered = long_answer is not None and index < 2
        if answered and long_answer is not None:
            long_answers.append(
                {**NO_ANSWER, "start_token": long_answer[0], "end_token": long_answer[1]}
            )
        else:
            long_answers.append(dict(NO_ANSWER))
        if answered and short_answer is not None:
            short_answers.append(
                {
                    "start_token": [short_answer[0]],
                    "end_token": [short_answer[1]],
                    "start_byte": [-1],
                    "end_byte": [-1],
                    "text": ["short"],
                }
            )
        else:
            short_answers.append(
                {"start_token": [], "end_token": [], "start_byte": [], "end_byte": [], "text": []}
            )
    return {
        "id": example_id,
        "document": {"title": title, "url": url, "tokens": tokens_of(spec), "html": ""},
        "question": {"text": question, "tokens": question.split()},
        "annotations": {
            "id": [str(i) for i in range(annotators)],
            "long_answer": long_answers,
            "short_answers": short_answers,
            "yes_no_answer": [-1] * annotators,
        },
    }


def write_dataset(path: Path, rows: list[dict[str, object]]) -> Path:
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


DUBLIN = (
    "<H1> Dublin </H1> <P> Dublin is the capital of Ireland ~. </P> "
    "<H2> History </H2> <P> Vikings founded it in 841 ~. </P>"
)
PARIS = "<H1> Paris </H1> <P> Paris is the capital of France ~. </P>"


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    write_dataset(
        tmp_path / "part-0.parquet",
        [
            example(
                example_id="q1",
                title="Dublin",
                url="https://en.wikipedia.org/wiki/Dublin?oldid=1",
                spec=DUBLIN,
                question="who founded dublin",
                long_answer=(15, 23),
                short_answer=(16, 18),
            ),
            example(
                example_id="q2",
                title="Paris",
                url="https://en.wikipedia.org/wiki/Paris?oldid=1",
                spec=PARIS,
                question="what is the capital of france",
                long_answer=(3, 12),
                short_answer=(4, 5),
            ),
        ],
    )
    return tmp_path


class TestBuild:
    def test_pages_and_questions_are_written(self, dataset: Path, tmp_path: Path) -> None:
        out = tmp_path / "built"
        report = build_corpus([dataset / "part-0.parquet"], out)
        assert report.pages == 2
        assert report.questions == 2
        pages = read_jsonl(out / "pages.jsonl.gz")
        assert {page["title"] for page in pages} == {"Dublin", "Paris"}

    def test_a_question_points_at_the_text_that_answers_it(
        self, dataset: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "built"
        build_corpus([dataset / "part-0.parquet"], out)
        pages = {page["page_id"]: page for page in read_jsonl(out / "pages.jsonl.gz")}
        questions = read_jsonl(out / "questions.jsonl.gz")
        dublin = next(q for q in questions if q["question_id"] == "q1")
        page = pages[dublin["page_id"]]
        assert (
            page["text"][dublin["answer_start"] : dublin["answer_end"]]
            == "Vikings founded it in 841."
        )
        assert dublin["question"] == "who founded dublin"
        assert dublin["answer_type"] == "paragraph"
        assert dublin["short_answers"] == ["Vikings founded"]

    def test_questions_without_an_agreed_answer_are_left_out(self, tmp_path: Path) -> None:
        path = write_dataset(
            tmp_path / "part.parquet",
            [
                example(
                    example_id="none",
                    title="Paris",
                    url="u1",
                    spec=PARIS,
                    question="unanswerable",
                    long_answer=None,
                    short_answer=None,
                )
            ],
        )
        report = build_corpus([path], tmp_path / "built")
        assert report.pages == 1
        assert report.questions == 0
        assert report.without_agreed_answer == 1


class TestRevisions:
    """One revision per article, with answers relocated into it by text match (decision 29)."""

    def test_the_fullest_revision_is_kept_and_answers_relocate(self, tmp_path: Path) -> None:
        short_revision = "<P> Vikings founded it in 841 ~. </P>"
        path = write_dataset(
            tmp_path / "part.parquet",
            [
                example(
                    example_id="short",
                    title="Dublin",
                    url="https://en.wikipedia.org/wiki/Dublin?oldid=1",
                    spec=short_revision,
                    question="who founded dublin",
                    long_answer=(0, 8),
                    short_answer=(1, 3),
                ),
                example(
                    example_id="full",
                    title="Dublin",
                    url="https://en.wikipedia.org/wiki/Dublin?oldid=2",
                    spec=DUBLIN,
                    question="what is the capital of ireland",
                    long_answer=(3, 12),
                    short_answer=(4, 5),
                ),
            ],
        )
        out = tmp_path / "built"
        report = build_corpus([path], out)
        assert report.pages == 1  # the fuller revision only
        assert report.questions == 2  # both questions kept
        assert report.relocated == 1
        pages = read_jsonl(out / "pages.jsonl.gz")
        questions = {q["question_id"]: q for q in read_jsonl(out / "questions.jsonl.gz")}
        text = pages[0]["text"]
        moved = questions["short"]
        assert text[moved["answer_start"] : moved["answer_end"]] == "Vikings founded it in 841."

    def test_an_answer_missing_from_the_kept_revision_drops_its_question(
        self, tmp_path: Path
    ) -> None:
        path = write_dataset(
            tmp_path / "part.parquet",
            [
                example(
                    example_id="gone",
                    title="Dublin",
                    url="https://en.wikipedia.org/wiki/Dublin?oldid=1",
                    spec="<P> A paragraph that later editors deleted ~. </P>",
                    question="deleted fact",
                    long_answer=(0, 9),
                    short_answer=(1, 3),
                ),
                example(
                    example_id="kept",
                    title="Dublin",
                    url="https://en.wikipedia.org/wiki/Dublin?oldid=2",
                    spec=DUBLIN,
                    question="what is the capital of ireland",
                    long_answer=(3, 12),
                    short_answer=(4, 5),
                ),
            ],
        )
        report = build_corpus([path], tmp_path / "built")
        assert report.pages == 1
        assert report.questions == 1
        assert report.not_relocatable == 1


class TestSource:
    def test_every_file_is_pinned(self) -> None:
        assert NATURAL_QUESTIONS.revision
        assert len(NATURAL_QUESTIONS.files) == 7
        for file in NATURAL_QUESTIONS.files:
            assert len(file.sha256) == 64
            assert file.url.startswith("https://huggingface.co/")
            assert NATURAL_QUESTIONS.revision in file.url


def test_the_report_says_how_much_of_the_source_reached_the_pages(
    dataset: Path, tmp_path: Path
) -> None:
    # Content left out must be visible in the record, not discovered later by someone counting.
    report = build_corpus([dataset / "part-0.parquet"], tmp_path / "built")
    assert report.mapped_tokens > 0
    assert report.mapped_tokens <= report.source_tokens
    meta = json.loads((tmp_path / "built" / "meta.json").read_text(encoding="utf-8"))
    assert meta["report"]["mapped_tokens"] == report.mapped_tokens
