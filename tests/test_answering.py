"""Stage two's answer half: the question sample, and answers written from a saved ranking."""

import json
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from groundwork.answering import (
    AnsweringError,
    answer_setup,
    latest_run,
    load_answers,
    read_ranking,
    stratified_sample,
    summarise,
)
from groundwork.artefact import (
    CodeVersion,
    RunRecord,
    StageTwoRecord,
    write_artefact,
    write_stage_two_artefact,
)
from groundwork.config import ExperimentConfig, StageTwoConfig
from groundwork.evaluation import EvalQuestion, EvaluationSet
from groundwork.experiment import run_experiment
from groundwork.generation import Generation
from groundwork.page import Block, Page

NOW = datetime(2026, 9, 24, 18, 0, 0, tzinfo=UTC)
CODE = CodeVersion(package_version="0.1.0", git_commit="a" * 40, git_dirty=False)


def question(question_id: str, answer_type: str, text: str = "q") -> EvalQuestion:
    return EvalQuestion(
        question_id=question_id,
        text=text,
        page_relevance={"p": 1},
        spans=(("p", 0, 5),),
        short_answers=("x",),
        answer_type=answer_type,
    )


POOL = tuple(
    [question(f"p{i:03d}", "paragraph") for i in range(70)]
    + [question(f"t{i:03d}", "table") for i in range(20)]
    + [question(f"l{i:03d}", "list") for i in range(10)]
)


class TestStratifiedSample:
    def test_each_type_keeps_its_share(self) -> None:
        sample = stratified_sample(POOL, 10, seed=1)
        assert Counter(q.answer_type for q in sample) == {"paragraph": 7, "table": 2, "list": 1}

    def test_leftover_places_go_to_the_largest_remainders(self) -> None:
        # Exact shares of 11 are 7.7, 2.2 and 1.1: floors 7, 2, 1 leave one place, which goes to
        # paragraph, the largest remainder.
        sample = stratified_sample(POOL, 11, seed=1)
        assert Counter(q.answer_type for q in sample) == {"paragraph": 8, "table": 2, "list": 1}

    def test_the_same_seed_draws_the_same_questions(self) -> None:
        first = [q.question_id for q in stratified_sample(POOL, 20, seed=5)]
        assert first == [q.question_id for q in stratified_sample(POOL, 20, seed=5)]
        assert first != [q.question_id for q in stratified_sample(POOL, 20, seed=6)]
        assert first == sorted(first)

    @pytest.mark.parametrize("size", [0, 101])
    def test_rejects_an_impossible_size(self, size: int) -> None:
        with pytest.raises(ValueError, match="between 1 and 100"):
            stratified_sample(POOL, size, seed=1)


def page(page_id: str, title: str, text: str) -> Page:
    block = Block(kind="paragraph", section="", text=text, start=0, end=len(text))
    return Page(page_id=page_id, title=title, url="u", text=text, blocks=(block,))


RIVER = page(
    "p-river",
    "River Shannon",
    "The Shannon is the longest river in Ireland. It reaches the sea at Limerick.",
)
PLANET = page(
    "p-planet", "Jupiter", "Jupiter is the largest planet. It is a gas giant with a great red spot."
)
QUESTIONS = (
    EvalQuestion(
        question_id="q1",
        text="longest river in ireland",
        page_relevance={"p-river": 1},
        spans=(("p-river", 0, 44),),
        short_answers=("Shannon",),
        answer_type="paragraph",
    ),
    EvalQuestion(
        question_id="q2",
        text="which planet is a gas giant",
        page_relevance={"p-planet": 1},
        spans=(("p-planet", 31, 71),),
        short_answers=("Jupiter",),
        answer_type="table",
    ),
)
EVALUATION_SET = EvaluationSet(name="tiny", pages=(RIVER, PLANET), questions=QUESTIONS, meta={})
CONFIG = ExperimentConfig.model_validate(
    {
        "name": "tiny-bm25",
        "description": "Fixture run.",
        "seed": 1,
        "corpus": {"name": "natural-questions", "split": "validation"},
        "chunking": {"strategy": "fixed_words", "size": 6, "overlap": 1},
        "retrieval": {"method": "bm25", "depth": 5, "bm25": {"k1": 0.9, "b": 0.4, "stem": True}},
        "evaluation": {"cutoffs": [1, 5]},
    }
)


class RecordingWriter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def __call__(self, question_text: str, passages: Sequence[str]) -> Generation:
        self.calls.append((question_text, list(passages)))
        return Generation(
            text="I don't know" if "planet" in question_text else "The Shannon.",
            declined="planet" in question_text,
            prompt_tokens=100,
            output_tokens=5,
            seconds=2.0,
            cached=False,
        )


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    record = RunRecord(
        config=CONFIG,
        config_path=Path("configs/tiny.toml"),
        corpus={"name": "natural-questions"},
        result=run_experiment(CONFIG, EVALUATION_SET),
        started_at=NOW,
        finished_at=NOW,
        code=CODE,
        environment={},
    )
    return write_artefact(record, tmp_path / "results")


class TestAnswerSetup:
    def test_the_writer_reads_the_top_chunks_of_the_saved_ranking(self, run_dir: Path) -> None:
        writer = RecordingWriter()
        answers = answer_setup(run_dir, EVALUATION_SET, QUESTIONS, writer, passages=2)
        ranking = read_ranking(run_dir, 2)
        assert [a.chunk_ids for a in answers] == [ranking["q1"], ranking["q2"]]
        question_text, passages = writer.calls[0]
        assert question_text == "longest river in ireland"
        assert len(passages) == 2
        assert passages[0].startswith("River Shannon\n")
        assert answers[0].passages == tuple(passages)
        assert answers[0].setup == "tiny-bm25"
        assert answers[1].declined

    def test_a_changed_chunker_is_caught_before_any_answer(self, run_dir: Path) -> None:
        longer = EvaluationSet(
            name="tiny",
            pages=(RIVER, PLANET, page("p-extra", "Extra", "one two three four five six seven")),
            questions=QUESTIONS,
            meta={},
        )
        writer = RecordingWriter()
        with pytest.raises(AnsweringError, match="the corpus or the chunker has changed"):
            answer_setup(run_dir, longer, QUESTIONS, writer, passages=2)
        assert writer.calls == []

    def test_the_latest_finished_run_is_chosen(self, run_dir: Path) -> None:
        (run_dir.parent / ".incomplete-x").mkdir()
        (run_dir.parent / "99999999T000000Z-unfinished").mkdir()
        assert latest_run(run_dir.parent.parent, "tiny-bm25") == run_dir

    def test_a_setup_without_a_run_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(AnsweringError, match="no finished retrieval run for 'absent'"):
            latest_run(tmp_path, "absent")


def test_the_stage_two_artefact_records_the_answers_and_what_made_them(
    tmp_path: Path, run_dir: Path
) -> None:
    answers = answer_setup(run_dir, EVALUATION_SET, QUESTIONS, RecordingWriter(), passages=2)
    config = StageTwoConfig.model_validate(
        {
            "name": "stage-two",
            "description": "Fixture.",
            "seed": 1,
            "questions": 2,
            "passages": 2,
            "writer": "qwen3.8-27b-iq4xs",
            "setups": ["tiny-bm25"],
            "judge": {"model": "gpt-6-luna", "thinking": "high"},
        }
    )
    directory = write_stage_two_artefact(
        StageTwoRecord(
            config=config,
            config_path=Path("configs/stage2.toml"),
            runs={"tiny-bm25": run_dir},
            sample={"sampled": 2},
            writer_digest="8a45235b15fb",
            answers=answers,
            started_at=NOW,
            finished_at=NOW,
            code=CODE,
            environment={},
        ),
        tmp_path / "out",
    )
    document = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    assert document["writer"]["digest"] == "8a45235b15fb"
    assert document["writer"]["temperature"] == 0
    assert document["runs"]["tiny-bm25"]["git_commit"] == "a" * 40
    assert document["answers"]["tiny-bm25"]["declined"] == 1
    assert document["answers"]["tiny-bm25"]["decline_rate"] == 0.5
    assert load_answers(directory / "answers.jsonl") == answers


def test_summary_reports_tokens_and_time() -> None:
    summary = summarise([])
    assert summary["answers"] == 0
    assert summary["decline_rate"] == 0.0
