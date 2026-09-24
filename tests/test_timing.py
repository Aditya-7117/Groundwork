"""Re-timing: the same sample per setup, real calls, mismatches counted."""

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from groundwork.answering import Answer
from groundwork.artefact import TimingRecord, write_timing_artefact
from groundwork.config import StageTwoConfig
from groundwork.generation import Generation
from groundwork.timing import retime, summarise_timings, timing_sample


def answer(setup: str, number: int) -> Answer:
    return Answer(
        setup=setup,
        question_id=f"q{number:03d}",
        question=f"question {number}",
        answer_type="paragraph",
        chunk_ids=("c",),
        passages=("p",),
        text="The Shannon.",
        declined=False,
        prompt_tokens=1,
        output_tokens=1,
        seconds=99.0,
    )


ANSWERS = tuple(answer(setup, n) for setup in ("a", "b") for n in range(20))


def test_each_setup_gets_the_same_questions() -> None:
    sample = timing_sample(ANSWERS, size=5, seed=1)
    ids = {setup: [a.question_id for a in sample if a.setup == setup] for setup in ("a", "b")}
    assert ids["a"] == ids["b"]
    assert len(ids["a"]) == 5


def test_a_small_setup_is_timed_in_full() -> None:
    assert len(timing_sample(ANSWERS[:3], size=5, seed=1)) == 3


def test_times_are_measured_afresh_and_mismatches_counted() -> None:
    calls: list[str] = []

    def fresh(question: str, passages: Sequence[str]) -> Generation:
        del passages
        calls.append(question)
        changed = question.endswith("3")
        return Generation(
            text="The Liffey." if changed else "The Shannon.",
            declined=False,
            prompt_tokens=100,
            output_tokens=10,
            seconds=2.0 + len(calls),
            cached=False,
        )

    sample = timing_sample(ANSWERS, size=4, seed=1)
    timings = retime(sample, fresh)
    assert len(calls) == 8
    summary = summarise_timings(timings)
    assert set(summary) == {"a", "b"}
    # The stored 99 seconds never appears: every figure comes from the new calls.
    assert summary["a"]["p50_seconds"] < 99
    assert summary["a"]["mean_prompt_tokens"] == pytest.approx(100)
    mismatches = sum(not t.same_text for t in timings)
    assert summary["a"]["text_mismatches"] + summary["b"]["text_mismatches"] == mismatches


def test_the_timing_artefact_records_the_method_and_summary(tmp_path: Path) -> None:
    timings = retime(
        timing_sample(ANSWERS, size=2, seed=1),
        lambda question, passages: Generation(
            text="The Shannon.",
            declined=False,
            prompt_tokens=50,
            output_tokens=5,
            seconds=3.0,
            cached=False,
        ),
    )
    config = StageTwoConfig.model_validate(
        {
            "name": "stage-two",
            "description": "Fixture.",
            "seed": 1,
            "questions": 20,
            "passages": 1,
            "writer": "qwen3.8-27b-iq4xs",
            "setups": ["a", "b"],
            "judge": {"model": "gemini-3.8-flash", "thinking": "medium"},
        }
    )
    now = datetime(2026, 9, 25, 10, 0, tzinfo=UTC)
    directory = write_timing_artefact(
        TimingRecord(
            config=config,
            answers_dir=Path("results/stage-two/x"),
            timings=timings,
            writer_digest="8a45235b15fb",
            started_at=now,
            finished_at=now,
        ),
        tmp_path,
    )
    document = json.loads((directory / "timing.json").read_text(encoding="utf-8"))
    assert directory.parent.name == "stage-two-timing"
    assert "uncached" in document["method"]
    assert document["setups"]["a"]["p50_seconds"] == pytest.approx(3.0)
    assert document["setups"]["b"]["text_mismatches"] == 0
