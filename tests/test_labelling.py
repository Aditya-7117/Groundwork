"""Hand labelling: the blind sample, resuming, and refusing labels for changed text."""

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from groundwork.answering import Answer
from groundwork.labelling import (
    LabellingError,
    label_sample,
    load_labels,
    run_labelling,
)


def answer(setup: str, number: int, *, declined: bool = False, text: str = "Shannon.") -> Answer:
    return Answer(
        setup=setup,
        question_id=f"q{number:03d}",
        question=f"question {number}",
        answer_type="paragraph",
        chunk_ids=("c1", "c2"),
        passages=("Title > Section\nfirst passage", "Other\nsecond passage"),
        text=text,
        declined=declined,
        prompt_tokens=1,
        output_tokens=1,
        seconds=1.0,
    )


ANSWERS = tuple(
    answer(setup, number, declined=number % 5 == 0)
    for setup in ("setup-a", "setup-b")
    for number in range(30)
)


class Scripted:
    def __init__(self, *keys: str) -> None:
        self.keys = list(keys)
        self.prompts = 0

    def __call__(self, prompt: str) -> str:
        del prompt
        self.prompts += 1
        return self.keys.pop(0)


class TestSample:
    def test_equal_numbers_per_setup_and_no_declined_answers(self) -> None:
        sample = label_sample(ANSWERS, seed=1, per_setup=10)
        assert Counter(a.setup for a in sample) == {"setup-a": 10, "setup-b": 10}
        assert not any(a.declined for a in sample)

    def test_setups_are_interleaved_so_order_gives_nothing_away(self) -> None:
        setups = [a.setup for a in label_sample(ANSWERS, seed=1, per_setup=10)]
        assert setups != sorted(setups)

    def test_the_same_seed_draws_the_same_sample(self) -> None:
        assert label_sample(ANSWERS, seed=3, per_setup=10) == label_sample(
            ANSWERS, seed=3, per_setup=10
        )

    def test_too_few_answers_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="setup-a has only 24 answers to label, needs 25"):
            label_sample(ANSWERS, seed=1, per_setup=25)


class TestRunLabelling:
    def test_labels_are_saved_one_by_one_and_the_screen_is_blind(self, tmp_path: Path) -> None:
        sample = label_sample(ANSWERS, seed=1, per_setup=2)
        screens: list[str] = []
        path = tmp_path / "labels.jsonl"
        done = run_labelling(
            sample, path, ask=Scripted("1", "2", "x", "3", "1"), show=screens.append
        )
        assert done == 4
        assert [row["label"] for row in load_labels(path).values()] == [
            "supported",
            "partly_supported",
            "not_supported",
            "supported",
        ]
        assert "question" in screens[0]
        assert "setup-" not in "\n".join(screens)

    def test_quitting_keeps_what_was_labelled_and_resuming_skips_it(self, tmp_path: Path) -> None:
        sample = label_sample(ANSWERS, seed=1, per_setup=2)
        path = tmp_path / "labels.jsonl"
        assert run_labelling(sample, path, ask=Scripted("1", "q"), show=lambda _: None) == 1
        resumed = Scripted("3", "3", "3")
        assert run_labelling(sample, path, ask=resumed, show=lambda _: None) == 4
        assert resumed.prompts == 3

    def test_a_label_for_changed_text_is_refused(self, tmp_path: Path) -> None:
        sample = label_sample(ANSWERS, seed=1, per_setup=2)
        path = tmp_path / "labels.jsonl"
        run_labelling(sample, path, ask=Scripted("1", "q"), show=lambda _: None)
        altered = [replace(sample[0], text="Liffey."), *sample[1:]]
        with pytest.raises(LabellingError, match="given for different text"):
            run_labelling(altered, path, ask=Scripted(), show=lambda _: None)
