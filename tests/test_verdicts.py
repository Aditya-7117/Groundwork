"""Stage two's scoring half: every rung per answer, the cost guard, and the summaries."""

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from groundwork.answering import Answer
from groundwork.artefact import CodeVersion, VerdictsRecord, write_verdicts_artefact
from groundwork.cache import ResponseCache
from groundwork.config import StageTwoConfig
from groundwork.judging import GeminiJudge, Verdict
from groundwork.remote import JsonClient, Reply
from groundwork.verdicts import (
    BudgetExceededError,
    CostGuard,
    baseline_agreement,
    judge_all,
    score_answers,
    summarise_setup,
    verdict_cost,
)


def answer(number: int, text: str, *, declined: bool = False) -> Answer:
    return Answer(
        setup="fixed-bm25",
        question_id=f"q{number}",
        question="longest river in ireland",
        answer_type="paragraph",
        chunk_ids=("c1",),
        passages=("River Shannon\nThe Shannon is the longest river in Ireland.",),
        text=text,
        declined=declined,
        prompt_tokens=1,
        output_tokens=1,
        seconds=1.0,
    )


ANSWERS = (
    answer(1, "The Shannon is the longest river in Ireland."),
    answer(2, "The Liffey is the longest river in Ireland."),
    answer(3, "I don't know", declined=True),
)
REFERENCES = {"q1": ("Shannon",), "q2": ("Shannon",), "q3": ("Shannon",)}


class FakeGemini:
    """Labels by looking for the word Shannon, and counts calls by rubric."""

    def __init__(self, *, thinking_tokens: int = 100) -> None:
        self.thinking_tokens = thinking_tokens
        self.calls: list[str] = []

    def __call__(
        self, url: str, body: bytes | None, headers: Mapping[str, str], timeout: float
    ) -> Reply:
        del url, headers, timeout
        assert body is not None
        request = json.loads(body)
        rubric = request["systemInstruction"]["parts"][0]["text"]
        prompt = request["contents"][0]["parts"][0]["text"]
        answer_text = prompt.rsplit("Answer: ", 1)[1]
        right = "Shannon" in answer_text
        if "supported by the passages" in rubric:
            self.calls.append("groundedness")
            label = "supported" if right else "not_supported"
        else:
            self.calls.append("correctness")
            label = "correct" if right else "incorrect"
        reply = {
            "candidates": [
                {"content": {"parts": [{"text": json.dumps({"reason": "r", "label": label})}]}}
            ],
            "usageMetadata": {
                "promptTokenCount": 1_000,
                "candidatesTokenCount": 20,
                "thoughtsTokenCount": self.thinking_tokens,
            },
            "modelVersion": "gemini-3.8-flash",
        }
        return Reply(status=200, body=json.dumps(reply).encode())


def judge(tmp_path: Path, server: FakeGemini) -> GeminiJudge:
    return GeminiJudge(
        cache=ResponseCache(tmp_path / "judge.jsonl"),
        thinking="medium",
        api_key="k",
        client=JsonClient(transport=server),
    )


def verdict(prompt: int, output: int, thinking: int) -> Verdict:
    return Verdict(
        rubric="groundedness",
        label="supported",
        reason="",
        prompt_tokens=prompt,
        output_tokens=output,
        thinking_tokens=thinking,
        seconds=1.0,
        model_version="m",
        cached=False,
    )


def test_cost_bills_thinking_as_output() -> None:
    # 1,000 in at $0.75/M plus (20 + 100) out at $3.75/M = $0.00075 + $0.00045.
    assert verdict_cost(verdict(1_000, 20, 100)) == pytest.approx(0.0012)


class TestCostGuard:
    def test_stays_quiet_before_enough_calls(self) -> None:
        guard = CostGuard(budget=0.01, expected_calls=1_000, check_after=3)
        guard.record(verdict(1_000_000, 0, 0))
        guard.record(verdict(1_000_000, 0, 0))

    def test_stops_when_the_projection_is_far_over_budget(self) -> None:
        # Each call costs $0.0012; 10,000 calls project to $12 against a $5 budget.
        guard = CostGuard(budget=5.0, expected_calls=10_000, check_after=2)
        guard.record(verdict(1_000, 20, 100))
        with pytest.raises(BudgetExceededError, match=r"projected judging cost \$12\.00"):
            guard.record(verdict(1_000, 20, 100))

    def test_allows_up_to_half_again_over(self) -> None:
        # $12 projected against a $10 budget is within the 50% tolerance.
        guard = CostGuard(budget=10.0, expected_calls=10_000, check_after=2)
        guard.record(verdict(1_000, 20, 100))
        guard.record(verdict(1_000, 20, 100))


def test_declined_answers_are_never_judged(tmp_path: Path) -> None:
    server = FakeGemini()
    guard = CostGuard(budget=100.0, expected_calls=4)
    judged = judge_all(ANSWERS, REFERENCES, judge(tmp_path, server), guard, workers=2)
    assert set(judged) == {("fixed-bm25", "q1"), ("fixed-bm25", "q2")}
    assert sorted(server.calls) == ["correctness"] * 2 + ["groundedness"] * 2


def test_a_budget_overrun_stops_the_run(tmp_path: Path) -> None:
    server = FakeGemini(thinking_tokens=10_000)
    guard = CostGuard(budget=0.001, expected_calls=4, check_after=1)
    with pytest.raises(BudgetExceededError):
        judge_all(ANSWERS, REFERENCES, judge(tmp_path, server), guard, workers=1)


def test_every_rung_is_recorded_per_answer(tmp_path: Path) -> None:
    guard = CostGuard(budget=100.0, expected_calls=4)
    judged = judge_all(ANSWERS, REFERENCES, judge(tmp_path, FakeGemini()), guard, workers=2)

    def entailment(passages: Sequence[str], text: str) -> float:
        del passages
        return 0.9 if "Shannon" in text else 0.1

    right, wrong, declined = score_answers(ANSWERS, REFERENCES, judged, entailment)
    assert (right.contains_reference, right.groundedness, right.correctness) == (
        True,
        "supported",
        "correct",
    )
    assert (right.lexical, right.nli) == (1.0, 0.9)
    assert (wrong.contains_reference, wrong.groundedness, wrong.lexical) == (
        False,
        "not_supported",
        0.0,
    )
    assert declined.declined
    assert (declined.groundedness, declined.nli, declined.judge_cost) == (None, None, 0.0)

    summary = summarise_setup((right, wrong, declined))
    assert summary["decline_rate"] == pytest.approx(1 / 3)
    assert summary["correct_containment"] == pytest.approx(1 / 3)
    assert summary["correct_judge"] == pytest.approx(1 / 3)
    assert summary["groundedness_judge"] == {
        "supported": 0.5,
        "partly_supported": 0.0,
        "not_supported": 0.5,
    }
    assert summary["supported_nli"] == 0.5

    agreement = baseline_agreement((right, wrong, declined), seed=1)
    assert agreement["answers"] == 2
    nli = agreement["nli"]
    assert isinstance(nli, dict)
    assert nli["kappa"] == pytest.approx(1.0)
    assert nli["auc"] == pytest.approx(1.0)


def test_the_verdicts_artefact_records_the_judge_price_and_spend(tmp_path: Path) -> None:
    guard = CostGuard(budget=100.0, expected_calls=4)
    judged = judge_all(ANSWERS, REFERENCES, judge(tmp_path, FakeGemini()), guard, workers=1)
    scored = score_answers(ANSWERS, REFERENCES, judged, lambda _passages, _text: 0.9)
    config = StageTwoConfig.model_validate(
        {
            "name": "stage-two",
            "description": "Fixture.",
            "seed": 1,
            "questions": 3,
            "passages": 1,
            "writer": "qwen3.8-27b-iq4xs",
            "setups": ["fixed-bm25"],
            "judge": {"model": "gemini-3.8-flash", "thinking": "medium"},
        }
    )
    now = datetime(2026, 9, 25, 9, 0, tzinfo=UTC)
    directory = write_verdicts_artefact(
        VerdictsRecord(
            config=config,
            answers_dir=Path("results/stage-two/20260925T080000Z-abc"),
            scored=scored,
            budget=20.0,
            spent=guard.spent,
            nli_device="cpu",
            started_at=now,
            finished_at=now,
            code=CodeVersion(package_version="0.1.0", git_commit="b" * 40, git_dirty=False),
            environment={},
        ),
        tmp_path / "results",
    )
    assert directory.parent.name == "stage-two-verdicts"
    document = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    assert document["judge"]["thinking"] == "medium"
    assert document["judge"]["price"]["checked"] == "2026-09-24"
    assert document["judge"]["spent"] == pytest.approx(guard.spent)
    assert document["setups"]["fixed-bm25"]["decline_rate"] == pytest.approx(1 / 3)
    assert len((directory / "scored.jsonl").read_text(encoding="utf-8").splitlines()) == 3
