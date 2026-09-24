"""The judge: what it is sent, what it accepts back, and its cache."""

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from groundwork.cache import ResponseCache
from groundwork.judging import (
    CORRECTNESS,
    GROUNDEDNESS,
    Judge,
    JudgeError,
    correctness_prompt,
    groundedness_prompt,
)
from groundwork.remote import JsonClient, Reply


def openai_reply(
    label: str, *, status: str = "completed", refusal: bool = False
) -> dict[str, object]:
    content: dict[str, object] = (
        {"type": "refusal", "refusal": "I can't help with that."}
        if refusal
        else {
            "type": "output_text",
            "text": json.dumps({"reason": "Passage 1 says so.", "label": label}),
        }
    )
    return {
        "status": status,
        "model": "gpt-6-luna",
        "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "content": [content]},
        ],
        "usage": {
            "input_tokens": 950,
            "output_tokens": 201,
            "output_tokens_details": {"reasoning_tokens": 180},
        },
    }


class FakeOpenAI:
    def __init__(self, reply: Mapping[str, object]) -> None:
        self.reply = reply
        self.calls: list[tuple[str, dict[str, object], Mapping[str, str]]] = []

    def __call__(
        self, url: str, body: bytes | None, headers: Mapping[str, str], timeout: float
    ) -> Reply:
        del timeout
        assert body is not None
        self.calls.append((url, json.loads(body), headers))
        return Reply(status=200, body=json.dumps(self.reply).encode())


def judge(tmp_path: Path, server: FakeOpenAI) -> Judge:
    return Judge(
        cache=ResponseCache(tmp_path / "judge.jsonl"),
        thinking="high",
        api_key="test-key",
        client=JsonClient(transport=server),
    )


def test_the_groundedness_judge_never_sees_the_reference() -> None:
    prompt = groundedness_prompt("longest river in ireland", ["A > B\nThe Shannon."], "Shannon.")
    assert prompt == (
        "Passages:\n[1] A > B\nThe Shannon.\n\nQuestion: longest river in ireland\n\n"
        "Answer: Shannon."
    )
    assert "Reference" not in prompt


def test_the_correctness_prompt_lists_every_reference() -> None:
    prompt = correctness_prompt("q", ["River Shannon", "the Shannon"], "It is the Shannon.")
    assert "- River Shannon\n- the Shannon" in prompt


def test_the_request_constrains_the_label_and_sends_the_key_in_a_header(tmp_path: Path) -> None:
    server = FakeOpenAI(openai_reply("supported"))
    verdict = judge(tmp_path, server).judge(GROUNDEDNESS, "prompt")
    [(url, body, headers)] = server.calls
    assert url == "https://api.openai.com/v1/responses"
    assert headers["Authorization"] == "Bearer test-key"
    assert body["model"] == "gpt-6-luna"
    assert body["reasoning"] == {"effort": "high"}
    assert body["store"] is False
    text = body["text"]
    assert isinstance(text, dict)
    assert text["format"]["strict"] is True
    assert text["format"]["schema"]["properties"]["label"]["enum"] == list(GROUNDEDNESS.labels)
    assert (verdict.label, verdict.reason) == ("supported", "Passage 1 says so.")
    # Reasoning is counted inside output tokens by the API; the verdict separates the two.
    assert (verdict.prompt_tokens, verdict.output_tokens, verdict.thinking_tokens) == (950, 21, 180)
    assert verdict.model_version == "gpt-6-luna"
    assert not verdict.cached


def test_the_reasoning_item_is_not_read_as_the_answer(tmp_path: Path) -> None:
    server = FakeOpenAI(openai_reply("not_supported"))
    assert judge(tmp_path, server).judge(GROUNDEDNESS, "prompt").label == "not_supported"


def test_an_incomplete_reply_is_rejected(tmp_path: Path) -> None:
    server = FakeOpenAI(openai_reply("supported", status="incomplete"))
    with pytest.raises(JudgeError, match="is incomplete, not completed"):
        judge(tmp_path, server).judge(GROUNDEDNESS, "prompt")


def test_a_refusal_is_rejected(tmp_path: Path) -> None:
    server = FakeOpenAI(openai_reply("supported", refusal=True))
    with pytest.raises(JudgeError, match="the judge refused groundedness"):
        judge(tmp_path, server).judge(GROUNDEDNESS, "prompt")


def test_a_repeated_judgement_comes_from_the_cache(tmp_path: Path) -> None:
    server = FakeOpenAI(openai_reply("correct"))
    judge(tmp_path, server).judge(CORRECTNESS, "prompt")
    again = judge(tmp_path, server).judge(CORRECTNESS, "prompt")
    assert len(server.calls) == 1
    assert again.cached
    assert again.label == "correct"


def test_the_api_key_is_not_part_of_the_cache_key(tmp_path: Path) -> None:
    server = FakeOpenAI(openai_reply("correct"))
    judge(tmp_path, server).judge(CORRECTNESS, "prompt")
    Judge(
        cache=ResponseCache(tmp_path / "judge.jsonl"),
        thinking="high",
        api_key="another-key",
        client=JsonClient(transport=server),
    ).judge(CORRECTNESS, "prompt")
    assert len(server.calls) == 1
    assert "test-key" not in (tmp_path / "judge.jsonl").read_text(encoding="utf-8")


def test_a_label_outside_the_rubric_is_rejected(tmp_path: Path) -> None:
    server = FakeOpenAI(openai_reply("maybe"))
    with pytest.raises(JudgeError, match="'maybe' is not in the groundedness rubric"):
        judge(tmp_path, server).judge(GROUNDEDNESS, "prompt")


def test_an_unreadable_reply_is_rejected(tmp_path: Path) -> None:
    server = FakeOpenAI(
        {
            "status": "completed",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "not json"}]}
            ],
        }
    )
    with pytest.raises(JudgeError, match="unreadable judge reply"):
        judge(tmp_path, server).judge(GROUNDEDNESS, "prompt")


def test_a_missing_key_says_where_to_set_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(JudgeError, match=r"set OPENAI_API_KEY \(see \.env\.example\)"):
        Judge(cache=ResponseCache(tmp_path / "j.jsonl"), thinking="low")
