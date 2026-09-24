"""The judge: what it is sent, what it accepts back, and its cache."""

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from groundwork.cache import ResponseCache
from groundwork.judging import (
    CORRECTNESS,
    GROUNDEDNESS,
    GeminiJudge,
    JudgeError,
    correctness_prompt,
    groundedness_prompt,
)
from groundwork.remote import JsonClient, Reply


def gemini_reply(label: str, *, thought: bool = False) -> dict[str, object]:
    parts: list[dict[str, object]] = [
        {"text": json.dumps({"reason": "Passage 1 says so.", "label": label})}
    ]
    if thought:
        parts.insert(0, {"text": "thinking about it", "thought": True})
    return {
        "candidates": [{"content": {"parts": parts}, "finishReason": "STOP"}],
        "usageMetadata": {
            "promptTokenCount": 950,
            "candidatesTokenCount": 21,
            "thoughtsTokenCount": 180,
        },
        "modelVersion": "gemini-3.8-flash",
    }


class FakeGemini:
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


def judge(tmp_path: Path, server: FakeGemini) -> GeminiJudge:
    return GeminiJudge(
        cache=ResponseCache(tmp_path / "judge.jsonl"),
        thinking="low",
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
    server = FakeGemini(gemini_reply("supported"))
    verdict = judge(tmp_path, server).judge(GROUNDEDNESS, "prompt")
    [(url, body, headers)] = server.calls
    assert url.endswith("/models/gemini-3.8-flash:generateContent")
    assert "key=" not in url
    assert headers["x-goog-api-key"] == "test-key"
    config = body["generationConfig"]
    assert isinstance(config, dict)
    assert config["thinkingConfig"] == {"thinkingLevel": "low"}
    assert config["responseSchema"]["properties"]["label"]["enum"] == list(GROUNDEDNESS.labels)
    assert (verdict.label, verdict.reason) == ("supported", "Passage 1 says so.")
    assert (verdict.prompt_tokens, verdict.output_tokens, verdict.thinking_tokens) == (950, 21, 180)
    assert not verdict.cached


def test_thinking_text_is_not_read_as_the_answer(tmp_path: Path) -> None:
    server = FakeGemini(gemini_reply("not_supported", thought=True))
    assert judge(tmp_path, server).judge(GROUNDEDNESS, "prompt").label == "not_supported"


def test_a_repeated_judgement_comes_from_the_cache(tmp_path: Path) -> None:
    server = FakeGemini(gemini_reply("correct"))
    judge(tmp_path, server).judge(CORRECTNESS, "prompt")
    again = judge(tmp_path, server).judge(CORRECTNESS, "prompt")
    assert len(server.calls) == 1
    assert again.cached
    assert again.label == "correct"


def test_the_api_key_is_not_part_of_the_cache_key(tmp_path: Path) -> None:
    server = FakeGemini(gemini_reply("correct"))
    judge(tmp_path, server).judge(CORRECTNESS, "prompt")
    GeminiJudge(
        cache=ResponseCache(tmp_path / "judge.jsonl"),
        thinking="low",
        api_key="another-key",
        client=JsonClient(transport=server),
    ).judge(CORRECTNESS, "prompt")
    assert len(server.calls) == 1
    assert "test-key" not in (tmp_path / "judge.jsonl").read_text(encoding="utf-8")


def test_a_label_outside_the_rubric_is_rejected(tmp_path: Path) -> None:
    server = FakeGemini(gemini_reply("maybe"))
    with pytest.raises(JudgeError, match="'maybe' is not in the groundedness rubric"):
        judge(tmp_path, server).judge(GROUNDEDNESS, "prompt")


def test_an_unreadable_reply_is_rejected(tmp_path: Path) -> None:
    server = FakeGemini({"candidates": [{"content": {"parts": [{"text": "not json"}]}}]})
    with pytest.raises(JudgeError, match="unreadable judge reply"):
        judge(tmp_path, server).judge(GROUNDEDNESS, "prompt")


def test_a_missing_key_says_where_to_set_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(JudgeError, match=r"set GEMINI_API_KEY \(see \.env\.example\)"):
        GeminiJudge(cache=ResponseCache(tmp_path / "j.jsonl"), thinking="low")
