"""The answer writer: its prompt, its decline rule, and its use of the cache."""

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from groundwork.cache import ResponseCache
from groundwork.generation import (
    DECLINE,
    GenerationError,
    OllamaWriter,
    build_messages,
    is_declined,
)
from groundwork.remote import JsonClient, Reply

TAGS = {"models": [{"name": "qwen3.8-27b-iq4xs:latest", "digest": "8a45235b15fb"}]}


def chat_reply(content: str) -> dict[str, object]:
    return {
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": 812,
        "eval_count": 14,
        "total_duration": 2_500_000_000,
        "done_reason": "stop",
    }


class FakeOllama:
    """Serves /api/tags and /api/chat, recording each chat request it receives."""

    def __init__(self, answer: str = "The Shannon is the longest river in Ireland.") -> None:
        self.answer = answer
        self.chats: list[dict[str, object]] = []

    def __call__(
        self, url: str, body: bytes | None, headers: Mapping[str, str], timeout: float
    ) -> Reply:
        del headers, timeout
        if url.endswith("/api/tags"):
            return Reply(status=200, body=json.dumps(TAGS).encode())
        assert body is not None
        self.chats.append(json.loads(body))
        return Reply(status=200, body=json.dumps(chat_reply(self.answer)).encode())


def writer(tmp_path: Path, server: FakeOllama) -> OllamaWriter:
    return OllamaWriter(
        model="qwen3.8-27b-iq4xs",
        cache=ResponseCache(tmp_path / "answers.jsonl"),
        seed=1,
        client=JsonClient(transport=server),
    )


def test_passages_are_numbered_after_the_fixed_instruction() -> None:
    system, user = build_messages("longest river in ireland", ["A > B\ntext one", "C\ntext two"])
    assert system["role"] == "system"
    assert system["content"].endswith(DECLINE)
    assert user["content"] == (
        "Passages:\n[1] A > B\ntext one\n\n[2] C\ntext two\n\nQuestion: longest river in ireland"
    )


@pytest.mark.parametrize(
    ("answer", "declined"),
    [
        ("I don't know", True),
        ("I don't know.", True),
        ("i do not know", True),
        ('"I don\N{RIGHT SINGLE QUOTATION MARK}t know."', True),
        ("I don't know who wrote it, but it was published in 1851.", False),
        ("The Shannon.", False),
    ],
)
def test_only_the_decline_sentence_counts_as_declining(*, answer: str, declined: bool) -> None:
    assert is_declined(answer) is declined


def test_the_request_pins_determinism_and_disables_thinking(tmp_path: Path) -> None:
    server = FakeOllama()
    result = writer(tmp_path, server).write("longest river in ireland", ["p1", "p2"])
    [chat] = server.chats
    assert chat["think"] is False
    assert chat["stream"] is False
    assert chat["options"] == {"temperature": 0, "seed": 1, "num_ctx": 16_384}
    assert result.text == "The Shannon is the longest river in Ireland."
    assert (result.prompt_tokens, result.output_tokens, result.seconds) == (812, 14, 2.5)
    assert not result.declined
    assert not result.cached


def test_a_repeated_request_is_served_from_the_cache(tmp_path: Path) -> None:
    server = FakeOllama()
    writer(tmp_path, server).write("q", ["p"])
    again = writer(tmp_path, server).write("q", ["p"])
    assert len(server.chats) == 1
    assert again.cached
    # The cached answer keeps the cost it had when it was generated.
    assert again.seconds == 2.5


def test_different_passages_are_a_different_request(tmp_path: Path) -> None:
    server = FakeOllama()
    answers = writer(tmp_path, server)
    answers.write("q", ["p"])
    answers.write("q", ["other"])
    assert len(server.chats) == 2


def test_a_declined_answer_is_marked(tmp_path: Path) -> None:
    assert writer(tmp_path, FakeOllama(answer=DECLINE)).write("q", ["p"]).declined


def test_a_missing_model_names_the_available_ones(tmp_path: Path) -> None:
    with pytest.raises(GenerationError, match=r"no model 'llama'; it has: qwen3\.8-27b-iq4xs"):
        OllamaWriter(
            model="llama",
            cache=ResponseCache(tmp_path / "a.jsonl"),
            seed=1,
            client=JsonClient(transport=FakeOllama()),
        )


def test_a_fresh_call_bypasses_and_leaves_the_cache(tmp_path: Path) -> None:
    server = FakeOllama()
    answers = writer(tmp_path, server)
    answers.write("q", ["p"])
    timed = answers.write("q", ["p"], fresh=True)
    assert len(server.chats) == 2
    assert not timed.cached
    assert len(ResponseCache(tmp_path / "answers.jsonl")) == 1
