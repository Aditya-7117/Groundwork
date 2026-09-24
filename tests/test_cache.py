from pathlib import Path

import pytest

from groundwork.cache import CacheError, ResponseCache, request_key


def test_the_key_ignores_key_order_but_not_values() -> None:
    assert request_key({"a": 1, "b": [1, 2]}) == request_key({"b": [1, 2], "a": 1})
    assert request_key({"a": 1}) != request_key({"a": 2})


def test_responses_survive_reopening(tmp_path: Path) -> None:
    path = tmp_path / "cache" / "responses.jsonl"
    ResponseCache(path).put("k1", {"text": "Limerick"})
    reopened = ResponseCache(path)
    assert reopened.get("k1") == {"text": "Limerick"}
    assert reopened.get("k2") is None
    assert len(reopened) == 1


def test_a_half_written_last_line_is_dropped_and_writing_continues(tmp_path: Path) -> None:
    path = tmp_path / "responses.jsonl"
    ResponseCache(path).put("k1", {"text": "one"})
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"key": "k2", "respo')
    cache = ResponseCache(path)
    assert cache.get("k1") == {"text": "one"}
    assert cache.get("k2") is None
    cache.put("k3", {"text": "three"})
    reopened = ResponseCache(path)
    assert (reopened.get("k1"), reopened.get("k3")) == ({"text": "one"}, {"text": "three"})


def test_damage_before_the_last_line_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "responses.jsonl"
    path.write_text('not json\n{"key": "k1", "response": {}}\n', encoding="utf-8")
    with pytest.raises(CacheError, match=r"responses\.jsonl:1: not valid JSON"):
        ResponseCache(path)
