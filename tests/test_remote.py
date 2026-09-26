"""Retries happen only for failures that can pass, and every other failure stops at once."""

import json
import random
from collections.abc import Mapping

import pytest

from groundwork.remote import JsonClient, RemoteError, Reply, _retry_after, urllib_transport


class Scripted:
    """Answers with a fixed sequence of replies and records every request."""

    def __init__(self, *replies: Reply) -> None:
        self.replies = list(replies)
        self.requests: list[tuple[str, bytes | None]] = []

    def __call__(
        self, url: str, body: bytes | None, headers: Mapping[str, str], timeout: float
    ) -> Reply:
        del headers, timeout
        self.requests.append((url, body))
        return self.replies.pop(0)


OK = Reply(status=200, body=b'{"answer": 42}')


def call(transport: Scripted, sleeps: list[float], attempts: int = 6) -> dict[str, object]:
    client = JsonClient(
        transport=transport,
        sleep=sleeps.append,
        rng=random.Random(0),  # noqa: S311 -- seeded jitter, so the waits are predictable
        attempts=attempts,
    )
    return client.request("https://api.example/v1?key=secret", {"q": 1})


def test_a_success_returns_the_decoded_object() -> None:
    transport = Scripted(OK)
    assert call(transport, []) == {"answer": 42}
    assert transport.requests == [
        ("https://api.example/v1?key=secret", json.dumps({"q": 1}).encode())
    ]


@pytest.mark.parametrize("status", [0, 429, 500, 502, 503, 504])
def test_overload_and_connection_failures_are_retried(status: int) -> None:
    sleeps: list[float] = []
    transport = Scripted(Reply(status=status, body=b"busy"), OK)
    assert call(transport, sleeps) == {"answer": 42}
    assert len(transport.requests) == 2
    assert len(sleeps) == 1


def test_the_wait_doubles_with_each_attempt() -> None:
    sleeps: list[float] = []
    busy = Reply(status=503, body=b"busy")
    call(Scripted(busy, busy, busy, OK), sleeps)
    # 2, 4 and 8 seconds, each plus under one second of jitter.
    assert [int(delay) for delay in sleeps] == [2, 4, 8]


def test_the_servers_retry_after_is_honoured() -> None:
    sleeps: list[float] = []
    call(Scripted(Reply(status=429, body=b"", retry_after=30.0), OK), sleeps)
    assert 30 <= sleeps[0] < 31


def test_a_bad_request_is_not_retried() -> None:
    transport = Scripted(Reply(status=400, body=b"thinking level not supported"))
    with pytest.raises(RemoteError, match=r"answered 400.*thinking level not supported"):
        call(transport, [])
    assert len(transport.requests) == 1


def test_gives_up_after_the_last_attempt_without_leaking_the_query_string() -> None:
    busy = Reply(status=503, body=b"busy")
    with pytest.raises(RemoteError, match="answered 503 \\(attempt 3\\)") as raised:
        call(Scripted(busy, busy, busy), [], attempts=3)
    assert "secret" not in str(raised.value)


def test_a_reply_that_is_not_a_json_object_is_rejected() -> None:
    with pytest.raises(RemoteError, match="expected an object"):
        call(Scripted(Reply(status=200, body=b"[1, 2]")), [])
    with pytest.raises(RemoteError, match="invalid JSON"):
        call(Scripted(Reply(status=200, body=b"<html>")), [])


def test_the_real_transport_refuses_plain_http_to_other_hosts() -> None:
    with pytest.raises(RemoteError, match="only https or a local server"):
        urllib_transport("http://example.invalid/x", None, {}, 1.0)


@pytest.mark.parametrize(("header", "seconds"), [("2", 2.0), ("1.524", 1.524), (None, None)])
def test_retry_after_accepts_fractions(header: str | None, seconds: float | None) -> None:
    assert _retry_after(header) == seconds


def test_an_http_date_in_retry_after_is_ignored() -> None:
    assert _retry_after("Wed, 21 Oct 2026 07:28:00 GMT") is None
