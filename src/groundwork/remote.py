"""Calling JSON APIs over HTTP, retrying only the failures that are worth retrying.

Two services are called: the local Ollama server that writes answers, and the Gemini API that
judges them. Under load Gemini answers 429 (rate limited) or 503 (overloaded), and both pass, so
those are retried with exponential backoff and a random jitter. Any other failure stops at once:
a 400 means the request itself is wrong, and retrying it would only hide that.
"""

import json
import logging
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_SUCCESS = range(200, 300)
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_MAX_DELAY_SECONDS = 120.0


class RemoteError(RuntimeError):
    """A request failed and was not worth retrying, or kept failing after every retry."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Reply:
    """A raw HTTP reply.

    Attributes:
        status: HTTP status code.
        body: Response body.
        retry_after: Seconds the server asked the client to wait, if it said.
    """

    status: int
    body: bytes
    retry_after: float | None = None


type Transport = Callable[[str, bytes | None, Mapping[str, str], float], Reply]
type Sleep = Callable[[float], None]


def urllib_transport(
    url: str, body: bytes | None, headers: Mapping[str, str], timeout: float
) -> Reply:
    """Send one request with the standard library.

    Raises:
        RemoteError: If the URL is neither https nor a local http address, or the connection
            fails outright.
    """
    if not url.startswith(("https://", "http://localhost", "http://127.0.0.1")):
        raise RemoteError(f"refusing to call {url!r}: only https or a local server is allowed")
    method = "POST" if body is not None else "GET"
    # The scheme check above covers S310, which warns that urllib also opens file: URLs.
    request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)  # noqa: S310
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return Reply(status=response.status, body=response.read())
    except urllib.error.HTTPError as error:
        retry_after = error.headers.get("Retry-After") if error.headers else None
        return Reply(
            status=error.code,
            body=error.read(),
            retry_after=float(retry_after) if retry_after and retry_after.isdigit() else None,
        )
    except (urllib.error.URLError, TimeoutError) as error:
        return Reply(status=0, body=str(error).encode("utf-8"))


class JsonClient:
    """Sends JSON requests through a transport, retrying overloads with backoff."""

    def __init__(
        self,
        *,
        transport: Transport = urllib_transport,
        sleep: Sleep = time.sleep,
        rng: random.Random | None = None,
        attempts: int = 6,
    ) -> None:
        """Set how requests are sent and how patiently failures are retried."""
        self._transport = transport
        self._sleep = sleep
        # Jitter only spreads retries out in time; it needs no secrecy (S311).
        self._rng = rng or random.Random()  # noqa: S311
        self._attempts = attempts

    def request(
        self,
        url: str,
        payload: Mapping[str, object] | None = None,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 300.0,
    ) -> dict[str, object]:
        """Send a request (POST with a payload, GET without) and return the decoded object.

        Status 0 stands for a connection failure or timeout, which is retried like an overload.

        Raises:
            RemoteError: On a non-retryable status, on the last failed attempt, or if the reply
                is not a JSON object.
        """
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        all_headers = {"Content-Type": "application/json", **(headers or {})}
        for attempt in range(1, self._attempts + 1):
            reply = self._transport(url, body, all_headers, timeout)
            if reply.status in _SUCCESS:
                return _decode(url, reply.body)
            retryable = reply.status == 0 or reply.status in _RETRY_STATUSES
            if not retryable or attempt == self._attempts:
                detail = reply.body.decode("utf-8", errors="replace")[:500]
                raise RemoteError(
                    f"{_redact(url)} answered {reply.status} (attempt {attempt}): {detail}"
                )
            delay = reply.retry_after or min(_MAX_DELAY_SECONDS, 2.0**attempt)
            delay += self._rng.uniform(0, 1)
            logger.warning(
                "request failed, retrying",
                extra={
                    "url": _redact(url),
                    "status": reply.status,
                    "attempt": attempt,
                    "delay": delay,
                },
            )
            self._sleep(delay)
        raise RemoteError(f"{_redact(url)}: no attempts were made")


def _decode(url: str, body: bytes) -> dict[str, object]:
    try:
        value = json.loads(body)
    except json.JSONDecodeError as error:
        raise RemoteError(f"{_redact(url)} returned invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise RemoteError(f"{_redact(url)} returned {type(value).__name__}, expected an object")
    return value


def _redact(url: str) -> str:
    """Drop any query string, where some APIs accept keys, before a URL reaches a log."""
    return url.split("?", 1)[0]
