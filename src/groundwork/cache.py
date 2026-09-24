"""An append-only cache of model responses, keyed by a digest of the exact request.

Every call to the answer writer and the judge goes through one of these. A rerun reads the cached
response instead of calling the model again, so verdicts can be re-scored for free, published
numbers can be reproduced after a hosted model is retired, and an interrupted run resumes where it
stopped. The key covers the whole request, including the model and its settings, so a changed
prompt or model can never be served an old answer.

The file is JSON Lines, one response per line, appended and flushed as each arrives. A run killed
mid-write can leave a half-written last line; it is cut off with a warning and the response is
simply requested again.
"""

import hashlib
import json
import logging
from collections.abc import Mapping
from pathlib import Path

logger = logging.getLogger(__name__)


class CacheError(ValueError):
    """A cache file is damaged somewhere other than its last line."""


def request_key(request: Mapping[str, object]) -> str:
    """Return a SHA-256 over the request, independent of key order and whitespace."""
    canonical = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ResponseCache:
    """Responses on disk, loaded into memory once."""

    def __init__(self, path: Path) -> None:
        """Open the cache file, creating its directory if needed, and load what it holds."""
        self._path = path
        self._entries: dict[str, dict[str, object]] = {}
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            self._load()

    def __len__(self) -> int:
        """Return how many responses are cached."""
        return len(self._entries)

    def get(self, key: str) -> dict[str, object] | None:
        """Return the cached response for a request key, if there is one."""
        return self._entries.get(key)

    def put(self, key: str, response: Mapping[str, object]) -> None:
        """Store a response and append it to the file straight away."""
        record = {"key": key, "response": dict(response)}
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._entries[key] = dict(response)

    def _load(self) -> None:
        lines = self._path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                if number == len(lines):
                    # Cut the fragment off, or the next response would be appended onto it.
                    logger.warning(
                        "dropping a half-written last line", extra={"path": str(self._path)}
                    )
                    kept = "".join(f"{kept_line}\n" for kept_line in lines[:-1])
                    self._path.write_text(kept, encoding="utf-8")
                    continue
                raise CacheError(f"{self._path}:{number}: not valid JSON: {error}") from error
            self._entries[record["key"]] = record["response"]
