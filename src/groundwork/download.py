"""Downloading source files over HTTPS.

Only HTTPS is allowed, and callers check every downloaded file against a pinned SHA-256 digest,
so an upstream change fails the build instead of quietly changing every number computed from it.
"""

import shutil
import urllib.request
from collections.abc import Callable
from pathlib import Path

from groundwork import __version__

_DOWNLOAD_TIMEOUT_SECONDS = 60

type Downloader = Callable[[str, Path], None]


class SourceError(RuntimeError):
    """A file could not be fetched, or what was fetched is not what was pinned."""


def download_https(url: str, destination: Path) -> None:
    """Save the resource at an HTTPS URL to a local file.

    Raises:
        SourceError: If the URL is not HTTPS or the request fails.
    """
    if not url.startswith("https://"):
        raise SourceError(f"only https downloads are allowed, got {url!r}")
    # S310 warns that urllib also opens file: and custom schemes. The check above allows only
    # https, so both suppressions below are covered by it.
    user_agent = {"User-Agent": f"groundwork/{__version__}"}
    request = urllib.request.Request(url, headers=user_agent)  # noqa: S310
    try:
        with (
            urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response,  # noqa: S310
            destination.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
    except OSError as error:
        raise SourceError(f"download of {url} failed: {error}") from error
