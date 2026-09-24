"""Where each corpus comes from, and fetching it into a local cache.

A download is checked against a pinned SHA-256 digest before anything is extracted. If the
upstream file ever changes, the fetch fails instead of quietly changing every number computed
from it.
"""

import hashlib
import logging
import shutil
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from groundwork import __version__

logger = logging.getLogger(__name__)

_MARKER = ".verified-sha256"
_DOWNLOAD_TIMEOUT_SECONDS = 60

type Downloader = Callable[[str, Path], None]


class SourceError(RuntimeError):
    """A corpus could not be fetched, or what was fetched is not what was pinned."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CorpusSource:
    """A pinned, downloadable corpus archive.

    Attributes:
        name: Id that configs refer to.
        url: HTTPS location of the archive.
        sha256: Expected digest of the archive.
        archive_root: Directory inside the archive that holds the corpus files.
        licence: Licence terms, with where they were checked.
        provisional: True for a corpus used only to exercise the pipeline, whose numbers are not
            results of the project.
    """

    name: str
    url: str
    sha256: str
    archive_root: str
    licence: str
    provisional: bool


SOURCES: Mapping[str, CorpusSource] = {
    "beir-scifact": CorpusSource(
        name="beir-scifact",
        url="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        sha256="536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
        archive_root="scifact",
        licence=(
            "Claims CC BY 4.0, abstracts ODC-By 1.0, per LICENSE.md in the allenai/scifact "
            "repository, checked 2026-09-19."
        ),
        provisional=True,
    ),
}


def get_source(name: str) -> CorpusSource:
    """Return the registered source with this name.

    Raises:
        SourceError: If no source has this name.
    """
    try:
        return SOURCES[name]
    except KeyError as error:
        known = ", ".join(sorted(SOURCES))
        raise SourceError(f"unknown corpus {name!r}; known: {known}") from error


def fetch_archive(
    source: CorpusSource, data_dir: Path, *, download: Downloader | None = None
) -> Path:
    """Return the directory holding the corpus files, downloading and verifying them if needed.

    The archive is downloaded and extracted inside a temporary directory next to the cache, and
    moved into place only after its digest matches. An interrupted fetch therefore never leaves
    a directory that looks complete.

    Args:
        source: The corpus to fetch.
        data_dir: Cache directory. The corpus lands in data_dir / source.name.
        download: Function that saves a URL to a path. Defaults to an HTTPS download.

    Raises:
        SourceError: If the download fails or its digest does not match the pinned one.
    """
    target = data_dir / source.name
    corpus_dir = target / source.archive_root
    marker = target / _MARKER
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == source.sha256:
        logger.debug("corpus cache hit", extra={"corpus": source.name, "path": str(corpus_dir)})
        return corpus_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=data_dir, prefix=f".{source.name}-") as scratch:
        archive = Path(scratch) / "archive.zip"
        logger.info("downloading corpus", extra={"corpus": source.name, "url": source.url})
        (download or download_https)(source.url, archive)

        actual = _sha256(archive)
        if actual != source.sha256:
            raise SourceError(
                f"SHA-256 mismatch for {source.name}: expected {source.sha256}, got {actual}. "
                "The upstream file has changed or the download is corrupt."
            )

        staging = Path(scratch) / "extracted"
        with zipfile.ZipFile(archive) as contents:
            contents.extractall(staging)
        (staging / _MARKER).write_text(f"{source.sha256}\n", encoding="utf-8")

        if target.exists():
            logger.warning(
                "replacing unverified corpus directory",
                extra={"corpus": source.name, "path": str(target)},
            )
            shutil.rmtree(target)
        staging.rename(target)

    logger.info("corpus verified", extra={"corpus": source.name, "sha256": source.sha256})
    return corpus_dir


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
