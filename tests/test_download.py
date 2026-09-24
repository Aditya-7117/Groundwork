from pathlib import Path

import pytest

from groundwork.download import SourceError, download_https


def test_plain_http_download_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SourceError, match="only https"):
        download_https("http://example.invalid/x.zip", tmp_path / "x.zip")


def test_a_failed_request_is_reported_as_a_source_error(tmp_path: Path) -> None:
    # The .invalid top-level domain is reserved and never resolves.
    with pytest.raises(SourceError, match=r"download of https://example\.invalid/x failed"):
        download_https("https://example.invalid/x", tmp_path / "x")
