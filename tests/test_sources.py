import re
from pathlib import Path

import pytest

from groundwork.corpus import load_beir
from groundwork.sources import (
    SOURCES,
    CorpusSource,
    SourceError,
    download_https,
    fetch_archive,
    get_source,
)


class FakeDownloader:
    """Serves a local archive in place of the network, and counts how often it was asked."""

    def __init__(self, archive: Path) -> None:
        self.archive = archive
        self.calls = 0

    def __call__(self, url: str, destination: Path) -> None:
        self.calls += 1
        destination.write_bytes(self.archive.read_bytes())


def _source(sha256: str) -> CorpusSource:
    return CorpusSource(
        name="tiny-beir",
        url="https://example.invalid/tiny-beir.zip",
        sha256=sha256,
        archive_root="tiny-beir",
        licence="Written for this repository's tests.",
        provisional=True,
    )


class TestRejects:
    def test_unknown_corpus_name_lists_the_known_ones(self) -> None:
        with pytest.raises(SourceError, match="unknown corpus 'nope'; known: beir-scifact"):
            get_source("nope")

    def test_checksum_mismatch_leaves_nothing_behind(
        self, tmp_path: Path, tiny_beir_archive: tuple[Path, str]
    ) -> None:
        data_dir = tmp_path / "data"
        source = _source("0" * 64)
        with pytest.raises(SourceError, match="SHA-256 mismatch"):
            fetch_archive(source, data_dir, download=FakeDownloader(tiny_beir_archive[0]))
        assert not (data_dir / source.name).exists()
        assert list(data_dir.iterdir()) == []

    def test_plain_http_download_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(SourceError, match="only https"):
            download_https("http://example.invalid/x.zip", tmp_path / "x.zip")


class TestFetch:
    def test_fetches_verifies_and_extracts(
        self, tmp_path: Path, tiny_beir_archive: tuple[Path, str]
    ) -> None:
        corpus_dir = fetch_archive(
            _source(tiny_beir_archive[1]),
            tmp_path / "data",
            download=FakeDownloader(tiny_beir_archive[0]),
        )
        assert len(load_beir(corpus_dir, split="test").documents) == 6

    def test_second_fetch_uses_the_verified_cache(
        self, tmp_path: Path, tiny_beir_archive: tuple[Path, str]
    ) -> None:
        downloader = FakeDownloader(tiny_beir_archive[0])
        first = fetch_archive(_source(tiny_beir_archive[1]), tmp_path / "data", download=downloader)
        second = fetch_archive(
            _source(tiny_beir_archive[1]), tmp_path / "data", download=downloader
        )
        assert first == second
        assert downloader.calls == 1

    def test_unverified_leftover_directory_is_replaced(
        self, tmp_path: Path, tiny_beir_archive: tuple[Path, str]
    ) -> None:
        # A directory without the verification marker is what an interrupted extraction leaves.
        leftover = tmp_path / "data" / "tiny-beir" / "tiny-beir"
        leftover.mkdir(parents=True)
        (leftover / "corpus.jsonl").write_text("partial", encoding="utf-8")
        downloader = FakeDownloader(tiny_beir_archive[0])
        corpus_dir = fetch_archive(
            _source(tiny_beir_archive[1]), tmp_path / "data", download=downloader
        )
        assert downloader.calls == 1
        assert len(load_beir(corpus_dir, split="test").documents) == 6


class TestRegistry:
    @pytest.mark.parametrize("source", SOURCES.values(), ids=lambda source: source.name)
    def test_sources_are_pinned_and_served_over_https(self, source: CorpusSource) -> None:
        assert re.fullmatch(r"[0-9a-f]{64}", source.sha256)
        assert source.url.startswith("https://")
        assert source.licence
