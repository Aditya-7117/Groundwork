import json
import logging
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from groundwork import sources
from groundwork.cli import main
from groundwork.sources import SOURCES, CorpusSource

FIXED_NOW = datetime(2026, 9, 19, 3, 4, 5, tzinfo=UTC)

TINY_CONFIG = """
name = "tiny-bm25"
description = "End-to-end run over the fixture corpus."
seed = 3

[corpus]
name = "tiny-beir"
split = "test"

[chunking]
strategy = "fixed_words"
size = 200
overlap = 50

[retrieval]
method = "bm25"
depth = 5
k1 = 0.9
b = 0.4

[evaluation]
cutoffs = [1, 5]
"""


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    # The CLI installs its own root handler; put back whatever pytest had afterwards.
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.fixture
def tiny_source(
    monkeypatch: pytest.MonkeyPatch, tiny_beir_archive: tuple[Path, str]
) -> CorpusSource:
    """Register the fixture corpus and serve its archive locally instead of over the network."""
    archive, sha256 = tiny_beir_archive
    source = CorpusSource(
        name="tiny-beir",
        url="https://example.invalid/tiny-beir.zip",
        sha256=sha256,
        archive_root="tiny-beir",
        licence="Written for this repository's tests.",
        provisional=True,
    )
    monkeypatch.setitem(SOURCES, source.name, source)
    monkeypatch.setattr(
        sources, "download_https", lambda url, destination: shutil.copyfile(archive, destination)
    )
    return source


def _log_events(stderr: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in stderr.splitlines() if line.strip()]


def _run(tmp_path: Path, config_path: Path) -> int:
    return main(
        [
            "run",
            str(config_path),
            "--data-dir",
            str(tmp_path / "data"),
            "--results-dir",
            str(tmp_path / "results"),
        ],
        now=lambda: FIXED_NOW,
    )


class TestFailures:
    def test_missing_config_fails_with_a_logged_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _run(tmp_path, tmp_path / "absent.toml") == 1
        [event] = [e for e in _log_events(capsys.readouterr().err) if e["level"] == "ERROR"]
        assert event["event"] == "run failed"
        assert "config file not found" in str(event["error"])

    def test_unknown_corpus_fails_before_any_download(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text(TINY_CONFIG.replace('name = "tiny-beir"', 'name = "nope"'))
        assert _run(tmp_path, config_path) == 1
        assert "unknown corpus 'nope'" in capsys.readouterr().err
        assert not (tmp_path / "data").exists()


@pytest.mark.usefixtures("tiny_source")
class TestEndToEnd:
    def test_run_writes_an_artefact_with_the_expected_metrics(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text(TINY_CONFIG)

        assert _run(tmp_path, config_path) == 0

        [run_dir] = (tmp_path / "results" / "tiny-bm25").iterdir()
        assert run_dir.name.startswith("20260919T030405Z-")
        document = json.loads((run_dir / "result.json").read_text())
        # The fixture's hand-derived results; see tests/test_experiment.py for the working.
        assert document["metrics"]["aggregate"] == pytest.approx(
            {
                "recall@1": 0.5,
                "recall@5": 0.666667,
                "ndcg@1": 0.666667,
                "ndcg@5": 0.666667,
                "mrr@1": 0.666667,
                "mrr@5": 0.666667,
            },
            abs=1e-6,
        )
        assert document["provisional"] is True
        assert document["corpus"]["documents"] == 6
        assert (run_dir / "run.trec").read_text().splitlines()[0].startswith("q1 Q0 d1 1 ")

        events = {event["event"] for event in _log_events(capsys.readouterr().err)}
        assert {"corpus is provisional", "run complete"} <= events

    def test_second_run_reuses_the_verified_corpus(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_path = tmp_path / "config.toml"
        config_path.write_text(TINY_CONFIG)
        assert _run(tmp_path, config_path) == 0

        def refuse_download(url: str, destination: Path) -> None:
            raise AssertionError("the verified cache should have been used")

        monkeypatch.setattr(sources, "download_https", refuse_download)
        later = datetime(2026, 9, 19, 4, 0, 0, tzinfo=UTC)
        exit_code = main(
            [
                "run",
                str(config_path),
                "--data-dir",
                str(tmp_path / "data"),
                "--results-dir",
                str(tmp_path / "results"),
            ],
            now=lambda: later,
        )
        assert exit_code == 0
        assert len(list((tmp_path / "results" / "tiny-bm25").iterdir())) == 2
