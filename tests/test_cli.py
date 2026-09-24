"""The command line, end to end, over a corpus written in the built-corpus format."""

import gzip
import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from groundwork.cli import main

FIXED_NOW = datetime(2026, 9, 20, 3, 4, 5, tzinfo=UTC)

CONFIG = """
name = "tiny-bm25"
description = "End-to-end run over a tiny built corpus."
seed = 3

[corpus]
name = "natural-questions"
split = "validation"

[chunking]
strategy = "fixed_words"
size = 12
overlap = 3

[retrieval]
method = "bm25"
depth = 5
k1 = 0.9
b = 0.4
stem = true

[evaluation]
cutoffs = [1, 5]
"""

PAGES = [
    {
        "page_id": "p-river",
        "title": "River Shannon",
        "url": "https://example.invalid/shannon",
        "text": "The Shannon is the longest river in Ireland.\n\nIt reaches the sea at Limerick.",
        "blocks": [
            {
                "kind": "paragraph",
                "section": "",
                "text": "The Shannon is the longest river in Ireland.",
                "start": 0,
                "end": 43,
            },
            {
                "kind": "paragraph",
                "section": "Course",
                "text": "It reaches the sea at Limerick.",
                "start": 45,
                "end": 76,
            },
        ],
    },
    {
        "page_id": "p-tea",
        "title": "Green tea",
        "url": "https://example.invalid/tea",
        "text": "Steeping green tea too hot makes it bitter.",
        "blocks": [
            {
                "kind": "paragraph",
                "section": "",
                "text": "Steeping green tea too hot makes it bitter.",
                "start": 0,
                "end": 42,
            }
        ],
    },
]

QUESTIONS = [
    {
        "question_id": "q1",
        "question": "what is the longest river in ireland",
        "page_id": "p-river",
        "answer_start": 0,
        "answer_end": 43,
        "short_answers": ["the Shannon"],
        "answer_type": "paragraph",
    },
    {
        "question_id": "q2",
        "question": "why does green tea taste bitter",
        "page_id": "p-tea",
        "answer_start": 0,
        "answer_end": 42,
        "short_answers": ["steeping too hot"],
        "answer_type": "paragraph",
    },
]


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    # The CLI installs its own root handler; put back whatever pytest had afterwards.
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    built = tmp_path / "data" / "natural-questions" / "built"
    built.mkdir(parents=True)
    for name, records in (("pages", PAGES), ("questions", QUESTIONS)):
        with gzip.open(built / f"{name}.jsonl.gz", "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
    (built / "meta.json").write_text(
        json.dumps(
            {
                "source": {"dataset": "example/natural_questions", "revision": "abc123"},
                "dropped_sections": ["References"],
                "report": {"pages": 2, "questions": 2},
            }
        ),
        encoding="utf-8",
    )
    return tmp_path / "data"


def run(tmp_path: Path, data_dir: Path, config_text: str = CONFIG) -> tuple[int, Path]:
    config_path = tmp_path / "config.toml"
    config_path.write_text(config_text, encoding="utf-8")
    code = main(
        [
            "--data-dir",
            str(data_dir),
            "run",
            str(config_path),
            "--results-dir",
            str(tmp_path / "results"),
        ],
        now=lambda: FIXED_NOW,
    )
    return code, tmp_path / "results"


class TestFailures:
    def test_missing_config_is_reported_not_raised(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = main(["run", str(tmp_path / "absent.toml")], now=lambda: FIXED_NOW)
        assert code == 1
        assert "config file not found" in capsys.readouterr().err

    def test_unknown_corpus_names_the_known_ones(
        self, tmp_path: Path, data_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, _ = run(
            tmp_path, data_dir, CONFIG.replace('name = "natural-questions"', 'name = "nope"')
        )
        assert code == 1
        assert "unknown corpus 'nope'" in capsys.readouterr().err

    def test_a_corpus_that_was_never_built_says_so(self, tmp_path: Path) -> None:
        code, _ = run(tmp_path, tmp_path / "empty")
        assert code == 1


class TestEndToEnd:
    def test_a_run_writes_an_artefact_with_both_levels_of_metrics(
        self, tmp_path: Path, data_dir: Path
    ) -> None:
        code, results_dir = run(tmp_path, data_dir)
        assert code == 0

        [run_dir] = (results_dir / "tiny-bm25").iterdir()
        assert run_dir.name.startswith("20260920T030405Z-")
        assert sorted(path.name for path in run_dir.iterdir()) == [
            "result.json",
            "run.pages.trec",
            "run.passages.trec",
        ]
        document = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        # Each question's own page shares its distinctive words, so both are found first.
        assert document["metrics"]["aggregate"]["page.recall@1"] == pytest.approx(1.0)
        assert document["metrics"]["aggregate"]["passage.recall@5"] == pytest.approx(1.0)
        assert document["questions"]["evaluated"] == 2
        assert document["corpus"]["source"]["revision"] == "abc123"
        assert document["corpus"]["pages"] == 2
        assert document["experiment"]["config"]["retrieval"]["stem"] is True

    def test_the_run_files_hold_one_line_per_retrieved_item(
        self, tmp_path: Path, data_dir: Path
    ) -> None:
        _, results_dir = run(tmp_path, data_dir)
        [run_dir] = (results_dir / "tiny-bm25").iterdir()
        lines = (run_dir / "run.pages.trec").read_text(encoding="utf-8").splitlines()
        assert lines[0].split()[:3] == ["q1", "Q0", "p-river"]
        assert all(len(line.split()) == 6 for line in lines)

    def test_two_runs_of_the_same_config_do_not_overwrite_each_other(
        self, tmp_path: Path, data_dir: Path
    ) -> None:
        run(tmp_path, data_dir)
        later = datetime(2026, 9, 20, 4, 0, 0, tzinfo=UTC)
        config_path = tmp_path / "config.toml"
        code = main(
            [
                "--data-dir",
                str(data_dir),
                "run",
                str(config_path),
                "--results-dir",
                str(tmp_path / "results"),
            ],
            now=lambda: later,
        )
        assert code == 0
        assert len(list((tmp_path / "results" / "tiny-bm25").iterdir())) == 2
