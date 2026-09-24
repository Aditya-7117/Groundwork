"""The golden slice and the regression gate, over a tiny built corpus written in the test."""

import gzip
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from fakes import FakeModels
from groundwork.gate import Check, GateError, record_baseline, run_gate
from groundwork.golden import build_golden

PAGES = [
    {
        "page_id": f"p{index}",
        "title": f"Page {index}",
        "url": "u",
        "text": text,
        "blocks": [
            {"kind": "paragraph", "section": "", "text": text, "start": 0, "end": len(text)}
        ],
    }
    for index, text in enumerate(
        [
            "The Shannon is the longest river in Ireland.",
            "Jupiter is the largest planet in the solar system.",
            "Green tea is made from unoxidised leaves.",
            "The Liffey flows through Dublin.",
            "Saturn has rings made of ice.",
            "Black tea is fully oxidised.",
        ]
    )
]
QUESTIONS = [
    {
        "question_id": f"q{index}",
        "question": question,
        "page_id": f"p{index}",
        "answer_start": 0,
        "answer_end": 10,
        "short_answers": ["x"],
        "answer_type": kind,
    }
    for index, (question, kind) in enumerate(
        [
            ("longest river in ireland", "paragraph"),
            ("largest planet", "paragraph"),
            ("how is green tea made", "table"),
        ]
    )
]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


@pytest.fixture
def built(tmp_path: Path) -> Path:
    directory = tmp_path / "built"
    directory.mkdir()
    write_jsonl(directory / "pages.jsonl.gz", PAGES)
    write_jsonl(directory / "questions.jsonl.gz", QUESTIONS)
    (directory / "meta.json").write_text(json.dumps({"source": {"revision": "r"}}), "utf-8")
    return directory


def read_ids(path: Path, field: str) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line)[field] for line in handle]


class TestGolden:
    def test_the_slice_keeps_the_questions_pages_and_some_distractors(
        self, built: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "golden"
        description = build_golden(built, out, questions=2, distractors=2, seed=1)
        questions = read_ids(out / "questions.jsonl.gz", "question_id")
        pages = read_ids(out / "pages.jsonl.gz", "page_id")
        assert len(questions) == 2
        answer_pages = {f"p{question_id[1:]}" for question_id in questions}
        assert answer_pages <= set(pages)
        assert len(pages) == 4
        assert description["distractor_pages"] == 2
        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        assert meta["source"] == {"revision": "r"}
        assert "CC BY-SA" in meta["golden"]["licence"]

    def test_rebuilding_gives_identical_bytes(self, built: Path, tmp_path: Path) -> None:
        build_golden(built, tmp_path / "a", questions=2, distractors=2, seed=1)
        build_golden(built, tmp_path / "b", questions=2, distractors=2, seed=1)
        for name in ("pages.jsonl.gz", "questions.jsonl.gz"):
            assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()

    def test_too_few_pages_for_the_distractors_is_an_error(
        self, built: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="cannot draw 9 distractors"):
            build_golden(built, tmp_path / "g", questions=2, distractors=9, seed=1)


CONFIG = """
name = "gate-bm25"
description = "Gate fixture."
seed = 1

[corpus]
name = "natural-questions"
split = "validation"

[chunking]
strategy = "fixed_words"
size = 20
overlap = 2

[retrieval]
method = "bm25"
depth = 10

[retrieval.bm25]
k1 = 0.9
b = 0.4
stem = true

[evaluation]
cutoffs = [1, 10]
"""


@pytest.fixture
def baseline(built: Path, tmp_path: Path) -> Path:
    config = tmp_path / "gate-bm25.toml"
    config.write_text(CONFIG, encoding="utf-8")
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps(
            {
                "corpus": str(built),
                "setups": {"gate-bm25": {"config": str(config), "tolerance": 0.0}},
            }
        ),
        encoding="utf-8",
    )
    return path


class TestGate:
    def test_recording_then_checking_passes(self, baseline: Path, tmp_path: Path) -> None:
        recorded = record_baseline(baseline, FakeModels(tmp_path / "v"))
        assert set(recorded["gate-bm25"]) == {"passage.recall@10", "passage.ndcg@10"}
        checks = run_gate(baseline, FakeModels(tmp_path / "v"))
        assert len(checks) == 2
        assert all(check.passed for check in checks)

    def test_a_drop_beyond_the_tolerance_fails(self, baseline: Path, tmp_path: Path) -> None:
        record_baseline(baseline, FakeModels(tmp_path / "v"))
        document = json.loads(baseline.read_text(encoding="utf-8"))
        document["setups"]["gate-bm25"]["metrics"]["passage.recall@10"] += 0.05
        baseline.write_text(json.dumps(document), encoding="utf-8")
        failed = [c for c in run_gate(baseline, FakeModels(tmp_path / "v")) if not c.passed]
        assert [check.metric for check in failed] == ["passage.recall@10"]

    def test_a_drop_within_the_tolerance_passes(self) -> None:
        check = Check(setup="s", metric="m", baseline=0.50, measured=0.495, tolerance=0.01)
        assert check.passed

    def test_a_setup_without_metrics_is_an_error(self, baseline: Path, tmp_path: Path) -> None:
        with pytest.raises(GateError, match="needs metrics and a tolerance"):
            run_gate(baseline, FakeModels(tmp_path / "v"))
