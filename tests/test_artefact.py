import json
import re
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from groundwork.artefact import (
    SCHEMA_VERSION,
    ArtefactError,
    CodeVersion,
    RunRecord,
    current_code_version,
    describe_environment,
    write_artefact,
)
from groundwork.config import ExperimentConfig, config_digest
from groundwork.experiment import ExperimentResult, QuestionResult

CONFIG = ExperimentConfig.model_validate(
    {
        "name": "tiny-bm25",
        "description": "Fixture run.",
        "seed": 3,
        "corpus": {"name": "natural-questions", "split": "validation"},
        "chunking": {"strategy": "fixed_words", "size": 150, "overlap": 30},
        "retrieval": {
            "method": "bm25",
            "depth": 5,
            "bm25": {"k1": 0.9, "b": 0.4, "stem": False},
        },
        "evaluation": {"cutoffs": (1,)},
    }
)
RESULT = ExperimentResult(
    questions=(
        QuestionResult(
            question_id="q1",
            answer_type="table",
            chunk_ranking=("p1#0", "p2#3"),
            page_ranking=("p1", "p2"),
            metrics={"passage.recall@1": 1.0, "page.recall@1": 1.0},
        ),
    ),
    aggregate={"passage.recall@1": 1.0, "page.recall@1": 1.0},
    by_answer_type={"table": {"passage.recall@1": 1.0, "page.recall@1": 1.0}},
    chunk_count=6,
    excluded_question_ids=("q9",),
    stage_seconds={"chunking": 0.1, "indexing": 0.2, "retrieval": 0.3, "evaluation": 0.4},
    retrieval_latency_ms={"mean": 1.0, "p50": 1.0, "p95": 1.0, "max": 1.0},
)
STARTED = datetime(2026, 9, 24, 3, 4, 5, tzinfo=UTC)
RECORD = RunRecord(
    config=CONFIG,
    config_path=Path("configs/tiny-bm25.toml"),
    corpus={"name": "natural-questions", "pages": 2, "source": {"revision": "abc123"}},
    result=RESULT,
    started_at=STARTED,
    finished_at=STARTED + timedelta(seconds=2),
    code=CodeVersion(package_version="0.1.0", git_commit="f" * 40, git_dirty=False),
    environment={"python_version": "3.12.0"},
)


def read(directory: Path) -> dict[str, object]:
    document = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


class TestRejects:
    def test_an_existing_run_directory_is_never_overwritten(self, tmp_path: Path) -> None:
        write_artefact(RECORD, tmp_path)
        with pytest.raises(ArtefactError, match="already exists"):
            write_artefact(RECORD, tmp_path)

    @pytest.mark.parametrize(
        "started_at",
        [
            datetime(2026, 9, 24, 3, 4, 5),  # noqa: DTZ001 -- the naive datetime is under test
            datetime(2026, 9, 24, 3, 4, 5, tzinfo=timezone(timedelta(hours=1))),
        ],
        ids=["naive", "not-utc"],
    )
    def test_timestamps_must_be_utc(self, tmp_path: Path, started_at: datetime) -> None:
        with pytest.raises(ValueError, match="must be timezone-aware UTC"):
            write_artefact(replace(RECORD, started_at=started_at), tmp_path)


class TestWrites:
    def test_the_directory_is_named_by_start_time_and_config_digest(self, tmp_path: Path) -> None:
        directory = write_artefact(RECORD, tmp_path)
        assert (
            directory == tmp_path / "tiny-bm25" / f"20260924T030405Z-{config_digest(CONFIG)[:12]}"
        )
        assert sorted(path.name for path in directory.iterdir()) == [
            "result.json",
            "run.pages.trec",
            "run.passages.trec",
        ]

    def test_the_result_records_what_produced_the_numbers(self, tmp_path: Path) -> None:
        document = read(write_artefact(RECORD, tmp_path))
        assert document["schema_version"] == SCHEMA_VERSION == 3
        assert document["provisional"] is False
        experiment = document["experiment"]
        assert isinstance(experiment, dict)
        assert experiment["config_digest"] == config_digest(CONFIG)
        assert experiment["config"]["retrieval"]["bm25"]["stem"] is False
        assert document["code"] == {
            "groundwork_version": "0.1.0",
            "git_commit": "f" * 40,
            "git_dirty": False,
        }
        assert document["corpus"] == {
            "name": "natural-questions",
            "pages": 2,
            "source": {"revision": "abc123"},
            "split": "validation",
            "chunks": 6,
        }

    def test_questions_excluded_and_counted_by_type_are_listed(self, tmp_path: Path) -> None:
        document = read(write_artefact(RECORD, tmp_path))
        assert document["questions"] == {
            "evaluated": 1,
            "selection": "all evaluable queries",
            "excluded_no_chunk_covers_the_answer": ["q9"],
            "by_answer_type": {"table": 1},
        }

    def test_metrics_are_recorded_overall_by_type_and_per_question(self, tmp_path: Path) -> None:
        metrics = read(write_artefact(RECORD, tmp_path))["metrics"]
        assert isinstance(metrics, dict)
        assert metrics["aggregate"]["passage.recall@1"] == 1.0
        assert metrics["by_answer_type"]["table"]["page.recall@1"] == 1.0
        assert metrics["per_question"]["q1"]["passage.recall@1"] == 1.0

    def test_both_run_files_are_in_trec_format_with_descending_scores(self, tmp_path: Path) -> None:
        # query id, the literal Q0, item id, rank, score, run tag. Scores descend with rank so an
        # independent tool re-sorting by score reproduces exactly the evaluated order.
        directory = write_artefact(RECORD, tmp_path)
        assert (directory / "run.passages.trec").read_text(encoding="utf-8").splitlines() == [
            "q1 Q0 p1#0 1 2 tiny-bm25",
            "q1 Q0 p2#3 2 1 tiny-bm25",
        ]
        assert (directory / "run.pages.trec").read_text(encoding="utf-8").splitlines() == [
            "q1 Q0 p1 1 2 tiny-bm25",
            "q1 Q0 p2 2 1 tiny-bm25",
        ]

    def test_a_question_sample_is_described_with_its_seed(self, tmp_path: Path) -> None:
        sampled = CONFIG.model_copy(
            update={"corpus": CONFIG.corpus.model_copy(update={"query_limit": 1})}
        )
        document = read(write_artefact(replace(RECORD, config=sampled), tmp_path))
        questions = document["questions"]
        assert isinstance(questions, dict)
        assert questions["selection"] == "seeded random sample of 1 (seed 3)"


class TestProvenance:
    def test_the_code_version_comes_from_this_checkout(self) -> None:
        code = current_code_version()
        assert code.package_version
        # The tests run from a git checkout, locally and in CI.
        assert code.git_commit is not None
        assert re.fullmatch(r"[0-9a-f]{40}", code.git_commit)
        assert isinstance(code.git_dirty, bool)

    def test_the_environment_describes_software_and_hardware(self) -> None:
        environment = describe_environment()
        assert {"python_version", "platform", "machine", "cpu_model", "cpu_count"} <= set(
            environment
        )
