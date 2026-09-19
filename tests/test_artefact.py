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
from groundwork.config import (
    ChunkingConfig,
    CorpusConfig,
    EvaluationConfig,
    ExperimentConfig,
    RetrievalConfig,
    config_digest,
)
from groundwork.experiment import ExperimentResult, QueryResult
from groundwork.ranking import RankedDocument
from groundwork.sources import CorpusSource

CONFIG = ExperimentConfig(
    name="tiny-bm25",
    description="Fixture run.",
    seed=3,
    corpus=CorpusConfig(name="tiny-beir", split="test", query_limit=None),
    chunking=ChunkingConfig(strategy="fixed_words", size=200, overlap=50),
    retrieval=RetrievalConfig(method="bm25", depth=5, k1=0.9, b=0.4),
    evaluation=EvaluationConfig(cutoffs=(1,)),
)
SOURCE = CorpusSource(
    name="tiny-beir",
    url="https://example.invalid/tiny-beir.zip",
    sha256="a" * 64,
    archive_root="tiny-beir",
    licence="Written for the tests.",
    provisional=True,
)
RESULT = ExperimentResult(
    queries=(
        QueryResult(
            query_id="q1",
            ranking=(
                RankedDocument(doc_id="d1", score=2.5, best_chunk_id="d1#0"),
                RankedDocument(doc_id="d2", score=0.125, best_chunk_id="d2#0"),
            ),
            metrics={"recall@1": 1.0, "ndcg@1": 1.0, "rr@1": 1.0},
        ),
    ),
    aggregate={"recall@1": 1.0, "ndcg@1": 1.0, "mrr@1": 1.0},
    excluded_query_ids=("q9",),
    chunk_count=6,
    stage_seconds={"chunking": 0.1, "indexing": 0.2, "retrieval": 0.3, "evaluation": 0.4},
    retrieval_latency_ms={"mean": 1.0, "p50": 1.0, "p95": 1.0, "max": 1.0},
)
STARTED = datetime(2026, 9, 19, 3, 4, 5, tzinfo=UTC)


RECORD = RunRecord(
    config=CONFIG,
    config_path=Path("configs/tiny-bm25.toml"),
    source=SOURCE,
    document_count=6,
    result=RESULT,
    started_at=STARTED,
    finished_at=STARTED + timedelta(seconds=2),
    code=CodeVersion(package_version="0.1.0", git_commit="f" * 40, git_dirty=False),
    environment={"python_version": "3.12.0"},
)


class TestRejects:
    def test_existing_run_directory_is_never_overwritten(self, tmp_path: Path) -> None:
        write_artefact(RECORD, tmp_path)
        with pytest.raises(ArtefactError, match="already exists"):
            write_artefact(RECORD, tmp_path)

    @pytest.mark.parametrize(
        "started_at",
        [
            datetime(2026, 9, 19, 3, 4, 5),  # noqa: DTZ001 -- the naive datetime is under test
            datetime(2026, 9, 19, 3, 4, 5, tzinfo=timezone(timedelta(hours=1))),
        ],
        ids=["naive", "not-utc"],
    )
    def test_timestamps_must_be_utc(self, tmp_path: Path, started_at: datetime) -> None:
        with pytest.raises(ValueError, match="must be timezone-aware UTC"):
            write_artefact(replace(RECORD, started_at=started_at), tmp_path)


class TestWrites:
    def test_directory_is_named_by_start_time_and_config_digest(self, tmp_path: Path) -> None:
        directory = write_artefact(RECORD, tmp_path)
        expected = f"20260919T030405Z-{config_digest(CONFIG)[:12]}"
        assert directory == tmp_path / "tiny-bm25" / expected
        assert sorted(path.name for path in directory.iterdir()) == ["result.json", "run.trec"]

    def test_result_records_what_produced_the_numbers(self, tmp_path: Path) -> None:
        document = json.loads((write_artefact(RECORD, tmp_path) / "result.json").read_text())
        assert document["schema_version"] == SCHEMA_VERSION
        assert document["provisional"] is True
        assert document["experiment"]["config_digest"] == config_digest(CONFIG)
        assert document["experiment"]["config"]["retrieval"] == {
            "method": "bm25",
            "depth": 5,
            "k1": 0.9,
            "b": 0.4,
        }
        assert document["experiment"]["config_path"] == "configs/tiny-bm25.toml"
        assert document["code"] == {
            "groundwork_version": "0.1.0",
            "git_commit": "f" * 40,
            "git_dirty": False,
        }
        assert document["corpus"]["sha256"] == "a" * 64
        assert document["corpus"]["documents"] == 6
        assert document["corpus"]["chunks"] == 6
        assert document["queries"] == {
            "evaluated": 1,
            "selection": "all evaluable queries",
            "excluded_no_relevant_document": ["q9"],
        }
        assert document["timing"]["started_at"] == "2026-09-19T03:04:05+00:00"
        assert document["timing"]["stage_seconds"]["retrieval"] == 0.3

    def test_metrics_are_recorded_in_aggregate_and_per_query(self, tmp_path: Path) -> None:
        document = json.loads((write_artefact(RECORD, tmp_path) / "result.json").read_text())
        assert document["metrics"]["aggregate"] == {"recall@1": 1.0, "ndcg@1": 1.0, "mrr@1": 1.0}
        assert document["metrics"]["per_query"]["q1"]["rr@1"] == 1.0

    def test_run_file_is_in_trec_format(self, tmp_path: Path) -> None:
        # query-id, the literal Q0, document id, rank, score, run tag.
        lines = (write_artefact(RECORD, tmp_path) / "run.trec").read_text().splitlines()
        assert lines == ["q1 Q0 d1 1 2.5 tiny-bm25", "q1 Q0 d2 2 0.125 tiny-bm25"]

    def test_query_sample_is_described_with_its_seed(self, tmp_path: Path) -> None:
        sampled = replace(CONFIG, corpus=replace(CONFIG.corpus, query_limit=1))
        directory = write_artefact(replace(RECORD, config=sampled), tmp_path)
        document = json.loads((directory / "result.json").read_text())
        assert document["queries"]["selection"] == "seeded random sample of 1 (seed 3)"


class TestProvenance:
    def test_code_version_comes_from_this_checkout(self) -> None:
        code = current_code_version()
        assert code.package_version
        # The tests run from a git checkout, both locally and in CI.
        assert code.git_commit is not None
        assert re.fullmatch(r"[0-9a-f]{40}", code.git_commit)
        assert isinstance(code.git_dirty, bool)

    def test_environment_describes_software_and_hardware(self) -> None:
        environment = describe_environment()
        assert {"python_version", "platform", "machine", "cpu_model", "cpu_count"} <= set(
            environment
        )
