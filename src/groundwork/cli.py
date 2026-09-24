"""Command-line entry point.

    groundwork build-corpus   download the dataset and build the evaluation corpus
    groundwork run CONFIG     run one experiment and write its results artefact

Paths that depend on the machine, such as where the corpus is cached and where results are
written, are command-line options rather than config fields, so the same experiment has the same
config digest everywhere.
"""

import argparse
import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from groundwork.artefact import (
    ArtefactError,
    RunRecord,
    current_code_version,
    describe_environment,
    write_artefact,
)
from groundwork.config import ConfigError, ExperimentConfig, load_config
from groundwork.corpus import CorpusError, load_beir
from groundwork.evaluation import EvaluationSet, EvaluationSetError, from_beir, load_built_corpus
from groundwork.experiment import run_experiment
from groundwork.logs import configure_logging
from groundwork.natural_questions import NATURAL_QUESTIONS, build_corpus, fetch
from groundwork.sentences import model_digest
from groundwork.sources import SOURCES, SourceError, fetch_archive, get_source

logger = logging.getLogger(__name__)

type Now = Callable[[], datetime]

_FAILURES = (ConfigError, SourceError, EvaluationSetError, ArtefactError, CorpusError)


def main(argv: Sequence[str] | None = None, *, now: Now | None = None) -> int:
    """Parse arguments, run the requested command, and return the process exit code.

    Returns:
        0 on success, 1 if the run failed for a reason reported in the log.
    """
    arguments = _parser().parse_args(argv)
    configure_logging(arguments.log_level)
    try:
        if arguments.command == "build-corpus":
            return _build(arguments.data_dir)
        return _run(arguments.config, arguments.data_dir, arguments.results_dir, now or _utc_now)
    except _FAILURES as error:
        logger.error("command failed", extra={"error": str(error)})  # noqa: TRY400 -- the message is the diagnosis
        return 1


def _build(data_dir: Path) -> int:
    """Download the dataset if needed and build the evaluation corpus."""
    paths = fetch(NATURAL_QUESTIONS, data_dir)
    out_dir = data_dir / NATURAL_QUESTIONS.name / "built"
    report = build_corpus(paths, out_dir)
    logger.info(
        "corpus ready",
        extra={
            "path": str(out_dir),
            "pages": report.pages,
            "questions": report.questions,
            "words_kept": f"{report.mapped_tokens / report.source_tokens:.1%}",
        },
    )
    return 0


def _run(config_path: Path, data_dir: Path, results_dir: Path, now: Now) -> int:
    config = load_config(config_path)
    started_at = now()
    evaluation_set = _load(config, data_dir)
    result = run_experiment(config, evaluation_set)
    finished_at = now()

    code = current_code_version()
    if code.git_dirty:
        logger.warning(
            "working tree has uncommitted changes",
            extra={"note": "this result cannot be reproduced from its commit alone"},
        )
    directory = write_artefact(
        RunRecord(
            config=config,
            config_path=config_path,
            corpus=_corpus_record(evaluation_set, config),
            result=result,
            started_at=started_at,
            finished_at=finished_at,
            code=code,
            environment=describe_environment(),
        ),
        results_dir,
    )
    headline = {
        name: value
        for name, value in result.aggregate.items()
        if name.startswith("passage.") and name.endswith("@10")
    }
    logger.info("run complete", extra={"artefact": str(directory), **headline})
    return 0


def _load(config: ExperimentConfig, data_dir: Path) -> EvaluationSet:
    """Load the evaluation set the config names.

    Raises:
        EvaluationSetError: If the corpus is unknown or has not been built.
    """
    if config.corpus.name == NATURAL_QUESTIONS.name:
        return load_built_corpus(data_dir / NATURAL_QUESTIONS.name / "built")
    if config.corpus.name in SOURCES:
        source = get_source(config.corpus.name)
        directory = fetch_archive(source, data_dir)
        return from_beir(load_beir(directory, split=config.corpus.split), config.corpus.name)
    known = ", ".join(sorted({NATURAL_QUESTIONS.name, *SOURCES}))
    raise EvaluationSetError(f"unknown corpus {config.corpus.name!r}; known: {known}")


def _corpus_record(evaluation_set: EvaluationSet, config: ExperimentConfig) -> dict[str, object]:
    """Everything about the corpus that a published number has to carry."""
    meta = evaluation_set.meta
    source = meta.get("source", {}) if isinstance(meta, dict) else {}
    return {
        "name": config.corpus.name,
        "source": source,
        "dropped_sections": meta.get("dropped_sections", []) if isinstance(meta, dict) else [],
        "build_report": meta.get("report", {}) if isinstance(meta, dict) else {},
        "pages": len(evaluation_set.pages),
        "questions_available": len(evaluation_set.questions),
        "sentence_model_sha256": model_digest(),
        "provisional": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="groundwork", description="Evaluate retrieval configurations defined in config files."
    )
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="corpus cache (default: data)"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build-corpus", help="download the dataset and build the corpus")
    run = commands.add_parser("run", help="run one experiment and write its results artefact")
    run.add_argument("config", type=Path, help="experiment config file, e.g. configs/x.toml")
    run.add_argument(
        "--results-dir", type=Path, default=Path("results"), help="output (default: results)"
    )
    return parser


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)
