"""Command-line entry point: `groundwork run <config>`.

Paths that depend on the machine, such as where corpora are cached and where results are written,
are command-line options rather than config fields, so the same experiment has the same config
digest everywhere.
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
from groundwork.config import ConfigError, load_config
from groundwork.corpus import CorpusError, load_beir
from groundwork.experiment import run_experiment
from groundwork.logs import configure_logging
from groundwork.sources import SourceError, fetch, get_source

logger = logging.getLogger(__name__)

type Now = Callable[[], datetime]


def main(argv: Sequence[str] | None = None, *, now: Now | None = None) -> int:
    """Parse arguments, run the requested command, and return the process exit code.

    Args:
        argv: Arguments without the program name. Defaults to the process arguments.
        now: Wall clock returning timezone-aware UTC. Defaults to the system clock.

    Returns:
        0 on success, 1 if the run failed for a reason reported in the log.
    """
    arguments = _parser().parse_args(argv)
    configure_logging(arguments.log_level)
    try:
        return _run(arguments.config, arguments.data_dir, arguments.results_dir, now or _utc_now)
    except (ConfigError, SourceError, CorpusError, ArtefactError) as error:
        logger.error("run failed", extra={"error": str(error)})  # noqa: TRY400 -- the message is the diagnosis
        return 1


def _run(config_path: Path, data_dir: Path, results_dir: Path, now: Now) -> int:
    config = load_config(config_path)
    source = get_source(config.corpus.name)
    if source.provisional:
        logger.warning(
            "corpus is provisional",
            extra={"corpus": source.name, "note": "pipeline plumbing, not a project result"},
        )

    started_at = now()
    corpus = load_beir(fetch(source, data_dir), split=config.corpus.split)
    result = run_experiment(config, corpus)
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
            source=source,
            document_count=len(corpus.documents),
            result=result,
            started_at=started_at,
            finished_at=finished_at,
            code=code,
            environment=describe_environment(),
        ),
        results_dir,
    )
    logger.info("run complete", extra={"artefact": str(directory), **result.aggregate})
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="groundwork", description="Evaluate retrieval configurations defined in config files."
    )
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run one experiment and write its results artefact")
    run.add_argument("config", type=Path, help="experiment config file, e.g. configs/x.toml")
    run.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="corpus cache (default: data)"
    )
    run.add_argument(
        "--results-dir", type=Path, default=Path("results"), help="output (default: results)"
    )
    return parser


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)
