"""Writing a run's results as a versioned artefact that records everything that produced them.

Each run gets its own directory, results/<experiment>/<UTC start>-<config digest>/, which is
never overwritten. It holds:

- result.json: the full config and its digest, the code version (git commit and whether the
  working tree had uncommitted changes), the corpus and its checksum, the hardware, the timings,
  and every metric, both averaged and per query.
- run.trec: the rankings in the standard TREC run format, so the numbers can be recomputed with
  an independent tool such as trec_eval.
"""

import json
import logging
import os
import platform
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from groundwork import __version__
from groundwork.config import ExperimentConfig, config_digest
from groundwork.experiment import ExperimentResult
from groundwork.sources import CorpusSource

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
"""Version of the result.json layout. Increment it whenever a field changes meaning or moves."""


class ArtefactError(RuntimeError):
    """A results artefact could not be written."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CodeVersion:
    """The code that produced a result.

    Attributes:
        package_version: The installed groundwork version.
        git_commit: The commit checked out, or None when not running from a git checkout.
        git_dirty: Whether the working tree had uncommitted changes, or None when unknown. A
            result from a dirty tree cannot be reproduced from its commit alone.
    """

    package_version: str
    git_commit: str | None
    git_dirty: bool | None


@dataclass(frozen=True, slots=True, kw_only=True)
class RunRecord:
    """Everything that goes into one artefact."""

    config: ExperimentConfig
    config_path: Path
    source: CorpusSource
    document_count: int
    result: ExperimentResult
    started_at: datetime
    finished_at: datetime
    code: CodeVersion
    environment: Mapping[str, object]


def write_artefact(record: RunRecord, results_dir: Path) -> Path:
    """Write the artefact for one run and return its directory.

    Files are written to a hidden temporary directory and renamed into place, so a crash never
    leaves a directory that looks complete.

    Raises:
        ValueError: If a timestamp is not timezone-aware UTC.
        ArtefactError: If the run directory already exists.
    """
    for label, moment in (("started_at", record.started_at), ("finished_at", record.finished_at)):
        if moment.utcoffset() != timedelta(0):
            raise ValueError(f"{label} must be timezone-aware UTC, got {moment.isoformat()}")

    digest = config_digest(record.config)
    parent = results_dir / record.config.name
    directory = parent / f"{record.started_at:%Y%m%dT%H%M%SZ}-{digest[:12]}"
    if directory.exists():
        raise ArtefactError(f"run directory already exists, refusing to overwrite: {directory}")

    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=parent, prefix=".incomplete-"))
    (staging / "run.trec").write_text(_trec_run(record), encoding="utf-8")
    (staging / "result.json").write_text(
        json.dumps(_result_document(record, digest), indent=2) + "\n", encoding="utf-8"
    )
    staging.rename(directory)
    return directory


def current_code_version() -> CodeVersion:
    """Describe the code that is running, including its git commit when there is one."""
    git = shutil.which("git")
    checkout = Path(__file__).resolve().parent
    if git is None:
        logger.warning("git not found; the result will not record a commit")
        return CodeVersion(package_version=__version__, git_commit=None, git_dirty=None)
    commit = _run([git, "-C", str(checkout), "rev-parse", "HEAD"])
    if commit is None:
        logger.warning("not running from a git checkout; the result will not record a commit")
        return CodeVersion(package_version=__version__, git_commit=None, git_dirty=None)
    status = _run([git, "-C", str(checkout), "status", "--porcelain"])
    return CodeVersion(
        package_version=__version__,
        git_commit=commit,
        git_dirty=None if status is None else bool(status),
    )


def describe_environment() -> dict[str, object]:
    """Describe the software and hardware a run executed on.

    CPU model and memory are read from the operating system, and are None where it does not
    report them.
    """
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "memory_bytes": _memory_bytes(),
    }


def _result_document(record: RunRecord, digest: str) -> dict[str, object]:
    config, result = record.config, record.result
    limit = config.corpus.query_limit
    selection = (
        "all evaluable queries"
        if limit is None
        else f"seeded random sample of {limit} (seed {config.seed})"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "provisional": record.source.provisional,
        "experiment": {
            "name": config.name,
            "description": config.description,
            "config_path": record.config_path.as_posix(),
            "config_digest": digest,
            "config": asdict(config),
        },
        "code": {
            "groundwork_version": record.code.package_version,
            "git_commit": record.code.git_commit,
            "git_dirty": record.code.git_dirty,
        },
        "corpus": {
            "name": record.source.name,
            "url": record.source.url,
            "sha256": record.source.sha256,
            "licence": record.source.licence,
            "split": config.corpus.split,
            "documents": record.document_count,
            "chunks": result.chunk_count,
        },
        "queries": {
            "evaluated": len(result.queries),
            "selection": selection,
            "excluded_no_relevant_document": list(result.excluded_query_ids),
        },
        "environment": dict(record.environment),
        "timing": {
            "started_at": record.started_at.isoformat(),
            "finished_at": record.finished_at.isoformat(),
            "stage_seconds": dict(result.stage_seconds),
            "retrieval_latency_ms": dict(result.retrieval_latency_ms),
        },
        "metrics": {
            "aggregate": dict(result.aggregate),
            "per_query": {query.query_id: dict(query.metrics) for query in result.queries},
        },
        "files": {"run": "run.trec"},
    }


def _trec_run(record: RunRecord) -> str:
    # repr keeps every digit of the score, so rounding cannot create ties that were not there.
    return "".join(
        f"{query.query_id} Q0 {ranked.doc_id} {rank} {ranked.score!r} {record.config.name}\n"
        for query in record.result.queries
        for rank, ranked in enumerate(query.ranking, start=1)
    )


def _cpu_model() -> str | None:
    match platform.system():
        case "Darwin":
            return _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        case "Linux":
            return _proc_field(Path("/proc/cpuinfo"), "model name")
        case _:
            return None


def _memory_bytes() -> int | None:
    match platform.system():
        case "Darwin":
            reported = _run(["sysctl", "-n", "hw.memsize"])
            return int(reported) if reported and reported.isdigit() else None
        case "Linux":
            reported = _proc_field(Path("/proc/meminfo"), "MemTotal")
            kibibytes = reported.split()[0] if reported else ""
            return int(kibibytes) * 1024 if kibibytes.isdigit() else None
        case _:
            return None


def _proc_field(path: Path, name: str) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        logger.debug("could not read system file", extra={"path": str(path)})
        return None
    for line in lines:
        key, _, value = line.partition(":")
        if key.strip() == name:
            return value.strip()
    return None


def _run(command: list[str]) -> str | None:
    """Run a read-only system command and return its trimmed output, or None if it failed."""
    executable = shutil.which(command[0])
    if executable is None:
        return None
    # The arguments are fixed strings from this module, never user input, which is the risk
    # S603 guards against.
    completed = subprocess.run(  # noqa: S603
        [executable, *command[1:]], capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        logger.debug("system command failed", extra={"command": command[0]})
        return None
    return completed.stdout.strip()
