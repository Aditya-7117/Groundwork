"""Writing a run's results as a versioned artefact that records everything that produced them.

Each run gets its own directory, results/<experiment>/<UTC start>-<config digest>/, which is
never overwritten. It holds:

- result.json: the full config and its digest, the code version (git commit and whether the
  working tree had uncommitted changes), the corpus and its provenance, the hardware, the
  timings, and every metric: averaged, broken down by answer type, and per question.
- run.passages.trec and run.pages.trec: the rankings in the standard TREC run format, so the
  numbers can be recomputed with an independent tool such as trec_eval.

A stage-two run, which writes answers from chosen setups' rankings, gets a directory of the same
shape under results/<stage-two name>/, holding result.json and answers.jsonl. Its verdicts go to
results/<stage-two name>-verdicts/, holding result.json and scored.jsonl, and point back to the
answers they scored.
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
from groundwork.answering import Answer, answer_rows, summarise
from groundwork.baselines import NLI_MODEL
from groundwork.config import ExperimentConfig, StageTwoConfig, config_digest
from groundwork.experiment import ExperimentResult
from groundwork.generation import CONTEXT_TOKENS, SYSTEM_PROMPT
from groundwork.judging import CORRECTNESS, GROUNDEDNESS
from groundwork.verdicts import (
    GEMINI_PRICE,
    LEXICAL_SUPPORTED,
    NLI_SUPPORTED,
    Scored,
    baseline_agreement,
    summarise_setup,
)

logger = logging.getLogger(__name__)

STAGE_TWO_SCHEMA_VERSION = 1
"""Version of a stage-two result.json layout, counted separately from retrieval runs."""

SCHEMA_VERSION = 4
"""Version of the result.json layout. Increment it whenever a field changes meaning or moves.

Version 2 reports metrics at passage and page level, adds a breakdown by answer type, and
describes the corpus as a record of its own rather than a single downloaded archive.

Version 3 moves the BM25 settings into their own section of the config, and records the neural
models used (name, pinned revision, device, precision) and the reranking latency.

Version 4 records each question's answer type, so comparisons within a type need no corpus.
"""


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
    corpus: Mapping[str, object]
    result: ExperimentResult
    started_at: datetime
    finished_at: datetime
    code: CodeVersion
    environment: Mapping[str, object]


@dataclass(frozen=True, slots=True, kw_only=True)
class StageTwoRecord:
    """Everything that goes into one stage-two artefact.

    Attributes:
        config: The stage-two config.
        config_path: Where it was read from.
        runs: The retrieval run directory answered for each setup.
        sample: How the questions were drawn, and how many of each answer type.
        writer_digest: Digest of the exact writer weights.
        answers: Every answer, all setups together.
        started_at: UTC start.
        finished_at: UTC finish.
        code: The code version.
        environment: The machine.
    """

    config: StageTwoConfig
    config_path: Path
    runs: Mapping[str, Path]
    sample: Mapping[str, object]
    writer_digest: str
    answers: tuple[Answer, ...]
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
    return _publish(
        results_dir / record.config.name,
        f"{record.started_at:%Y%m%dT%H%M%SZ}-{digest[:12]}",
        {
            "run.passages.trec": _trec_run(record, "passages"),
            "run.pages.trec": _trec_run(record, "pages"),
            "result.json": json.dumps(_result_document(record, digest), indent=2) + "\n",
        },
    )


def write_stage_two_artefact(record: StageTwoRecord, results_dir: Path) -> Path:
    """Write the artefact for one stage-two run and return its directory.

    Raises:
        ValueError: If a timestamp is not timezone-aware UTC.
        ArtefactError: If the run directory already exists.
    """
    for label, moment in (("started_at", record.started_at), ("finished_at", record.finished_at)):
        if moment.utcoffset() != timedelta(0):
            raise ValueError(f"{label} must be timezone-aware UTC, got {moment.isoformat()}")
    digest = config_digest(record.config)
    config = record.config
    document = {
        "schema_version": STAGE_TWO_SCHEMA_VERSION,
        "experiment": {
            "name": config.name,
            "description": config.description,
            "config_path": record.config_path.as_posix(),
            "config_digest": digest,
            "config": config.model_dump(mode="json"),
        },
        "code": {
            "groundwork_version": record.code.package_version,
            "git_commit": record.code.git_commit,
            "git_dirty": record.code.git_dirty,
        },
        "environment": dict(record.environment),
        "writer": {
            "model": config.writer,
            "digest": record.writer_digest,
            "system_prompt": SYSTEM_PROMPT,
            "temperature": 0,
            "seed": config.seed,
            "thinking": False,
            "context_tokens": CONTEXT_TOKENS,
            "passages": config.passages,
        },
        "questions": dict(record.sample),
        "runs": {setup: _run_provenance(run_dir) for setup, run_dir in sorted(record.runs.items())},
        "timing": {
            "started_at": record.started_at.isoformat(),
            "finished_at": record.finished_at.isoformat(),
        },
        "answers": {
            setup: summarise([a for a in record.answers if a.setup == setup])
            for setup in config.setups
        },
        "files": {"answers": "answers.jsonl"},
    }
    return _publish(
        results_dir / config.name,
        f"{record.started_at:%Y%m%dT%H%M%SZ}-{digest[:12]}",
        {
            "answers.jsonl": answer_rows(record.answers),
            "result.json": json.dumps(document, indent=2) + "\n",
        },
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class VerdictsRecord:
    """Everything that goes into one verdicts artefact.

    Attributes:
        config: The stage-two config the answers were written under.
        answers_dir: The answers that were scored.
        scored: Every answer's result from every rung.
        budget: The approved judging budget, in the price's currency.
        spent: What the judging cost at the list price.
        nli_device: Where the NLI classifier ran.
        started_at: UTC start.
        finished_at: UTC finish.
        code: The code version.
        environment: The machine.
    """

    config: StageTwoConfig
    answers_dir: Path
    scored: tuple[Scored, ...]
    budget: float
    spent: float
    nli_device: str
    started_at: datetime
    finished_at: datetime
    code: CodeVersion
    environment: Mapping[str, object]


def write_verdicts_artefact(record: VerdictsRecord, results_dir: Path) -> Path:
    """Write the verdicts for one stage-two run and return their directory.

    Raises:
        ValueError: If a timestamp is not timezone-aware UTC.
        ArtefactError: If the run directory already exists.
    """
    for label, moment in (("started_at", record.started_at), ("finished_at", record.finished_at)):
        if moment.utcoffset() != timedelta(0):
            raise ValueError(f"{label} must be timezone-aware UTC, got {moment.isoformat()}")
    config = record.config
    digest = config_digest(config)
    document = {
        "schema_version": STAGE_TWO_SCHEMA_VERSION,
        "answers": record.answers_dir.as_posix(),
        "code": {
            "groundwork_version": record.code.package_version,
            "git_commit": record.code.git_commit,
            "git_dirty": record.code.git_dirty,
        },
        "environment": dict(record.environment),
        "judge": {
            "model": config.judge.model,
            "thinking": config.judge.thinking,
            "rubrics": {
                rubric.name: {"instruction": rubric.instruction, "labels": list(rubric.labels)}
                for rubric in (GROUNDEDNESS, CORRECTNESS)
            },
            "price": asdict(GEMINI_PRICE),
            "budget": record.budget,
            "spent": record.spent,
        },
        "baselines": {
            "nli": {
                "model": NLI_MODEL.name,
                "revision": NLI_MODEL.revision,
                "device": record.nli_device,
                "precision": "float32",
                "supported_at": NLI_SUPPORTED,
            },
            "lexical": {"supported_at": LEXICAL_SUPPORTED},
        },
        "timing": {
            "started_at": record.started_at.isoformat(),
            "finished_at": record.finished_at.isoformat(),
        },
        "setups": {
            setup: summarise_setup([s for s in record.scored if s.setup == setup])
            for setup in config.setups
        },
        "agreement_with_judge": baseline_agreement(record.scored, seed=config.seed),
        "files": {"scored": "scored.jsonl"},
    }
    rows = "".join(json.dumps(asdict(s), ensure_ascii=False) + "\n" for s in record.scored)
    return _publish(
        results_dir / f"{config.name}-verdicts",
        f"{record.started_at:%Y%m%dT%H%M%SZ}-{digest[:12]}",
        {"scored.jsonl": rows, "result.json": json.dumps(document, indent=2) + "\n"},
    )


def write_report(
    report: Mapping[str, object], summary: str, results_dir: Path, moment: datetime
) -> Path:
    """Write a report as results/report/<UTC time>/report.json and report.md.

    Raises:
        ArtefactError: If the directory already exists.
    """
    return _publish(
        results_dir / "report",
        f"{moment:%Y%m%dT%H%M%SZ}",
        {"report.json": json.dumps(report, indent=2) + "\n", "report.md": summary},
    )


def _run_provenance(run_dir: Path) -> dict[str, object]:
    """Which retrieval run was answered, and the code and config that produced it."""
    document = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    return {
        "path": run_dir.as_posix(),
        "config_digest": document["experiment"]["config_digest"],
        "git_commit": document["code"]["git_commit"],
        "git_dirty": document["code"]["git_dirty"],
    }


def _publish(parent: Path, name: str, files: Mapping[str, str]) -> Path:
    """Write files into a hidden directory, then rename it into place in one step.

    A crash therefore never leaves a directory that looks complete.

    Raises:
        ArtefactError: If the directory already exists.
    """
    directory = parent / name
    if directory.exists():
        raise ArtefactError(f"run directory already exists, refusing to overwrite: {directory}")
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=parent, prefix=".incomplete-"))
    for filename, content in files.items():
        (staging / filename).write_text(content, encoding="utf-8")
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
        "provisional": bool(record.corpus.get("provisional", False)),
        "experiment": {
            "name": config.name,
            "description": config.description,
            "config_path": record.config_path.as_posix(),
            "config_digest": digest,
            "config": config.model_dump(mode="json"),
        },
        "code": {
            "groundwork_version": record.code.package_version,
            "git_commit": record.code.git_commit,
            "git_dirty": record.code.git_dirty,
        },
        "corpus": {
            **dict(record.corpus),
            "split": config.corpus.split,
            "chunks": result.chunk_count,
        },
        "questions": {
            "evaluated": len(result.questions),
            "selection": selection,
            "excluded_no_chunk_covers_the_answer": list(result.excluded_question_ids),
            "by_answer_type": {
                answer_type: sum(1 for q in result.questions if q.answer_type == answer_type)
                for answer_type in sorted({q.answer_type for q in result.questions})
            },
            "answer_types": {q.question_id: q.answer_type for q in result.questions},
        },
        "environment": dict(record.environment),
        "timing": {
            "started_at": record.started_at.isoformat(),
            "finished_at": record.finished_at.isoformat(),
            "stage_seconds": dict(result.stage_seconds),
            "retrieval_latency_ms": dict(result.retrieval_latency_ms),
            "rerank_latency_ms": dict(result.rerank_latency_ms),
        },
        "models": dict(result.models),
        "metrics": {
            "aggregate": dict(result.aggregate),
            "by_answer_type": {
                answer_type: dict(values) for answer_type, values in result.by_answer_type.items()
            },
            "per_question": {
                question.question_id: dict(question.metrics) for question in result.questions
            },
        },
        "files": {"passages": "run.passages.trec", "pages": "run.pages.trec"},
    }


def _trec_run(record: RunRecord, level: str) -> str:
    """Write one ranking per question in the TREC run format.

    Scores descend with rank rather than carrying the retriever's own scores, because the ranking
    is what the metrics are computed from. An independent tool re-scoring this file therefore
    checks the metric code, not the retriever's arithmetic.
    """
    lines = []
    for question in record.result.questions:
        ranking = question.chunk_ranking if level == "passages" else question.page_ranking
        for rank, item in enumerate(ranking, start=1):
            score = len(ranking) - rank + 1
            lines.append(f"{question.question_id} Q0 {item} {rank} {score} {record.config.name}\n")
    return "".join(lines)


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
