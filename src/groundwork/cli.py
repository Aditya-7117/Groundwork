"""Command-line entry point.

    groundwork build-corpus         download the dataset and build the evaluation corpus
    groundwork grid                 write the config file of every setup in the grid
    groundwork run CONFIG [...]     run experiments and write one results artefact each
    groundwork answer CONFIG        write answers from chosen setups' latest retrieval runs
    groundwork label ANSWERS_DIR    label a blind sample of answers by hand

Paths that depend on the machine, such as where the corpus is cached and where results are
written, are command-line options rather than config fields, so the same experiment has the same
config digest everywhere.
"""

import argparse
import json
import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from groundwork.answering import (
    AnsweringError,
    answer_setup,
    latest_run,
    load_answers,
    stratified_sample,
)
from groundwork.artefact import (
    ArtefactError,
    RunRecord,
    StageTwoRecord,
    current_code_version,
    describe_environment,
    write_artefact,
    write_stage_two_artefact,
)
from groundwork.cache import ResponseCache
from groundwork.config import ConfigError, ExperimentConfig, load_config, load_stage_two_config
from groundwork.download import SourceError
from groundwork.embeddings import EmbeddingError
from groundwork.evaluation import EvaluationSet, EvaluationSetError, load_built_corpus
from groundwork.experiment import LocalModels, ModelProvider, run_experiment
from groundwork.generation import GenerationError, OllamaWriter
from groundwork.grid import write_grid
from groundwork.labelling import LabellingError, label_sample, run_labelling
from groundwork.logs import configure_logging
from groundwork.natural_questions import NATURAL_QUESTIONS, build_corpus, fetch
from groundwork.rerank import RerankError
from groundwork.sentences import model_digest

logger = logging.getLogger(__name__)

type Now = Callable[[], datetime]

_FAILURES = (
    ConfigError,
    SourceError,
    EvaluationSetError,
    ArtefactError,
    EmbeddingError,
    RerankError,
    AnsweringError,
    GenerationError,
    LabellingError,
)


def main(
    argv: Sequence[str] | None = None,
    *,
    now: Now | None = None,
    models: ModelProvider | None = None,
) -> int:
    """Parse arguments, run the requested command, and return the process exit code.

    Args:
        argv: Command-line arguments; defaults to the process's own.
        now: Wall clock for the run record; tests pass a fixed one.
        models: Where neural models come from; defaults to real models cached under the data
            directory. Tests pass small fakes.

    Returns:
        0 on success, 1 if the run failed for a reason reported in the log.
    """
    arguments = _parser().parse_args(argv)
    configure_logging(arguments.log_level)
    data_dir: Path = arguments.data_dir
    try:
        if arguments.command == "build-corpus":
            return _build(data_dir)
        if arguments.command == "grid":
            write_grid(arguments.out)
            return 0
        if arguments.command == "label":
            return _label(arguments.answers, arguments.results_dir)
        if arguments.command == "answer":
            return _answer(arguments.config, data_dir, arguments.results_dir, now=now or _utc_now)
        return _run(
            arguments.configs,
            data_dir,
            arguments.results_dir,
            now=now or _utc_now,
            models=models
            or LocalModels(
                cache_dir=data_dir / "embeddings", weights_dir=data_dir / "huggingface" / "hub"
            ),
        )
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


def _run(
    config_paths: Sequence[Path],
    data_dir: Path,
    results_dir: Path,
    *,
    now: Now,
    models: ModelProvider,
) -> int:
    """Run each config in turn, loading each corpus and each model once.

    Every config is validated before anything runs, so a typo in the last file of a grid fails
    in a second rather than after hours of earlier runs.
    """
    configs = [(path, load_config(path)) for path in config_paths]
    corpora: dict[tuple[str, str], EvaluationSet] = {}
    code = current_code_version()
    if code.git_dirty:
        logger.warning(
            "working tree has uncommitted changes",
            extra={"note": "these results cannot be reproduced from their commit alone"},
        )
    for config_path, config in configs:
        started_at = now()
        key = (config.corpus.name, config.corpus.split)
        if key not in corpora:
            corpora[key] = _load(config, data_dir)
        evaluation_set = corpora[key]
        result = run_experiment(config, evaluation_set, models=models)
        finished_at = now()
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


def _answer(config_path: Path, data_dir: Path, results_dir: Path, *, now: Now) -> int:
    """Write answers for every setup a stage-two config names.

    Every setup's retrieval run is located before any answer is written, so a missing run fails
    at once. Answers are cached, so an interrupted run resumes where it stopped.
    """
    config = load_stage_two_config(config_path)
    runs = {setup: latest_run(results_dir, setup) for setup in config.setups}
    started_at = now()
    evaluation_set = load_built_corpus(data_dir / NATURAL_QUESTIONS.name / "built")
    questions = stratified_sample(evaluation_set.questions, config.questions, seed=config.seed)
    writer = OllamaWriter(
        model=config.writer,
        cache=ResponseCache(data_dir / "cache" / "answers.jsonl"),
        seed=config.seed,
    )
    code = current_code_version()
    answers = tuple(
        answer
        for setup in config.setups
        for answer in answer_setup(
            runs[setup], evaluation_set, questions, writer.write, passages=config.passages
        )
    )
    by_type = {
        kind: sum(1 for question in questions if question.answer_type == kind)
        for kind in sorted({question.answer_type for question in questions})
    }
    directory = write_stage_two_artefact(
        StageTwoRecord(
            config=config,
            config_path=config_path,
            runs=runs,
            sample={
                "sampled": len(questions),
                "from": len(evaluation_set.questions),
                "selection": f"stratified by answer type, seed {config.seed}",
                "by_answer_type": by_type,
            },
            writer_digest=writer.digest,
            answers=answers,
            started_at=started_at,
            finished_at=now(),
            code=code,
            environment=describe_environment(),
        ),
        results_dir,
    )
    logger.info("answers complete", extra={"artefact": str(directory), "answers": len(answers)})
    return 0


def _label(answers_dir: Path, results_dir: Path) -> int:
    """Label the blind sample of a stage-two run's answers, resuming where the last session ended.

    Labels are saved under results/labels/, next to the runs they describe, because the agreement
    numbers cannot be reproduced without them.
    """
    document = json.loads((answers_dir / "result.json").read_text(encoding="utf-8"))
    name = document["experiment"]["name"]
    sample = label_sample(
        load_answers(answers_dir / "answers.jsonl"), seed=document["experiment"]["config"]["seed"]
    )
    labels_path = results_dir / "labels" / f"{name}.jsonl"
    done = run_labelling(sample, labels_path)
    logger.info(
        "labelling session ended",
        extra={"labelled": done, "of": len(sample), "path": str(labels_path)},
    )
    return 0


def _load(config: ExperimentConfig, data_dir: Path) -> EvaluationSet:
    """Load the evaluation set the config names.

    Raises:
        EvaluationSetError: If the corpus is unknown or has not been built.
    """
    if config.corpus.name == NATURAL_QUESTIONS.name:
        return load_built_corpus(data_dir / NATURAL_QUESTIONS.name / "built")
    raise EvaluationSetError(
        f"unknown corpus {config.corpus.name!r}; known: {NATURAL_QUESTIONS.name}"
    )


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
    grid = commands.add_parser("grid", help="write the config file of every setup in the grid")
    grid.add_argument(
        "--out", type=Path, default=Path("configs/grid"), help="output (default: configs/grid)"
    )
    run = commands.add_parser("run", help="run experiments and write one results artefact each")
    run.add_argument(
        "configs", type=Path, nargs="+", help="experiment config files, e.g. configs/grid/*.toml"
    )
    run.add_argument(
        "--results-dir", type=Path, default=Path("results"), help="output (default: results)"
    )
    answer = commands.add_parser(
        "answer", help="write answers from chosen setups' latest retrieval runs"
    )
    answer.add_argument("config", type=Path, help="stage-two config file, e.g. configs/stage2.toml")
    answer.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="runs and output (default: results)",
    )
    label = commands.add_parser("label", help="label a blind sample of answers by hand")
    label.add_argument("answers", type=Path, help="a stage-two run directory")
    label.add_argument(
        "--results-dir", type=Path, default=Path("results"), help="output (default: results)"
    )
    return parser


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)
