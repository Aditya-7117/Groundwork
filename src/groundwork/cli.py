"""Command-line entry point.

    groundwork build-corpus         download the dataset and build the evaluation corpus
    groundwork grid                 write the config file of every setup in the grid
    groundwork run CONFIG [...]     run experiments and write one results artefact each
    groundwork answer CONFIG        write answers from chosen setups' latest retrieval runs
    groundwork judge ANSWERS_DIR    score answers by the judge, NLI, word overlap and containment
    groundwork label ANSWERS_DIR    label a blind sample of answers by hand
    groundwork retime ANSWERS_DIR   measure generation latency on a fixed sample, uncached
    groundwork golden               write the golden slice of the corpus for the regression gate
    groundwork report               assemble every published number from the run artefacts
    groundwork site                 write the explorer's data files from the artefacts
    groundwork serve                serve the explorer, with live search over chosen setups
    groundwork gate BASELINE        re-run the golden slice and fail if retrieval got worse

Paths that depend on the machine, such as where the corpus is cached and where results are
written, are command-line options rather than config fields, so the same experiment has the same
config digest everywhere.
"""

import argparse
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import uvicorn

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
    TimingRecord,
    VerdictsRecord,
    current_code_version,
    describe_environment,
    write_artefact,
    write_report,
    write_stage_two_artefact,
    write_timing_artefact,
    write_verdicts_artefact,
)
from groundwork.baselines import NliChecker
from groundwork.cache import ResponseCache
from groundwork.config import (
    ConfigError,
    ExperimentConfig,
    StageTwoConfig,
    load_config,
    load_stage_two_config,
)
from groundwork.download import SourceError
from groundwork.embeddings import EmbeddingError
from groundwork.evaluation import EvaluationSet, EvaluationSetError, load_built_corpus
from groundwork.experiment import LocalModels, ModelProvider, run_experiment
from groundwork.gate import GateError, record_baseline, run_gate
from groundwork.generation import GenerationError, OllamaWriter
from groundwork.golden import build_golden
from groundwork.grid import grid, write_grid
from groundwork.judging import Judge, JudgeError
from groundwork.labelling import LabellingError, label_sample, run_labelling
from groundwork.logs import configure_logging
from groundwork.natural_questions import NATURAL_QUESTIONS, build_corpus, fetch
from groundwork.report import ReportError, build_report, markdown
from groundwork.rerank import RerankError
from groundwork.sentences import model_digest
from groundwork.serve import build_live_setup, create_app
from groundwork.site import StageTwoInputs, export_site
from groundwork.timing import SAMPLE, retime, timing_sample
from groundwork.verdicts import CostGuard, entailment_with, judge_all, score_answers

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
    JudgeError,
    GateError,
    ReportError,
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
    clock = now or _utc_now
    commands: dict[str, Callable[[], int]] = {
        "build-corpus": lambda: _build(data_dir),
        "grid": lambda: _grid(arguments.out),
        "run": lambda: _run(
            arguments.configs,
            data_dir,
            arguments.results_dir,
            now=clock,
            models=models
            or LocalModels(
                cache_dir=data_dir / "embeddings", weights_dir=data_dir / "huggingface" / "hub"
            ),
        ),
        "answer": lambda: _answer(arguments.config, data_dir, arguments.results_dir, now=clock),
        "judge": lambda: _judge(
            arguments.answers,
            data_dir,
            arguments.results_dir,
            budget=_Budget(dollars=arguments.budget, workers=arguments.workers),
            now=clock,
        ),
        "label": lambda: _label(arguments.answers, arguments.results_dir),
        "retime": lambda: _retime(
            arguments.answers, data_dir, arguments.results_dir, sample=arguments.sample, now=clock
        ),
        "golden": lambda: _golden(data_dir, arguments.out),
        "site": lambda: _site(arguments, data_dir),
        "serve": lambda: _serve(arguments, data_dir),
        "report": lambda: _report(
            arguments.results_dir, arguments.verdicts, arguments.labels, now=clock
        ),
        "gate": lambda: _gate(
            arguments.baseline,
            LocalModels(
                cache_dir=data_dir / "embeddings",
                weights_dir=data_dir / "huggingface" / "hub",
                device=arguments.device,
            ),
            record=arguments.record,
        ),
    }
    try:
        return commands[arguments.command]()
    except _FAILURES as error:
        logger.error("command failed", extra={"error": str(error)})  # noqa: TRY400 -- the message is the diagnosis
        return 1


@dataclass(frozen=True, slots=True, kw_only=True)
class _Budget:
    """What judging may cost, and how many judge requests may be in flight at once."""

    dollars: float
    workers: int


def _grid(out: Path) -> int:
    write_grid(out)
    return 0


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


def _retime(answers_dir: Path, data_dir: Path, results_dir: Path, *, sample: int, now: Now) -> int:
    """Re-send a fixed sample of each setup's prompts as real calls and record their latency."""
    document = json.loads((answers_dir / "result.json").read_text(encoding="utf-8"))
    config = StageTwoConfig.model_validate(document["experiment"]["config"])
    answers = load_answers(answers_dir / "answers.jsonl")
    writer = OllamaWriter(
        model=config.writer,
        cache=ResponseCache(data_dir / "cache" / "answers.jsonl"),
        seed=config.seed,
    )
    chosen = timing_sample(answers, size=sample, seed=config.seed)
    # One unrecorded call first, so loading the model into memory is not counted as latency.
    writer.write(chosen[0].question, chosen[0].passages, fresh=True)
    started_at = now()
    timings = retime(
        chosen, lambda question, passages: writer.write(question, passages, fresh=True)
    )
    directory = write_timing_artefact(
        TimingRecord(
            config=config,
            answers_dir=answers_dir,
            timings=timings,
            writer_digest=writer.digest,
            started_at=started_at,
            finished_at=now(),
        ),
        results_dir,
    )
    logger.info("re-timing written", extra={"path": str(directory), "answers": len(timings)})
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


def _judge(
    answers_dir: Path, data_dir: Path, results_dir: Path, *, budget: _Budget, now: Now
) -> int:
    """Score a stage-two run's answers by every rung, and write the verdicts as their own run.

    The answers directory is never modified; the verdicts point back to it.
    """
    document = json.loads((answers_dir / "result.json").read_text(encoding="utf-8"))
    config = StageTwoConfig.model_validate(document["experiment"]["config"])
    answers = load_answers(answers_dir / "answers.jsonl")
    evaluation_set = load_built_corpus(data_dir / NATURAL_QUESTIONS.name / "built")
    references = {q.question_id: q.short_answers for q in evaluation_set.questions}
    started_at = now()
    judge = Judge(
        cache=ResponseCache(data_dir / "cache" / "judge.jsonl"),
        model=config.judge.model,
        thinking=config.judge.thinking,
    )
    answered = sum(not answer.declined for answer in answers)
    guard = CostGuard(budget=budget.dollars, expected_calls=2 * answered)
    judged = judge_all(answers, references, judge, guard, workers=budget.workers)
    checker = NliChecker(cache_dir=data_dir / "huggingface" / "hub")
    scored = score_answers(answers, references, judged, entailment_with(checker))
    directory = write_verdicts_artefact(
        VerdictsRecord(
            config=config,
            answers_dir=answers_dir,
            scored=scored,
            budget=budget.dollars,
            spent=guard.spent,
            nli_device=checker.device,
            started_at=started_at,
            finished_at=now(),
            code=current_code_version(),
            environment=describe_environment(),
        ),
        results_dir,
    )
    logger.info(
        "judging complete",
        extra={"artefact": str(directory), "spent_usd": round(guard.spent, 2)},
    )
    return 0


def _report(results_dir: Path, verdicts: Path | None, labels: Path | None, *, now: Now) -> int:
    """Write the report over the whole grid, plus stage two and the labels once they exist."""
    report = build_report(
        results_dir,
        [setup.name for setup in grid()],
        verdicts_dir=verdicts,
        labels_path=labels,
        seed=1,
    )
    directory = write_report(report, markdown(report), results_dir, now())
    logger.info("report written", extra={"path": str(directory), "winner": report["significance"]})
    return 0


def _site(arguments: argparse.Namespace, data_dir: Path) -> int:
    """Write the explorer's data files: the report plus every grid setup, and stage two if given."""
    results_dir: Path = arguments.results_dir
    setups = [setup.name for setup in grid()]
    report = build_report(
        results_dir,
        setups,
        verdicts_dir=arguments.verdicts,
        labels_path=arguments.labels,
        seed=1,
    )
    stage_two = None
    if arguments.answers is not None and arguments.verdicts is not None:
        stage_two = StageTwoInputs(
            answers_dir=arguments.answers,
            verdicts_dir=arguments.verdicts,
            labels_path=arguments.labels,
        )
    export_site(
        report,
        {setup: latest_run(results_dir, setup) for setup in setups},
        load_built_corpus(data_dir / NATURAL_QUESTIONS.name / "built"),
        arguments.out,
        stage_two=stage_two,
    )
    logger.info("site data written", extra={"path": str(arguments.out)})
    return 0


def _serve(arguments: argparse.Namespace, data_dir: Path) -> int:
    """Build each live setup once, then serve the site and live search until stopped."""
    configs = [load_config(path) for path in arguments.live]
    evaluation_set = load_built_corpus(data_dir / NATURAL_QUESTIONS.name / "built")
    models = LocalModels(
        cache_dir=data_dir / "embeddings",
        weights_dir=data_dir / "huggingface" / "hub",
        device=arguments.device,
    )
    live = {config.name: build_live_setup(config, evaluation_set, models) for config in configs}
    uvicorn.run(
        create_app(live, arguments.site), host=arguments.host, port=arguments.port, log_config=None
    )
    return 0


def _golden(data_dir: Path, out: Path) -> int:
    """Write the golden slice: 100 stratified questions, their pages and 200 distractors."""
    build_golden(
        data_dir / NATURAL_QUESTIONS.name / "built", out, questions=100, distractors=200, seed=1
    )
    return 0


def _gate(baseline: Path, models: ModelProvider, *, record: bool) -> int:
    """Run the regression gate, or re-record its baseline when a change is meant to move it."""
    if record:
        recorded = record_baseline(baseline, models)
        logger.info("baseline recorded", extra={"path": str(baseline), **_flatten(recorded)})
        return 0
    checks = run_gate(baseline, models)
    failed = [check for check in checks if not check.passed]
    if failed:
        logger.error("regression gate failed", extra={"failed": len(failed), "of": len(checks)})
        return 1
    logger.info("regression gate passed", extra={"checks": len(checks)})
    return 0


def _flatten(recorded: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        f"{setup}.{metric}": value
        for setup, row in recorded.items()
        for metric, value in row.items()
    }


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
    judge = commands.add_parser("judge", help="score answers by every rung of the ladder")
    judge.add_argument("answers", type=Path, help="a stage-two run directory")
    judge.add_argument(
        "--budget", type=float, required=True, help="approved judging budget in US dollars"
    )
    judge.add_argument(
        "--workers", type=int, default=8, help="judge requests in flight at once (default: 8)"
    )
    judge.add_argument(
        "--results-dir", type=Path, default=Path("results"), help="output (default: results)"
    )
    report = commands.add_parser("report", help="assemble every published number")
    report.add_argument("--verdicts", type=Path, default=None, help="a stage-two verdicts run")
    report.add_argument("--labels", type=Path, default=None, help="the hand labels file")
    report.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="runs and output (default: results)",
    )
    site = commands.add_parser("site", help="write the explorer's data files")
    site.add_argument("--answers", type=Path, default=None, help="a stage-two answers run")
    site.add_argument("--verdicts", type=Path, default=None, help="its verdicts run")
    site.add_argument("--labels", type=Path, default=None, help="the hand labels file")
    site.add_argument("--results-dir", type=Path, default=Path("results"), help="runs")
    site.add_argument(
        "--out", type=Path, default=Path("results/site/data"), help="output directory"
    )
    serve = commands.add_parser("serve", help="serve the explorer with live search")
    serve.add_argument(
        "--live", type=Path, nargs="*", default=[], help="config files of the live setups"
    )
    serve.add_argument("--site", type=Path, default=None, help="the built explorer directory")
    serve.add_argument("--host", default="127.0.0.1", help="interface (default: 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8000, help="port (default: 8000)")
    serve.add_argument(
        "--device", choices=["cpu", "mps"], default=None, help="where models run (default: best)"
    )
    golden = commands.add_parser("golden", help="write the golden slice for the regression gate")
    golden.add_argument(
        "--out",
        type=Path,
        default=Path("tests/golden/corpus"),
        help="output (default: tests/golden/corpus)",
    )
    gate = commands.add_parser("gate", help="re-run the golden slice and fail on a regression")
    gate.add_argument("baseline", type=Path, help="baseline file, e.g. tests/golden/baseline.json")
    gate.add_argument(
        "--device", choices=["cpu", "mps"], default=None, help="where models run (default: best)"
    )
    gate.add_argument(
        "--record", action="store_true", help="re-record the baseline metrics instead of checking"
    )
    retime_parser = commands.add_parser(
        "retime", help="measure generation latency on a fixed sample of uncached calls"
    )
    retime_parser.add_argument("answers", type=Path, help="a stage-two answers run")
    retime_parser.add_argument(
        "--sample", type=int, default=SAMPLE, help=f"prompts per setup (default: {SAMPLE})"
    )
    retime_parser.add_argument(
        "--results-dir", type=Path, default=Path("results"), help="output (default: results)"
    )
    label = commands.add_parser("label", help="label a blind sample of answers by hand")
    label.add_argument("answers", type=Path, help="a stage-two run directory")
    label.add_argument(
        "--results-dir", type=Path, default=Path("results"), help="output (default: results)"
    )
    return parser


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)
