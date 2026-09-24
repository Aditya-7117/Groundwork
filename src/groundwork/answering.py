"""Stage two, first half: answers written from each chosen setup's retrieval run.

Every setup answers the same questions: a sample stratified by answer type, drawn once with a
fixed seed, so paragraph, table and list questions appear in the same proportions as in the full
set (decision 62). For each question the writer reads the top chunks of that setup's saved
ranking, rebuilt from the setup's own chunking settings. The rebuilt chunk count is checked
against the count the retrieval run recorded, so answers can never be written from chunks that
differ from the ones that were ranked.
"""

import json
import logging
import math
import random
import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from groundwork.chunking import ChunkingSettings, chunk_pages
from groundwork.config import ExperimentConfig
from groundwork.evaluation import EvalQuestion, EvaluationSet
from groundwork.generation import Generation

logger = logging.getLogger(__name__)

type Writer = Callable[[str, Sequence[str]], Generation]


class AnsweringError(ValueError):
    """A retrieval run is missing, or does not match the corpus it is being answered from."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Answer:
    """One setup's answer to one question.

    Attributes:
        setup: The grid setup whose ranking supplied the passages.
        question_id: The question.
        answer_type: Where its reference answer sits: paragraph, table or list.
        chunk_ids: The chunks the writer read, best first.
        text: The answer.
        declined: Whether the writer said it did not know.
        prompt_tokens: Tokens read.
        output_tokens: Tokens written.
        seconds: Server time to write it, recorded when it was first generated.
    """

    setup: str
    question_id: str
    answer_type: str
    chunk_ids: tuple[str, ...]
    text: str
    declined: bool
    prompt_tokens: int
    output_tokens: int
    seconds: float


def stratified_sample(
    questions: Sequence[EvalQuestion], size: int, *, seed: int
) -> tuple[EvalQuestion, ...]:
    """Draw a seeded sample with each answer type in proportion to the whole set.

    Each type's share is size times its fraction, rounded down; the questions left over go to the
    types with the largest remainders (the largest-remainder method), so the sizes add up exactly.

    Raises:
        ValueError: If size is below 1 or larger than the set.
    """
    if not 1 <= size <= len(questions):
        raise ValueError(f"sample size must be between 1 and {len(questions)}, got {size}")
    by_type: dict[str, list[EvalQuestion]] = defaultdict(list)
    for question in sorted(questions, key=lambda item: item.question_id):
        by_type[question.answer_type].append(question)
    exact = {kind: size * len(items) / len(questions) for kind, items in by_type.items()}
    quota = {kind: math.floor(value) for kind, value in exact.items()}
    leftover = size - sum(quota.values())
    for kind in sorted(exact, key=lambda k: (quota[k] - exact[k], k))[:leftover]:
        quota[kind] += 1
    # A seeded, reproducible sample is the point here; S311 concerns cryptographic randomness.
    rng = random.Random(seed)  # noqa: S311
    chosen = [q for kind in sorted(by_type) for q in rng.sample(by_type[kind], quota[kind])]
    return tuple(sorted(chosen, key=lambda item: item.question_id))


def latest_run(results_dir: Path, setup: str) -> Path:
    """Return the newest finished retrieval run of a setup.

    Run directories start with their UTC start time, so the newest sorts last.

    Raises:
        AnsweringError: If the setup has no finished run.
    """
    runs = sorted(
        path
        for path in (results_dir / setup).glob("*")
        if path.is_dir() and not path.name.startswith(".") and (path / "result.json").is_file()
    )
    if not runs:
        raise AnsweringError(f"no finished retrieval run for {setup!r} under {results_dir}")
    return runs[-1]


def read_ranking(run_dir: Path, depth: int) -> dict[str, tuple[str, ...]]:
    """Read the top depth chunk ids per question from a run's passage ranking file."""
    ranking: dict[str, list[str]] = defaultdict(list)
    with (run_dir / "run.passages.trec").open(encoding="utf-8") as handle:
        for line in handle:
            question_id, _, chunk_id, rank, _, _ = line.split()
            if int(rank) <= depth:
                ranking[question_id].append(chunk_id)
    return {question_id: tuple(ids) for question_id, ids in ranking.items()}


def answer_setup(
    run_dir: Path,
    evaluation_set: EvaluationSet,
    questions: Sequence[EvalQuestion],
    write: Writer,
    *,
    passages: int,
) -> tuple[Answer, ...]:
    """Answer the questions from one retrieval run.

    Raises:
        AnsweringError: If the rebuilt chunks do not match the run, or a question has no ranking.
    """
    document = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    config = ExperimentConfig.model_validate(document["experiment"]["config"])
    settings = ChunkingSettings(
        strategy=config.chunking.strategy,
        size=config.chunking.size,
        overlap=config.chunking.overlap,
        flatten_tables=config.chunking.flatten_tables,
    )
    texts = {chunk.chunk_id: chunk.text for chunk in chunk_pages(evaluation_set.pages, settings)}
    recorded = document["corpus"]["chunks"]
    if len(texts) != recorded:
        raise AnsweringError(
            f"{run_dir}: rebuilt {len(texts)} chunks, but the run ranked {recorded}; the corpus "
            "or the chunker has changed since it ran"
        )
    ranking = read_ranking(run_dir, passages)

    answers = []
    started = time.perf_counter()
    for number, question in enumerate(questions, start=1):
        chunk_ids = ranking.get(question.question_id)
        if not chunk_ids:
            raise AnsweringError(f"{run_dir}: no ranking for question {question.question_id}")
        generation = write(question.text, [texts[chunk_id] for chunk_id in chunk_ids])
        answers.append(
            Answer(
                setup=config.name,
                question_id=question.question_id,
                answer_type=question.answer_type,
                chunk_ids=chunk_ids,
                text=generation.text,
                declined=generation.declined,
                prompt_tokens=generation.prompt_tokens,
                output_tokens=generation.output_tokens,
                seconds=generation.seconds,
            )
        )
        if number % 50 == 0 or number == len(questions):
            elapsed = time.perf_counter() - started
            logger.info(
                "answers written",
                extra={
                    "setup": config.name,
                    "done": number,
                    "total": len(questions),
                    "minutes_left": round(elapsed / number * (len(questions) - number) / 60, 1),
                },
            )
    return tuple(answers)


def summarise(answers: Sequence[Answer]) -> dict[str, object]:
    """Counts, decline rate, tokens and generation time for one setup's answers."""
    seconds = sorted(answer.seconds for answer in answers)
    declined = sum(answer.declined for answer in answers)
    return {
        "answers": len(answers),
        "declined": declined,
        "decline_rate": declined / len(answers) if answers else 0.0,
        "prompt_tokens": sum(answer.prompt_tokens for answer in answers),
        "output_tokens": sum(answer.output_tokens for answer in answers),
        "seconds": {
            "total": sum(seconds),
            "mean": statistics.fmean(seconds) if seconds else 0.0,
            "p50": statistics.median(seconds) if seconds else 0.0,
            "p95": seconds[math.ceil(0.95 * len(seconds)) - 1] if seconds else 0.0,
        },
    }


def answer_rows(answers: Sequence[Answer]) -> str:
    """Serialise answers as JSON Lines."""
    return "".join(json.dumps(asdict(answer), ensure_ascii=False) + "\n" for answer in answers)


def load_answers(path: Path) -> tuple[Answer, ...]:
    """Read answers written by answer_rows."""
    answers = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row: Mapping[str, object] = json.loads(line)
            answers.append(
                Answer(
                    setup=str(row["setup"]),
                    question_id=str(row["question_id"]),
                    answer_type=str(row["answer_type"]),
                    chunk_ids=tuple(str(item) for item in _sequence(row["chunk_ids"])),
                    text=str(row["text"]),
                    declined=bool(row["declined"]),
                    prompt_tokens=_integer(row["prompt_tokens"]),
                    output_tokens=_integer(row["output_tokens"]),
                    seconds=float(_number(row["seconds"])),
                )
            )
    return tuple(answers)


def _sequence(value: object) -> Sequence[object]:
    if not isinstance(value, list):
        raise AnsweringError(f"expected a list, got {type(value).__name__}")
    return value


def _integer(value: object) -> int:
    if not isinstance(value, int):
        raise AnsweringError(f"expected an integer, got {value!r}")
    return value


def _number(value: object) -> float:
    if not isinstance(value, int | float):
        raise AnsweringError(f"expected a number, got {value!r}")
    return float(value)
