"""Generation latency, measured under the same conditions for every setup (decision 75).

Answers are written over hours, and the time each took depends on what else the machine was doing:
the baseline's were written while it swapped heavily. So latency is not read from the answers.
Instead, for each stage-two setup, the same seeded sample of its prompts is sent again as real,
uncached calls, one setup after another in one session with nothing else running, after one
unrecorded warm-up call so model loading is not counted. At temperature 0 the text should come
back unchanged; any answer that does not is counted and reported rather than hidden.
"""

import logging
import math
import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from groundwork.answering import Answer
from groundwork.generation import Generation

logger = logging.getLogger(__name__)

SAMPLE = 100

type FreshWriter = Callable[[str, Sequence[str]], Generation]


@dataclass(frozen=True, slots=True, kw_only=True)
class Timing:
    """One re-timed answer.

    Attributes:
        setup: The setup whose prompt was re-sent.
        question_id: The question.
        seconds: Server time for the call, reading the prompt included.
        prompt_tokens: Tokens read.
        output_tokens: Tokens written.
        same_text: Whether the answer matched the one originally written.
    """

    setup: str
    question_id: str
    seconds: float
    prompt_tokens: int
    output_tokens: int
    same_text: bool


def timing_sample(answers: Sequence[Answer], *, size: int, seed: int) -> tuple[Answer, ...]:
    """The same seeded sample of each setup's answers, setups in name order."""
    chosen: list[Answer] = []
    for setup in sorted({answer.setup for answer in answers}):
        pool = sorted((a for a in answers if a.setup == setup), key=lambda a: a.question_id)
        # A seeded, reproducible sample is the point here; S311 concerns cryptographic randomness.
        chosen += random.Random(seed).sample(pool, min(size, len(pool)))  # noqa: S311
    return tuple(chosen)


def retime(sample: Sequence[Answer], write_fresh: FreshWriter) -> tuple[Timing, ...]:
    """Re-send each sampled prompt as a real call and record how long it took."""
    timings = []
    for number, answer in enumerate(sample, start=1):
        generation = write_fresh(answer.question, answer.passages)
        timings.append(
            Timing(
                setup=answer.setup,
                question_id=answer.question_id,
                seconds=generation.seconds,
                prompt_tokens=generation.prompt_tokens,
                output_tokens=generation.output_tokens,
                same_text=generation.text == answer.text,
            )
        )
        if number % 25 == 0:
            logger.info("re-timed", extra={"done": number, "total": len(sample)})
    return tuple(timings)


def summarise_timings(timings: Sequence[Timing]) -> dict[str, dict[str, float]]:
    """Per setup: median, 95th percentile and mean seconds, token counts, and text mismatches."""
    summary = {}
    for setup in sorted({timing.setup for timing in timings}):
        rows = [timing for timing in timings if timing.setup == setup]
        seconds = sorted(row.seconds for row in rows)
        summary[setup] = {
            "answers": float(len(rows)),
            "p50_seconds": statistics.median(seconds),
            "p95_seconds": seconds[math.ceil(0.95 * len(seconds)) - 1],
            "mean_seconds": statistics.fmean(seconds),
            "mean_prompt_tokens": statistics.fmean(row.prompt_tokens for row in rows),
            "mean_output_tokens": statistics.fmean(row.output_tokens for row in rows),
            "text_mismatches": float(sum(not row.same_text for row in rows)),
        }
    return summary
