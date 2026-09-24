"""Hand labels: the human rung of the groundedness ladder (decision 67).

Two hundred answers, forty from each stage-two setup, are drawn with a fixed seed from the answers
the writer did not decline, then shuffled together. The labeller sees the question, the passages
and the answer, and nothing else: not which setup wrote it, not the reference answer and not the
judge's label, so none of those can sway the label. The scale is the judge's own (decision 54),
which is what lets the two be compared with Cohen's kappa.

Each label is appended to a JSON Lines file the moment it is given, with a digest of exactly what
was shown, so a session can stop and resume at any point, and a label can never be matched to an
answer that has since changed.
"""

import hashlib
import json
import random
import shutil
import textwrap
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from groundwork.answering import Answer
from groundwork.judging import GROUNDEDNESS

LABELS_PER_SETUP = 40
_KEYS = dict(zip(("1", "2", "3"), GROUNDEDNESS.labels, strict=True))

type Ask = Callable[[str], str]
type Show = Callable[[str], None]


class LabellingError(ValueError):
    """The labels file does not match the answers it claims to label."""


def label_sample(
    answers: Sequence[Answer], *, seed: int, per_setup: int = LABELS_PER_SETUP
) -> tuple[Answer, ...]:
    """Draw the answers to label: per_setup non-declined answers per setup, then shuffled.

    Raises:
        ValueError: If a setup has fewer non-declined answers than per_setup.
    """
    # A seeded, reproducible sample is the point here; S311 concerns cryptographic randomness.
    rng = random.Random(seed)  # noqa: S311
    chosen: list[Answer] = []
    for setup in sorted({answer.setup for answer in answers}):
        pool = sorted(
            (a for a in answers if a.setup == setup and not a.declined),
            key=lambda a: a.question_id,
        )
        if len(pool) < per_setup:
            raise ValueError(f"{setup} has only {len(pool)} answers to label, needs {per_setup}")
        chosen += rng.sample(pool, per_setup)
    rng.shuffle(chosen)
    return tuple(chosen)


def shown_digest(answer: Answer) -> str:
    """SHA-256 of exactly what the labeller sees for one answer."""
    payload = json.dumps([answer.question, list(answer.passages), answer.text], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_labels(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Read saved labels, keyed by (setup, question id)."""
    if not path.is_file():
        return {}
    labels = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            labels[(row["setup"], row["question_id"])] = row
    return labels


def run_labelling(
    sample: Sequence[Answer],
    labels_path: Path,
    *,
    ask: Ask = input,
    show: Show = print,
) -> int:
    """Ask for a label on every unlabelled answer in the sample, saving each at once.

    Returns:
        How many labels the file now holds for this sample.

    Raises:
        LabellingError: If a saved label was given for different text than is shown now.
    """
    saved = load_labels(labels_path)
    for answer in sample:
        existing = saved.get((answer.setup, answer.question_id))
        digest = shown_digest(answer)
        if existing is not None and existing["shown_sha256"] != digest:
            raise LabellingError(
                f"the label for {answer.question_id} was given for different text; "
                "the answers have changed since labelling began"
            )
    width = min(shutil.get_terminal_size((100, 40)).columns, 100)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    for position, answer in enumerate(sample, start=1):
        if (answer.setup, answer.question_id) in saved:
            continue
        show(_screen(position, len(sample), answer, width))
        choice = ""
        while choice not in {*_KEYS, "q"}:
            choice = ask("Label [1 supported / 2 partly / 3 not / q quit]: ").strip().lower()
        if choice == "q":
            break
        row = {
            "setup": answer.setup,
            "question_id": answer.question_id,
            "label": _KEYS[choice],
            "shown_sha256": shown_digest(answer),
            "position": str(position),
            "labelled_at": datetime.now(tz=UTC).isoformat(),
        }
        with labels_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        saved[(answer.setup, answer.question_id)] = row
    return sum((a.setup, a.question_id) in saved for a in sample)


def _screen(position: int, total: int, answer: Answer, width: int) -> str:
    rule = "-" * width
    lines = [
        "\n" * 2 + rule,
        f"Answer {position} of {total}. Is everything it says supported by the passages?",
        "Judge only against the passages, not your own knowledge.",
        rule,
    ]
    for index, passage in enumerate(answer.passages, start=1):
        title, _, body = passage.partition("\n")
        lines.append(f"[{index}] {title}")
        lines += textwrap.wrap(body, width=width, initial_indent="    ", subsequent_indent="    ")
        lines.append("")
    lines += [
        rule,
        *textwrap.wrap(f"QUESTION: {answer.question}", width=width),
        *textwrap.wrap(f"ANSWER:   {answer.text}", width=width),
        rule,
        "1 supported: every claim is stated in, or directly follows from, the passages",
        "2 partly:    the main claim is supported, but a detail is not",
        "3 not:       the main claim is missing from the passages or contradicts them",
    ]
    return "\n".join(lines)
