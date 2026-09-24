"""Stage two, second half: every answer scored by every rung of the ladder.

For each answer the stage records the mechanical containment check against the references, and,
unless the writer declined, word overlap, NLI entailment and the judge's two verdicts. Declined
answers are never judged: they are counted as declined, and as not correct.

The judge runs several requests at once, because one at a time at medium thinking would take most
of a day. Spending is watched as it happens (decision 61): after the first fifty judgements the
cost so far is projected over every call the run will make, and the run stops if that projection
exceeds the approved budget by more than half.
"""

import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from groundwork.agreement import agreement, roc_auc
from groundwork.answering import Answer
from groundwork.baselines import (
    NliChecker,
    contains_reference,
    lexical_support,
    strongest_entailment,
)
from groundwork.judging import (
    CORRECTNESS,
    GROUNDEDNESS,
    Judge,
    JudgeError,
    Verdict,
    correctness_prompt,
    groundedness_prompt,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class Price:
    """A dated list price, so a cost is always reported with where and when it came from."""

    model: str
    input_per_million: float
    output_per_million: float
    currency: str
    checked: str
    source: str


JUDGE_PRICE = Price(
    model="gpt-6-luna",
    input_per_million=0.10,
    output_per_million=0.50,
    currency="USD",
    checked="2026-09-25",
    source=(
        "https://developers.openai.com/api/docs/pricing, standard tier, short context; reasoning "
        "tokens are counted as output tokens in the API's usage report"
    ),
)


class BudgetExceededError(JudgeError):
    """The projected judging cost is more than half again over the approved budget."""


def verdict_cost(verdict: Verdict, price: Price = JUDGE_PRICE) -> float:
    """What one judgement costs at the list price; thinking is billed as output."""
    output = verdict.output_tokens + verdict.thinking_tokens
    return (
        verdict.prompt_tokens * price.input_per_million + output * price.output_per_million
    ) / 1_000_000


class CostGuard:
    """Projects total spend from the calls so far and stops a run heading far over budget."""

    def __init__(
        self, *, budget: float, expected_calls: int, check_after: int = 50, tolerance: float = 1.5
    ) -> None:
        """Set the approved budget and when to start checking against it."""
        self.budget = budget
        self.expected_calls = expected_calls
        self.check_after = check_after
        self.tolerance = tolerance
        self.calls = 0
        self.spent = 0.0
        self._lock = threading.Lock()

    def record(self, verdict: Verdict) -> None:
        """Count one judgement's cost, and stop if the projection is over the limit.

        Raises:
            BudgetExceededError: If, after check_after calls, the projected total exceeds
                tolerance times the budget.
        """
        with self._lock:
            self.calls += 1
            self.spent += verdict_cost(verdict)
            if self.calls < self.check_after:
                return
            projected = self.spent / self.calls * self.expected_calls
            if projected > self.tolerance * self.budget:
                raise BudgetExceededError(
                    f"projected judging cost ${projected:.2f} over {self.expected_calls} calls "
                    f"exceeds the approved ${self.budget:.2f} by more than "
                    f"{(self.tolerance - 1):.0%}; stopped after {self.calls} calls"
                )


@dataclass(frozen=True, slots=True, kw_only=True)
class Scored:
    """One answer, scored by every rung.

    Attributes:
        setup: The setup that wrote it.
        question_id: The question.
        answer_type: Paragraph, table or list.
        declined: Whether the writer said it did not know.
        contains_reference: Mechanical correctness; False when declined.
        lexical: Share of the answer's new words found in the passages; None when declined.
        nli: Strongest entailment probability across the passages; None when declined.
        groundedness: The judge's groundedness label; None when declined.
        correctness: The judge's correctness label; None when declined.
        groundedness_reason: The judge's reason for its groundedness label.
        correctness_reason: The judge's reason for its correctness label.
        judge_cost: What the two judgements cost at the list price.
        judge_seconds: Time the two judgements took, including retries.
    """

    setup: str
    question_id: str
    answer_type: str
    declined: bool
    contains_reference: bool
    lexical: float | None
    nli: float | None
    groundedness: str | None
    correctness: str | None
    groundedness_reason: str | None
    correctness_reason: str | None
    judge_cost: float
    judge_seconds: float


type Key = tuple[str, str]
type Entailment = Callable[[Sequence[str], str], float]


def entailment_with(checker: NliChecker) -> Entailment:
    """Adapt the NLI checker to 'strongest entailment of this answer by these passages'."""

    def score(passages: Sequence[str], answer: str) -> float:
        return strongest_entailment(checker.probabilities(passages, answer))

    return score


def judge_all(
    answers: Sequence[Answer],
    references: Mapping[str, Sequence[str]],
    judge: Judge,
    guard: CostGuard,
    *,
    workers: int,
) -> dict[Key, tuple[Verdict, Verdict]]:
    """Judge every non-declined answer for groundedness and correctness, several at a time.

    Raises:
        BudgetExceededError: If spending heads too far over budget; unstarted work is cancelled.
        JudgeError: If a judgement fails after its retries.
    """
    pending = [answer for answer in answers if not answer.declined]
    done: dict[Key, tuple[Verdict, Verdict]] = {}

    def one(answer: Answer) -> tuple[Key, tuple[Verdict, Verdict]]:
        grounded = judge.judge(
            GROUNDEDNESS, groundedness_prompt(answer.question, answer.passages, answer.text)
        )
        guard.record(grounded)
        correct = judge.judge(
            CORRECTNESS,
            correctness_prompt(answer.question, references[answer.question_id], answer.text),
        )
        guard.record(correct)
        return (answer.setup, answer.question_id), (grounded, correct)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, answer) for answer in pending]
        try:
            for number, future in enumerate(as_completed(futures), start=1):
                key, verdicts = future.result()
                done[key] = verdicts
                if number % 200 == 0:
                    logger.info(
                        "judged",
                        extra={
                            "done": number,
                            "total": len(pending),
                            "spent": round(guard.spent, 2),
                        },
                    )
        except JudgeError:
            pool.shutdown(wait=True, cancel_futures=True)
            raise
    logger.info("judging finished", extra={"judged": len(done), "spent": round(guard.spent, 4)})
    return done


def score_answers(
    answers: Sequence[Answer],
    references: Mapping[str, Sequence[str]],
    judged: Mapping[Key, tuple[Verdict, Verdict]],
    entailment: Entailment,
) -> tuple[Scored, ...]:
    """Combine every rung's result for each answer."""
    scored = []
    for answer in answers:
        key = (answer.setup, answer.question_id)
        if answer.declined:
            scored.append(
                Scored(
                    setup=answer.setup,
                    question_id=answer.question_id,
                    answer_type=answer.answer_type,
                    declined=True,
                    contains_reference=False,
                    lexical=None,
                    nli=None,
                    groundedness=None,
                    correctness=None,
                    groundedness_reason=None,
                    correctness_reason=None,
                    judge_cost=0.0,
                    judge_seconds=0.0,
                )
            )
            continue
        grounded, correct = judged[key]
        scored.append(
            Scored(
                setup=answer.setup,
                question_id=answer.question_id,
                answer_type=answer.answer_type,
                declined=False,
                contains_reference=contains_reference(answer.text, references[answer.question_id]),
                lexical=lexical_support(answer.question, answer.passages, answer.text),
                nli=entailment(answer.passages, answer.text),
                groundedness=grounded.label,
                correctness=correct.label,
                groundedness_reason=grounded.reason,
                correctness_reason=correct.reason,
                judge_cost=verdict_cost(grounded) + verdict_cost(correct),
                judge_seconds=grounded.seconds + correct.seconds,
            )
        )
    return tuple(scored)


NLI_SUPPORTED = 0.5
"""Entailment probability at or above which NLI counts an answer as supported (decision 66)."""

LEXICAL_SUPPORTED = 1.0
"""Word-overlap share at which it counts an answer as supported: every new word found."""


def summarise_setup(scored: Sequence[Scored]) -> dict[str, object]:
    """One setup's stage-two results.

    Correctness rates are over every question, with declined answers counted as not correct,
    because declining is the right call only when the answer truly is absent, and correctness
    alone cannot tell. Groundedness shares are over answered questions only.
    """
    answered = [s for s in scored if not s.declined]
    total = len(scored)
    labels = GROUNDEDNESS.labels

    def share(count: int, of: int) -> float:
        return count / of if of else 0.0

    return {
        "questions": total,
        "declined": total - len(answered),
        "decline_rate": share(total - len(answered), total),
        "correct_containment": share(sum(s.contains_reference for s in scored), total),
        "correct_judge": share(sum(s.correctness == "correct" for s in answered), total),
        "groundedness_judge": {
            label: share(sum(s.groundedness == label for s in answered), len(answered))
            for label in labels
        },
        "supported_nli": share(
            sum(s.nli is not None and s.nli >= NLI_SUPPORTED for s in answered), len(answered)
        ),
        "supported_lexical": share(
            sum(s.lexical is not None and s.lexical >= LEXICAL_SUPPORTED for s in answered),
            len(answered),
        ),
        "judge_cost": sum(s.judge_cost for s in scored),
    }


def baseline_agreement(scored: Sequence[Scored], *, seed: int) -> dict[str, object]:
    """How well each cheaper rung agrees with the judge on 'fully supported or not'.

    Reported as kappa at the cut-offs fixed in advance, and as ROC AUC, which needs no cut-off.
    """
    answered = [s for s in scored if not s.declined]
    judge_supported = [s.groundedness == "supported" for s in answered]
    judge_labels = ["supported" if flag else "not" for flag in judge_supported]
    result: dict[str, object] = {"answers": len(answered)}
    for name, values, cut in (
        ("nli", [s.nli or 0.0 for s in answered], NLI_SUPPORTED),
        ("lexical", [s.lexical or 0.0 for s in answered], LEXICAL_SUPPORTED),
    ):
        labels = ["supported" if value >= cut else "not" for value in values]
        try:
            measured = agreement(judge_labels, labels, seed=seed)
            auc = roc_auc(values, judge_supported)
        except ValueError as error:
            # Too few answers, or one label throughout: say so rather than report a number.
            result[name] = {"cut_off": cut, "undefined": str(error)}
            continue
        result[name] = {
            "cut_off": cut,
            "kappa": measured.kappa,
            "kappa_interval": [measured.kappa_low, measured.kappa_high],
            "observed": measured.observed,
            "confusion": measured.confusion,
            "auc": auc,
        }
    return result
