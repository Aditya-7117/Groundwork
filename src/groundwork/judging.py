"""The language-model judge: Gemini labels each answer's groundedness and correctness.

Groundedness asks whether everything the answer says is backed by the passages it was written
from, whatever the truth of the matter; correctness asks whether it matches the reference answer.
They are separate calls with separate rubrics, and the groundedness judge never sees the
reference, so it cannot mark an answer grounded merely because it is right.

The judge is from a different model family than the writer (decision record 0008), because a
model grades its own family's text more kindly. Its labels are themselves measured: against a
word-overlap baseline, an NLI classifier and 200 hand labels (decision 54).

Gemini 3 models are documented to work best at their default temperature, so none is set. The
same request can therefore get a different label on a fresh call; the response cache is what
makes a published verdict reproducible.
"""

import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from groundwork.cache import ResponseCache, request_key
from groundwork.remote import JsonClient, RemoteError

logger = logging.getLogger(__name__)

JUDGE_MODEL = "gemini-3.8-flash"
API_KEY_VARIABLE = "GEMINI_API_KEY"
_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

type ThinkingLevel = Literal["low", "medium", "high"]


class JudgeError(RuntimeError):
    """The judge could not be called, or answered outside its rubric."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Rubric:
    """What the judge is asked, and the labels it may answer with.

    Attributes:
        name: Short id, recorded with every verdict.
        instruction: The system instruction.
        labels: The only labels accepted, in order from best to worst.
    """

    name: str
    instruction: str
    labels: tuple[str, ...]


GROUNDEDNESS = Rubric(
    name="groundedness",
    instruction=(
        "You check whether an answer is supported by the passages it was written from.\n"
        "Judge only against the passages, never against your own knowledge: a true statement\n"
        "that the passages do not contain is not supported.\n\n"
        "Labels:\n"
        "- supported: every claim in the answer is stated in, or directly follows from, the\n"
        "  passages.\n"
        "- partly_supported: the main claim is supported, but the answer adds a detail the\n"
        "  passages do not support.\n"
        "- not_supported: the main claim is missing from the passages or contradicts them.\n\n"
        "Give a one-sentence reason, then the label."
    ),
    labels=("supported", "partly_supported", "not_supported"),
)

CORRECTNESS = Rubric(
    name="correctness",
    instruction=(
        "You check whether an answer to a question is correct, by comparing it with reference\n"
        "answers written by human annotators. Any one reference is enough. Wording, spelling\n"
        "and level of detail may differ; what matters is whether the answer gives the same\n"
        "fact.\n\n"
        "Labels:\n"
        "- correct: the answer gives the fact in a reference answer.\n"
        "- incorrect: it gives a different fact, or none.\n\n"
        "Give a one-sentence reason, then the label."
    ),
    labels=("correct", "incorrect"),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Verdict:
    """One judgement and what it cost.

    Attributes:
        rubric: Which rubric was applied.
        label: One of the rubric's labels.
        reason: The judge's one-sentence reason.
        prompt_tokens: Tokens read.
        output_tokens: Tokens in the visible reply.
        thinking_tokens: Tokens spent thinking, billed as output.
        seconds: Wall-clock time of the call, including any retries.
        model_version: The exact model version the API reported.
        cached: Whether this came from the cache.
    """

    rubric: str
    label: str
    reason: str
    prompt_tokens: int
    output_tokens: int
    thinking_tokens: int
    seconds: float
    model_version: str
    cached: bool


class _Part(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str = ""
    thought: bool = False


class _Content(BaseModel):
    model_config = ConfigDict(extra="ignore")

    parts: tuple[_Part, ...] = ()


class _Candidate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    content: _Content
    finishReason: str = ""  # noqa: N815 -- the API's field name


class _Usage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    promptTokenCount: int = 0  # noqa: N815
    candidatesTokenCount: int = 0  # noqa: N815
    thoughtsTokenCount: int = 0  # noqa: N815


class _Response(BaseModel):
    model_config = ConfigDict(extra="ignore")

    candidates: tuple[_Candidate, ...]
    usageMetadata: _Usage = _Usage()  # noqa: N815
    modelVersion: str = ""  # noqa: N815


class _Label(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str
    label: str


def groundedness_prompt(question: str, passages: Sequence[str], answer: str) -> str:
    """The user message for a groundedness judgement: passages, question, answer, no reference."""
    numbered = "\n\n".join(f"[{index}] {text}" for index, text in enumerate(passages, start=1))
    return f"Passages:\n{numbered}\n\nQuestion: {question}\n\nAnswer: {answer}"


def correctness_prompt(question: str, references: Sequence[str], answer: str) -> str:
    """The user message for a correctness judgement: question, references, answer."""
    listed = "\n".join(f"- {reference}" for reference in references)
    return f"Question: {question}\n\nReference answers:\n{listed}\n\nAnswer: {answer}"


class GeminiJudge:
    """Applies a rubric through the Gemini API, one cached call per judgement."""

    def __init__(
        self,
        *,
        cache: ResponseCache,
        thinking: ThinkingLevel,
        model: str = JUDGE_MODEL,
        api_key: str | None = None,
        client: JsonClient | None = None,
    ) -> None:
        """Set the model and thinking level, and find the API key.

        Raises:
            JudgeError: If no key is given and GEMINI_API_KEY is not set.
        """
        key = api_key or os.environ.get(API_KEY_VARIABLE)
        if not key:
            raise JudgeError(f"set {API_KEY_VARIABLE} (see .env.example) to call the judge")
        self._key = key
        self.model = model
        self._cache = cache
        self.thinking: ThinkingLevel = thinking
        self._client = client or JsonClient()

    def judge(self, rubric: Rubric, prompt: str) -> Verdict:
        """Apply one rubric to one prompt.

        Raises:
            JudgeError: If the call fails, or the reply is not one of the rubric's labels.
        """
        payload: dict[str, object] = {
            "systemInstruction": {"parts": [{"text": rubric.instruction}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "reason": {"type": "STRING"},
                        "label": {"type": "STRING", "enum": list(rubric.labels)},
                    },
                    "required": ["reason", "label"],
                    "propertyOrdering": ["reason", "label"],
                },
                "thinkingConfig": {"thinkingLevel": self.thinking},
            },
        }
        key = request_key({"model": self.model, **payload})
        stored = self._cache.get(key)
        cached = stored is not None
        if stored is None:
            started = time.perf_counter()
            try:
                raw = self._client.request(
                    _ENDPOINT.format(model=self.model),
                    payload,
                    headers={"x-goog-api-key": self._key},
                )
            except RemoteError as error:
                raise JudgeError(str(error)) from error
            stored = {"reply": raw, "seconds": time.perf_counter() - started}
            self._cache.put(key, stored)
        return _verdict(rubric, stored, cached=cached)


def _verdict(rubric: Rubric, stored: dict[str, object], *, cached: bool) -> Verdict:
    try:
        response = _Response.model_validate(stored["reply"])
        [candidate] = response.candidates
        text = "".join(part.text for part in candidate.content.parts if not part.thought)
        parsed = _Label.model_validate_json(text)
    except (ValidationError, ValueError, KeyError) as error:
        raise JudgeError(f"unreadable judge reply for {rubric.name}: {error}") from error
    if parsed.label not in rubric.labels:
        raise JudgeError(f"label {parsed.label!r} is not in the {rubric.name} rubric")
    seconds = stored.get("seconds", 0.0)
    return Verdict(
        rubric=rubric.name,
        label=parsed.label,
        reason=parsed.reason,
        prompt_tokens=response.usageMetadata.promptTokenCount,
        output_tokens=response.usageMetadata.candidatesTokenCount,
        thinking_tokens=response.usageMetadata.thoughtsTokenCount,
        seconds=float(seconds) if isinstance(seconds, int | float) else 0.0,
        model_version=response.modelVersion,
        cached=cached,
    )
