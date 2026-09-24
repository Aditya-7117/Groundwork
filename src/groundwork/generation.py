"""Writing answers from retrieved passages with a local model.

The writer is Qwen3.8-27B, served by Ollama on this machine (decision record 0008). The prompt is
fixed for every setup, so the only thing that changes between setups is what retrieval handed
over. The writer sees the top five chunks (decision 53) and must answer in one short sentence from
them alone, or say exactly "I don't know" (decision 57). Declining is counted separately, never
scored as a grounded answer, so a writer cannot look careful by guessing.

Thinking is switched off and temperature is 0 with a fixed seed, so the same prompt gives the same
answer. Every response is cached with its token counts and timings from when it was generated.
"""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, ValidationError

from groundwork.cache import ResponseCache, request_key
from groundwork.remote import JsonClient, RemoteError

logger = logging.getLogger(__name__)

CONTEXT_PASSAGES = 5
CONTEXT_TOKENS = 16_384
"""The context window the model was built with; the Modelfile sets the same value."""
OLLAMA_URL = "http://localhost:11434"
DECLINE = "I don't know"
SYSTEM_PROMPT = (
    "Answer the question using only the numbered passages.\n"
    "Reply with one short sentence.\n"
    "If the passages do not contain the answer, reply exactly:\n"
    f"{DECLINE}"
)

_NANOSECONDS = 1e9
_DECLINE_FORM = re.compile(r"i (?:don't|do not) know")


class GenerationError(RuntimeError):
    """The writer could not be reached, or replied with something unusable."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Generation:
    """One written answer and what it cost.

    Attributes:
        text: The answer.
        declined: Whether the writer said it did not know.
        prompt_tokens: Tokens read, as counted by the model's tokenizer.
        output_tokens: Tokens written.
        seconds: Wall-clock time the server spent, including reading the prompt.
        cached: Whether this came from the cache rather than a fresh call.
    """

    text: str
    declined: bool
    prompt_tokens: int
    output_tokens: int
    seconds: float
    cached: bool


class _Message(BaseModel):
    model_config = ConfigDict(extra="ignore")

    content: str


class _ChatReply(BaseModel):
    """The fields used from Ollama's /api/chat reply."""

    model_config = ConfigDict(extra="ignore")

    message: _Message
    prompt_eval_count: int = 0
    eval_count: int = 0
    total_duration: int = 0
    done_reason: str = ""


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    digest: str


class _Tags(BaseModel):
    model_config = ConfigDict(extra="ignore")

    models: tuple[_Model, ...]


def build_messages(question: str, passages: Sequence[str]) -> list[dict[str, str]]:
    """Return the chat messages for one question: the fixed instruction, then numbered passages."""
    numbered = "\n\n".join(f"[{index}] {text}" for index, text in enumerate(passages, start=1))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Passages:\n{numbered}\n\nQuestion: {question}"},
    ]


def is_declined(answer: str) -> bool:
    """Return whether an answer is the decline sentence, ignoring case, quotes and a full stop."""
    curly = "\N{RIGHT SINGLE QUOTATION MARK}"
    normalised = answer.strip().strip("\"'.").replace(curly, "'").lower()
    return _DECLINE_FORM.fullmatch(normalised) is not None


class OllamaWriter:
    """Answers questions with a model served by a local Ollama server."""

    def __init__(
        self,
        *,
        model: str,
        cache: ResponseCache,
        seed: int,
        client: JsonClient | None = None,
    ) -> None:
        """Look up the model's digest, which pins the exact weights every answer came from.

        Raises:
            GenerationError: If the server is unreachable or does not have the model.
        """
        self._model = model
        self._cache = cache
        self._seed = seed
        self._client = client or JsonClient()
        self.digest = self._find_digest()

    def write(self, question: str, passages: Sequence[str]) -> Generation:
        """Answer one question from its passages.

        Raises:
            GenerationError: If the call fails or the reply cannot be read.
        """
        payload: dict[str, object] = {
            "model": self._model,
            "messages": build_messages(question, passages),
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "seed": self._seed,
                "num_ctx": CONTEXT_TOKENS,
            },
        }
        key = request_key({"digest": self.digest, **payload})
        response = self._cache.get(key)
        cached = response is not None
        if response is None:
            try:
                response = self._client.request(
                    f"{OLLAMA_URL}/api/chat", {**payload, "keep_alive": "30m"}
                )
            except RemoteError as error:
                raise GenerationError(str(error)) from error
            self._cache.put(key, response)
        reply = _parse(_ChatReply, response)
        if reply.done_reason and reply.done_reason != "stop":
            logger.warning("answer did not finish cleanly", extra={"reason": reply.done_reason})
        text = reply.message.content.strip()
        return Generation(
            text=text,
            declined=is_declined(text),
            prompt_tokens=reply.prompt_eval_count,
            output_tokens=reply.eval_count,
            seconds=reply.total_duration / _NANOSECONDS,
            cached=cached,
        )

    def _find_digest(self) -> str:
        try:
            raw = self._client.request(f"{OLLAMA_URL}/api/tags")
        except RemoteError as error:
            raise GenerationError(f"the Ollama server is not reachable: {error}") from error
        tags = _parse(_Tags, raw)
        wanted = {self._model, f"{self._model}:latest"}
        for model in tags.models:
            if model.name in wanted:
                return model.digest
        available = ", ".join(sorted(model.name for model in tags.models))
        raise GenerationError(f"Ollama has no model {self._model!r}; it has: {available}")


def _parse[T: BaseModel](model: type[T], raw: dict[str, object]) -> T:
    try:
        return model.model_validate(raw)
    except ValidationError as error:
        raise GenerationError(f"unexpected reply from Ollama: {error}") from error
