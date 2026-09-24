"""The cheaper measures the judge is compared against, and the mechanical correctness check.

The groundedness ladder runs from cheapest to dearest: word overlap, an NLI classifier, the
language-model judge, and hand labels (decision 54). Each rung is only worth its cost if it
agrees with the humans better than the rung below, which is what the agreement statistics
measure.

- Word overlap: which of the answer's own content words, the ones it adds beyond the question,
  appear in the passages. Restating the question proves nothing, since the passages were
  retrieved for it; the new words are where the claim is.
- NLI: a classifier trained to decide whether a premise entails a hypothesis. Each passage is the
  premise and the answer the hypothesis; the strongest entailment across the passages counts,
  because the answer only has to follow from one of them (decision 55).
- Containment: whether the answer contains a reference answer after the standard normalisation
  (lowercase, punctuation and articles removed, whitespace collapsed), the rule SQuAD and
  Natural Questions evaluations use.
"""

import re
import string
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from groundwork.bm25 import tokenize

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = str.maketrans("", "", string.punctuation)

# Function words carry no claim. Kept short and explicit so the baseline is easy to audit.
_FUNCTION_WORDS = frozenset(
    tokenize(
        "a an the and or but of in on at to for from by with as is are was were be been being "
        "it its this that these those he she they them his her their there here which who whom "
        "whose what when where why how do does did has have had not no yes so than then also "
        "about into over after before during i you we us our",
        stem=True,
    )
)


@dataclass(frozen=True, slots=True, kw_only=True)
class NliModel:
    """The NLI classifier, pinned to one published revision."""

    name: str
    revision: str
    max_length: int
    entailment: int
    contradiction: int


NLI_MODEL = NliModel(
    name="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    revision="b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7",
    max_length=512,
    entailment=0,
    contradiction=2,
)


def normalise_answer(text: str) -> str:
    """Lowercase, drop punctuation and articles, and collapse whitespace."""
    lowered = text.lower().translate(_PUNCTUATION)
    return " ".join(_ARTICLES.sub(" ", lowered).split())


def contains_reference(answer: str, references: Sequence[str]) -> bool:
    """Whether the normalised answer contains any normalised reference as whole words."""
    padded = f" {normalise_answer(answer)} "
    return any(
        f" {normalise_answer(reference)} " in padded
        for reference in references
        if normalise_answer(reference)
    )


def lexical_support(question: str, passages: Sequence[str], answer: str) -> float:
    """The share of the answer's new content words that appear in the passages.

    New content words are the answer's stemmed words, less function words and less the words of
    the question. An answer that adds nothing beyond the question scores 1.0: it claims nothing
    the passages could fail to support.
    """
    asked = set(tokenize(question, stem=True))
    new = set(tokenize(answer, stem=True)) - asked - _FUNCTION_WORDS
    if not new:
        return 1.0
    available = set(tokenize(" ".join(passages), stem=True))
    return len(new & available) / len(new)


class NliChecker:
    """Scores how strongly any passage entails the answer."""

    def __init__(self, *, cache_dir: Path, batch_size: int = 16) -> None:
        """Load the pinned classifier.

        It runs in full precision: it scores only a few thousand answers, so half precision
        would save minutes while adding a source of numerical doubt.
        """
        import torch  # noqa: PLC0415 -- heavy import, only paid when NLI is used
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.precision = "float32"
        self._model = CrossEncoder(
            NLI_MODEL.name,
            revision=NLI_MODEL.revision,
            device=self.device,
            max_length=NLI_MODEL.max_length,
            cache_folder=str(cache_dir),
        )
        self._batch_size = batch_size

    def probabilities(self, premises: Sequence[str], hypothesis: str) -> NDArray[np.float64]:
        """Return (entailment, neutral, contradiction) probabilities for each premise."""
        pairs = [(premise, hypothesis) for premise in premises]
        scores = self._model.predict(
            pairs, batch_size=self._batch_size, apply_softmax=True, show_progress_bar=False
        )
        return np.asarray(scores, dtype=np.float64)


def strongest_entailment(probabilities: NDArray[np.float64]) -> float:
    """The highest entailment probability across passages, or 0.0 with no passages."""
    if probabilities.size == 0:
        return 0.0
    return float(probabilities[:, NLI_MODEL.entailment].max())
