"""Splitting text into sentences, with the character positions of each one.

Wikipedia prose is full of abbreviations ("Dr.", "U.S.", "c. 1850") that a full-stop rule splits
in the wrong place, which would cut chunks mid-sentence and corrupt the sentence-aware strategy.
NLTK's Punkt splitter carries a learned list of abbreviations instead. Its English model is small
(about 0.24 MB) and is fetched once into the data directory, where its digest is recorded with
every run.
"""

import hashlib
import logging
import os
from functools import cache
from pathlib import Path

import nltk
from nltk.tokenize import PunktTokenizer

logger = logging.getLogger(__name__)

MODEL = "punkt_tab"
LANGUAGE = "english"


def data_dir() -> Path:
    """Where the sentence model lives. Override with the NLTK_DATA environment variable."""
    return Path(os.environ.get("NLTK_DATA", "data/nltk_data"))


def ensure_model(directory: Path | None = None) -> Path:
    """Download the sentence model unless it is already present, and return its directory."""
    target = directory or data_dir()
    english = target / "tokenizers" / MODEL / LANGUAGE
    if not english.is_dir():
        logger.info("downloading sentence model", extra={"model": MODEL, "path": str(target)})
        target.mkdir(parents=True, exist_ok=True)
        nltk.download(MODEL, download_dir=str(target), quiet=True)
    if str(target) not in nltk.data.path:
        nltk.data.path.insert(0, str(target))
    return english


def model_digest(directory: Path | None = None) -> str:
    """Return a SHA-256 over the sentence model's files, for the run record."""
    english = ensure_model(directory)
    digest = hashlib.sha256()
    for path in sorted(english.iterdir()):
        if path.is_file():
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


@cache
def _tokenizer() -> PunktTokenizer:
    ensure_model()
    return PunktTokenizer(LANGUAGE)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return the character span of each sentence in text, in order."""
    if not text.strip():
        return []
    return [(start, end) for start, end in _tokenizer().span_tokenize(text)]
