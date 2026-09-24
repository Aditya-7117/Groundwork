"""The golden set: a small slice of the corpus the CI regression gate searches on every push.

CI cannot download and rebuild the multi-gigabyte corpus on each push, so a slice is committed
(decision 56): 100 questions drawn with a fixed seed in the full set's paragraph, table and list
proportions, their own pages, and 200 more pages drawn at random as distractors, pages that
answer none of the questions but make finding the right passage harder. Records are copied line
for line from the built corpus, so the slice is loaded by exactly the same code as the full set.

The pages are Wikipedia text, shared under CC BY-SA 3.0; the slice carries that notice.
"""

import gzip
import json
import logging
import random
from pathlib import Path

from groundwork.answering import stratified_sample
from groundwork.evaluation import load_built_corpus

logger = logging.getLogger(__name__)

LICENCE = (
    "Page text from English Wikipedia via the Natural Questions dataset (Google), shared under "
    "CC BY-SA 3.0. Questions and annotations from Natural Questions, CC BY-SA 3.0."
)


def build_golden(
    built_dir: Path, out_dir: Path, *, questions: int, distractors: int, seed: int
) -> dict[str, object]:
    """Write the golden slice of a built corpus and return its description.

    Raises:
        ValueError: If there are too few pages for the distractors asked for.
    """
    full = load_built_corpus(built_dir)
    chosen = stratified_sample(full.questions, questions, seed=seed)
    chosen_ids = {question.question_id for question in chosen}
    answer_pages = {page_id for question in chosen for page_id in question.page_relevance}
    others = sorted(page.page_id for page in full.pages if page.page_id not in answer_pages)
    if len(others) < distractors:
        raise ValueError(f"only {len(others)} other pages, cannot draw {distractors} distractors")
    # A seeded, reproducible sample is the point here; S311 concerns cryptographic randomness.
    keep_pages = answer_pages | set(random.Random(seed).sample(others, distractors))  # noqa: S311

    out_dir.mkdir(parents=True, exist_ok=True)
    _filter(built_dir / "pages.jsonl.gz", out_dir / "pages.jsonl.gz", "page_id", keep_pages)
    _filter(
        built_dir / "questions.jsonl.gz", out_dir / "questions.jsonl.gz", "question_id", chosen_ids
    )
    description: dict[str, object] = {
        "questions": len(chosen),
        "answer_pages": len(answer_pages),
        "distractor_pages": distractors,
        "seed": seed,
        "selection": "questions stratified by answer type; distractors uniform over other pages",
        "by_answer_type": {
            kind: sum(q.answer_type == kind for q in chosen)
            for kind in sorted({q.answer_type for q in chosen})
        },
        "licence": LICENCE,
    }
    meta = {**full.meta, "golden": description}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    logger.info("golden set written", extra={"path": str(out_dir), **description})
    return description


def _filter(source: Path, target: Path, field: str, keep: set[str]) -> None:
    """Copy the JSON Lines records whose field is in keep, byte for byte, in source order."""
    with gzip.open(source, "rt", encoding="utf-8") as reader:
        lines = [line for line in reader if json.loads(line)[field] in keep]
    # A fixed modification time keeps the compressed file identical across rebuilds.
    with (
        target.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as writer,
    ):
        writer.write("".join(lines).encode("utf-8"))
