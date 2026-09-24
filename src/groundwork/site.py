"""The explorer's data: the report and the run artefacts, reshaped into files a browser can load.

The same files serve both ways the explorer runs: as a static site anyone can open, and behind
the local server that adds live search. Nothing here computes a new number; it only selects and
reshapes what the report and the runs recorded, so the explorer can never disagree with them.

Files, under the output directory:
- report.json: the report, unchanged.
- questions.json: each question's text, answer type, reference answers and page title.
- setups/<setup>.json: one setup's aggregate metrics and, per question, its metrics and top ten
  passages, each with its heading and whether it holds the answer.
- stage2/<setup>.json: that setup's answers, the passages each was written from, and every
  rung's verdict.
- judge.json: agreement between the rungs, and every answer where the judge and the hand label
  disagree, for the disagreement browser.
"""

import json
import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from groundwork.answering import Answer, load_answers, read_ranking
from groundwork.chunking import Chunk, ChunkingSettings, chunk_pages
from groundwork.config import ExperimentConfig
from groundwork.evaluation import EvaluationSet, chunk_relevance

logger = logging.getLogger(__name__)

TOP = 10
"""Passages shown per question in the drill-down."""


@dataclass(frozen=True, slots=True, kw_only=True)
class StageTwoInputs:
    """Where stage two's outputs are, once they exist.

    Attributes:
        answers_dir: The stage-two answers run.
        verdicts_dir: Its verdicts run.
        labels_path: The hand labels, or None before labelling.
    """

    answers_dir: Path
    verdicts_dir: Path
    labels_path: Path | None


def export_site(
    report: Mapping[str, object],
    run_dirs: Mapping[str, Path],
    evaluation_set: EvaluationSet,
    out_dir: Path,
    *,
    stage_two: StageTwoInputs | None = None,
) -> None:
    """Write every data file the explorer reads.

    Args:
        report: The assembled report.
        run_dirs: Each grid setup's run directory.
        evaluation_set: The corpus the runs were made on.
        out_dir: Where the files go.
        stage_two: Stage two's outputs, once it has run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "report.json", report)
    pages = {page.page_id: page.title for page in evaluation_set.pages}
    _write(
        out_dir / "questions.json",
        {
            q.question_id: {
                "question": q.text,
                "type": q.answer_type,
                "references": list(q.short_answers),
                "page": pages.get(next(iter(q.page_relevance), ""), ""),
            }
            for q in evaluation_set.questions
        },
    )
    chunks_by_settings: dict[ChunkingSettings, dict[str, Chunk]] = {}
    for setup, run_dir in sorted(run_dirs.items()):
        document = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        config = ExperimentConfig.model_validate(document["experiment"]["config"])
        settings = ChunkingSettings(
            strategy=config.chunking.strategy,
            size=config.chunking.size,
            overlap=config.chunking.overlap,
            flatten_tables=config.chunking.flatten_tables,
        )
        if settings not in chunks_by_settings:
            chunks_by_settings[settings] = {
                chunk.chunk_id: chunk for chunk in chunk_pages(evaluation_set.pages, settings)
            }
        _write(
            out_dir / "setups" / f"{setup}.json",
            setup_file(
                document, read_ranking(run_dir, TOP), chunks_by_settings[settings], evaluation_set
            ),
        )
        logger.info("setup exported", extra={"setup": setup})
    if stage_two is not None:
        answers = load_answers(stage_two.answers_dir / "answers.jsonl")
        scored = _scored(stage_two.verdicts_dir / "scored.jsonl")
        for setup, rows in stage_two_files(answers, scored).items():
            _write(out_dir / "stage2" / f"{setup}.json", rows)
        verdicts = json.loads((stage_two.verdicts_dir / "result.json").read_text(encoding="utf-8"))
        labels = _labels(stage_two.labels_path) if stage_two.labels_path is not None else {}
        _write(
            out_dir / "judge.json",
            {
                "agreement_with_judge": verdicts["agreement_with_judge"],
                "human_agreement": report.get("human_agreement"),
                "disagreements": disagreements(answers, scored, labels),
            },
        )


def setup_file(
    document: Mapping[str, object],
    ranking: Mapping[str, Sequence[str]],
    chunks: Mapping[str, Chunk],
    evaluation_set: EvaluationSet,
) -> dict[str, object]:
    """One setup's metrics and, per question, its top passages with headings and relevance."""
    metrics = _mapping(document["metrics"])
    per_question = _mapping(metrics["per_question"])
    chunks_by_page: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks.values():
        chunks_by_page[chunk.page_id].append(chunk)
    questions: dict[str, object] = {}
    for question in evaluation_set.questions:
        if question.question_id not in per_question:
            continue
        relevant = chunk_relevance(question, chunks_by_page)
        questions[question.question_id] = {
            "metrics": per_question[question.question_id],
            "top": [
                {
                    "id": chunk_id,
                    "heading": chunks[chunk_id].text.partition("\n")[0],
                    "kind": chunks[chunk_id].kind,
                    "relevant": chunk_id in relevant,
                }
                for chunk_id in ranking.get(question.question_id, ())
            ],
        }
    experiment = _mapping(document["experiment"])
    return {
        "setup": experiment["name"],
        "description": experiment["description"],
        "config": experiment["config"],
        "aggregate": metrics["aggregate"],
        "by_answer_type": metrics["by_answer_type"],
        "timing": document["timing"],
        "models": document.get("models", {}),
        "chunks": _mapping(document["corpus"])["chunks"],
        "questions": questions,
    }


def stage_two_files(
    answers: Sequence[Answer], scored: Mapping[tuple[str, str], Mapping[str, object]]
) -> dict[str, list[dict[str, object]]]:
    """Each setup's answers with the passages they were written from and every verdict."""
    files: dict[str, list[dict[str, object]]] = defaultdict(list)
    for answer in answers:
        verdict = scored.get((answer.setup, answer.question_id), {})
        files[answer.setup].append(
            {
                "question_id": answer.question_id,
                "answer": answer.text,
                "declined": answer.declined,
                "passages": list(answer.passages),
                "chunk_ids": list(answer.chunk_ids),
                "verdicts": {
                    key: verdict.get(key)
                    for key in (
                        "groundedness",
                        "groundedness_reason",
                        "correctness",
                        "correctness_reason",
                        "contains_reference",
                        "nli",
                        "lexical",
                    )
                },
            }
        )
    return dict(files)


def disagreements(
    answers: Sequence[Answer],
    scored: Mapping[tuple[str, str], Mapping[str, object]],
    labels: Mapping[tuple[str, str], str],
) -> list[dict[str, object]]:
    """Every hand-labelled answer whose judge label differs from the hand label."""
    by_key = {(answer.setup, answer.question_id): answer for answer in answers}
    rows = []
    for key, human in sorted(labels.items()):
        verdict = scored.get(key, {})
        if verdict.get("groundedness") == human or key not in by_key:
            continue
        answer = by_key[key]
        rows.append(
            {
                "setup": key[0],
                "question_id": key[1],
                "question": answer.question,
                "answer": answer.text,
                "passages": list(answer.passages),
                "human": human,
                "judge": verdict.get("groundedness"),
                "judge_reason": verdict.get("groundedness_reason"),
                "nli": verdict.get("nli"),
                "lexical": verdict.get("lexical"),
            }
        )
    return rows


def _scored(path: Path) -> dict[tuple[str, str], dict[str, object]]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows[(row["setup"], row["question_id"])] = row
    return rows


def _labels(path: Path) -> dict[tuple[str, str], str]:
    labels = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        labels[(row["setup"], row["question_id"])] = str(row["label"])
    return labels


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"expected an object, got {type(value).__name__}")
    return value


def _write(path: Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, separators=(",", ":")), encoding="utf-8")
