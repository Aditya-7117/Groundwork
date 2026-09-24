"""The report: every published number, assembled from the run artefacts and nothing else.

The report reads the newest finished run of each grid setup, the stage-two verdicts and the hand
labels, and writes one JSON document and one Markdown summary. It computes nothing a run did not
record, apart from the comparisons between runs: significance tests (decision 59), agreement with
the hand labels, and the table ablation. Each figure carries where it came from, so any number in
the README can be traced to an artefact.
"""

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

from groundwork.agreement import agreement, roc_auc
from groundwork.answering import latest_run
from groundwork.judging import GROUNDEDNESS
from groundwork.significance import compare
from groundwork.verdicts import LEXICAL_SUPPORTED, NLI_SUPPORTED

logger = logging.getLogger(__name__)

PRIMARY = "passage.ndcg@10"
"""The metric that picks the winner and the stage-two setups (decision 62)."""

TESTED = ("passage.ndcg@10", "passage.recall@10")
"""The metrics every significance family is tested on (decision 59)."""

TABLE_COLUMNS = (
    "passage.recall@10",
    "passage.ndcg@10",
    "passage.mrr@10",
    "passage.recall@100",
    "page.recall@10",
)

type Document = dict[str, object]


class ReportError(ValueError):
    """An artefact the report needs is missing or inconsistent."""


def load_runs(results_dir: Path, setups: Sequence[str]) -> dict[str, Document]:
    """Read the newest finished run of each setup.

    Raises:
        ReportError: If a setup has no run, or runs were made from different commits.
    """
    runs: dict[str, Document] = {}
    for setup in setups:
        directory = latest_run(results_dir, setup)
        document = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        document["path"] = directory.as_posix()
        runs[setup] = document
    commits = {str(_get(run, "code", "git_commit")) for run in runs.values()}
    if len(commits) > 1:
        raise ReportError(
            f"runs come from {len(commits)} different commits; re-run so one commit made them all"
        )
    return runs


def retrieval_table(runs: Mapping[str, Document]) -> list[dict[str, object]]:
    """One row per setup: headline metrics, recall@10 by answer type, cost and latency."""
    rows = []
    for setup, run in runs.items():
        aggregate = _mapping(_get(run, "metrics", "aggregate"))
        by_type = _mapping(_get(run, "metrics", "by_answer_type"))
        timing = _mapping(run["timing"])
        rerank = _mapping(timing.get("rerank_latency_ms", {}))
        stages = _mapping(timing["stage_seconds"])
        rows.append(
            {
                "setup": setup,
                **{metric: aggregate[metric] for metric in TABLE_COLUMNS},
                "recall@10_by_type": {
                    kind: _mapping(values)["passage.recall@10"] for kind, values in by_type.items()
                },
                "retrieval_p50_ms": _mapping(timing["retrieval_latency_ms"])["p50"],
                "rerank_p50_ms": rerank.get("p50"),
                "embedding_seconds": stages.get("embedding"),
                "chunks": _get(run, "corpus", "chunks"),
                "path": run["path"],
            }
        )
    return sorted(rows, key=lambda row: (-float(str(row[PRIMARY])), str(row["setup"])))


def significance(runs: Mapping[str, Document], *, seed: int) -> dict[str, object]:
    """The two families of decision 59, each tested on nDCG@10 and recall@10 with Holm."""
    per_question = {setup: _per_question(run) for setup, run in runs.items()}
    winner = retrieval_table(runs)[0]["setup"]
    against_winner = [(str(winner), setup) for setup in sorted(runs) if setup != winner]
    reranker = [
        (f"{setup}-rerank", setup)
        for setup in sorted(runs)
        if not setup.endswith("-rerank") and f"{setup}-rerank" in runs
    ]
    families: dict[str, object] = {"winner": winner}
    for name, pairs in (("winner_against_each", against_winner), ("reranker", reranker)):
        families[name] = {
            metric: [asdict(row) for row in compare(per_question, pairs, metric, seed=seed)]
            for metric in TESTED
            if pairs
        }
    return families


def table_ablation(runs: Mapping[str, Document], winner: str) -> dict[str, object] | None:
    """The winner against itself with tables flattened, on table questions (decision 40)."""
    flat = f"{winner}-flat-tables"
    if flat not in runs:
        return None
    per_question = {setup: _per_question(runs[setup]) for setup in (winner, flat)}
    tables = _table_questions(runs[winner])
    subset = {
        setup: {question: values for question, values in rows.items() if question in tables}
        for setup, rows in per_question.items()
    }
    return {
        "questions": len(tables),
        "comparisons": {
            metric: [asdict(row) for row in compare(subset, [(winner, flat)], metric, seed=1)]
            for metric in TESTED
        },
    }


def human_agreement(scored_path: Path, labels_path: Path, *, seed: int) -> dict[str, object]:
    """Agreement of the judge and each cheaper rung with the hand labels.

    The judge is compared on the full three-level scale. The cheaper rungs give scores, so they are
    compared on "fully supported or not", at the cut-offs fixed in advance and by ROC AUC.

    Raises:
        ReportError: If a label has no scored answer.
    """
    scored = {}
    for line in scored_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        scored[(row["setup"], row["question_id"])] = row
    labels = [json.loads(line) for line in labels_path.read_text(encoding="utf-8").splitlines()]
    rows = []
    for label in labels:
        key = (label["setup"], label["question_id"])
        if key not in scored:
            raise ReportError(f"label for {key} has no scored answer")
        rows.append((label["label"], scored[key]))
    human = [label for label, _ in rows]
    human_supported = [label == "supported" for label in human]
    result: dict[str, object] = {"labels": len(rows), "scale": list(GROUNDEDNESS.labels)}
    judge = agreement(human, [row["groundedness"] for _, row in rows], seed=seed)
    result["judge"] = asdict(judge)
    for name, cut in (("nli", NLI_SUPPORTED), ("lexical", LEXICAL_SUPPORTED)):
        values = [float(row[name]) for _, row in rows]
        binary = ["supported" if value >= cut else "not" for value in values]
        result[name] = {
            "cut_off": cut,
            **asdict(
                agreement(
                    ["supported" if flag else "not" for flag in human_supported], binary, seed=seed
                )
            ),
            "auc": roc_auc(values, human_supported),
        }
    return result


def markdown(report: Mapping[str, object]) -> str:
    """The report's headline tables as Markdown, for the README."""
    lines = [
        "| Setup | Recall@10 | nDCG@10 | MRR@10 | Recall@100 | Page recall@10 |",
        "|---|---|---|---|---|---|",
    ]
    for row in _rows(report["retrieval"]):
        lines.append(
            f"| {row['setup']} | "
            + " | ".join(f"{float(str(row[metric])):.3f}" for metric in TABLE_COLUMNS)
            + " |"
        )
    return "\n".join(lines) + "\n"


def _per_question(run: Document) -> dict[str, dict[str, float]]:
    rows = _mapping(_get(run, "metrics", "per_question"))
    return {
        question: {name: float(str(value)) for name, value in _mapping(values).items()}
        for question, values in rows.items()
    }


def _table_questions(run: Document) -> set[str]:
    by_question = _mapping(_get(run, "questions", "answer_types"))
    return {question for question, kind in by_question.items() if kind == "table"}


def _get(document: Mapping[str, object], *keys: str) -> object:
    value: object = document
    for key in keys:
        value = _mapping(value)[key]
    return value


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReportError(f"expected an object, got {type(value).__name__}")
    return value


def _rows(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        raise ReportError("expected a list of rows")
    return [_mapping(row) for row in value]


def build_report(
    results_dir: Path,
    setups: Sequence[str],
    *,
    verdicts_dir: Path | None,
    labels_path: Path | None,
    seed: int,
) -> dict[str, object]:
    """Assemble every section the artefacts allow; later stages are left out until they exist.

    Raises:
        ReportError: If an artefact is missing or the runs come from different commits.
    """
    runs = load_runs(results_dir, setups)
    retrieval = retrieval_table(runs)
    winner = str(retrieval[0]["setup"])
    report: dict[str, object] = {
        "primary_metric": PRIMARY,
        "commit": _get(next(iter(runs.values())), "code", "git_commit"),
        "retrieval": retrieval,
        "significance": significance(runs, seed=seed),
    }
    ablation_runs = dict(runs)
    flat = f"{winner}-flat-tables"
    if (results_dir / flat).is_dir():
        ablation_runs.update(load_runs(results_dir, [flat]))
    report["table_ablation"] = table_ablation(ablation_runs, winner)
    if verdicts_dir is not None:
        verdicts = json.loads((verdicts_dir / "result.json").read_text(encoding="utf-8"))
        report["stage_two"] = {
            "path": verdicts_dir.as_posix(),
            "setups": verdicts["setups"],
            "agreement_with_judge": verdicts["agreement_with_judge"],
            "judge": {key: verdicts["judge"][key] for key in ("model", "thinking", "spent")},
        }
        if labels_path is not None:
            report["human_agreement"] = human_agreement(
                verdicts_dir / "scored.jsonl", labels_path, seed=seed
            )
    return report
