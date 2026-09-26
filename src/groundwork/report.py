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
from dataclasses import asdict, dataclass
from pathlib import Path

from groundwork.agreement import agreement, roc_auc
from groundwork.answering import latest_run, read_ranking
from groundwork.baselines import contains_reference
from groundwork.chunking import ChunkingSettings, chunk_pages
from groundwork.config import ExperimentConfig
from groundwork.evaluation import EvaluationSet
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

BASELINE = "fixed-bm25"
"""The keyword baseline every setup is compared with."""

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


def stage_two_setups(rows: Sequence[Mapping[str, object]], *, top: int = 3) -> tuple[str, ...]:
    """Pick the stage-two setups by the rule fixed before the grid ran (decision 62).

    The BM25 baseline, the lowest-scoring other setup as the weak reference, and the top three by
    the primary metric, in that order and without repeats.

    Raises:
        ReportError: If the baseline is missing from the rows.
    """
    ranked = sorted(rows, key=lambda row: (-float(str(row[PRIMARY])), str(row["setup"])))
    names = [str(row["setup"]) for row in ranked]
    if BASELINE not in names:
        raise ReportError(f"the baseline {BASELINE} has no run")
    weakest = next(name for name in reversed(names) if name != BASELINE)
    chosen = [BASELINE, weakest]
    for name in names:
        if len(chosen) == top + 2:
            break
        if name not in chosen:
            chosen.append(name)
    return tuple(chosen)


def run_commit(runs: Mapping[str, Document]) -> str:
    """The commit the runs were made from (load_runs guarantees there is only one)."""
    return str(_get(next(iter(runs.values())), "code", "git_commit"))


def stage_two_config_text(chosen: Sequence[str], scores: Mapping[str, float], commit: str) -> str:
    """The stage-two config for the chosen setups, with the rule and the scores that chose them."""
    listed = ", ".join(f"{name} {scores[name]:.3f}" for name in chosen)
    return "\n".join(
        [
            f"# Written by `groundwork stage-two-config` from the runs of commit {commit[:12]}.",
            "# The setups follow the rule fixed before the grid ran (decision 62): the BM25",
            "# baseline, the lowest-scoring other setup, and the top three by passage nDCG@10.",
            f"# nDCG@10: {listed}.",
            "",
            'name = "stage-two"',
            'description = "Answers and judgements for the baseline, the weak reference and the '
            'top three setups."',
            "seed = 1",
            "questions = 1000",
            "passages = 5",
            'writer = "qwen3.8-27b-iq4xs"',
            "setups = [" + ", ".join(f'"{name}"' for name in chosen) + "]",
            "",
            "[judge]",
            'model = "gpt-6-luna"',
            'thinking = "high"',
            "",
        ]
    )


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


ABLATION_METRICS = ("answer_hit@10", "answer_rr@10", "span_hit@10", "span_rr@10")
"""Table-ablation measures, primary first (decision 81). The answer-text measures ask whether a
top-10 passage contains a reference answer, which counts the same way for both table layouts.
The span measures ask whether a passage overlaps the answer's marked span; for a table answer that
span is often the whole table, and each layout splits a table into a different number of chunks,
so they are shown for comparison with the grid, not as the verdict."""


def answer_text_ranks(
    run_dir: Path, evaluation_set: EvaluationSet, question_ids: set[str]
) -> dict[str, int | None]:
    """For each question, the rank of the first top-10 passage containing a reference answer."""
    document = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    config = ExperimentConfig.model_validate(document["experiment"]["config"])
    settings = ChunkingSettings(
        strategy=config.chunking.strategy,
        size=config.chunking.size,
        overlap=config.chunking.overlap,
        flatten_tables=config.chunking.flatten_tables,
    )
    texts = {chunk.chunk_id: chunk.text for chunk in chunk_pages(evaluation_set.pages, settings)}
    ranking = read_ranking(run_dir, 10)
    references = {q.question_id: q.short_answers for q in evaluation_set.questions}
    ranks: dict[str, int | None] = {}
    for question_id in question_ids:
        ranks[question_id] = next(
            (
                rank
                for rank, chunk_id in enumerate(ranking.get(question_id, ()), start=1)
                # The body only: the "Title > Section" line is not retrieved content.
                if contains_reference(texts[chunk_id].partition("\n")[2], references[question_id])
            ),
            None,
        )
    return ranks


def table_ablation(
    runs: Mapping[str, Document],
    winner: str,
    run_dirs: Mapping[str, Path],
    evaluation_set: EvaluationSet,
) -> dict[str, object] | None:
    """The winner against itself with tables flattened, on table questions (decisions 40, 81)."""
    flat = f"{winner}-flat-tables"
    if flat not in runs:
        return None
    with_references = {q.question_id for q in evaluation_set.questions if q.short_answers}
    tables = _table_questions(runs[winner]) & with_references
    per_question: dict[str, dict[str, dict[str, float]]] = {}
    for setup in (winner, flat):
        spans = _per_question(runs[setup])
        ranks = answer_text_ranks(run_dirs[setup], evaluation_set, tables)
        per_question[setup] = {}
        for question in tables:
            rank = ranks[question]
            per_question[setup][question] = {
                "answer_hit@10": float(rank is not None),
                "answer_rr@10": 1 / rank if rank is not None else 0.0,
                "span_hit@10": float(spans[question]["passage.rr@10"] > 0),
                "span_rr@10": spans[question]["passage.rr@10"],
            }
    return {
        "questions": len(tables),
        "primary": list(ABLATION_METRICS[:2]),
        "comparisons": {
            metric: [asdict(row) for row in compare(per_question, [(flat, winner)], metric, seed=1)]
            for metric in ABLATION_METRICS
        },
    }


def table_writer(structured: Path, flattened: Path, winner: str, *, seed: int) -> dict[str, object]:
    """The writer's answers to the same table questions from structured and flattened tables.

    Correct and grounded count over every question, with declined answers as neither.

    Raises:
        ReportError: If the two runs share no answered question.
    """
    flat = f"{winner}-flat-tables"
    rows = {winner: _scored_rows(structured, winner), flat: _scored_rows(flattened, flat)}
    shared = rows[winner].keys() & rows[flat].keys()
    if not shared:
        raise ReportError("the structured and flattened runs share no question")

    def measures(row: Mapping[str, object]) -> dict[str, float]:
        return {
            "correct_judge": float(row.get("correctness") == "correct"),
            "correct_containment": float(bool(row.get("contains_reference"))),
            "declined": float(bool(row.get("declined"))),
            "supported": float(row.get("groundedness") == "supported"),
        }

    per_question = {
        setup: {question: measures(rows[setup][question]) for question in shared} for setup in rows
    }
    return {
        "questions": len(shared),
        "comparisons": {
            metric: [
                asdict(row) for row in compare(per_question, [(flat, winner)], metric, seed=seed)
            ]
            for metric in ("correct_judge", "correct_containment", "declined", "supported")
        },
    }


def _scored_rows(verdicts_dir: Path, setup: str) -> dict[str, dict[str, object]]:
    rows = {}
    for line in (verdicts_dir / "scored.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["setup"] == setup and row["answer_type"] == "table":
            rows[str(row["question_id"])] = row
    return rows


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


@dataclass(frozen=True, slots=True, kw_only=True)
class ReportInputs:
    """Where everything the report reads is.

    Attributes:
        results_dir: The runs.
        setups: The grid setups.
        evaluation_set: The corpus, for the table ablation's answer-text measures.
        verdicts_dir: The stage-two verdicts, once they exist.
        tables_verdicts_dir: The verdicts of the table-question follow-up (decision 80).
        labels_path: The hand labels, once they exist.
        seed: Seeds every significance test.
    """

    results_dir: Path
    setups: Sequence[str]
    evaluation_set: EvaluationSet
    verdicts_dir: Path | None = None
    tables_verdicts_dir: Path | None = None
    labels_path: Path | None = None
    seed: int = 1


def build_report(inputs: ReportInputs) -> dict[str, object]:
    """Assemble every section the artefacts allow; later stages are left out until they exist.

    Raises:
        ReportError: If an artefact is missing or the runs come from different commits.
    """
    runs = load_runs(inputs.results_dir, inputs.setups)
    retrieval = retrieval_table(runs)
    winner = str(retrieval[0]["setup"])
    report: dict[str, object] = {
        "primary_metric": PRIMARY,
        "commit": run_commit(runs),
        "retrieval": retrieval,
        "significance": significance(runs, seed=inputs.seed),
    }
    flat = f"{winner}-flat-tables"
    if (inputs.results_dir / flat).is_dir():
        # The ablation ran later, from its own commit; retrieval code was unchanged in between.
        ablation_runs = {**runs, **load_runs(inputs.results_dir, [flat])}
        run_dirs = {name: latest_run(inputs.results_dir, name) for name in (winner, flat)}
        ablation = table_ablation(ablation_runs, winner, run_dirs, inputs.evaluation_set)
        if ablation is not None:
            ablation["commit"] = run_commit({flat: ablation_runs[flat]})
        report["table_ablation"] = ablation
    if inputs.verdicts_dir is not None:
        verdicts = json.loads((inputs.verdicts_dir / "result.json").read_text(encoding="utf-8"))
        report["stage_two"] = {
            "path": inputs.verdicts_dir.as_posix(),
            "setups": verdicts["setups"],
            "agreement_with_judge": verdicts["agreement_with_judge"],
            "judge": {key: verdicts["judge"][key] for key in ("model", "thinking", "spent")},
        }
        if inputs.tables_verdicts_dir is not None:
            report["table_writer"] = table_writer(
                inputs.verdicts_dir, inputs.tables_verdicts_dir, winner, seed=inputs.seed
            )
        if inputs.labels_path is not None:
            report["human_agreement"] = human_agreement(
                inputs.verdicts_dir / "scored.jsonl", inputs.labels_path, seed=inputs.seed
            )
    return report
