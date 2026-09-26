"""The report, over small run artefacts written in the test."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from groundwork.config import load_stage_two_config
from groundwork.evaluation import EvaluationSet
from groundwork.report import (
    ReportError,
    answer_text_ranks,
    human_agreement,
    load_runs,
    markdown,
    retrieval_table,
    significance,
    stage_two_config_text,
    stage_two_setups,
    table_writer,
)
from runs import EVALUATION_SET, write_run


@pytest.fixture
def results(tmp_path: Path) -> Path:
    directory = tmp_path / "results"
    write_run(directory, "alpha", stem=True)
    write_run(directory, "alpha-rerank", stem=False)
    write_run(directory, "beta", stem=False)
    return directory


def test_the_table_is_sorted_by_the_primary_metric(results: Path) -> None:
    rows = retrieval_table(load_runs(results, ["alpha", "alpha-rerank", "beta"]))
    ndcg = [float(str(row["passage.ndcg@10"])) for row in rows]
    assert ndcg == sorted(ndcg, reverse=True)
    by_type = rows[0]["recall@10_by_type"]
    assert isinstance(by_type, dict)
    assert set(by_type) == {"paragraph", "table"}
    assert "| Setup | Recall@10" in markdown({"retrieval": rows})


def test_the_two_families_are_tested(results: Path) -> None:
    families = significance(load_runs(results, ["alpha", "alpha-rerank", "beta"]), seed=1)
    winner = families["winner"]
    against = families["winner_against_each"]
    assert isinstance(against, dict)
    assert {row["a"] for row in against["passage.ndcg@10"]} == {winner}
    assert len(against["passage.ndcg@10"]) == 2
    reranker = families["reranker"]
    assert isinstance(reranker, dict)
    assert [(row["a"], row["b"]) for row in reranker["passage.recall@10"]] == [
        ("alpha-rerank", "alpha")
    ]


def test_runs_from_different_commits_are_refused(tmp_path: Path) -> None:
    write_run(tmp_path, "alpha", stem=True, commit="a" * 40)
    write_run(tmp_path, "beta", stem=True, commit="b" * 40)
    with pytest.raises(ReportError, match="different commits"):
        load_runs(tmp_path, ["alpha", "beta"])


def test_agreement_with_the_hand_labels(tmp_path: Path) -> None:
    scored = tmp_path / "scored.jsonl"
    labels = tmp_path / "labels.jsonl"
    rows = [
        ("q1", "supported", 0.9, 1.0, "supported"),
        ("q2", "not_supported", 0.1, 0.0, "not_supported"),
        ("q3", "supported", 0.8, 1.0, "partly_supported"),
        ("q4", "partly_supported", 0.2, 0.5, "partly_supported"),
    ]
    scored.write_text(
        "".join(
            json.dumps(
                {"setup": "s", "question_id": q, "groundedness": judge, "nli": nli, "lexical": lex}
            )
            + "\n"
            for q, judge, nli, lex, _ in rows
        ),
        encoding="utf-8",
    )
    labels.write_text(
        "".join(
            json.dumps({"setup": "s", "question_id": q, "label": human}) + "\n"
            for q, _, _, _, human in rows
        ),
        encoding="utf-8",
    )
    result = human_agreement(scored, labels, seed=1)
    assert result["labels"] == 4
    judge = result["judge"]
    assert isinstance(judge, dict)
    assert judge["observed"] == pytest.approx(0.75)
    nli = result["nli"]
    assert isinstance(nli, dict)
    assert nli["auc"] == pytest.approx(1.0)


def test_stage_two_takes_the_baseline_the_weakest_and_the_top_three() -> None:
    rows = [
        {"setup": name, "passage.ndcg@10": value}
        for name, value in [
            ("fixed-bm25", 0.32),
            ("sentence-bm25", 0.30),
            ("a", 0.50),
            ("b", 0.48),
            ("c", 0.45),
            ("d", 0.40),
        ]
    ]
    assert stage_two_setups(rows) == ("fixed-bm25", "sentence-bm25", "a", "b", "c")


def test_the_weak_reference_is_never_the_baseline() -> None:
    rows = [
        {"setup": "fixed-bm25", "passage.ndcg@10": 0.10},
        {"setup": "x", "passage.ndcg@10": 0.20},
        {"setup": "y", "passage.ndcg@10": 0.30},
    ]
    assert stage_two_setups(rows, top=1) == ("fixed-bm25", "x", "y")


def test_the_stage_two_config_loads_and_records_the_rule(tmp_path: Path) -> None:
    chosen = ("fixed-bm25", "sentence-bm25", "a", "b", "c")
    scores = dict.fromkeys(chosen, 0.5)
    path = tmp_path / "stage2.toml"
    path.write_text(stage_two_config_text(chosen, scores, "d" * 40), encoding="utf-8")
    config = load_stage_two_config(path)
    assert config.setups == chosen
    assert (config.judge.model, config.judge.thinking) == ("gpt-6-luna", "high")
    assert "decision 62" in path.read_text(encoding="utf-8")


def test_answer_text_ranks_find_the_first_passage_holding_a_reference(tmp_path: Path) -> None:
    references = {"q1": ("360 km",), "q2": ("absent words",)}
    evaluation_set = EvaluationSet(
        name="tiny",
        pages=EVALUATION_SET.pages,
        questions=tuple(
            replace(q, short_answers=references.get(q.question_id, ("x",)))
            for q in EVALUATION_SET.questions
        ),
        meta={},
    )
    run_dir = write_run(tmp_path, "alpha", stem=True)
    ranks = answer_text_ranks(run_dir, evaluation_set, {"q1", "q2"})
    assert ranks["q1"] is not None
    assert ranks["q1"] >= 1
    assert ranks["q2"] is None


def write_scored(path: Path, setup: str, rows: list[tuple[str, str, bool, bool]]) -> Path:
    path.mkdir(parents=True)
    (path / "scored.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "setup": setup,
                    "question_id": question,
                    "answer_type": "table",
                    "correctness": correctness,
                    "groundedness": "supported" if supported else "not_supported",
                    "contains_reference": correctness == "correct",
                    "declined": declined,
                }
            )
            + "\n"
            for question, correctness, supported, declined in rows
        ),
        encoding="utf-8",
    )
    return path


def test_the_writer_is_compared_on_the_same_table_questions(tmp_path: Path) -> None:
    structured = write_scored(
        tmp_path / "structured",
        "win",
        [
            ("q1", "correct", True, False),
            ("q2", "incorrect", False, False),
            ("q3", "", False, True),
        ],
    )
    flattened = write_scored(
        tmp_path / "flat",
        "win-flat-tables",
        [
            ("q1", "correct", True, False),
            ("q2", "correct", True, False),
            ("q9", "correct", True, False),
        ],
    )
    result = table_writer(structured, flattened, "win", seed=1)
    assert result["questions"] == 2
    comparisons = result["comparisons"]
    assert isinstance(comparisons, dict)
    [correct] = comparisons["correct_judge"]
    assert (correct["a"], correct["b"]) == ("win-flat-tables", "win")
    assert correct["mean_a"] == pytest.approx(1.0)
    assert correct["mean_b"] == pytest.approx(0.5)
