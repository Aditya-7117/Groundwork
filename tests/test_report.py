"""The report, over small run artefacts written in the test."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from groundwork.artefact import CodeVersion, RunRecord, write_artefact
from groundwork.config import ExperimentConfig
from groundwork.evaluation import EvalQuestion, EvaluationSet
from groundwork.experiment import run_experiment
from groundwork.page import Block, Page
from groundwork.report import (
    ReportError,
    human_agreement,
    load_runs,
    markdown,
    retrieval_table,
    significance,
)

NOW = datetime(2026, 9, 25, 9, 0, tzinfo=UTC)


def page(page_id: str, title: str, text: str) -> Page:
    block = Block(kind="paragraph", section="", text=text, start=0, end=len(text))
    return Page(page_id=page_id, title=title, url="u", text=text, blocks=(block,))


PAGES = (
    page("p-river", "River Shannon", "The Shannon is the longest river in Ireland at 360 km."),
    page("p-planet", "Jupiter", "Jupiter is the largest planet and a gas giant with rings."),
    page("p-tea", "Green tea", "Green tea is made from unoxidised leaves and can be bitter."),
)


def question(question_id: str, text: str, page_obj: Page, kind: str) -> EvalQuestion:
    return EvalQuestion(
        question_id=question_id,
        text=text,
        page_relevance={page_obj.page_id: 1},
        spans=((page_obj.page_id, 0, 20),),
        short_answers=("x",),
        answer_type=kind,
    )


EVALUATION_SET = EvaluationSet(
    name="tiny",
    pages=PAGES,
    questions=(
        question("q1", "longest river in ireland", PAGES[0], "paragraph"),
        question("q2", "largest gas giant planet", PAGES[1], "table"),
        question("q3", "green tea leaves", PAGES[2], "paragraph"),
    ),
    meta={},
)


def config(name: str, *, stem: bool) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": name,
            "description": "Fixture.",
            "seed": 1,
            "corpus": {"name": "natural-questions", "split": "validation"},
            "chunking": {"strategy": "fixed_words", "size": 5, "overlap": 1},
            "retrieval": {
                "method": "bm25",
                "depth": 100,
                "bm25": {"k1": 0.9, "b": 0.4, "stem": stem},
            },
            "evaluation": {"cutoffs": [1, 10, 100]},
        }
    )


def write_run(results: Path, name: str, *, stem: bool, commit: str = "c" * 40) -> Path:
    run_config = config(name, stem=stem)
    return write_artefact(
        RunRecord(
            config=run_config,
            config_path=Path(f"configs/{name}.toml"),
            corpus={"name": "natural-questions"},
            result=run_experiment(run_config, EVALUATION_SET),
            started_at=NOW,
            finished_at=NOW,
            code=CodeVersion(package_version="0.1.0", git_commit=commit, git_dirty=False),
            environment={},
        ),
        results,
    )


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
