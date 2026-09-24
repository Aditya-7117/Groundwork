"""The explorer's data files: reshaped from the artefacts, never recomputed."""

import json
from pathlib import Path

from groundwork.answering import Answer
from groundwork.report import load_runs, retrieval_table
from groundwork.site import disagreements, export_site, stage_two_files
from runs import EVALUATION_SET, write_run


def answer(question_id: str, text: str) -> Answer:
    return Answer(
        setup="alpha",
        question_id=question_id,
        question="q",
        answer_type="paragraph",
        chunk_ids=("c1",),
        passages=("Title\npassage",),
        text=text,
        declined=False,
        prompt_tokens=1,
        output_tokens=1,
        seconds=1.0,
    )


def test_every_setup_gets_its_questions_with_top_passages(tmp_path: Path) -> None:
    results = tmp_path / "results"
    run_dir = write_run(results, "alpha", stem=True)
    report = {"retrieval": retrieval_table(load_runs(results, ["alpha"]))}
    out = tmp_path / "site"
    export_site(report, {"alpha": run_dir}, EVALUATION_SET, out)

    setup = json.loads((out / "setups" / "alpha.json").read_text(encoding="utf-8"))
    assert setup["setup"] == "alpha"
    assert setup["question_metrics"][:2] == ["passage.recall@10", "passage.ndcg@10"]
    q1 = setup["questions"]["q1"]
    assert len(q1["metrics"]) == 4
    heading, _, table = q1["top"][0]
    assert setup["headings"][heading] == "River Shannon"
    assert table == 0
    assert any(passage[1] for passage in q1["top"])
    questions = json.loads((out / "questions.json").read_text(encoding="utf-8"))
    assert questions["q2"] == {
        "question": "largest gas giant planet",
        "type": "table",
        "references": ["x"],
        "page": "Jupiter",
    }
    assert json.loads((out / "report.json").read_text(encoding="utf-8")) == report


def test_stage_two_rows_carry_every_verdict() -> None:
    scored = {("alpha", "q1"): {"groundedness": "supported", "nli": 0.9, "lexical": 1.0}}
    files = stage_two_files([answer("q1", "Shannon.")], scored)
    [row] = files["alpha"]
    verdicts = row["verdicts"]
    assert isinstance(verdicts, dict)
    assert verdicts["groundedness"] == "supported"
    assert row["passages"] == ["Title\npassage"]


def test_only_disagreeing_labels_are_listed() -> None:
    answers = [answer("q1", "Shannon."), answer("q2", "Liffey.")]
    scored = {
        ("alpha", "q1"): {"groundedness": "supported"},
        ("alpha", "q2"): {"groundedness": "supported"},
    }
    labels = {("alpha", "q1"): "supported", ("alpha", "q2"): "not_supported"}
    [row] = disagreements(answers, scored, labels)
    assert (row["question_id"], row["human"], row["judge"]) == ("q2", "not_supported", "supported")
