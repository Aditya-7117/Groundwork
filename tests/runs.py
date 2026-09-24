"""Small run artefacts for tests that read results: three pages, three questions, BM25 runs."""

from datetime import UTC, datetime
from pathlib import Path

from groundwork.artefact import CodeVersion, RunRecord, write_artefact
from groundwork.config import ExperimentConfig
from groundwork.evaluation import EvalQuestion, EvaluationSet
from groundwork.experiment import run_experiment
from groundwork.page import Block, Page

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
