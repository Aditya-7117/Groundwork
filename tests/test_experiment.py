"""Running one experiment over a small evaluation set built in code.

Three pages, three questions, and a chunk size small enough that every expected ranking can be
reasoned about by hand.
"""

from typing import Any

import pytest

from groundwork.config import ExperimentConfig
from groundwork.evaluation import EvalQuestion, EvaluationSet
from groundwork.experiment import run_experiment
from groundwork.page import Block, Page

TOLERANCE = 1e-6


def page(page_id: str, title: str, text: str) -> Page:
    block = Block(kind="paragraph", section="", text=text, start=0, end=len(text))
    return Page(page_id=page_id, title=title, url="u", text=text, blocks=(block,))


TEA = page(
    "p-tea",
    "Green tea",
    "Green tea is made from unoxidised leaves. Steeping it too hot makes it bitter.",
)
RIVER = page(
    "p-river",
    "River Shannon",
    "The Shannon is the longest river in Ireland. It reaches the sea at Limerick.",
)
PLANET = page(
    "p-planet", "Jupiter", "Jupiter is the largest planet. It is a gas giant with a great red spot."
)


def question(
    question_id: str, text: str, page_obj: Page, answer: str, answer_type: str
) -> EvalQuestion:
    start = page_obj.text.index(answer)
    return EvalQuestion(
        question_id=question_id,
        text=text,
        page_relevance={page_obj.page_id: 1},
        spans=((page_obj.page_id, start, start + len(answer)),),
        short_answers=(answer,),
        answer_type=answer_type,
    )


EVALUATION_SET = EvaluationSet(
    name="tiny",
    pages=(TEA, RIVER, PLANET),
    questions=(
        question(
            "q1",
            "why is green tea bitter",
            TEA,
            "Steeping it too hot makes it bitter.",
            "paragraph",
        ),
        question(
            "q2",
            "longest river in Ireland",
            RIVER,
            "The Shannon is the longest river in Ireland.",
            "paragraph",
        ),
        question(
            "q3",
            "which planet is a gas giant",
            PLANET,
            "It is a gas giant with a great red spot.",
            "table",
        ),
    ),
    meta={},
)

CONFIG_FIELDS: dict[str, Any] = {
    "name": "tiny-bm25",
    "description": "Fixture run.",
    "seed": 3,
    "corpus": {"name": "natural-questions", "split": "validation"},
    "chunking": {"strategy": "fixed_words", "size": 8, "overlap": 2},
    "retrieval": {"method": "bm25", "depth": 5, "k1": 0.9, "b": 0.4},
    "evaluation": {"cutoffs": (1, 5)},
}


def config(**overrides: object) -> ExperimentConfig:
    fields = {**CONFIG_FIELDS, **overrides}
    return ExperimentConfig.model_validate(fields)


class TestRun:
    def test_every_question_is_scored_at_passage_and_page_level(self) -> None:
        result = run_experiment(config(), EVALUATION_SET)
        assert [q.question_id for q in result.questions] == ["q1", "q2", "q3"]
        assert set(result.aggregate) == {
            f"{level}.{name}@{k}"
            for level in ("passage", "page")
            for name in ("precision", "recall", "ndcg", "mrr")
            for k in (1, 5)
        }

    def test_the_right_page_is_retrieved_first(self) -> None:
        # Each question shares distinctive words with exactly one page.
        result = run_experiment(config(), EVALUATION_SET)
        first_pages = {q.question_id: q.page_ranking[0] for q in result.questions}
        assert first_pages == {"q1": "p-tea", "q2": "p-river", "q3": "p-planet"}
        assert result.aggregate["page.recall@1"] == pytest.approx(1.0)

    def test_passage_scores_are_not_page_scores(self) -> None:
        # Pages are split into several chunks, so finding the page is easier than finding the
        # passage. Reporting only the page number would overstate the system.
        result = run_experiment(config(), EVALUATION_SET)
        assert result.aggregate["passage.recall@1"] <= result.aggregate["page.recall@1"]

    def test_results_are_broken_down_by_answer_type(self) -> None:
        result = run_experiment(config(), EVALUATION_SET)
        assert set(result.by_answer_type) == {"paragraph", "table"}
        assert result.by_answer_type["table"]["page.recall@1"] == pytest.approx(1.0)

    def test_counts_and_timings_are_reported(self) -> None:
        result = run_experiment(config(), EVALUATION_SET)
        assert result.chunk_count > 3
        assert set(result.stage_seconds) == {"chunking", "indexing", "retrieval", "evaluation"}
        assert set(result.retrieval_latency_ms) == {"mean", "p50", "p95", "max"}


class TestQuestionSelection:
    def test_a_question_no_chunk_covers_is_excluded_and_reported(self) -> None:
        stray = EvalQuestion(
            question_id="q9",
            text="unanswerable",
            page_relevance={"p-tea": 1},
            spans=(("p-tea", 10_000, 10_010),),
            short_answers=(),
            answer_type="paragraph",
        )
        with_stray = EvaluationSet(
            name="tiny",
            pages=EVALUATION_SET.pages,
            questions=(*EVALUATION_SET.questions, stray),
            meta={},
        )
        result = run_experiment(config(), with_stray)
        assert result.excluded_question_ids == ("q9",)
        assert [q.question_id for q in result.questions] == ["q1", "q2", "q3"]

    def test_a_seeded_sample_is_reproducible(self) -> None:
        sampled = config(
            corpus={"name": "natural-questions", "split": "validation", "query_limit": 2}
        )
        first = [q.question_id for q in run_experiment(sampled, EVALUATION_SET).questions]
        second = [q.question_id for q in run_experiment(sampled, EVALUATION_SET).questions]
        assert first == second
        assert len(first) == 2

    def test_asking_for_more_questions_than_exist_is_an_error(self) -> None:
        sampled = config(
            corpus={"name": "natural-questions", "split": "validation", "query_limit": 9}
        )
        with pytest.raises(ValueError, match="query_limit 9 exceeds the 3 evaluable questions"):
            run_experiment(sampled, EVALUATION_SET)


def test_stemming_changes_what_matches() -> None:
    # The pages say "river"; only stemming lets the plural "rivers" match it. The query uses no
    # other word from the page, so the match can only come from stemming.
    asked = EvalQuestion(
        question_id="q-stem",
        text="rivers",
        page_relevance={"p-river": 1},
        spans=(("p-river", 0, 43),),
        short_answers=(),
        answer_type="paragraph",
    )
    single = EvaluationSet(name="tiny", pages=(TEA, RIVER, PLANET), questions=(asked,), meta={})
    plain = run_experiment(config(retrieval={**CONFIG_FIELDS["retrieval"], "stem": False}), single)
    stemmed = run_experiment(config(retrieval={**CONFIG_FIELDS["retrieval"], "stem": True}), single)
    assert plain.questions[0].page_ranking == ()
    assert stemmed.questions[0].page_ranking[0] == "p-river"
