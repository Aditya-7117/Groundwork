import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from groundwork.config import (
    ChunkingConfig,
    CorpusConfig,
    EvaluationConfig,
    ExperimentConfig,
    RetrievalConfig,
)
from groundwork.corpus import Corpus, Query, load_beir
from groundwork.experiment import run_experiment

FIXTURE = Path(__file__).parent / "fixtures" / "tiny-beir"
TOLERANCE = 1e-6

CONFIG = ExperimentConfig(
    name="tiny-bm25",
    description="Fixture run.",
    seed=3,
    corpus=CorpusConfig(name="tiny-beir", split="test", query_limit=None),
    chunking=ChunkingConfig(strategy="fixed_words", size=200, overlap=50),
    retrieval=RetrievalConfig(method="bm25", depth=5, k1=0.9, b=0.4),
    evaluation=EvaluationConfig(cutoffs=(1, 5)),
)


@pytest.fixture
def corpus() -> Corpus:
    return load_beir(FIXTURE, split="test")


class TestRejects:
    def test_query_limit_larger_than_the_question_set(self, corpus: Corpus) -> None:
        config = replace(CONFIG, corpus=replace(CONFIG.corpus, query_limit=4))
        with pytest.raises(ValueError, match="query_limit 4 exceeds the 3 evaluable queries"):
            run_experiment(config, corpus)


class TestFixtureRun:
    def test_rankings(self, corpus: Corpus) -> None:
        # q1 "bitter green tea": d1 has all three terms, d2 only "tea".
        # q2 "longest river Ireland": d4 has all three, d3 only "river".
        # q3 "ringed gas giant": only d6 shares terms ("gas", "giant"). The relevant d5 says
        #    "ring", which does not match "ringed" without stemming.
        result = run_experiment(CONFIG, corpus)
        rankings = {
            query.query_id: [ranked.doc_id for ranked in query.ranking] for query in result.queries
        }
        assert rankings == {"q1": ["d1", "d2"], "q2": ["d4", "d3"], "q3": ["d6"]}

    def test_per_query_metrics(self, corpus: Corpus) -> None:
        # q2 has two relevant documents, d4 (grade 2) and d3 (grade 1), ranked in ideal order:
        # recall@1 = 1/2, recall@5 = 2/2, nDCG = 1 at both cutoffs, first relevant at rank 1.
        result = run_experiment(CONFIG, corpus)
        q2 = next(query for query in result.queries if query.query_id == "q2")
        assert q2.metrics == {
            "recall@1": 0.5,
            "recall@5": 1.0,
            "ndcg@1": 1.0,
            "ndcg@5": 1.0,
            "rr@1": 1.0,
            "rr@5": 1.0,
        }

    def test_aggregate_metrics(self, corpus: Corpus) -> None:
        # Per query (q1, q2, q3), then the mean:
        #   recall@1  1, 0.5, 0  -> 0.5
        #   recall@5  1, 1,   0  -> 0.666667
        #   ndcg@1    1, 1,   0  -> 0.666667
        #   ndcg@5    1, 1,   0  -> 0.666667
        #   mrr@1     1, 1,   0  -> 0.666667
        #   mrr@5     1, 1,   0  -> 0.666667
        result = run_experiment(CONFIG, corpus)
        assert dict(result.aggregate) == pytest.approx(
            {
                "recall@1": 0.5,
                "recall@5": 0.666667,
                "ndcg@1": 0.666667,
                "ndcg@5": 0.666667,
                "mrr@1": 0.666667,
                "mrr@5": 0.666667,
            },
            abs=TOLERANCE,
        )

    def test_counts_and_timings_are_reported(self, corpus: Corpus) -> None:
        result = run_experiment(CONFIG, corpus)
        assert result.chunk_count == 6
        assert set(result.stage_seconds) == {"chunking", "indexing", "retrieval", "evaluation"}
        assert all(seconds >= 0 for seconds in result.stage_seconds.values())
        assert set(result.retrieval_latency_ms) == {"mean", "p50", "p95", "max"}


class TestQuerySelection:
    def test_query_with_no_relevant_document_is_excluded_and_reported(self, corpus: Corpus) -> None:
        with_unanswerable = Corpus(
            documents=corpus.documents,
            queries={**corpus.queries, "q4": Query(query_id="q4", text="tea")},
            judgements={**corpus.judgements, "q4": {"d2": 0}},
        )
        result = run_experiment(CONFIG, with_unanswerable)
        assert result.excluded_query_ids == ("q4",)
        assert [query.query_id for query in result.queries] == ["q1", "q2", "q3"]

    def test_seeded_sample_is_reproducible(self, corpus: Corpus) -> None:
        config = replace(CONFIG, corpus=replace(CONFIG.corpus, query_limit=2))
        first = [query.query_id for query in run_experiment(config, corpus).queries]
        second = [query.query_id for query in run_experiment(config, corpus).queries]
        assert first == second
        assert len(first) == 2

    def test_different_seeds_can_select_different_queries(self, corpus: Corpus) -> None:
        selections = {
            tuple(
                query.query_id
                for query in run_experiment(
                    replace(CONFIG, seed=seed, corpus=replace(CONFIG.corpus, query_limit=1)),
                    corpus,
                ).queries
            )
            for seed in range(20)
        }
        assert len(selections) > 1


_RUN_IN_SUBPROCESS = """
import json, sys
from pathlib import Path
from groundwork.corpus import load_beir
from groundwork.experiment import run_experiment
sys.path.insert(0, sys.argv[2])
from test_experiment import CONFIG
result = run_experiment(CONFIG, load_beir(Path(sys.argv[1]), split="test"))
print(json.dumps({
    "rankings": {q.query_id: [(r.doc_id, r.score) for r in q.ranking] for q in result.queries},
    "aggregate": dict(result.aggregate),
}))
"""


def test_results_do_not_depend_on_hash_randomisation() -> None:
    # Python randomises string hashing per process. Anything that iterated over a set of
    # strings while summing scores would give different floats, and possibly different
    # rankings, from one process to the next. Identical output under different seeds shows the
    # pipeline does not depend on it.
    outputs = set()
    for hash_seed in ("0", "1", "2"):
        completed = subprocess.run(  # noqa: S603 -- fixed argv: this interpreter and a literal script
            [sys.executable, "-c", _RUN_IN_SUBPROCESS, str(FIXTURE), str(Path(__file__).parent)],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
        )
        outputs.add(completed.stdout)
    assert len(outputs) == 1
    assert json.loads(outputs.pop())["rankings"]["q1"][0][0] == "d1"
