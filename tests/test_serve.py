"""The explorer's server: live search over setups built with the evaluation's own code."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fakes import FakeModels
from groundwork.config import ExperimentConfig
from groundwork.serve import build_live_setup, create_app
from runs import EVALUATION_SET, config

DENSE = ExperimentConfig.model_validate(
    {
        **config("live-dense", stem=True).model_dump(mode="json"),
        "retrieval": {"method": "dense", "depth": 100, "dense": {"model": "all-MiniLM-L6-v2"}},
        "rerank": {"model": "bge-reranker-v2-m3", "depth": 5},
    }
)


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    models = FakeModels(tmp_path / "vectors", rerank_word="jupiter")
    live = {
        "live-bm25": build_live_setup(config("live-bm25", stem=True), EVALUATION_SET, models),
        "live-dense": build_live_setup(DENSE, EVALUATION_SET, models),
    }
    return TestClient(create_app(live, None))


def test_the_live_setups_are_listed(client: TestClient) -> None:
    names = [setup["name"] for setup in client.get("/api/live").json()["setups"]]
    assert names == ["live-bm25", "live-dense"]


def test_bm25_finds_the_page_that_shares_the_words(client: TestClient) -> None:
    reply = client.post("/api/search", json={"setup": "live-bm25", "question": "longest river"})
    assert reply.status_code == 200
    body = reply.json()
    assert body["passages"][0]["heading"] == "River Shannon"
    assert body["rerank_ms"] is None
    assert body["retrieval_ms"] >= 0


def test_a_reranked_setup_reports_its_reranking_time(client: TestClient) -> None:
    # The fake reranker prefers passages mentioning Jupiter, whatever the question.
    body = client.post("/api/search", json={"setup": "live-dense", "question": "tea"}).json()
    assert body["passages"][0]["heading"] == "Jupiter"
    assert body["rerank_ms"] is not None


def test_an_unknown_setup_is_a_404(client: TestClient) -> None:
    reply = client.post("/api/search", json={"setup": "nope", "question": "q"})
    assert reply.status_code == 404


def test_an_empty_question_is_rejected(client: TestClient) -> None:
    reply = client.post("/api/search", json={"setup": "live-bm25", "question": ""})
    assert reply.status_code == 422
