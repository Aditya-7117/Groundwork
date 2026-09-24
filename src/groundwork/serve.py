"""The explorer's local server: the built site, plus live search over a few setups.

Live mode (decision 73) answers a typed question with a setup's ranked passages, using exactly the
retrieval code the evaluation ran: the same chunking, index, fusion and reranker, built from the
setup's own config file. It is meant to run in Docker on a laptop, so models run on CPU; the
winner's chunk vectors are computed ahead and read from the vector cache, so only the question is
encoded live. Every response says how long retrieval and reranking took.
"""

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from groundwork.chunking import ChunkingSettings, chunk_pages
from groundwork.config import ExperimentConfig
from groundwork.evaluation import EvaluationSet
from groundwork.experiment import ModelProvider, build_first_stage
from groundwork.rerank import get_reranker, rerank
from groundwork.retrieval import ChunkTable

logger = logging.getLogger(__name__)

SHOWN = 10
"""Passages returned per live search."""


class SearchRequest(BaseModel):
    """A live search: which setup, and the question."""

    model_config = ConfigDict(extra="forbid")

    setup: str
    question: str = Field(min_length=1, max_length=500)


class Passage(BaseModel):
    """One retrieved passage."""

    id: str
    heading: str
    text: str
    kind: str


class SearchResponse(BaseModel):
    """A live search result, with where the time went."""

    setup: str
    passages: list[Passage]
    retrieval_ms: float
    rerank_ms: float | None


@dataclass(frozen=True, slots=True, kw_only=True)
class LiveSetup:
    """One setup, built and ready to search."""

    config: ExperimentConfig
    search: Callable[[str], SearchResponse]


def build_live_setup(
    config: ExperimentConfig, evaluation_set: EvaluationSet, models: ModelProvider
) -> LiveSetup:
    """Build a setup's chunks, index and models once, and return its search function."""
    chunks = chunk_pages(
        evaluation_set.pages,
        ChunkingSettings(
            strategy=config.chunking.strategy,
            size=config.chunking.size,
            overlap=config.chunking.overlap,
            flatten_tables=config.chunking.flatten_tables,
        ),
    )
    table = ChunkTable(chunks)
    first_stage = build_first_stage(config, chunks, table, models, time.perf_counter)
    scorer = models.scorer(get_reranker(config.rerank.model)) if config.rerank is not None else None

    def search(question: str) -> SearchResponse:
        started = time.perf_counter()
        retrieval = first_stage.search(question)
        retrieval_ms = (time.perf_counter() - started) * 1000
        rerank_ms = None
        if scorer is not None and config.rerank is not None:
            started = time.perf_counter()
            retrieval = rerank(retrieval, question, scorer, depth=config.rerank.depth)
            rerank_ms = (time.perf_counter() - started) * 1000
        return SearchResponse(
            setup=config.name,
            passages=[
                Passage(
                    id=item.chunk.chunk_id,
                    heading=item.chunk.text.partition("\n")[0],
                    text=item.chunk.text.partition("\n")[2],
                    kind=item.chunk.kind,
                )
                for item in retrieval.chunks[:SHOWN]
            ],
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
        )

    logger.info("live setup ready", extra={"setup": config.name, "chunks": len(chunks)})
    return LiveSetup(config=config, search=search)


def create_app(live: Mapping[str, LiveSetup], site_dir: Path | None) -> FastAPI:
    """The server: live search under /api, and the built site at the root when given."""
    app = FastAPI(title="Groundwork explorer", docs_url=None, redoc_url=None)

    @app.get("/api/live")
    def live_setups() -> dict[str, Sequence[dict[str, str]]]:
        return {
            "setups": [
                {"name": name, "description": setup.config.description}
                for name, setup in live.items()
            ]
        }

    @app.post("/api/search")
    def search(request: SearchRequest) -> SearchResponse:
        if request.setup not in live:
            raise HTTPException(status_code=404, detail=f"setup {request.setup!r} is not live")
        return live[request.setup].search(request.question)

    if site_dir is not None:
        app.mount("/", StaticFiles(directory=site_dir, html=True), name="site")
    return app
