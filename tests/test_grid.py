"""The grid is defined once, and the committed config files are exactly what it generates."""

from pathlib import Path

import pytest

from groundwork.config import load_config
from groundwork.grid import Setup, grid, render, write_grid

GRID_DIR = Path(__file__).resolve().parents[1] / "configs" / "grid"


def test_thirty_setups_with_unique_names() -> None:
    setups = grid()
    assert len(setups) == 30
    assert len({setup.name for setup in setups}) == 30


def test_every_retriever_is_crossed_with_every_chunker_and_the_reranker() -> None:
    setups = grid()
    assert {s.chunker for s in setups} == {"fixed", "sentence", "section"}
    assert {s.retriever for s in setups} == {
        "bm25",
        "minilm",
        "qwen3",
        "hybrid-minilm",
        "hybrid-qwen3",
    }
    assert sum(s.rerank for s in setups) == 15


def test_the_committed_configs_are_exactly_the_generated_ones() -> None:
    # If this fails, run `groundwork grid` and commit the result; never edit the files by hand.
    committed = {path.name: path.read_text(encoding="utf-8") for path in GRID_DIR.glob("*.toml")}
    generated = {f"{setup.name}.toml": render(setup) for setup in grid()}
    assert committed == generated


@pytest.mark.parametrize("setup", grid(), ids=lambda setup: setup.name)
def test_every_setup_is_a_valid_config(tmp_path: Path, setup: Setup) -> None:
    path = tmp_path / f"{setup.name}.toml"
    path.write_text(render(setup), encoding="utf-8")
    config = load_config(path)
    assert config.name == setup.name
    assert (config.rerank is not None) == setup.rerank


def test_setups_differ_only_in_what_the_grid_varies(tmp_path: Path) -> None:
    configs = []
    for setup in grid():
        path = tmp_path / f"{setup.name}.toml"
        path.write_text(render(setup), encoding="utf-8")
        configs.append(load_config(path))
    assert {c.seed for c in configs} == {1}
    assert {(c.chunking.size, c.chunking.overlap) for c in configs} == {(150, 30)}
    assert {c.evaluation.cutoffs for c in configs} == {(1, 5, 10, 100)}
    assert {c.retrieval.bm25 for c in configs if c.retrieval.bm25 is not None} == {
        configs[0].retrieval.bm25
    }


def test_a_hybrid_reranked_setup_carries_every_section(tmp_path: Path) -> None:
    setup = Setup(chunker="section", retriever="hybrid-qwen3", rerank=True)
    path = tmp_path / "s.toml"
    path.write_text(render(setup), encoding="utf-8")
    config = load_config(path)
    assert config.name == "section-hybrid-qwen3-rerank"
    assert config.chunking.strategy == "section_aware"
    assert config.retrieval.method == "hybrid"
    assert config.retrieval.dense is not None
    assert config.retrieval.dense.model == "Qwen3-Embedding-0.6B"
    assert config.retrieval.fusion is not None
    assert (config.retrieval.fusion.k, config.retrieval.fusion.candidates) == (60, 100)
    assert config.rerank is not None
    assert (config.rerank.model, config.rerank.depth) == ("bge-reranker-v2-m3", 50)


def test_the_table_ablation_flattens_tables(tmp_path: Path) -> None:
    setup = Setup(chunker="section", retriever="bm25", rerank=False, flatten_tables=True)
    path = tmp_path / "a.toml"
    path.write_text(render(setup), encoding="utf-8")
    config = load_config(path)
    assert config.name == "section-bm25-flat-tables"
    assert config.chunking.flatten_tables is True
    assert "tables flattened" in config.description


def test_writing_the_grid_removes_configs_no_longer_in_it(tmp_path: Path) -> None:
    (tmp_path / "old-setup.toml").write_text("", encoding="utf-8")
    paths = write_grid(tmp_path)
    assert len(paths) == 30
    assert not (tmp_path / "old-setup.toml").exists()


@pytest.mark.parametrize(
    "setup",
    [
        Setup(chunker="paragraph", retriever="bm25", rerank=False),
        Setup(chunker="fixed", retriever="colbert", rerank=False),
        Setup(chunker="fixed", retriever="hybrid-colbert", rerank=False),
    ],
    ids=["chunker", "retriever", "hybrid"],
)
def test_unknown_parts_are_rejected(setup: Setup) -> None:
    with pytest.raises(ValueError, match="unknown"):
        render(setup)
