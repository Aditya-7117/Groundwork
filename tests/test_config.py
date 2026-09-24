from pathlib import Path

import pytest

from groundwork.config import (
    ChunkingConfig,
    ConfigError,
    CorpusConfig,
    EvaluationConfig,
    ExperimentConfig,
    RetrievalConfig,
    config_digest,
    load_config,
)

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"

VALID = """
name = "example-bm25"
description = "A config used by the tests."
seed = 7

[corpus]
name = "beir-scifact"
split = "test"

[chunking]
strategy = "fixed_words"
size = 200
overlap = 50

[retrieval]
method = "bm25"
depth = 100
k1 = 0.9
b = 0.4

[evaluation]
cutoffs = [1, 5, 10]
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "experiment.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _replace(old: str, new: str) -> str:
    assert old in VALID
    return VALID.replace(old, new)


class TestRejectsInvalidConfig:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "absent.toml")

    def test_malformed_toml_names_the_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match=r"experiment\.toml.*not valid TOML"):
            load_config(_write(tmp_path, "name = "))

    def test_unknown_key_is_rejected_so_typos_cannot_pass_silently(self, tmp_path: Path) -> None:
        text = _replace("overlap = 50", "overlap = 50\noverlab = 60")
        with pytest.raises(
            ConfigError, match=r"\[chunking\.overlab\] Extra inputs are not permitted"
        ):
            load_config(_write(tmp_path, text))

    def test_unknown_section_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match=r"\[reranking\] Extra inputs are not permitted"):
            load_config(_write(tmp_path, VALID + "\n[reranking]\nmethod = 'none'\n"))

    def test_missing_key(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match=r"\[retrieval\.depth\] Field required"):
            load_config(_write(tmp_path, _replace("depth = 100\n", "")))

    def test_missing_seed(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match=r"\[seed\] Field required"):
            load_config(_write(tmp_path, _replace("seed = 7\n", "")))

    def test_string_where_integer_expected(self, tmp_path: Path) -> None:
        with pytest.raises(
            ConfigError, match=r"\[chunking\.size\] Input should be a valid integer"
        ):
            load_config(_write(tmp_path, _replace("size = 200", 'size = "200"')))

    def test_boolean_is_not_accepted_as_integer(self, tmp_path: Path) -> None:
        # In Python, True is an int. A config that says seed = true is a mistake, not seed 1.
        with pytest.raises(ConfigError, match=r"\[seed\] Input should be a valid integer"):
            load_config(_write(tmp_path, _replace("seed = 7", "seed = true")))

    def test_float_is_not_accepted_as_integer(self, tmp_path: Path) -> None:
        with pytest.raises(
            ConfigError, match=r"\[retrieval\.depth\] Input should be a valid integer"
        ):
            load_config(_write(tmp_path, _replace("depth = 100", "depth = 100.0")))

    def test_overlap_must_be_smaller_than_chunk_size(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="overlap must be less than size"):
            load_config(_write(tmp_path, _replace("overlap = 50", "overlap = 200")))

    def test_negative_overlap(self, tmp_path: Path) -> None:
        with pytest.raises(
            ConfigError, match=r"\[chunking\.overlap\] Input should be greater than or equal to 0"
        ):
            load_config(_write(tmp_path, _replace("overlap = 50", "overlap = -1")))

    def test_bm25_b_must_lie_between_zero_and_one(self, tmp_path: Path) -> None:
        with pytest.raises(
            ConfigError, match=r"\[retrieval\.b\] Input should be less than or equal to 1"
        ):
            load_config(_write(tmp_path, _replace("b = 0.4", "b = 1.5")))

    def test_cutoff_deeper_than_retrieval_depth(self, tmp_path: Path) -> None:
        # Asking for recall@200 from a 100-deep ranking would report a number that could not
        # have been measured.
        text = _replace("cutoffs = [1, 5, 10]", "cutoffs = [1, 200]")
        with pytest.raises(ConfigError, match="cutoff 200 exceeds retrieval depth 100"):
            load_config(_write(tmp_path, text))

    def test_duplicate_cutoffs(self, tmp_path: Path) -> None:
        text = _replace("cutoffs = [1, 5, 10]", "cutoffs = [5, 5]")
        with pytest.raises(ConfigError, match="cutoffs must not repeat"):
            load_config(_write(tmp_path, text))

    def test_unsupported_chunking_strategy(self, tmp_path: Path) -> None:
        text = _replace('strategy = "fixed_words"', 'strategy = "semantic"')
        with pytest.raises(
            ConfigError, match=r"\[chunking\.strategy\] Input should be 'fixed_words'"
        ):
            load_config(_write(tmp_path, text))

    @pytest.mark.parametrize("name", ["../escape", "Upper", "has space", "double--hyphen"])
    def test_name_must_be_a_safe_slug(self, tmp_path: Path, name: str) -> None:
        # The name becomes a directory under results/, so it must not be able to leave it.
        text = _replace('name = "example-bm25"', f'name = "{name}"')
        with pytest.raises(ConfigError, match="name must be lowercase letters, digits and"):
            load_config(_write(tmp_path, text))

    def test_query_limit_must_be_positive(self, tmp_path: Path) -> None:
        text = _replace('split = "test"', 'split = "test"\nquery_limit = 0')
        with pytest.raises(
            ConfigError, match=r"\[corpus\.query_limit\] Input should be greater than or equal to 1"
        ):
            load_config(_write(tmp_path, text))


class TestLoadsValidConfig:
    def test_every_field_is_read(self, tmp_path: Path) -> None:
        assert load_config(_write(tmp_path, VALID)) == ExperimentConfig(
            name="example-bm25",
            description="A config used by the tests.",
            seed=7,
            corpus=CorpusConfig(name="beir-scifact", split="test", query_limit=None),
            chunking=ChunkingConfig(strategy="fixed_words", size=200, overlap=50),
            retrieval=RetrievalConfig(method="bm25", depth=100, k1=0.9, b=0.4),
            evaluation=EvaluationConfig(cutoffs=(1, 5, 10)),
        )

    def test_integer_is_accepted_where_float_expected(self, tmp_path: Path) -> None:
        config = load_config(_write(tmp_path, _replace("b = 0.4", "b = 1")))
        assert config.retrieval.b == 1.0

    def test_optional_query_limit(self, tmp_path: Path) -> None:
        text = _replace('split = "test"', 'split = "test"\nquery_limit = 50')
        assert load_config(_write(tmp_path, text)).corpus.query_limit == 50

    @pytest.mark.parametrize("path", sorted(CONFIGS_DIR.glob("*.toml")), ids=lambda p: p.name)
    def test_committed_configs_are_valid(self, path: Path) -> None:
        load_config(path)


class TestConfigDigest:
    def test_formatting_and_key_order_do_not_change_the_digest(self, tmp_path: Path) -> None:
        reordered = _replace("k1 = 0.9\nb = 0.4", "b = 0.4   # length normalisation\nk1 = 0.9")
        first = load_config(_write(tmp_path, VALID))
        second = load_config(_write(tmp_path, reordered))
        assert config_digest(first) == config_digest(second)

    def test_any_value_change_changes_the_digest(self, tmp_path: Path) -> None:
        first = load_config(_write(tmp_path, VALID))
        second = load_config(_write(tmp_path, _replace("seed = 7", "seed = 8")))
        assert config_digest(first) != config_digest(second)

    def test_digest_is_a_sha256_hex_string(self, tmp_path: Path) -> None:
        digest = config_digest(load_config(_write(tmp_path, VALID)))
        assert len(digest) == 64
        assert set(digest) <= set("0123456789abcdef")
