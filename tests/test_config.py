import tomllib
from pathlib import Path

import pytest

from groundwork.config import (
    BM25Config,
    ChunkingConfig,
    ConfigError,
    CorpusConfig,
    EvaluationConfig,
    ExperimentConfig,
    RerankConfig,
    RetrievalConfig,
    config_digest,
    load_config,
    load_stage_two_config,
)

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"

VALID = """
name = "example-bm25"
description = "A config used by the tests."
seed = 7

[corpus]
name = "natural-questions"
split = "validation"

[chunking]
strategy = "fixed_words"
size = 200
overlap = 50

[retrieval]
method = "bm25"
depth = 100

[retrieval.bm25]
k1 = 0.9
b = 0.4
stem = false

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
            ConfigError, match=r"\[retrieval\.bm25\.b\] Input should be less than or equal to 1"
        ):
            load_config(_write(tmp_path, _replace("b = 0.4", "b = 1.5")))

    def test_stemming_must_be_stated(self, tmp_path: Path) -> None:
        # Stemming moves recall by several points, so it is never left to a default.
        with pytest.raises(ConfigError, match=r"\[retrieval\.bm25\.stem\] Field required"):
            load_config(_write(tmp_path, _replace("stem = false\n", "")))

    def test_dense_without_its_section(self, tmp_path: Path) -> None:
        text = _replace('method = "bm25"', 'method = "dense"')
        with pytest.raises(ConfigError, match=r"method dense needs \[retrieval\.dense\]"):
            load_config(_write(tmp_path, text))

    def test_a_section_the_method_does_not_use(self, tmp_path: Path) -> None:
        # A bm25 config carrying a [retrieval.dense] section would read as if embeddings ran.
        text = VALID.replace("[evaluation]", '[retrieval.dense]\nmodel = "x"\n\n[evaluation]')
        with pytest.raises(ConfigError, match=r"method bm25 does not use \[retrieval\.dense\]"):
            load_config(_write(tmp_path, text))

    def test_hybrid_needs_every_part(self, tmp_path: Path) -> None:
        text = _replace('method = "bm25"', 'method = "hybrid"')
        with pytest.raises(ConfigError, match=r"method hybrid needs \[retrieval\.dense, fusion\]"):
            load_config(_write(tmp_path, text))

    def test_rerank_deeper_than_retrieval(self, tmp_path: Path) -> None:
        # A reranker can only re-order what the first stage returned.
        text = VALID + '\n[rerank]\nmodel = "bge-reranker-v2-m3"\ndepth = 200\n'
        with pytest.raises(ConfigError, match="rerank depth 200 exceeds retrieval depth 100"):
            load_config(_write(tmp_path, text))

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
        text = _replace('split = "validation"', 'split = "validation"\nquery_limit = 0')
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
            corpus=CorpusConfig(name="natural-questions", split="validation", query_limit=None),
            chunking=ChunkingConfig(strategy="fixed_words", size=200, overlap=50),
            retrieval=RetrievalConfig(
                method="bm25", depth=100, bm25=BM25Config(k1=0.9, b=0.4, stem=False)
            ),
            evaluation=EvaluationConfig(cutoffs=(1, 5, 10)),
        )

    def test_integer_is_accepted_where_float_expected(self, tmp_path: Path) -> None:
        config = load_config(_write(tmp_path, _replace("b = 0.4", "b = 1")))
        assert config.retrieval.bm25 is not None
        assert config.retrieval.bm25.b == 1.0

    def test_hybrid_with_reranking(self, tmp_path: Path) -> None:
        text = VALID.replace('method = "bm25"', 'method = "hybrid"').replace(
            "[evaluation]",
            '[retrieval.dense]\nmodel = "all-MiniLM-L6-v2"\n\n'
            "[retrieval.fusion]\nk = 60\ncandidates = 100\n\n"
            '[rerank]\nmodel = "bge-reranker-v2-m3"\ndepth = 50\n\n[evaluation]',
        )
        config = load_config(_write(tmp_path, text))
        assert config.retrieval.fusion is not None
        assert config.retrieval.fusion.k == 60
        assert config.rerank == RerankConfig(model="bge-reranker-v2-m3", depth=50)

    def test_optional_query_limit(self, tmp_path: Path) -> None:
        text = _replace('split = "validation"', 'split = "validation"\nquery_limit = 50')
        assert load_config(_write(tmp_path, text)).corpus.query_limit == 50

    @pytest.mark.parametrize(
        "path",
        sorted(CONFIGS_DIR.rglob("*.toml")),
        ids=lambda p: p.relative_to(CONFIGS_DIR).as_posix(),
    )
    def test_committed_configs_are_valid(self, path: Path) -> None:
        # A stage-two config names the setups to answer and the judge; everything else defines
        # one experiment.
        with path.open("rb") as handle:
            is_stage_two = "judge" in tomllib.load(handle)
        (load_stage_two_config if is_stage_two else load_config)(path)


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


STAGE_TWO = """
name = "stage-two"
description = "Answers for the chosen setups."
seed = 1
questions = 1000
passages = 5
writer = "qwen3.8-27b-iq4xs"
setups = ["fixed-bm25", "sentence-bm25"]

[judge]
model = "gpt-6-luna"
thinking = "high"
"""


class TestStageTwoConfig:
    def test_a_valid_config_loads(self, tmp_path: Path) -> None:
        path = tmp_path / "stage2.toml"
        path.write_text(STAGE_TWO, encoding="utf-8")
        config = load_stage_two_config(path)
        assert config.setups == ("fixed-bm25", "sentence-bm25")
        assert config.judge.thinking == "high"

    @pytest.mark.parametrize(
        ("old", "new", "message"),
        [
            ('thinking = "high"', 'thinking = "minimal"', r"\[judge\.thinking\]"),
            ('"sentence-bm25"]', '"fixed-bm25"]', "setups must not repeat"),
            ('["fixed-bm25", "sentence-bm25"]', "[]", "at least one setup"),
            ('"sentence-bm25"]', '"../escape"]', "must be lowercase letters"),
            ("questions = 1000", "questions = 0", r"\[questions\]"),
        ],
    )
    def test_rejects_invalid_values(self, tmp_path: Path, old: str, new: str, message: str) -> None:
        path = tmp_path / "stage2.toml"
        path.write_text(STAGE_TWO.replace(old, new), encoding="utf-8")
        with pytest.raises(ConfigError, match=message):
            load_stage_two_config(path)
