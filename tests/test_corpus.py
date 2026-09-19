import shutil
from pathlib import Path

import pytest

from groundwork.corpus import CorpusError, Document, Query, load_beir

FIXTURE = Path(__file__).parent / "fixtures" / "tiny-beir"


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    """A writable copy of the fixture corpus that a test can corrupt."""
    destination = tmp_path / "corpus"
    shutil.copytree(FIXTURE, destination)
    return destination


def _append(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


class TestRejectsMalformedCorpus:
    def test_missing_split(self, corpus_dir: Path) -> None:
        with pytest.raises(CorpusError, match=r"qrels/dev\.tsv.*not found"):
            load_beir(corpus_dir, split="dev")

    def test_malformed_json_line_is_located(self, corpus_dir: Path) -> None:
        _append(corpus_dir / "corpus.jsonl", "{not json\n")
        with pytest.raises(CorpusError, match=r"corpus\.jsonl:7: not valid JSON"):
            load_beir(corpus_dir, split="test")

    def test_document_without_id(self, corpus_dir: Path) -> None:
        _append(corpus_dir / "corpus.jsonl", '{"title": "t", "text": "x"}\n')
        with pytest.raises(CorpusError, match=r"corpus\.jsonl:7: _id must be a non-empty string"):
            load_beir(corpus_dir, split="test")

    def test_duplicate_document_id(self, corpus_dir: Path) -> None:
        _append(corpus_dir / "corpus.jsonl", '{"_id": "d1", "title": "", "text": "again"}\n')
        with pytest.raises(CorpusError, match=r"corpus\.jsonl:7: duplicate id d1"):
            load_beir(corpus_dir, split="test")

    def test_qrels_header_is_checked(self, corpus_dir: Path) -> None:
        (corpus_dir / "qrels" / "test.tsv").write_text("q1\td1\t1\n", encoding="utf-8")
        with pytest.raises(CorpusError, match=r"test\.tsv:1: expected header"):
            load_beir(corpus_dir, split="test")

    def test_qrels_grade_must_be_an_integer(self, corpus_dir: Path) -> None:
        _append(corpus_dir / "qrels" / "test.tsv", "q1\td2\thigh\n")
        with pytest.raises(CorpusError, match=r"test\.tsv:6: grade must be an integer"):
            load_beir(corpus_dir, split="test")

    def test_qrels_row_must_have_three_fields(self, corpus_dir: Path) -> None:
        _append(corpus_dir / "qrels" / "test.tsv", "q1\td2\n")
        with pytest.raises(CorpusError, match=r"test\.tsv:6: expected 3 tab-separated fields"):
            load_beir(corpus_dir, split="test")

    def test_judgement_for_unknown_query(self, corpus_dir: Path) -> None:
        _append(corpus_dir / "qrels" / "test.tsv", "q9\td1\t1\n")
        with pytest.raises(CorpusError, match="unknown query id q9"):
            load_beir(corpus_dir, split="test")

    def test_judgement_for_unknown_document(self, corpus_dir: Path) -> None:
        # A judged document that is not in the corpus can never be retrieved, so recall would be
        # capped below 1 for a reason that has nothing to do with the retriever.
        _append(corpus_dir / "qrels" / "test.tsv", "q1\td9\t1\n")
        with pytest.raises(CorpusError, match="unknown document id d9"):
            load_beir(corpus_dir, split="test")

    def test_repeated_judgement(self, corpus_dir: Path) -> None:
        _append(corpus_dir / "qrels" / "test.tsv", "q1\td1\t0\n")
        with pytest.raises(CorpusError, match="repeats the judgement for q1, d1"):
            load_beir(corpus_dir, split="test")


class TestLoadsCorpus:
    def test_documents(self) -> None:
        corpus = load_beir(FIXTURE, split="test")
        assert len(corpus.documents) == 6
        assert corpus.documents["d4"] == Document(
            doc_id="d4",
            title="River Shannon",
            text="The Shannon is the longest river in Ireland, running for over three hundred "
            "kilometres.",
        )

    def test_only_judged_queries_are_kept(self) -> None:
        corpus = load_beir(FIXTURE, split="test")
        assert sorted(corpus.queries) == ["q1", "q2", "q3"]
        assert corpus.queries["q2"] == Query(query_id="q2", text="longest river Ireland")

    def test_graded_judgements(self) -> None:
        corpus = load_beir(FIXTURE, split="test")
        assert corpus.judgements == {"q1": {"d1": 1}, "q2": {"d4": 2, "d3": 1}, "q3": {"d5": 1}}

    def test_indexable_text_joins_title_and_body(self) -> None:
        document = Document(doc_id="x", title="Title", text="Body text.")
        assert document.indexable_text == "Title\n\nBody text."

    def test_indexable_text_without_title(self) -> None:
        assert Document(doc_id="x", title="", text="Body.").indexable_text == "Body."
