# Groundwork

Groundwork measures which retrieval-augmented generation configuration answers questions correctly,
how quickly, and at what cost. Every experiment is defined by a file in `configs/`, run with a
fixed seed, and written to a results artefact that records the exact configuration, code version,
corpus checksum and hardware that produced it. The evaluation is the product; the retrieval
pipeline exists to be measured.

## Status

The retrieval half runs end to end: BM25 over fixed-size chunks, scored with recall@k, nDCG@k and
MRR@k. Generation, the groundedness judge, the full configuration grid and the regression gate are
not built yet.

The one corpus wired in, BEIR SciFact, is **provisional plumbing**. It exists to exercise the
pipeline, and its numbers are not results of this project. Configs and artefacts that use it are
marked `provisional`.

## Architecture

```
configs/<experiment>.toml
        │  validated, digested
        ▼
corpus source ──► download, verify SHA-256, extract ──► load (BEIR layout)
        │
        ▼
chunk (fixed-size word windows) ──► BM25 index over chunks
        │
        ▼
per query: score chunks ──► rank documents by best chunk ──► recall@k, nDCG@k, RR@k
        │
        ▼
results/<experiment>/<UTC start>-<config digest>/
        ├── result.json   config, code commit, corpus checksum, hardware, timings, metrics
        └── run.trec      rankings in TREC format, re-scorable with trec_eval
```

| Module | Responsibility |
|---|---|
| `config.py` | Parse and strictly validate an experiment file; compute its digest. |
| `sources.py` | Registry of pinned corpus archives; checksummed download and extraction. |
| `corpus.py` | Document, query and judgement data model; BEIR-format loader. |
| `chunking.py` | Split documents into chunks that keep their character spans. |
| `bm25.py` | BM25 scoring over chunks. |
| `ranking.py` | Turn chunk scores into a document ranking. |
| `metrics.py` | recall@k, nDCG@k, reciprocal rank at k, and averaging over queries. |
| `experiment.py` | Run one config over one corpus. |
| `artefact.py` | Write the versioned results artefact with provenance. |
| `cli.py` | The `groundwork run` command. |

Why each piece is built the way it is: [`docs/decisions/`](docs/decisions/).

## Setup

Requires [uv](https://docs.astral.sh/uv/). uv installs Python 3.12 if it is not already present.

```sh
uv sync
```

## Running an experiment

```sh
uv run groundwork run configs/provisional-scifact-bm25.toml
```

The first run downloads the corpus (2.8 MB) into `data/` and verifies its checksum. Results land in
`results/`. Logs are JSON lines on stderr. The corpus cache and results locations can be changed
with `--data-dir` and `--results-dir`.

## Checks

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

CI runs the same four on every push.

## Metric conventions

Metrics follow trec_eval, so results are comparable with published baselines. A document is
relevant when its grade is above zero; unjudged documents count as grade zero. nDCG uses linear
gain and builds its ideal ranking from every judged document. MRR is cut off at k. Queries with no
relevant document are excluded and listed in the artefact, never scored as zero. Every convention
is pinned by hand-worked tests in `tests/test_metrics.py`; the reasoning is in
[decision 0002](docs/decisions/0002-metric-conventions.md).

## Limitations

- Only lexical retrieval exists. BM25 here has no stemming and no stopword list, so it will not
  exactly match Anserini's BM25 on the same corpus.
- BM25 is pure Python and single-threaded, which suits tens of thousands of chunks, not millions.
- Documents are ranked by their best chunk, which suits document-level judgements. A corpus with
  passage-level judgements would need a different scoring unit.
- Latency figures are wall-clock times on one machine, recorded with the hardware in each
  artefact. They are not a statement about production performance.
