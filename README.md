# Groundwork

Groundwork measures which retrieval setup for retrieval-augmented generation actually finds the
passage that answers a real question, how fast, and at what cost. It scores 30 setups (three ways
of chunking, five first-stage retrievers, with and without a reranker) on all 3,220 questions of
the Natural Questions validation split, then has the best few write answers that a language-model
judge scores for groundedness and correctness. The judge is itself judged: its groundedness labels
are compared with two cheaper checks and with 200 answers labelled by hand, blind.

**Result.** Four setups that pair the Qwen3-Embedding model with a cross-encoder reranker tie at
the top: the best, fixed-qwen3-rerank, reaches passage nDCG@10 of 0.696 against 0.321 for the BM25
baseline, and it is not significantly ahead of the other three. The reranker is the decisive
component: it raises nDCG@10 for every one of the fifteen first stages, by 0.15 to 0.29. The
judge's groundedness labels held up against 200 blind hand labels: it gave the same label on 95% of
them (Cohen's kappa 0.53) and flagged every answer the hand labels flagged, where the cheaper NLI
check caught two in six.

## Results

All numbers: passage level (a passage counts only if it holds the human-marked answer), all
3,220 questions, seed 1, every run from commit 199dd2a, Apple M5 Pro with 24 GB. Page recall@10 is
the share of questions whose answer page appears in the top ten. Every number is in a run artefact
under `results/`.

| Setup | Recall@10 | nDCG@10 | MRR@10 | Recall@100 | Page recall@10 |
|---|---|---|---|---|---|
| fixed-qwen3-rerank | 0.795 | 0.696 | 0.716 | 0.943 | 0.994 |
| fixed-hybrid-qwen3-rerank | 0.792 | 0.694 | 0.716 | 0.937 | 0.997 |
| section-hybrid-qwen3-rerank | 0.799 | 0.690 | 0.699 | 0.932 | 0.997 |
| section-qwen3-rerank | 0.800 | 0.690 | 0.696 | 0.942 | 0.995 |
| fixed-hybrid-minilm-rerank | 0.778 | 0.686 | 0.710 | 0.923 | 0.993 |
| fixed-minilm-rerank | 0.777 | 0.682 | 0.705 | 0.929 | 0.987 |
| section-hybrid-minilm-rerank | 0.785 | 0.682 | 0.692 | 0.918 | 0.996 |
| section-minilm-rerank | 0.785 | 0.680 | 0.690 | 0.927 | 0.990 |
| sentence-qwen3-rerank | 0.773 | 0.676 | 0.704 | 0.934 | 0.993 |
| sentence-hybrid-qwen3-rerank | 0.768 | 0.673 | 0.706 | 0.924 | 0.997 |
| sentence-hybrid-minilm-rerank | 0.753 | 0.664 | 0.699 | 0.910 | 0.995 |
| sentence-minilm-rerank | 0.754 | 0.661 | 0.694 | 0.921 | 0.988 |
| fixed-bm25-rerank | 0.669 | 0.600 | 0.635 | 0.793 | 0.972 |
| section-bm25-rerank | 0.672 | 0.597 | 0.618 | 0.786 | 0.973 |
| sentence-bm25-rerank | 0.639 | 0.576 | 0.622 | 0.771 | 0.972 |
| section-qwen3 | 0.717 | 0.537 | 0.511 | 0.942 | 0.992 |
| fixed-qwen3 | 0.716 | 0.534 | 0.519 | 0.943 | 0.989 |
| sentence-qwen3 | 0.692 | 0.523 | 0.517 | 0.934 | 0.989 |
| fixed-hybrid-qwen3 | 0.633 | 0.467 | 0.460 | 0.937 | 0.992 |
| section-hybrid-qwen3 | 0.644 | 0.463 | 0.439 | 0.932 | 0.992 |
| sentence-hybrid-qwen3 | 0.599 | 0.444 | 0.443 | 0.924 | 0.991 |
| section-minilm | 0.634 | 0.437 | 0.400 | 0.927 | 0.989 |
| section-hybrid-minilm | 0.614 | 0.436 | 0.409 | 0.918 | 0.991 |
| fixed-hybrid-minilm | 0.597 | 0.431 | 0.420 | 0.923 | 0.989 |
| sentence-minilm | 0.602 | 0.420 | 0.402 | 0.921 | 0.987 |
| fixed-minilm | 0.621 | 0.417 | 0.386 | 0.929 | 0.986 |
| sentence-hybrid-minilm | 0.570 | 0.415 | 0.411 | 0.910 | 0.990 |
| fixed-bm25 | 0.468 | 0.321 | 0.309 | 0.793 | 0.942 |
| section-bm25 | 0.460 | 0.310 | 0.288 | 0.786 | 0.938 |
| sentence-bm25 | 0.432 | 0.300 | 0.293 | 0.771 | 0.941 |

### The winning setup, and why

- **The reranker matters most.** bge-reranker-v2-m3 re-ordering the top 50 raised nDCG@10 for all
  fifteen first stages, by 0.15 (Qwen3) to 0.29 (BM25), every gain significant after Holm
  correction (paired randomisation test, 10,000 resamples). It costs about 0.56 s per question on
  the laptop's GPU (median; 95th percentile 0.70 s), against 46 to 166 ms for the first stage.
- **The embedding model matters next.** The winner is significantly ahead of every MiniLM setup
  with the reranker, by 0.010 to 0.035 nDCG@10. Without a reranker the gap between the models is
  larger: 0.534 against 0.417 on fixed chunks.
- **Hybrid search did not help the stronger model.** Fusing BM25 with Qwen3 by reciprocal rank
  fusion lowered nDCG@10 from 0.534 to 0.467 without a reranker; with one, hybrid and plain Qwen3
  tie. With MiniLM the effect was mixed: +0.014 on fixed chunks, about zero on the others.
- **Chunking mattered least.** Fixed and section-aware chunks tie at the top; sentence-aware chunks
  trail by about 0.02.
- **A tie at the top.** fixed-qwen3-rerank is not significantly ahead of fixed-hybrid-qwen3-rerank
  (+0.002, 95% interval −0.002 to +0.006), section-hybrid-qwen3-rerank or section-qwen3-rerank.
  The simplest of the four, one embedding model plus the reranker, is the one to use.
- **Cost of the winner.** Encoding the 341,126 fixed chunks with Qwen3-Embedding took 221 minutes
  on the laptop (MiniLM: 4.3 minutes); reranking all 3,220 questions took 31 minutes.

### Answers and the judge

Five setups, chosen by a rule fixed before the grid ran (the BM25 baseline, the weakest setup and
the top three), each answered the same 1,000 questions, sampled with a fixed seed in the full set's
mix (738 paragraph, 189 table, 73 list). A local Qwen3.8-27B wrote one sentence from the top five
passages, or said "I don't know". GPT-6 Luna at high reasoning effort judged each answer's
groundedness against its passages, without seeing the reference, and its correctness against the
reference answers. Correct and declined are shares of all 1,000 questions; fully supported is a
share of the answered ones.

| Setup | Correct (judge) | Correct (contains a reference) | Declined | Fully supported |
|---|---|---|---|---|
| section-hybrid-qwen3-rerank | 80.7% | 62.7% | 6.4% | 93.1% |
| fixed-qwen3-rerank | 80.3% | 62.3% | 7.1% | 92.5% |
| fixed-hybrid-qwen3-rerank | 80.3% | 62.5% | 6.6% | 92.6% |
| fixed-bm25 | 60.2% | 42.3% | 27.6% | 89.2% |
| sentence-bm25 | 58.4% | 42.0% | 29.5% | 89.6% |

- Better retrieval shows up as fewer declines and more correct answers. Groundedness barely moves:
  when the passages lack the answer, the writer declines rather than inventing one.
- The judge's correctness runs about 18 points above mechanical containment, because containment
  misses paraphrases: "The Shannon is the longest river" does not contain the reference "River
  Shannon". Both are reported.
- Judging cost $0.98 for 8,456 calls at the list price of 25 September 2026.
- Writing an answer takes 8.0 to 9.3 s at the median on the laptop, measured by re-sending the same
  100 prompts per setup, uncached, in one session; reading the ~1,200-token prompt dominates.

Is the judge right? Two hundred answers were labelled by hand, blind: 40 answered questions per
setup, drawn with seed 1 and shuffled, each shown with its passages and question but not the judge's
label or the setup, and labelled for groundedness. On the same three-level scale, the judge gave the
same label on 190 (95%), Cohen's kappa 0.53 (95% interval 0.22 to 0.77). It flagged every answer the
hand labels flagged (4 partly supported, 2 not supported), with the same label each time; all ten
disagreements are answers labelled supported by hand that the judge marked partly (9) or not (1)
supported. The cheaper checks, cut to "fully supported or not" at their fixed cut-offs, do worse on
the same 200:

| Check against the hand labels | Agreement | Cohen's kappa (95% interval) | ROC AUC | Flagged answers caught |
|---|---|---|---|---|
| GPT-6 Luna judge (three levels) | 95.0% | 0.53 (0.22 to 0.77) | | 6 of 6 |
| NLI, entailment 0.5 or more | 96.5% | 0.35 (−0.02 to 0.69) | 0.74 | 2 of 6 |
| Word overlap, every word found | 82.0% | 0.05 (−0.05 to 0.19) | 0.58 | 2 of 6 |

Kappa sits far below raw agreement because 97% of the hand labels are "supported": two raters who
nearly always say "supported" agree often by chance, and kappa discounts that. NLI's raw agreement
is the highest for the same reason; it says "supported" almost every time, including for four of
the six flagged answers. With six flagged answers, every interval is wide. Labels 1 to 80 were given
on screen and 81 to 200 from a PDF of the same screens; the judge agreed with 95.0% of each part.

Against the judge on all 4,228 answered questions, the cheaper checks agree little beyond chance:
an NLI classifier (DeBERTa-v3-large, supported at entailment probability 0.5 or more, a cut-off
fixed in advance) reaches Cohen's kappa 0.22 (95% interval 0.17 to 0.27, ROC AUC 0.74), and word
overlap 0.12 (0.08 to 0.15, AUC 0.59).

### Tables

Keeping table structure did not help. Every setup above chunks tables with their structure kept:
each table its own chunks, rows as "Column: value", the header repeated in each chunk. The winner
was re-run with its tables flattened into loose text, and compared on the table questions:

| Table questions | Flattened | Structured | Difference (95% interval) | p |
|---|---|---|---|---|
| A top-10 passage contains a reference answer (607) | 0.908 | 0.895 | +0.013 (−0.010, +0.036) | 0.32 |
| Reciprocal rank of the first such passage (607) | 0.681 | 0.647 | +0.034 (+0.008, +0.061) | 0.012 |
| Written answer judged correct (189) | 73.0% | 73.5% | −0.5 points (−5.3, +4.2) | 1.00 |
| Written answer fully supported (189) | 80.4% | 81.5% | −1.1 points (−6.9, +4.2) | 0.86 |

Flattened tables find the answer as often, rank it slightly higher, and give the writer the same
accuracy. Span-overlap measures, used for the grid, show a larger gap in flattening's favour
(hit@10 0.904 against 0.832), but mostly because of counting: a table answer's marked span is often
the whole table, and the structured layout splits a table into more chunks, so there are more
"relevant" chunks to find. The comparison above uses measures that count the same way for both.

## How it works

```
Natural Questions (pinned commit, SHA-256 per file)
  -> pages rebuilt from tokens (tables keep rows and headers)
  -> chunks: fixed | sentence-aware | section-aware, ~150 words, "Title > Section" prefix
  -> first stage: BM25 | MiniLM | Qwen3-Embedding | BM25+MiniLM | BM25+Qwen3 (reciprocal rank fusion)
  -> optional reranker: bge-reranker-v2-m3 over the top 50
  -> metrics at passage and page level, by answer type; significance tests (Holm)
  -> stage two, five setups x 1,000 questions:
       Qwen3.8-27B (local) writes one sentence from the top 5, or "I don't know"
       GPT-6 Luna (high reasoning effort) judges groundedness and correctness
       word overlap, NLI and containment score the same answers
       200 blind hand labels -> Cohen's kappa for every rung
  -> report -> explorer (static site, or local server with live search)
```

- **Reproducible runs.** Each run is defined by a TOML file in `configs/` and writes its own
  directory with the config and its digest, the git commit, the machine, the pinned model
  revisions, per-stage timings, every per-question metric, and rankings in TREC format.
- **Metrics you can check.** recall@k, nDCG@k, MRR@k and precision@k follow trec_eval's
  conventions and are checked against trec_eval on random rankings in CI.
- **Significance.** Paired randomisation tests, bootstrap intervals and Holm's correction, on
  the two claims the results make: the winner beats each other setup, and the reranker helps.
- **Regression gate.** CI re-runs a committed 100-question slice with BM25 and MiniLM and fails if
  passage recall@10 or nDCG@10 falls below the committed baseline.

Design decisions and the options rejected are recorded in [`docs/decisions/`](docs/decisions/).

## Running it

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/). Stage two also needs
[Ollama](https://ollama.com) serving the writer model and an OpenAI API key. Times are from the
development machine (Apple M5 Pro, 24 GB).

```bash
uv sync --locked
uv run groundwork build-corpus                 # download and rebuild the corpus
uv run groundwork run configs/grid/*.toml      # all 30 setups: about 13 hours, mostly Qwen3 encoding
uv run groundwork report                       # results/report/<time>/report.md
```

Stage two, on the setups the rule picks from the grid:

```bash
uv run groundwork stage-two-config             # configs/stage2.toml, with the rule and scores
uv run groundwork answer configs/stage2.toml   # 5,000 answers, about 7 s each
uv run groundwork retime results/stage-two/<run>    # latency on a fixed sample, uncached
cp .env.example .env                           # then add OPENAI_API_KEY
uv run --env-file .env groundwork judge results/stage-two/<run> --budget 4
uv run groundwork label results/stage-two/<run>     # 200 blind hand labels
uv run groundwork report --verdicts results/stage-two-verdicts/<run> \
  --tables-verdicts results/stage-two-tables-verdicts/<run> --labels results/labels/stage-two.jsonl
```

The explorer:

```bash
uv run groundwork site --answers results/stage-two/<run> --verdicts results/stage-two-verdicts/<run> \
  --tables-verdicts results/stage-two-tables-verdicts/<run> \
  --labels results/labels/stage-two.jsonl      # results/site/data
cd src/explorer && npm ci && npm run build     # the static site, in src/explorer/dist
uv run groundwork serve --site src/explorer/dist --live configs/grid/fixed-bm25.toml configs/grid/fixed-qwen3-rerank.toml
```

Or in a container (about 12 GB of memory; the `Dockerfile` gives the command). Containers on a Mac
cannot use its GPU, so there the reranker takes about 18 s a question, against about 0.6 s when
served directly on Apple silicon.

Every model is pinned to a revision (the writer to its weights digest), every run is seeded and
written to its own directory under `results/`, and every model response is cached, so a rerun
reproduces the published numbers without calling a model again.

## Limitations

- One corpus: Wikipedia pages via Natural Questions. The findings may not transfer to other
  kinds of documents.
- Answers are written and judged for five setups on 1,000 questions, not for all thirty.
- One person made the hand labels, so agreement between people is not measured. Only 6 of the
  200 were not fully supported, so the agreement intervals are wide; sampling more of the
  answers the judge flags would narrow them.
- The hand labels check groundedness only. The judge's correctness labels are checked only against
  mechanical containment, which misses paraphrases.
- Chunk size, BM25 parameters, the fusion constant and the rerank depth are published defaults,
  not tuned.
- Every timing comes from one laptop; the artefacts record its state.
- Pages are rebuilt from the Natural Questions token stream, so section headings keep Wikipedia's
  "( edit )" link text and punctuation carries stray spaces. Every setup sees the same text.
- The top four setups tie statistically; the recommended one is the simplest of equals.
- The table ablation compares one setup on one corpus; with 189 written answers, a difference
  smaller than about five points could go unseen.

## Licence

Code: MIT. Corpus text and annotations: CC BY-SA 3.0 (Natural Questions, Wikipedia).
