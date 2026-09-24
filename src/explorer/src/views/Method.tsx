import type { ReactNode } from "react"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { Report } from "@/lib/data"

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 text-sm leading-relaxed text-muted-foreground">{children}</CardContent>
    </Card>
  )
}

const Formula = ({ children }: { children: ReactNode }) => (
  <code className="block rounded-md bg-muted px-3 py-2 font-mono text-xs text-foreground">{children}</code>
)

export function Method({ report }: { report: Report }) {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Method</h1>
        <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
          Every number on this site comes from a run artefact produced at commit{" "}
          <span className="font-mono">{report.commit.slice(0, 12)}</span>. Nothing here is computed in the
          browser.
        </p>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <Section title="Corpus">
          <p>
            The Natural Questions validation split: real Google queries, each with a human-marked answer in a
            Wikipedia page. Pages are rebuilt from the dataset's token stream, pinned by commit and SHA-256, with
            one revision per article: 6,930 pages and 3,220 questions (2,377 paragraph, 608 table and 235 list
            answers). Licensed CC BY-SA 3.0.
          </p>
        </Section>
        <Section title="Chunking">
          <p>
            About 150 words with about 30 words of overlap, three ways: cut blindly, whole sentences, or whole
            sentences without crossing a section heading. Every chunk starts with its page title and section.
            Tables are their own chunks, one "Column: value" row per line, with the header repeated in each chunk.
          </p>
        </Section>
        <Section title="Relevance">
          <p>
            A passage is relevant when it overlaps the human-marked answer span (passage level, the primary
            measure). Page level, where the page holding the answer counts, is reported beside it. Questions
            whose answer no chunk covers are excluded and listed, never scored as zero.
          </p>
        </Section>
        <Section title="Metrics (trec_eval conventions, checked against trec_eval in CI)">
          <p>recall@k: the share of relevant passages found in the top k.</p>
          <p>nDCG@k: relevant passages discounted by rank, divided by the best possible ranking's score.</p>
          <Formula>DCG@k = Σ gainᵢ / log₂(i + 1),  nDCG@k = DCG@k / IDCG@k</Formula>
          <p>
            Worked example: relevant at ranks 2 and 4 of five, three relevant in all. DCG = 0.631 + 0.431 = 1.062;
            ideal = 1 + 0.631 + 0.5 = 2.131; nDCG@5 = 0.498.
          </p>
          <p>MRR@k: the mean of 1 / (rank of the first relevant passage), 0 if none is in the top k.</p>
        </Section>
        <Section title="Significance">
          <p>
            Paired randomisation tests on per-question differences (10,000 sign flips), 95% bootstrap intervals,
            and Holm's correction within each family: the winner against every other setup, and each first stage
            with against without the reranker. Tested on nDCG@10 and recall@10.
          </p>
        </Section>
        <Section title="Answers and judging">
          <p>
            Five setups answer 1,000 questions stratified by answer type. Qwen3.8-27B, running locally with
            thinking off and temperature 0, writes one sentence from the top five passages or says "I don't know".
            Gemini 3.8 Flash labels groundedness (supported, partly, not) without seeing the reference, and
            correctness against the reference. Word overlap and an NLI classifier score the same answers; 200
            answers were labelled by hand, blind. Agreement is Cohen's κ with a bootstrap interval.
          </p>
        </Section>
        <Section title="Reproduce any number">
          <Formula>uv sync --locked</Formula>
          <Formula>uv run groundwork build-corpus</Formula>
          <Formula>uv run groundwork run configs/grid/*.toml</Formula>
          <Formula>uv run groundwork report</Formula>
          <p>
            Each run writes results/&lt;setup&gt;/&lt;UTC start&gt;-&lt;config digest&gt;/ with its config, commit,
            machine, models and every per-question number, and its rankings in TREC format.
          </p>
        </Section>
      </div>
    </div>
  )
}
