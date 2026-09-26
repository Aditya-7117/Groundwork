import { Kpi } from "@/components/Kpi"
import { SetupTable } from "@/components/SetupTable"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { ByTypeChart, QualityLatencyChart } from "@/components/charts"
import type { Comparison, Report } from "@/lib/data"
import { fixed3, pValue, percent, signed } from "@/lib/format"
import { href } from "@/lib/route"

export const BASELINE = "fixed-bm25"

const ABLATION_ROWS: [string, string][] = [
  ["answer_hit@10", "A top-10 passage contains a reference answer"],
  ["answer_rr@10", "Rank of the first such passage (reciprocal rank)"],
  ["span_hit@10", "A top-10 passage overlaps the answer's table (span measure)"],
  ["span_rr@10", "Rank of the first overlapping passage (span measure)"],
]

const WRITER_ROWS: [string, string][] = [
  ["correct_judge", "Written answer judged correct"],
  ["correct_containment", "Written answer contains a reference"],
  ["supported", "Written answer judged fully supported"],
  ["declined", "Writer declined"],
]
const ALPHA = 0.05

export function Significant({ row }: { row: Comparison }) {
  return row.p_holm < ALPHA ? (
    <Badge variant="secondary">significant</Badge>
  ) : (
    <Badge variant="outline">not significant</Badge>
  )
}

export function Overview({ report }: { report: Report }) {
  const { winner } = report.significance
  const rows = report.retrieval
  const best = rows.find((row) => row.setup === winner)
  const baseline = rows.find((row) => row.setup === BASELINE)
  const againstWinner = report.significance.winner_against_each["passage.ndcg@10"] ?? []
  const vsBaseline = againstWinner.find((row) => row.b === BASELINE)
  const reranker = report.significance.reranker["passage.ndcg@10"] ?? []
  const helped = reranker.filter((row) => row.difference > 0 && row.p_holm < ALPHA).length

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Which retrieval setup finds the answer?</h1>
        <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
          Thirty setups (three chunkers, five first stages, with and without a reranker) scored on all
          3,220 Natural Questions questions, at passage level: a passage counts only if it holds the
          human-marked answer.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Kpi
          label="Winner by nDCG@10"
          value={best ? fixed3(best["passage.ndcg@10"]) : "–"}
          detail={<a className="font-mono hover:underline" href={href("setups", winner)}>{winner}</a>}
        />
        <Kpi
          label="Against the BM25 baseline"
          value={vsBaseline ? signed(vsBaseline.difference) : "–"}
          detail={
            vsBaseline &&
            `95% interval [${signed(vsBaseline.ci_low)}, ${signed(vsBaseline.ci_high)}], Holm p ${pValue(vsBaseline.p_holm)}`
          }
        />
        <Kpi
          label="Reranker helped significantly"
          value={`${helped} of ${reranker.length}`}
          detail="First stages whose nDCG@10 rose with the reranker, Holm-corrected"
        />
        <Kpi
          label="Recall@10 on table questions"
          value={best ? fixed3(best["recall@10_by_type"].table) : "–"}
          detail={baseline && `BM25 baseline ${fixed3(baseline["recall@10_by_type"].table)}`}
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>All setups</CardTitle>
          <CardDescription>Sorted by nDCG@10. Select a setup to see every question.</CardDescription>
        </CardHeader>
        <CardContent>
          <SetupTable rows={rows} winner={winner} />
        </CardContent>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Quality against latency</CardTitle>
            <CardDescription>
              nDCG@10 against median time per question (log scale), retrieval plus reranking
            </CardDescription>
          </CardHeader>
          <CardContent>
            <QualityLatencyChart rows={rows} winner={winner} />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Recall@10 by answer type</CardTitle>
            <CardDescription>Where the answer sits in its page: a paragraph, a list or a table</CardDescription>
          </CardHeader>
          <CardContent>
            {best && baseline && (
              <ByTypeChart
                series={[
                  { name: winner, values: best["recall@10_by_type"] },
                  { name: BASELINE, values: baseline["recall@10_by_type"] },
                ]}
              />
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Is the winner really the winner?</CardTitle>
          <CardDescription>
            Paired randomisation test on nDCG@10 per question, 10,000 resamples, Holm-corrected across
            all {againstWinner.length} comparisons
          </CardDescription>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{winner} against</TableHead>
                <TableHead className="text-right">Difference</TableHead>
                <TableHead className="text-right">95% interval</TableHead>
                <TableHead className="text-right">Holm p</TableHead>
                <TableHead />
              </TableRow>
            </TableHeader>
            <TableBody>
              {[...againstWinner]
                .sort((a, b) => a.difference - b.difference)
                .map((row) => (
                  <TableRow key={row.b}>
                    <TableCell className="font-mono text-xs">{row.b}</TableCell>
                    <TableCell className="text-right tabular-nums">{signed(row.difference)}</TableCell>
                    <TableCell className="text-right tabular-nums">
                      [{signed(row.ci_low)}, {signed(row.ci_high)}]
                    </TableCell>
                    <TableCell className="text-right tabular-nums">{pValue(row.p_holm)}</TableCell>
                    <TableCell>
                      <Significant row={row} />
                    </TableCell>
                  </TableRow>
                ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>


      {report.table_ablation && (
        <Card>
          <CardHeader>
            <CardTitle>Did keeping table structure help?</CardTitle>
            <CardDescription>
              {winner} against the same setup with its tables flattened into loose text, on the{" "}
              {report.table_ablation.questions} table questions. Paired randomisation tests; the first two
              rows decide it, because they count the same way for both table layouts.
            </CardDescription>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Measure</TableHead>
                  <TableHead className="text-right">Flattened</TableHead>
                  <TableHead className="text-right">Structured</TableHead>
                  <TableHead className="text-right">Difference</TableHead>
                  <TableHead className="text-right">95% interval</TableHead>
                  <TableHead className="text-right">p</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {[
                  ...ABLATION_ROWS.map(([key, label]) => [label, report.table_ablation?.comparisons[key]?.[0]] as const),
                  ...WRITER_ROWS.map(([key, label]) => [label, report.table_writer?.comparisons[key]?.[0]] as const),
                ]
                  .filter((entry): entry is readonly [string, Comparison] => entry[1] !== undefined)
                  .map(([label, row]) => (
                    <TableRow key={label}>
                      <TableCell className="whitespace-normal">{label}</TableCell>
                      <TableCell className="text-right tabular-nums">{fixed3(row.mean_a)}</TableCell>
                      <TableCell className="text-right tabular-nums">{fixed3(row.mean_b)}</TableCell>
                      <TableCell className="text-right tabular-nums">{signed(row.difference)}</TableCell>
                      <TableCell className="text-right tabular-nums">
                        [{signed(row.ci_low)}, {signed(row.ci_high)}]
                      </TableCell>
                      <TableCell className="text-right tabular-nums">{pValue(row.p_value)}</TableCell>
                    </TableRow>
                  ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}

      {report.stage_two && (
        <Card>
          <CardHeader>
            <CardTitle>Answers from the stage-two setups</CardTitle>
            <CardDescription>
              1,000 questions each, stratified by answer type. Correct and declined rates are over all
              questions; supported is over answered questions.
            </CardDescription>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Setup</TableHead>
                  <TableHead className="text-right">Correct (judge)</TableHead>
                  <TableHead className="text-right">Correct (contains reference)</TableHead>
                  <TableHead className="text-right">Declined</TableHead>
                  <TableHead className="text-right">Supported (judge)</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {Object.entries(report.stage_two.setups).map(([setup, summary]) => (
                  <TableRow key={setup}>
                    <TableCell className="font-mono text-xs">
                      <a className="hover:underline" href={href("setups", setup)}>
                        {setup}
                      </a>
                    </TableCell>
                    <TableCell className="text-right tabular-nums">{percent(summary.correct_judge)}</TableCell>
                    <TableCell className="text-right tabular-nums">{percent(summary.correct_containment)}</TableCell>
                    <TableCell className="text-right tabular-nums">{percent(summary.decline_rate)}</TableCell>
                    <TableCell className="text-right tabular-nums">
                      {percent(summary.groundedness_judge.supported)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
      <Card>
        <CardHeader>
          <CardTitle>Limitations</CardTitle>
        </CardHeader>
        <CardContent>
          <ul className="list-disc space-y-1 pl-5 text-sm text-muted-foreground">
            <li>One corpus: Wikipedia pages from Natural Questions. Findings may not transfer to other documents.</li>
            <li>Answers are written and judged for five setups on 1,000 questions, not for all thirty.</li>
            <li>One person made the hand labels, so agreement between people is not measured.</li>
            <li>Chunk size, BM25 parameters, fusion constant and rerank depth are published defaults, not tuned.</li>
            <li>Every timing comes from one laptop (Apple M5 Pro, 24 GB); the artefacts record its state.</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  )
}
