import { useState } from "react"

import { Kpi } from "@/components/Kpi"
import { Confusion } from "@/components/charts"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import type { Agreement, BaselineAgreement, JudgeFile, Report } from "@/lib/data"
import { useData } from "@/lib/data"
import { GROUNDEDNESS_LABELS, fixed3, percent } from "@/lib/format"

const interval = (low?: number, high?: number) =>
  low === undefined || high === undefined ? "" : `95% interval [${fixed3(low)}, ${fixed3(high)}]`

/** Hand-label agreement carries its interval as two fields; the cards read the judge file's pair. */
const asBaseline = (agreement: BaselineAgreement & Agreement): BaselineAgreement => ({
  ...agreement,
  kappa_interval: [agreement.kappa_low, agreement.kappa_high],
})

function BaselineCard({ label, agreement }: { label: string; agreement: BaselineAgreement | undefined }) {
  if (!agreement || agreement.undefined) {
    return <Kpi label={label} value="–" detail={agreement?.undefined ?? "not measured yet"} />
  }
  return (
    <Kpi
      label={label}
      value={`κ ${fixed3(agreement.kappa)}`}
      detail={`${interval(agreement.kappa_interval?.[0], agreement.kappa_interval?.[1])} · AUC ${fixed3(agreement.auc)} · supported at ≥ ${agreement.cut_off}`}
    />
  )
}

export function Judge({ report }: { report: Report }) {
  const judge = useData<JudgeFile>(report.stage_two ? "judge.json" : null)
  const human = report.human_agreement
  const [open, setOpen] = useState<number | null>(null)

  if (!report.stage_two) {
    return (
      <p className="text-muted-foreground">
        Stage two has not been judged yet: this page fills in once answers are written and scored.
      </p>
    )
  }

  const scale = human?.scale ?? ["supported", "partly_supported", "not_supported"]
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Does the judge agree with people?</h1>
        <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
          The judge ({report.stage_two.judge.model}, {report.stage_two.judge.thinking} thinking) labels each
          answer's groundedness. It is compared with two cheaper checks and with {human?.labels ?? 200} answers
          labelled by hand, blind to the setup and to the judge. Cohen's κ is agreement corrected for
          chance: 1 is perfect, 0 is no better than chance.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Kpi
          label="Judge against hand labels"
          value={human ? `κ ${fixed3(human.judge.kappa)}` : "pending"}
          detail={
            human
              ? `${interval(human.judge.kappa_low, human.judge.kappa_high)} · raw agreement ${percent(human.judge.observed)}`
              : "Hand labels not yet made"
          }
        />
        <BaselineCard label="NLI against hand labels" agreement={human && asBaseline(human.nli)} />
        <BaselineCard label="Word overlap against hand labels" agreement={human && asBaseline(human.lexical)} />
        <Kpi label="Judging cost" value={`$${report.stage_two.judge.spent.toFixed(2)}`} detail="At the dated list price in the verdicts artefact" />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <BaselineCard label="NLI against the judge, all answers" agreement={report.stage_two.agreement_with_judge.nli} />
        <BaselineCard
          label="Word overlap against the judge, all answers"
          agreement={report.stage_two.agreement_with_judge.lexical}
        />
      </div>

      {human && (
        <div className="grid gap-4 lg:grid-cols-3">
          <ConfusionCard title="Hand label against judge" agreement={human.judge} labels={scale} columns="judge" />
          <ConfusionCard title="Hand label against NLI" agreement={human.nli} labels={["supported", "not"]} columns="NLI" />
          <ConfusionCard
            title="Hand label against word overlap"
            agreement={human.lexical}
            labels={["supported", "not"]}
            columns="overlap"
          />
        </div>
      )}

      {judge.state === "ready" && (
        <Card>
          <CardHeader>
            <CardTitle>Where the judge and the hand label disagree</CardTitle>
            <CardDescription>{judge.value.disagreements.length} answers</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {judge.value.disagreements.map((row, index) => (
              <div key={`${row.setup}-${row.question_id}`} className="rounded-md border">
                <button
                  className="flex w-full flex-wrap items-center gap-2 px-3 py-2 text-left text-sm"
                  onClick={() => setOpen(open === index ? null : index)}
                >
                  <span className="min-w-0 flex-1">{row.question}</span>
                  <Badge variant="outline">hand: {GROUNDEDNESS_LABELS[row.human]}</Badge>
                  <Badge variant="secondary">judge: {GROUNDEDNESS_LABELS[row.judge ?? ""] ?? "–"}</Badge>
                </button>
                {open === index && (
                  <div className="space-y-2 border-t px-3 py-3 text-sm">
                    <p>
                      <span className="text-muted-foreground">Answer: </span>
                      {row.answer}
                    </p>
                    {row.judge_reason && <p className="text-muted-foreground">Judge: {row.judge_reason}</p>}
                    <p className="text-xs text-muted-foreground">
                      NLI {fixed3(row.nli)} · word overlap {fixed3(row.lexical)} · setup{" "}
                      <span className="font-mono">{row.setup}</span>
                    </p>
                    <ol className="list-decimal space-y-2 pl-5 text-xs">
                      {row.passages.map((passage, position) => (
                        <li key={position} className="whitespace-pre-wrap">
                          {passage}
                        </li>
                      ))}
                    </ol>
                  </div>
                )}
              </div>
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function ConfusionCard({
  title,
  agreement,
  labels,
  columns,
}: {
  title: string
  agreement: Agreement | BaselineAgreement
  labels: string[]
  columns: string
}) {
  if (!agreement.confusion) return null
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent className="overflow-x-auto">
        <Confusion matrix={agreement.confusion} labels={labels} rowLabel="hand" columnLabel={columns} />
      </CardContent>
    </Card>
  )
}
