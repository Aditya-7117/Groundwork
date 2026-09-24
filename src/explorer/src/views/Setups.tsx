import { useMemo, useState } from "react"

import { Kpi } from "@/components/Kpi"
import { ByTypeChart } from "@/components/charts"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import type { Question, Report, SetupFile, SetupQuestion, StageTwoAnswer } from "@/lib/data"
import { questionMetric, rankedPassages, useData } from "@/lib/data"
import { GROUNDEDNESS_LABELS, fixed3, milliseconds } from "@/lib/format"
import { href } from "@/lib/route"
import { BASELINE } from "@/views/Overview"

const PAGE = 50

export function Setups({
  report,
  setup,
  question,
}: {
  report: Report
  setup: string | null
  question: string | null
}) {
  const name = setup ?? report.significance.winner
  const file = useData<SetupFile>(`setups/${name}.json`)
  const questions = useData<Record<string, Question>>("questions.json")
  const staged = report.stage_two?.setups[name] !== undefined
  const answers = useData<StageTwoAnswer[]>(staged ? `stage2/${name}.json` : null)
  const baseline = report.retrieval.find((row) => row.setup === BASELINE)

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="font-mono text-xl font-semibold">{name}</h1>
          {file.state === "ready" && <p className="text-sm text-muted-foreground">{file.value.description}</p>}
        </div>
        <label className="text-sm text-muted-foreground">
          Setup{" "}
          <select
            className="rounded-md border bg-background px-2 py-1 font-mono text-xs text-foreground"
            value={name}
            onChange={(event) => (window.location.hash = href("setups", event.target.value))}
          >
            {report.retrieval.map((row) => (
              <option key={row.setup} value={row.setup}>
                {row.setup}
              </option>
            ))}
          </select>
        </label>
      </div>

      {file.state === "loading" && <p className="text-muted-foreground">Loading {name}…</p>}
      {file.state === "missing" && <p className="text-muted-foreground">No data for {name}.</p>}
      {file.state === "ready" && questions.state === "ready" && (
        <>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Kpi label="Recall@10" value={fixed3(file.value.aggregate["passage.recall@10"])} />
            <Kpi label="nDCG@10" value={fixed3(file.value.aggregate["passage.ndcg@10"])} />
            <Kpi label="MRR@10" value={fixed3(file.value.aggregate["passage.mrr@10"])} />
            <Kpi
              label="p50 latency per question"
              value={milliseconds(
                (file.value.timing.retrieval_latency_ms.p50 ?? 0) + (file.value.timing.rerank_latency_ms.p50 ?? 0),
              )}
              detail={`${file.value.chunks.toLocaleString()} chunks indexed`}
            />
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Recall@10 by answer type</CardTitle>
                <CardDescription>Against the BM25 baseline over fixed chunks</CardDescription>
              </CardHeader>
              <CardContent>
                <ByTypeChart
                  series={[
                    {
                      name,
                      values: Object.fromEntries(
                        Object.entries(file.value.by_answer_type).map(([type, metrics]) => [
                          type,
                          metrics["passage.recall@10"],
                        ]),
                      ),
                    },
                    ...(baseline && name !== BASELINE
                      ? [{ name: BASELINE, values: baseline["recall@10_by_type"] }]
                      : []),
                  ]}
                />
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>How it ran</CardTitle>
                <CardDescription>Models with their pinned revisions, and time per stage</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                {Object.entries(file.value.models).map(([role, model]) => (
                  <div key={role}>
                    <span className="text-muted-foreground">{role}: </span>
                    <span className="font-mono text-xs">
                      {String(model.name ?? model.method ?? "")}
                      {model.revision ? ` @ ${String(model.revision).slice(0, 10)}` : ""}
                      {model.precision ? ` · ${String(model.precision)} on ${String(model.device)}` : ""}
                    </span>
                  </div>
                ))}
                <div className="flex flex-wrap gap-x-4 gap-y-1 text-muted-foreground">
                  {Object.entries(file.value.timing.stage_seconds).map(([stage, seconds]) => (
                    <span key={stage}>
                      {stage} {seconds >= 60 ? `${(seconds / 60).toFixed(1)} min` : `${seconds.toFixed(1)} s`}
                    </span>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
          <QuestionBrowser
            setup={name}
            file={file.value}
            questions={questions.value}
            answers={answers.state === "ready" ? answers.value : null}
            selected={question}
          />
        </>
      )}
    </div>
  )
}

function QuestionBrowser({
  setup,
  file,
  questions,
  answers,
  selected,
}: {
  setup: string
  file: SetupFile
  questions: Record<string, Question>
  answers: StageTwoAnswer[] | null
  selected: string | null
}) {
  const [text, setText] = useState("")
  const [type, setType] = useState("all")
  const [found, setFound] = useState("all")
  const [page, setPage] = useState(0)
  const byQuestion = useMemo(
    () => new Map((answers ?? []).map((answer) => [answer.question_id, answer])),
    [answers],
  )

  const rows = useMemo(() => {
    const needle = text.trim().toLowerCase()
    return Object.entries(file.questions).filter(([id, entry]) => {
      const q = questions[id]
      const hit = questionMetric(file, entry, "passage.recall@10") > 0
      return (
        (!needle || q.question.toLowerCase().includes(needle)) &&
        (type === "all" || q.type === type) &&
        (found === "all" || (found === "found") === hit)
      )
    })
  }, [file, questions, text, type, found])

  const shown = rows.slice(page * PAGE, (page + 1) * PAGE)
  const detail = selected !== null && file.questions[selected] ? selected : null

  return (
    <div className="grid gap-4 xl:grid-cols-[1fr_1.1fr]">
      <Card>
        <CardHeader>
          <CardTitle>Questions</CardTitle>
          <CardDescription>
            {rows.length.toLocaleString()} shown. "Found" means a passage holding the answer is in the top 10.
          </CardDescription>
          <div className="flex flex-wrap gap-2 pt-2">
            <Input
              className="max-w-56"
              placeholder="Search questions"
              value={text}
              onChange={(event) => {
                setText(event.target.value)
                setPage(0)
              }}
            />
            <select className="rounded-md border bg-background px-2 text-sm" value={type} onChange={(event) => setType(event.target.value)}>
              <option value="all">All types</option>
              <option value="paragraph">Paragraph</option>
              <option value="table">Table</option>
              <option value="list">List</option>
            </select>
            <select className="rounded-md border bg-background px-2 text-sm" value={found} onChange={(event) => setFound(event.target.value)}>
              <option value="all">Found or missed</option>
              <option value="found">Found</option>
              <option value="missed">Missed</option>
            </select>
          </div>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Question</TableHead>
                <TableHead>Type</TableHead>
                <TableHead className="text-right">nDCG@10</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {shown.map(([id, entry]) => (
                <TableRow key={id} className={id === detail ? "bg-muted" : undefined}>
                  <TableCell className="whitespace-normal">
                    <a className="hover:underline" href={href("setups", setup, { q: id })}>
                      {questions[id].question}
                    </a>
                    {byQuestion.has(id) && (
                      <Badge variant="outline" className="ml-2">
                        answered
                      </Badge>
                    )}
                  </TableCell>
                  <TableCell className="text-muted-foreground">{questions[id].type}</TableCell>
                  <TableCell className="text-right tabular-nums">
                    {fixed3(questionMetric(file, entry, "passage.ndcg@10"))}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <div className="mt-3 flex items-center justify-between text-sm text-muted-foreground">
            <Button variant="outline" size="sm" disabled={page === 0} onClick={() => setPage(page - 1)}>
              Previous
            </Button>
            <span>
              Page {page + 1} of {Math.max(1, Math.ceil(rows.length / PAGE))}
            </span>
            <Button
              variant="outline"
              size="sm"
              disabled={(page + 1) * PAGE >= rows.length}
              onClick={() => setPage(page + 1)}
            >
              Next
            </Button>
          </div>
        </CardContent>
      </Card>
      {detail ? (
        <QuestionDetail
          id={detail}
          question={questions[detail]}
          file={file}
          entry={file.questions[detail]}
          answer={byQuestion.get(detail) ?? null}
        />
      ) : (
        <Card>
          <CardContent className="py-10 text-center text-sm text-muted-foreground">
            Select a question to see what this setup retrieved for it.
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function QuestionDetail({
  id,
  question,
  file,
  entry,
  answer,
}: {
  id: string
  question: Question
  file: SetupFile
  entry: SetupQuestion
  answer: StageTwoAnswer | null
}) {
  return (
    <Card>
      <CardHeader>
        <CardDescription>
          Question {id} · {question.type} answer · page "{question.page}"
        </CardDescription>
        <CardTitle className="text-lg">{question.question}</CardTitle>
        <p className="text-sm text-muted-foreground">
          Reference answers: {question.references.length ? question.references.join(" · ") : "none"}
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        {answer && (
          <div className="space-y-2 rounded-md border p-3">
            <div className="text-xs uppercase tracking-wide text-muted-foreground">Written answer</div>
            <p className="font-medium">{answer.answer}</p>
            <div className="flex flex-wrap gap-2">
              {answer.declined ? (
                <Badge variant="outline">declined</Badge>
              ) : (
                <>
                  <Badge variant="secondary">
                    {GROUNDEDNESS_LABELS[answer.verdicts.groundedness ?? ""] ?? "not judged"}
                  </Badge>
                  <Badge variant="secondary">{answer.verdicts.correctness ?? "not judged"}</Badge>
                  <Badge variant="outline">NLI {fixed3(answer.verdicts.nli)}</Badge>
                  <Badge variant="outline">word overlap {fixed3(answer.verdicts.lexical)}</Badge>
                  <Badge variant="outline">
                    {answer.verdicts.contains_reference ? "contains a reference" : "no reference found"}
                  </Badge>
                </>
              )}
            </div>
            {answer.verdicts.groundedness_reason && (
              <p className="text-xs text-muted-foreground">Judge: {answer.verdicts.groundedness_reason}</p>
            )}
          </div>
        )}
        <div>
          <div className="mb-2 text-xs uppercase tracking-wide text-muted-foreground">Top 10 passages</div>
          <ol className="space-y-1">
            {rankedPassages(file, entry).map((passage, index) => (
              <li key={index} className="flex items-start gap-2 text-sm">
                <span className="w-5 shrink-0 text-right tabular-nums text-muted-foreground">{index + 1}</span>
                <span className="min-w-0 flex-1">
                  {passage.heading}
                  {passage.table && <span className="ml-1 text-xs text-muted-foreground">(table)</span>}
                  {answer && index < answer.passages.length && (
                    <details className="mt-1 text-xs text-muted-foreground">
                      <summary className="cursor-pointer">passage text</summary>
                      <p className="mt-1 whitespace-pre-wrap">{answer.passages[index].split("\n").slice(1).join("\n")}</p>
                    </details>
                  )}
                </span>
                {passage.relevant && <Badge>holds the answer</Badge>}
              </li>
            ))}
          </ol>
        </div>
      </CardContent>
    </Card>
  )
}
