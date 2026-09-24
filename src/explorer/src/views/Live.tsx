import { type FormEvent, useEffect, useState } from "react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import type { Report } from "@/lib/data"
import { milliseconds } from "@/lib/format"

interface LiveSetup {
  name: string
  description: string
}

interface LivePassage {
  id: string
  heading: string
  text: string
  kind: string
}

interface LiveResult {
  setup: string
  passages: LivePassage[]
  retrieval_ms: number
  rerank_ms: number | null
}

type Outcome = { state: "idle" } | { state: "running"; started: number } | { state: "done"; result: LiveResult } | { state: "failed"; error: string }

export function Live({ report }: { report: Report }) {
  const [setups, setSetups] = useState<LiveSetup[]>([])
  const [question, setQuestion] = useState("")
  const [outcomes, setOutcomes] = useState<Record<string, Outcome>>({})
  const [now, setNow] = useState(0)

  useEffect(() => {
    fetch("/api/live")
      .then((response) => response.json() as Promise<{ setups: LiveSetup[] }>)
      .then((body) => setSetups(body.setups))
      .catch(() => setSetups([]))
  }, [])

  const running = Object.values(outcomes).some((outcome) => outcome.state === "running")
  useEffect(() => {
    if (!running) return
    const timer = window.setInterval(() => setNow(Date.now()), 100)
    return () => window.clearInterval(timer)
  }, [running])

  const search = (event: FormEvent) => {
    event.preventDefault()
    if (!question.trim()) return
    setNow(Date.now())
    for (const setup of setups) {
      setOutcomes((current) => ({ ...current, [setup.name]: { state: "running", started: Date.now() } }))
      fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ setup: setup.name, question }),
      })
        .then(async (response) => {
          if (!response.ok) throw new Error(`${response.status}`)
          return (await response.json()) as LiveResult
        })
        .then(
          (result) => setOutcomes((current) => ({ ...current, [setup.name]: { state: "done", result } })),
          (error: unknown) =>
            setOutcomes((current) => ({ ...current, [setup.name]: { state: "failed", error: String(error) } })),
        )
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Live search</h1>
        <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
          Ask your own question. Each setup runs the same code the evaluation ran, on this machine's CPU. The
          winner of the evaluation was <span className="font-mono">{report.significance.winner}</span>.
        </p>
      </div>
      <form onSubmit={search} className="flex max-w-2xl gap-2">
        <Input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="who wrote the declaration of independence" maxLength={500} />
        <Button type="submit" disabled={running || setups.length === 0}>
          Search
        </Button>
      </form>
      <div className="grid gap-4 lg:grid-cols-2">
        {setups.map((setup) => {
          const outcome = outcomes[setup.name] ?? { state: "idle" }
          return (
            <Card key={setup.name}>
              <CardHeader>
                <CardTitle className="font-mono text-sm">{setup.name}</CardTitle>
                <CardDescription>{setup.description}</CardDescription>
                {outcome.state === "running" && (
                  <Badge variant="outline">searching… {milliseconds(now - outcome.started)}</Badge>
                )}
                {outcome.state === "done" && (
                  <div className="flex gap-2">
                    <Badge variant="secondary">retrieval {milliseconds(outcome.result.retrieval_ms)}</Badge>
                    {outcome.result.rerank_ms !== null && (
                      <Badge variant="secondary">reranking {milliseconds(outcome.result.rerank_ms)}</Badge>
                    )}
                  </div>
                )}
              </CardHeader>
              <CardContent>
                {outcome.state === "failed" && <p className="text-sm text-muted-foreground">Search failed: {outcome.error}</p>}
                {outcome.state === "done" && (
                  <ol className="space-y-3">
                    {outcome.result.passages.map((passage, index) => (
                      <li key={passage.id} className="text-sm">
                        <div className="font-medium">
                          {index + 1}. {passage.heading}
                          {passage.kind === "table" && <span className="ml-1 text-xs text-muted-foreground">(table)</span>}
                        </div>
                        <p className="line-clamp-3 text-muted-foreground">{passage.text}</p>
                      </li>
                    ))}
                  </ol>
                )}
              </CardContent>
            </Card>
          )
        })}
      </div>
    </div>
  )
}
