// Types and loaders for the files `groundwork site` writes. Nothing here computes a new number:
// the explorer only displays what the report and the runs recorded.

import { useEffect, useState } from "react"

export type Metrics = Record<string, number>

export interface RetrievalRow {
  setup: string
  "passage.recall@10": number
  "passage.ndcg@10": number
  "passage.mrr@10": number
  "passage.recall@100": number
  "page.recall@10": number
  "recall@10_by_type": Record<string, number>
  retrieval_p50_ms: number
  rerank_p50_ms: number | null
  embedding_seconds: number | null
  chunks: number
  path: string
}

export interface Comparison {
  a: string
  b: string
  metric: string
  questions: number
  mean_a: number
  mean_b: number
  difference: number
  ci_low: number
  ci_high: number
  p_value: number
  p_holm: number
}

export type Family = Record<string, Comparison[]>

export interface Agreement {
  items: number
  observed: number
  kappa: number
  kappa_low: number
  kappa_high: number
  confusion: Record<string, Record<string, number>>
}

export interface BaselineAgreement {
  cut_off: number
  kappa?: number
  kappa_interval?: [number, number]
  observed?: number
  confusion?: Record<string, Record<string, number>>
  auc?: number
  undefined?: string
}

export interface JudgeAgreement {
  answers: number
  nli: BaselineAgreement
  lexical: BaselineAgreement
}

export interface StageTwoSummary {
  questions: number
  declined: number
  decline_rate: number
  correct_containment: number
  correct_judge: number
  groundedness_judge: Record<string, number>
  supported_nli: number
  supported_lexical: number
  judge_cost: number
}

export interface Report {
  primary_metric: string
  commit: string
  retrieval: RetrievalRow[]
  significance: { winner: string; winner_against_each: Family; reranker: Family }
  table_ablation?: { questions: number; primary: string[]; comparisons: Family; commit?: string } | null
  table_writer?: { questions: number; comparisons: Family }
  stage_two?: {
    path: string
    setups: Record<string, StageTwoSummary>
    agreement_with_judge: JudgeAgreement
    judge: { model: string; thinking: string; spent: number }
  }
  human_agreement?: {
    labels: number
    scale: string[]
    judge: Agreement
    nli: BaselineAgreement & Agreement
    lexical: BaselineAgreement & Agreement
  }
}

export interface Question {
  question: string
  type: string
  references: string[]
  page: string
}

export interface RankedPassage {
  heading: string
  table: boolean
  relevant: boolean
}

/** One question in a setup file: metrics in `question_metrics` order, passages as
 * [heading index, holds the answer, is a table]. */
export interface SetupQuestion {
  metrics: number[]
  top: [number, number, number][]
}

export interface SetupFile {
  setup: string
  description: string
  config: Record<string, unknown>
  aggregate: Metrics
  by_answer_type: Record<string, Metrics>
  timing: {
    stage_seconds: Record<string, number>
    retrieval_latency_ms: Record<string, number>
    rerank_latency_ms: Record<string, number>
  }
  models: Record<string, Record<string, unknown>>
  chunks: number
  question_metrics: string[]
  headings: string[]
  questions: Record<string, SetupQuestion>
}

/** A question's value for one metric, looked up by name. */
export function questionMetric(file: SetupFile, question: SetupQuestion, name: string): number {
  return question.metrics[file.question_metrics.indexOf(name)] ?? 0
}

/** A question's top passages, expanded from their compact form. */
export function rankedPassages(file: SetupFile, question: SetupQuestion): RankedPassage[] {
  return question.top.map(([heading, relevant, table]) => ({
    heading: file.headings[heading] ?? "",
    relevant: relevant === 1,
    table: table === 1,
  }))
}

export interface Verdicts {
  groundedness: string | null
  groundedness_reason: string | null
  correctness: string | null
  correctness_reason: string | null
  contains_reference: boolean | null
  nli: number | null
  lexical: number | null
}

export interface StageTwoAnswer {
  question_id: string
  answer: string
  declined: boolean
  passages: string[]
  chunk_ids: string[]
  verdicts: Verdicts
}

export interface Disagreement {
  setup: string
  question_id: string
  question: string
  answer: string
  passages: string[]
  human: string
  judge: string | null
  judge_reason: string | null
  nli: number | null
  lexical: number | null
}

export interface JudgeFile {
  agreement_with_judge: JudgeAgreement
  human_agreement: Report["human_agreement"] | null
  disagreements: Disagreement[]
}

const cache = new Map<string, Promise<unknown>>()

/** Fetch a data file once; later calls share the same request. */
export function load<T>(path: string): Promise<T> {
  const url = `${import.meta.env.BASE_URL}data/${path}`
  if (!cache.has(url)) {
    cache.set(
      url,
      fetch(url).then((response) => {
        if (!response.ok) throw new Error(`${path}: ${response.status}`)
        return response.json()
      }),
    )
  }
  return cache.get(url) as Promise<T>
}

export type Loaded<T> =
  | { state: "loading" }
  | { state: "missing"; error: string }
  | { state: "ready"; value: T }

/** Load a data file into component state. A missing file is a normal state, not a crash. */
export function useData<T>(path: string | null): Loaded<T> {
  const [entry, setEntry] = useState<{ path: string | null; result: Loaded<T> }>({
    path: null,
    result: { state: "loading" },
  })
  useEffect(() => {
    if (path === null) return
    let current = true
    load<T>(path).then(
      (value) => current && setEntry({ path, result: { state: "ready", value } }),
      (error: unknown) => current && setEntry({ path, result: { state: "missing", error: String(error) } }),
    )
    return () => {
      current = false
    }
  }, [path])
  // A result for another path is stale: until this path's file arrives, it is loading.
  return entry.path === path ? entry.result : { state: "loading" }
}
