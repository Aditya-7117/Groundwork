// Formatting and setup-name parsing shared by every view.

export const fixed3 = (value: number | null | undefined): string =>
  value === null || value === undefined ? "–" : value.toFixed(3)

export const percent = (value: number | null | undefined, digits = 1): string =>
  value === null || value === undefined ? "–" : `${(value * 100).toFixed(digits)}%`

export const signed = (value: number): string => `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(3)}`

export const milliseconds = (value: number | null | undefined): string =>
  value === null || value === undefined
    ? "–"
    : value >= 1000
      ? `${(value / 1000).toFixed(1)} s`
      : `${value.toFixed(0)} ms`

export const pValue = (value: number): string => (value < 0.001 ? "< 0.001" : value.toFixed(3))

export interface SetupParts {
  chunker: "fixed" | "sentence" | "section"
  retriever: string
  rerank: boolean
  flatTables: boolean
}

const CHUNKERS = ["fixed", "sentence", "section"] as const

/** Split a grid setup name such as "section-hybrid-qwen3-rerank" into its parts. */
export function parseSetup(name: string): SetupParts {
  const parts = name.split("-")
  const chunker = CHUNKERS.find((candidate) => candidate === parts[0]) ?? "fixed"
  const flatTables = name.endsWith("-flat-tables")
  const rest = parts.slice(1, flatTables ? -2 : undefined)
  const rerank = rest.at(-1) === "rerank"
  return {
    chunker,
    retriever: (rerank ? rest.slice(0, -1) : rest).join("-"),
    rerank,
    flatTables,
  }
}

export const RETRIEVER_LABELS: Record<string, string> = {
  bm25: "BM25",
  minilm: "MiniLM",
  qwen3: "Qwen3-Embedding",
  "hybrid-minilm": "BM25 + MiniLM",
  "hybrid-qwen3": "BM25 + Qwen3",
}

export const GROUNDEDNESS_LABELS: Record<string, string> = {
  supported: "Supported",
  partly_supported: "Partly supported",
  not_supported: "Not supported",
}
