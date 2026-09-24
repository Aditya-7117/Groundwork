import { ArrowDown } from "lucide-react"
import { useMemo, useState } from "react"

import { Badge } from "@/components/ui/badge"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import type { RetrievalRow } from "@/lib/data"
import { RETRIEVER_LABELS, fixed3, milliseconds, parseSetup } from "@/lib/format"
import { href } from "@/lib/route"

const COLUMNS = [
  { key: "passage.recall@10", label: "Recall@10" },
  { key: "passage.ndcg@10", label: "nDCG@10" },
  { key: "passage.mrr@10", label: "MRR@10" },
  { key: "passage.recall@100", label: "Recall@100" },
  { key: "page.recall@10", label: "Page R@10" },
] as const

type SortKey = (typeof COLUMNS)[number]["key"] | "latency"

const latency = (row: RetrievalRow): number => row.retrieval_p50_ms + (row.rerank_p50_ms ?? 0)

export function SetupTable({ rows, winner }: { rows: RetrievalRow[]; winner: string }) {
  const [chunker, setChunker] = useState("all")
  const [retriever, setRetriever] = useState("all")
  const [rerank, setRerank] = useState("all")
  const [sort, setSort] = useState<SortKey>("passage.ndcg@10")

  const shown = useMemo(() => {
    const filtered = rows.filter((row) => {
      const parts = parseSetup(row.setup)
      return (
        (chunker === "all" || parts.chunker === chunker) &&
        (retriever === "all" || parts.retriever === retriever) &&
        (rerank === "all" || String(parts.rerank) === rerank)
      )
    })
    const value = (row: RetrievalRow) => (sort === "latency" ? -latency(row) : row[sort])
    return [...filtered].sort((a, b) => value(b) - value(a))
  }, [rows, chunker, retriever, rerank, sort])

  const select = (label: string, value: string, set: (value: string) => void, options: [string, string][]) => (
    <label className="flex items-center gap-2 text-sm text-muted-foreground">
      {label}
      <select
        className="rounded-md border bg-background px-2 py-1 text-foreground"
        value={value}
        onChange={(event) => set(event.target.value)}
      >
        <option value="all">All</option>
        {options.map(([option, text]) => (
          <option key={option} value={option}>
            {text}
          </option>
        ))}
      </select>
    </label>
  )

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-4">
        {select("Chunker", chunker, setChunker, [
          ["fixed", "Fixed"],
          ["sentence", "Sentence"],
          ["section", "Section"],
        ])}
        {select("First stage", retriever, setRetriever, Object.entries(RETRIEVER_LABELS))}
        {select("Reranker", rerank, setRerank, [
          ["true", "On"],
          ["false", "Off"],
        ])}
        <span className="ml-auto self-center text-xs text-muted-foreground">
          {shown.length} of {rows.length} setups · passage level unless stated
        </span>
      </div>
      <div className="overflow-x-auto rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Setup</TableHead>
              {COLUMNS.map((column) => (
                <TableHead key={column.key} className="text-right">
                  <button className="inline-flex items-center gap-1" onClick={() => setSort(column.key)}>
                    {column.label}
                    {sort === column.key && <ArrowDown className="size-3" />}
                  </button>
                </TableHead>
              ))}
              <TableHead className="text-right">
                <button className="inline-flex items-center gap-1" onClick={() => setSort("latency")}>
                  p50 latency
                  {sort === "latency" && <ArrowDown className="size-3" />}
                </button>
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {shown.map((row) => (
              <TableRow key={row.setup}>
                <TableCell>
                  <a className="font-mono text-xs hover:underline" href={href("setups", row.setup)}>
                    {row.setup}
                  </a>
                  {row.setup === winner && (
                    <Badge className="ml-2" variant="secondary">
                      winner
                    </Badge>
                  )}
                </TableCell>
                {COLUMNS.map((column) => (
                  <TableCell key={column.key} className="text-right tabular-nums">
                    {fixed3(row[column.key])}
                  </TableCell>
                ))}
                <TableCell className="text-right tabular-nums">{milliseconds(latency(row))}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  )
}
