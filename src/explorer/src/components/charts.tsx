// Charts follow one set of rules: at most two categorical colours (validated for colour-blind
// separation in both themes), thin marks, recessive grid, a legend for two series, direct labels
// only where the story is, and a tooltip on every mark.

import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts"

import type { RetrievalRow } from "@/lib/data"
import { fixed3, milliseconds, parseSetup } from "@/lib/format"

const AXIS = { stroke: "var(--chart-axis)", fontSize: 12, tickLine: false }
const GRID = <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="0" vertical={false} />
// Latency ticks for the log axis: round numbers a reader can hold in mind.
const LATENCY_TICKS = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

// Legend text stays in the text colour; the swatch beside it carries the series colour.
const legendText = (value: string) => <span style={{ color: "var(--foreground)" }}>{value}</span>

const TOOLTIP_STYLE = {
  background: "var(--popover)",
  border: "1px solid var(--border)",
  borderRadius: 8,
  color: "var(--popover-foreground)",
  fontSize: 12,
}

interface Point {
  setup: string
  latency: number
  ndcg: number
  label: string
}

/** nDCG@10 against per-question latency; reranked setups in one colour, the rest in another. */
export function QualityLatencyChart({ rows, winner }: { rows: RetrievalRow[]; winner: string }) {
  const points = rows.map((row) => ({
    setup: row.setup,
    latency: row.retrieval_p50_ms + (row.rerank_p50_ms ?? 0),
    ndcg: row["passage.ndcg@10"],
    label: row.setup === winner ? row.setup : "",
  }))
  const series = [
    { name: "Reranked", colour: "var(--series-1)", points: points.filter((p) => parseSetup(p.setup).rerank) },
    { name: "First stage only", colour: "var(--series-2)", points: points.filter((p) => !parseSetup(p.setup).rerank) },
  ]
  return (
    <ResponsiveContainer width="100%" height={320}>
      <ScatterChart margin={{ top: 24, right: 24, bottom: 8, left: 0 }}>
        {GRID}
        <XAxis
          {...AXIS}
          type="number"
          dataKey="latency"
          scale="log"
          domain={([low, high]: readonly number[]) => [low / 1.5, high * 1.5]}
          name="p50 latency"
          ticks={LATENCY_TICKS.filter(
            (tick) =>
              tick >= Math.min(...points.map((p) => p.latency)) / 2 &&
              tick <= Math.max(...points.map((p) => p.latency)) * 2,
          )}
          tickFormatter={(value: number) => milliseconds(value)}
        />
        <YAxis
          {...AXIS}
          type="number"
          dataKey="ndcg"
          name="nDCG@10"
          domain={([low, high]: readonly number[]) => [Math.floor(low / 0.05) * 0.05, Math.ceil(high / 0.05) * 0.05]}
          tickCount={5}
          tickFormatter={fixed3}
        />
        <ZAxis range={[110, 110]} />
        <Tooltip
          contentStyle={TOOLTIP_STYLE}
          cursor={{ stroke: "var(--chart-grid)" }}
          content={({ payload }) => {
            const point = payload?.[0]?.payload as Point | undefined
            if (!point) return null
            return (
              <div style={TOOLTIP_STYLE} className="px-3 py-2">
                <div className="font-mono">{point.setup}</div>
                <div>nDCG@10 {fixed3(point.ndcg)}</div>
                <div>p50 latency {milliseconds(point.latency)}</div>
              </div>
            )
          }}
        />
        <Legend verticalAlign="bottom" height={28} iconType="circle" formatter={legendText} />
        {series.map((item) => (
          <Scatter
            key={item.name}
            name={item.name}
            data={item.points}
            fill={item.colour}
            stroke="var(--background)"
            strokeWidth={2}
            isAnimationActive={false}
          >
            <LabelList dataKey="label" position="top" fontSize={11} fill="var(--foreground)" />
          </Scatter>
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  )
}

/** Recall@10 per answer type for up to two setups, side by side. */
export function ByTypeChart({ series }: { series: { name: string; values: Record<string, number> }[] }) {
  const types = ["paragraph", "list", "table"]
  const data = types.map((type) => ({
    type,
    ...Object.fromEntries(series.map((item) => [item.name, item.values[type] ?? 0])),
  }))
  const colours = ["var(--series-1)", "var(--series-2)"]
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 16, right: 16, bottom: 0, left: 0 }} barGap={2}>
        {GRID}
        <XAxis {...AXIS} dataKey="type" />
        <YAxis {...AXIS} domain={[0, 1]} tickFormatter={fixed3} />
        <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(value) => (typeof value === "number" ? fixed3(value) : String(value))} cursor={{ fill: "var(--muted)" }} />
        {series.length > 1 && (
          <Legend
            verticalAlign="top"
            height={28}
            iconType="circle"
            formatter={legendText}
            itemSorter={(item) => series.findIndex((entry) => entry.name === item.value)}
          />
        )}
        {series.slice(0, 2).map((item, index) => (
          <Bar
            key={item.name}
            dataKey={item.name}
            fill={colours[index]}
            maxBarSize={24}
            radius={[4, 4, 0, 0]}
            isAnimationActive={false}
          >
            <LabelList
              dataKey={item.name}
              position="top"
              fontSize={11}
              fill="var(--muted-foreground)"
              formatter={(value) => (typeof value === "number" ? fixed3(value) : "")}
            />
          </Bar>
        ))}
      </BarChart>
    </ResponsiveContainer>
  )
}

/** A confusion table as a single-hue heatmap: darker cells hold more answers. */
export function Confusion({
  matrix,
  rowLabel,
  columnLabel,
  labels,
}: {
  matrix: Record<string, Record<string, number>>
  rowLabel: string
  columnLabel: string
  labels: string[]
}) {
  const counts = labels.flatMap((row) => labels.map((column) => matrix[row]?.[column] ?? 0))
  const largest = Math.max(1, ...counts)
  const step = (count: number) => 1 + Math.round((count / largest) * 4)
  return (
    <table className="text-xs">
      <thead>
        <tr>
          <th className="p-1 text-left font-normal text-muted-foreground">
            {rowLabel} ↓ / {columnLabel} →
          </th>
          {labels.map((label) => (
            <th key={label} className="p-1 font-normal text-muted-foreground">
              {label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {labels.map((row) => (
          <tr key={row}>
            <th className="p-1 text-left font-normal text-muted-foreground">{row}</th>
            {labels.map((column) => {
              const count = matrix[row]?.[column] ?? 0
              return (
                <td
                  key={column}
                  className="h-10 w-16 rounded-sm border-2 border-background text-center tabular-nums"
                  style={{ background: `var(--seq-${step(count)})`, color: `var(--seq-ink-${step(count)})` }}
                  title={`${rowLabel} ${row}, ${columnLabel} ${column}: ${count}`}
                >
                  {count}
                </td>
              )
            })}
          </tr>
        ))}
      </tbody>
    </table>
  )
}
