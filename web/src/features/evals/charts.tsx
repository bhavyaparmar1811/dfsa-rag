import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import type { CacheMetrics, Compare } from "../../api/types";
import { cx } from "../../components/ui";
import { configLabel, num, pct } from "../../lib/format";
import { ChartTooltip } from "./parts";

const AXIS = { fontSize: 11.5, fill: "var(--ink-3)", fontFamily: "var(--font-sans)" };

// Colour follows the config, never its rank: the same config is the same colour
// on every chart. Slots are the validated categorical palette (tokens.css).
export const STAGE_SERIES: { config: string; color: string; swatch: string }[] = [
  { config: "dense", color: "var(--series-2)", swatch: "bg-series-2" },
  { config: "sparse", color: "var(--series-4)", swatch: "bg-series-4" },
  { config: "hybrid", color: "var(--series-3)", swatch: "bg-series-3" },
  { config: "hybrid_rerank", color: "var(--series-1)", swatch: "bg-series-1" },
];

export function Legend({
  items,
}: {
  items: { label: string; swatch: string; line?: boolean; emphasis?: boolean }[];
}) {
  return (
    <div className="flex flex-wrap gap-x-5 gap-y-1.5">
      {items.map((i) => (
        <span key={i.label} className="inline-flex items-center gap-2 text-[12px] text-ink-2">
          <span className={cx(i.line ? "h-[2px] w-4 rounded-full" : "h-2.5 w-2.5 rounded-[3px]", i.swatch)} />
          <span className={i.emphasis ? "font-medium text-ink" : undefined}>{i.label}</span>
        </span>
      ))}
    </div>
  );
}

// --- recall@k per retrieval stage -------------------------------------------------

const KS = [1, 3, 6, 20, 40];

export function RecallCurve({ compare }: { compare: Compare }) {
  const series = STAGE_SERIES.filter((s) => compare.runs.some((r) => r.config_name === s.config));
  const byConfig = Object.fromEntries(compare.runs.map((r) => [r.config_name, r.metrics]));
  const data = KS.map((k) => ({
    k: `@${k}`,
    ...Object.fromEntries(series.map((s) => [s.config, byConfig[s.config]?.[`recall@${k}`] as number])),
  }));
  const min = Math.min(
    ...data.flatMap((d) => series.map((s) => (d as Record<string, number | string>)[s.config] as number)),
  );
  const floor = Math.max(0, Math.floor(min * 10) / 10);
  const step = 1 - floor > 0.4 ? 0.2 : 0.1;
  const ticks = Array.from({ length: Math.round((1 - floor) / step) + 1 }, (_, i) =>
    Number((floor + i * step).toFixed(2)),
  );

  return (
    <div>
      <Legend
        items={series.map((s) => ({
          label: configLabel(s.config),
          swatch: s.swatch,
          line: true,
          emphasis: s.config === "hybrid_rerank",
        }))}
      />
      <div className="mt-4 h-[260px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
            <CartesianGrid vertical={false} stroke="var(--grid)" strokeWidth={1} />
            <XAxis dataKey="k" tick={AXIS} tickLine={false} axisLine={{ stroke: "var(--line)" }} />
            <YAxis
              domain={[floor, 1]}
              ticks={ticks}
              tick={AXIS}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
              width={52}
            />
            <Tooltip
              cursor={{ stroke: "var(--line-strong)", strokeWidth: 1 }}
              content={({ active, payload, label }) =>
                active && payload?.length ? (
                  <ChartTooltip
                    title={`Recall${label}: gold provision in the top ${String(label).slice(1)}`}
                    rows={[...payload]
                      .sort((a, b) => (b.value as number) - (a.value as number))
                      .map((p) => ({
                        color: String(p.color),
                        label: configLabel(String(p.dataKey)),
                        value: pct(p.value as number),
                        bold: p.dataKey === "hybrid_rerank",
                      }))}
                  />
                ) : null
              }
            />
            {series.map((s) => (
              <Line
                key={s.config}
                dataKey={s.config}
                stroke={s.color}
                strokeWidth={s.config === "hybrid_rerank" ? 2.5 : 2}
                strokeLinecap="round"
                strokeLinejoin="round"
                dot={{ r: 4, fill: s.color, stroke: "var(--surface)", strokeWidth: 2 }}
                activeDot={{ r: 5, fill: s.color, stroke: "var(--surface)", strokeWidth: 2 }}
                isAnimationActive
                animationDuration={900}
                animationEasing="ease-out"
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// --- the answer cache decision ---------------------------------------------------------

export function SimilarityStrip({ metrics }: { metrics: CacheMetrics }) {
  const jitter = (i: number) => ((i * 37) % 11) / 11 - 0.5; // deterministic spread
  const pos = metrics.pairs
    .filter((p) => p.kind === "positive")
    .map((p, i) => ({ ...p, x: p.similarity, y: 1 + jitter(i) * 0.36 }));
  const neg = metrics.pairs
    .filter((p) => p.kind === "negative")
    .map((p, i) => ({ ...p, x: p.similarity, y: jitter(i + 3) * 0.36 }));
  const worst = metrics.max_guard_passing_negative;
  const xmin = Math.floor(Math.min(...metrics.pairs.map((p) => p.similarity)) * 10) / 10;

  return (
    <div>
      <Legend
        items={[
          { label: "Paraphrase (should be served)", swatch: "bg-series-1" },
          { label: "Hard negative (must not be served)", swatch: "bg-series-2" },
        ]}
      />
      <div className="mt-3 h-[200px]">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 4, right: 16, bottom: 0, left: 0 }}>
            <CartesianGrid horizontal={false} stroke="var(--grid)" />
            <XAxis
              type="number"
              dataKey="x"
              domain={[xmin, 1]}
              ticks={Array.from({ length: Math.round((1 - xmin) * 10) + 1 }, (_, i) =>
                Number((xmin + i / 10).toFixed(1)),
              )}
              tick={AXIS}
              tickLine={false}
              axisLine={{ stroke: "var(--line)" }}
              tickFormatter={(v: number) => v.toFixed(1)}
            />
            <YAxis
              type="number"
              dataKey="y"
              domain={[-0.5, 1.5]}
              ticks={[0, 1]}
              tick={AXIS}
              tickLine={false}
              axisLine={false}
              width={92}
              tickFormatter={(v: number) => (v === 1 ? "Paraphrases" : "Hard negatives")}
            />
            <ZAxis range={[46, 46]} />
            {worst && (
              <ReferenceLine
                x={worst.similarity}
                stroke="var(--ink)"
                strokeWidth={1}
                label={{
                  value: `closest negative ${worst.similarity.toFixed(3)}`,
                  position: "insideTopRight",
                  offset: 0,
                  fill: "var(--ink-2)",
                  fontSize: 11.5,
                }}
              />
            )}
            <Tooltip
              cursor={false}
              content={({ active, payload }) => {
                const p = payload?.[0]?.payload as (typeof pos)[number] | undefined;
                return active && p ? (
                  <ChartTooltip
                    title={p.id}
                    rows={[
                      {
                        color: p.kind === "positive" ? "var(--series-1)" : "var(--series-2)",
                        label: p.kind === "positive" ? "Paraphrase" : "Hard negative",
                        value: `cosine ${p.similarity.toFixed(3)}`,
                      },
                      ...(p.guards.length
                        ? [{ color: "transparent", label: "stopped by guard", value: p.guards.join(", ") }]
                        : []),
                    ]}
                  />
                ) : null;
              }}
            />
            <Scatter data={pos} fill="var(--series-1)" stroke="var(--surface)" strokeWidth={2} />
            <Scatter data={neg} fill="var(--series-2)" stroke="var(--surface)" strokeWidth={2} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export function SweepChart({ metrics }: { metrics: CacheMetrics }) {
  const data = metrics.sweep.map((s) => ({
    threshold: s.threshold,
    naive: s.naive_false_hits,
    guarded: s.guarded_false_hits,
    rate: s.guarded_hit_rate,
  }));
  return (
    <div>
      <Legend
        items={[
          { label: "Similarity alone", swatch: "bg-series-2", line: true },
          { label: "Similarity + guards (terms, numbers, citations)", swatch: "bg-series-1", line: true },
        ]}
      />
      <div className="mt-3 h-[220px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: -16 }}>
            <CartesianGrid vertical={false} stroke="var(--grid)" />
            <XAxis
              dataKey="threshold"
              type="number"
              domain={["dataMin", "dataMax"]}
              tick={AXIS}
              tickLine={false}
              axisLine={{ stroke: "var(--line)" }}
              tickFormatter={(v: number) => v.toFixed(2)}
            />
            <YAxis tick={AXIS} tickLine={false} axisLine={false} allowDecimals={false} width={48} />
            <ReferenceLine y={0} stroke="var(--line-strong)" />
            <Tooltip
              cursor={{ stroke: "var(--line-strong)", strokeWidth: 1 }}
              content={({ active, payload, label }) =>
                active && payload?.length ? (
                  <ChartTooltip
                    title={`Threshold ${num(label as number, 3)}`}
                    rows={[
                      { color: "var(--series-2)", label: "False hits, similarity alone", value: String(payload[0]?.payload.naive) },
                      { color: "var(--series-1)", label: "False hits, with guards", value: String(payload[0]?.payload.guarded), bold: true },
                      { color: "transparent", label: "Paraphrases served", value: pct(payload[0]?.payload.rate, 0) },
                    ]}
                  />
                ) : null
              }
            />
            <Line dataKey="naive" stroke="var(--series-2)" strokeWidth={2} dot={false} type="stepAfter" animationDuration={900} />
            <Line dataKey="guarded" stroke="var(--series-1)" strokeWidth={2.5} dot={false} type="stepAfter" animationDuration={900} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-1 text-[11.5px] text-ink-4">Wrong answers served (of {metrics.n_negative} hard negatives), by similarity threshold</div>
    </div>
  );
}
