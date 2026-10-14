import { motion } from "motion/react";
import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useCostStats } from "../../api/hooks";
import type { CostStats } from "../../api/types";
import {
  Card,
  EmptyState,
  ErrorNote,
  Page,
  PageHeader,
  Section,
  Skeleton,
  Stat,
  Tabs,
  Td,
  Th,
  cx,
} from "../../components/ui";
import { REFUSAL_LABELS, day, int, ms, pct, usd } from "../../lib/format";
import { DUR, EASE, cascade } from "../../theme/motion";
import { Legend } from "../evals/charts";
import { ChartTooltip } from "../evals/parts";

// The same colour per stage as the answer's latency bar in the chat.
const STAGES = [
  { key: "embedding", label: "Query embeddings", provider: "OpenAI text-embedding-3-small", color: "var(--series-3)", swatch: "bg-series-3" },
  { key: "rewrite", label: "Follow-up rewriting", provider: "Azure OpenAI GPT-4o", color: "var(--series-4)", swatch: "bg-series-4" },
  { key: "answer", label: "Answers", provider: "Azure OpenAI GPT-4o", color: "var(--series-2)", swatch: "bg-series-2" },
] as const;

type Range = "all" | "30" | "7" | "today";

const isoDay = (d: Date) => d.toISOString().slice(0, 10);
function since(range: Range): string | null {
  if (range === "all") return null;
  const d = new Date();
  if (range !== "today") d.setUTCDate(d.getUTCDate() - (Number(range) - 1));
  return isoDay(d);
}

export function CostPage() {
  const [range, setRange] = useState<Range>("all");
  const { data, isLoading, error, isFetching } = useCostStats(since(range), null);

  return (
    <Page wide>
      <PageHeader
        eyebrow="Cost & usage"
        title="What each answer costs"
        actions={
          <Tabs
            layoutId="cost-range"
            value={range}
            onChange={setRange}
            options={[
              { value: "today", label: "Today" },
              { value: "7", label: "7 days" },
              { value: "30", label: "30 days" },
              { value: "all", label: "All time" },
            ]}
          />
        }
      >
        Production questions only, from the per-request traces in SQLite. Eval runs record their own
        spend on the run, and ingestion is reported by <span className="font-mono text-[13px]">scripts/cost_report.py</span>.
      </PageHeader>

      {error && <ErrorNote error={error} />}
      {isLoading && (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-[104px]" />
          ))}
        </div>
      )}
      {data && data.requests === 0 && (
        <EmptyState title="No questions in this period">Ask something on the Ask page and it shows up here.</EmptyState>
      )}
      {data && data.requests > 0 && (
        <div className={cx("transition-opacity duration-200", isFetching && "opacity-60")}>
          <Tiles data={data} />
          <Breakdown data={data} />
          <Daily data={data} />
          <Refusals data={data} />
        </div>
      )}
    </Page>
  );
}

function Tiles({ data }: { data: CostStats }) {
  const tiles = [
    { label: "Total spend", value: usd(data.cost_usd.total), note: `${int(data.requests)} questions` },
    { label: "Cost per question", value: usd(data.cost_per_request_usd), note: "rewrite, embeddings and answer" },
    {
      label: "Answer cache hit rate",
      value: pct(data.cache.hit_rate, 0),
      note: `${data.cache.hits} of ${data.cache.lookups} lookups, ${usd(data.cache.saved_usd)} saved`,
    },
    {
      label: "Refused",
      value: pct(data.refusal_rate, 0),
      note: `${data.refused} of ${data.requests - data.errors} questions`,
    },
    {
      label: "Latency, p50",
      value: ms(data.latency_ms.p50),
      note: `p95 ${ms(data.latency_ms.p95)}; cache hit ${ms(data.cache.hit_latency_p50_ms)}`,
    },
  ];
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
      {tiles.map((t, i) => (
        <motion.div
          key={t.label}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: DUR.slow, ease: EASE, delay: cascade(i, 0.05) }}
          className="h-full"
        >
          <Stat label={t.label} value={t.value} note={t.note} />
        </motion.div>
      ))}
    </div>
  );
}

function Breakdown({ data }: { data: CostStats }) {
  const total = data.cost_usd.total || 1;
  const t = data.tokens;
  const tokens: Record<string, string> = {
    embedding: `${int(t.embedding)} tokens`,
    rewrite: `${int(t.rewrite_input)} in, ${int(t.rewrite_output)} out`,
    answer: `${int(t.answer_input)} in, ${int(t.answer_output)} out`,
  };
  return (
    <Section
      title="Where the money goes"
      description="Two providers with separate credentials: OpenAI embeds the query, Azure OpenAI GPT-4o rewrites follow-ups and writes the answer."
    >
      <Card className="p-6">
        <div className="flex h-3 w-full gap-[2px] overflow-hidden rounded-[4px] bg-surface-3">
          {STAGES.map((s, i) => {
            const v = data.cost_usd[s.key];
            if (v <= 0) return null;
            return (
              <motion.div
                key={s.key}
                className="h-full origin-left"
                style={{ width: `${Math.max(0.6, (v / total) * 100)}%`, background: s.color }}
                initial={{ scaleX: 0 }}
                whileInView={{ scaleX: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, ease: EASE, delay: i * 0.08 }}
                title={`${s.label}: ${usd(v)}`}
              />
            );
          })}
        </div>
        <table className="mt-5 w-full border-collapse">
          <thead>
            <tr>
              <Th>Stage</Th>
              <Th>Provider</Th>
              <Th align="right">Tokens</Th>
              <Th align="right">Share</Th>
              <Th align="right">Cost</Th>
            </tr>
          </thead>
          <tbody>
            {STAGES.map((s) => (
              <tr key={s.key}>
                <Td>
                  <span className="inline-flex items-center gap-2">
                    <span className={cx("h-2.5 w-2.5 rounded-[3px]", s.swatch)} />
                    {s.label}
                  </span>
                </Td>
                <Td className="text-ink-3">{s.provider}</Td>
                <Td align="right" className="text-ink-3">
                  {tokens[s.key]}
                </Td>
                <Td align="right">{pct(data.cost_usd[s.key] / total, 1)}</Td>
                <Td align="right" className="font-medium">
                  {usd(data.cost_usd[s.key])}
                </Td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </Section>
  );
}

function Daily({ data }: { data: CostStats }) {
  const rows = useMemo(
    () =>
      data.by_day.map((d) => ({
        day: d.day,
        label: day(d.day),
        embedding: d.cost_usd.embedding,
        rewrite: d.cost_usd.rewrite,
        answer: d.cost_usd.answer,
        d,
      })),
    [data],
  );
  return (
    <Section title="By day" description="UTC days.">
      {rows.length > 1 && (
        <Card className="mb-6 p-5">
          <Legend items={STAGES.map((s) => ({ label: s.label, swatch: s.swatch }))} />
          <div className="mt-4 h-[240px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: -8 }} barCategoryGap="30%">
                <CartesianGrid vertical={false} stroke="var(--grid)" />
                <XAxis
                  dataKey="label"
                  tick={{ fontSize: 11.5, fill: "var(--ink-3)" }}
                  tickLine={false}
                  axisLine={{ stroke: "var(--line)" }}
                />
                <YAxis
                  tick={{ fontSize: 11.5, fill: "var(--ink-3)" }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(v: number) => `$${v.toFixed(2)}`}
                  width={56}
                />
                <Tooltip
                  cursor={{ fill: "var(--surface-2)" }}
                  content={({ active, payload, label }) =>
                    active && payload?.length ? (
                      <ChartTooltip
                        title={label}
                        rows={[
                          ...STAGES.map((s) => ({
                            color: s.color,
                            label: s.label,
                            value: usd(payload[0].payload[s.key]),
                          })),
                          {
                            color: "transparent",
                            label: "Total",
                            value: usd(payload[0].payload.d.cost_usd.total),
                            bold: true,
                          },
                        ]}
                      />
                    ) : null
                  }
                />
                {STAGES.map((s, i) => (
                  <Bar
                    key={s.key}
                    dataKey={s.key}
                    stackId="cost"
                    fill={s.color}
                    stroke="var(--surface)"
                    strokeWidth={1}
                    maxBarSize={24}
                    radius={i === STAGES.length - 1 ? [4, 4, 0, 0] : 0}
                    animationDuration={700}
                  />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}
      <div className="overflow-x-auto">
        <table className="w-full border-collapse">
          <thead>
            <tr>
              <Th>Day</Th>
              <Th align="right">Questions</Th>
              <Th align="right">Refused</Th>
              <Th align="right">Cache hits</Th>
              <Th align="right">p50 latency</Th>
              <Th align="right">Per question</Th>
              <Th align="right">Spend</Th>
            </tr>
          </thead>
          <tbody>
            {[...rows].reverse().map(({ day: key, label, d }) => (
              <tr key={key}>
                <Td>{label}</Td>
                <Td align="right">{d.requests}</Td>
                <Td align="right" className="text-ink-3">
                  {d.refused}
                </Td>
                <Td align="right" className="text-ink-3">
                  {d.cache.hits} / {d.cache.lookups}
                </Td>
                <Td align="right" className="text-ink-3">
                  {ms(d.latency_ms.p50)}
                </Td>
                <Td align="right">{usd(d.cost_per_request_usd)}</Td>
                <Td align="right" className="font-medium">
                  {usd(d.cost_usd.total)}
                </Td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Section>
  );
}

function Refusals({ data }: { data: CostStats }) {
  const codes = Object.entries(data.refusals_by_code).sort((a, b) => b[1] - a[1]);
  if (!codes.length) return null;
  const max = Math.max(...codes.map(([, n]) => n));
  return (
    <Section
      title="Refusals"
      description={`${data.gate_refusals} of ${data.refused} were refused by the confidence gate before any LLM call, so they cost only the query embedding.`}
    >
      <div className="max-w-2xl">
        {codes.map(([code, n], i) => (
          <div key={code} className="grid grid-cols-[220px_1fr_40px] items-center gap-4 border-b border-line py-2.5">
            <div className="text-[13px] text-ink">{REFUSAL_LABELS[code] ?? code}</div>
            <div className="h-1.5 rounded-full bg-surface-3">
              <motion.div
                className="h-1.5 origin-left rounded-full bg-ink-3"
                style={{ width: `${(n / max) * 100}%` }}
                initial={{ scaleX: 0 }}
                whileInView={{ scaleX: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, ease: EASE, delay: cascade(i) }}
              />
            </div>
            <div className="text-right text-[13px] tnum text-ink-2">{n}</div>
          </div>
        ))}
      </div>
    </Section>
  );
}
