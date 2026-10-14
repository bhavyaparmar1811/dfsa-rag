import { CheckCircle, Minus, XCircle } from "@phosphor-icons/react";
import { motion } from "motion/react";
import type { ReactNode } from "react";
import type { GateCheck, GateReport } from "../../api/types";
import { Tag, cx } from "../../components/ui";
import { isRate, isUnit, metricLabel, num, pct } from "../../lib/format";
import { DUR, EASE, cascade } from "../../theme/motion";

// --- formatting a metric value ----------------------------------------------------

export function fmtMetric(name: string, v: unknown): string {
  if (v == null || v === "") return name === "threshold" ? "unset" : "–";
  if (typeof v !== "number") return String(v);
  if (name.startsWith("relevance")) return num(v, 2);
  if (isRate(name)) return pct(v, 1);
  return Number.isInteger(v) ? String(v) : num(v, 3);
}

export function GateTag({ status }: { status: string }) {
  if (status === "passed")
    return (
      <Tag tone="green" icon={<CheckCircle size={11} weight="fill" />}>
        Gate passed
      </Tag>
    );
  if (status === "failed")
    return (
      <Tag tone="red" icon={<XCircle size={11} weight="fill" />}>
        Gate failed
      </Tag>
    );
  return <Tag>Not gated</Tag>;
}

// --- the release gate ---------------------------------------------------------------

type Group = { title: string; note: string; checks: GateCheck[] };

function groupChecks(checks: GateCheck[]): Group[] {
  const pre = new Set(["kind", "status", "reviewed_only", "items", "pairs", "threshold"]);
  const regression = checks.filter((c) => / vs #\d+$/.test(c.name));
  const rest = checks.filter((c) => !pre.has(c.name) && !regression.includes(c));
  return [
    { title: "Floors", note: "must be at or above", checks: rest.filter((c) => c.op === ">=") },
    { title: "Ceilings", note: "must be at or below", checks: rest.filter((c) => c.op === "<=") },
    {
      title: "No regression",
      note: "within 3% of the last passing run",
      checks: regression,
    },
    { title: "Preconditions", note: "what makes a run eligible", checks: checks.filter((c) => pre.has(c.name)) },
  ].filter((g) => g.checks.length > 0);
}

export function GateChecks({ report }: { report: GateReport }) {
  const groups = groupChecks(report.checks);
  let i = 0;
  return (
    <div className="grid gap-x-10 gap-y-8 lg:grid-cols-2">
      {groups.map((g) => (
        <div key={g.title}>
          <div className="mb-1 flex items-baseline justify-between border-b border-line pb-2">
            <span className="text-[12.5px] font-semibold text-ink">{g.title}</span>
            <span className="text-[11.5px] text-ink-4">{g.note}</span>
          </div>
          {g.checks.map((c) => (
            <CheckRow key={c.name} check={c} delay={cascade(i++, 0.025, 20)} />
          ))}
        </div>
      ))}
    </div>
  );
}

function CheckRow({ check: c, delay }: { check: GateCheck; delay: number }) {
  const base = c.name.replace(/ vs #\d+$/, "");
  const value = typeof c.value === "number" ? c.value : null;
  const limit = typeof c.limit === "number" ? c.limit : null;
  // A bar only where the scale is known: rates on 0-1, relevance on 1-5.
  const scale = base.startsWith("relevance") ? [1, 5] : isUnit(base) ? [0, 1] : null;
  const pos = (v: number) => (scale ? (v - scale[0]) / (scale[1] - scale[0]) : 0);
  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: DUR.slow, ease: EASE, delay }}
      className="grid grid-cols-[16px_1fr_auto] items-center gap-x-3 border-b border-line py-2.5 last:border-b-0"
    >
      {c.passed ? (
        <CheckCircle size={15} weight="fill" className="text-good" aria-label="passed" />
      ) : (
        <XCircle size={15} weight="fill" className="text-critical" aria-label="failed" />
      )}
      <div className="min-w-0">
        <div className="truncate text-[13px] text-ink">{metricLabel(c.name)}</div>
        {scale && value != null && limit != null && (
          <div className="relative mt-1.5 h-1 rounded-full bg-surface-3">
            <motion.div
              className={cx("absolute inset-y-0 left-0 origin-left rounded-full", c.passed ? "bg-ink-3" : "bg-critical")}
              style={{ width: `${Math.max(0, Math.min(1, pos(value))) * 100}%` }}
              initial={{ scaleX: 0 }}
              whileInView={{ scaleX: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, ease: EASE, delay: delay + 0.1 }}
            />
            <div
              className="absolute -top-[3px] h-[10px] w-[2px] rounded-full bg-ink"
              style={{ left: `calc(${Math.max(0, Math.min(1, pos(limit))) * 100}% - 1px)` }}
              title={`limit ${fmtMetric(base, limit)}`}
            />
          </div>
        )}
      </div>
      <div className="text-right text-[12.5px] tnum">
        <span className="font-medium text-ink">{fmtMetric(base, c.value)}</span>
        <span className="text-ink-4">
          {" "}
          {c.op === "==" ? "=" : c.op === ">=" ? "≥" : "≤"} {fmtMetric(base, c.limit)}
        </span>
        {c.note && !c.note.startsWith("baseline") && (
          <div className="max-w-[220px] text-[11px] text-ink-4">{c.note}</div>
        )}
        {c.note.startsWith("baseline") && (
          <div className="text-[11px] text-ink-4">
            was {fmtMetric(base, Number(c.note.replace("baseline ", "")))}
          </div>
        )}
      </div>
    </motion.div>
  );
}

// --- answer outcomes ------------------------------------------------------------------

export const OUTCOMES = [
  { key: "correct_answer", label: "Correct answer", color: "bg-good", tone: "green" as const },
  { key: "correct_refusal", label: "Correct refusal", color: "bg-series-1", tone: "blue" as const },
  { key: "false_refusal", label: "False refusal", color: "bg-series-4", tone: "yellow" as const },
  { key: "wrong_answer", label: "Wrong answer", color: "bg-critical", tone: "red" as const },
];

export function OutcomeBar({ outcomes }: { outcomes: Record<string, number> }) {
  const total = OUTCOMES.reduce((a, o) => a + (outcomes[o.key] ?? 0), 0) || 1;
  return (
    <div>
      <div className="flex h-3 w-full gap-[2px] overflow-hidden rounded-[4px]">
        {OUTCOMES.filter((o) => (outcomes[o.key] ?? 0) > 0).map((o, i) => (
          <motion.div
            key={o.key}
            className={cx("h-full origin-left", o.color)}
            style={{ width: `${((outcomes[o.key] ?? 0) / total) * 100}%` }}
            initial={{ scaleX: 0, opacity: 0 }}
            whileInView={{ scaleX: 1, opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, ease: EASE, delay: i * 0.08 }}
            title={`${o.label}: ${outcomes[o.key]}`}
          />
        ))}
      </div>
      <div className="mt-3 flex flex-wrap gap-x-6 gap-y-2">
        {OUTCOMES.map((o) => (
          <div key={o.key} className="flex items-center gap-2 text-[12.5px]">
            <span className={cx("h-2.5 w-2.5 rounded-[3px]", o.color)} />
            <span className="text-ink-2">{o.label}</span>
            <span className="tnum font-medium text-ink">{outcomes[o.key] ?? 0}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// --- small pieces --------------------------------------------------------------------

/** Key-value list, two columns. */
export function Facts({ items }: { items: [ReactNode, ReactNode][] }) {
  return (
    <dl className="grid grid-cols-[max-content_1fr] gap-x-6 gap-y-2 text-[13px]">
      {items.map(([k, v], i) => (
        <div key={i} className="contents">
          <dt className="text-ink-3">{k}</dt>
          <dd className="min-w-0 truncate text-ink">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function Dash() {
  return <Minus size={12} className="inline text-ink-4" />;
}

/** Tooltip body for Recharts: text in text tokens, a swatch carries identity. */
export function ChartTooltip({
  title,
  rows,
}: {
  title: ReactNode;
  rows: { color: string; label: ReactNode; value: ReactNode; bold?: boolean }[];
}) {
  return (
    <div className="min-w-[180px] rounded-lg border border-line bg-surface px-3 py-2.5 shadow-float">
      <div className="mb-1.5 text-[11.5px] font-medium text-ink-3">{title}</div>
      {rows.map((r, i) => (
        <div key={i} className="flex items-center gap-2 py-0.5 text-[12.5px]">
          <span className="h-2 w-2 shrink-0 rounded-[2px]" style={{ background: r.color }} />
          <span className={cx("flex-1", r.bold ? "font-medium text-ink" : "text-ink-2")}>{r.label}</span>
          <span className="tnum font-medium text-ink">{r.value}</span>
        </div>
      ))}
    </div>
  );
}
