import { ArrowRight } from "@phosphor-icons/react";
import { motion } from "motion/react";
import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useCompare, useEvalRun, useEvalRuns } from "../../api/hooks";
import type { CacheMetrics, Compare, EvalKind, EvalRun, EvalRunSummary, KindMetrics } from "../../api/types";
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
  Tag,
  Td,
  Th,
  cx,
} from "../../components/ui";
import { KIND_LABELS, configLabel, date, dateTime, ms, num, pct, shortHash, usd } from "../../lib/format";
import { DUR, EASE, cascade } from "../../theme/motion";
import { RecallCurve, SimilarityStrip, SweepChart } from "./charts";
import { GateChecks, GateTag, OutcomeBar } from "./parts";

export function EvalsPage() {
  const runs = useEvalRuns();
  const all = runs.data ?? [];
  const gated = all.find((r) => r.kind === "e2e" && r.status === "ok" && r.gate_status !== "not_gated");
  const cacheRun = all.find((r) => r.kind === "cache" && r.status === "ok");
  const e2e = useEvalRun(gated?.id);
  const cache = useEvalRun(cacheRun?.id);
  const compare = useCompare();

  return (
    <Page wide>
      <PageHeader eyebrow="Evaluation" title="Measured, not assumed">
        Every number here comes from a stored eval run over the hand-built golden set: questions
        across five Rulebook modules, terms defined in the DIFC Laws, exemptions, multi-turn
        follow-ups, and questions the assistant must refuse.
      </PageHeader>

      {runs.error && <ErrorNote error={runs.error} />}
      {runs.isLoading && <Skeleton className="h-28 w-full" />}
      {!runs.isLoading && !all.length && !runs.error && (
        <EmptyState title="No eval runs yet">
          Run <code className="font-mono">uv run python -m src.eval.run</code> to create one.
        </EmptyState>
      )}

      {e2e.data && <Headline run={e2e.data} />}
      {e2e.data && <GateSection run={e2e.data} />}
      {compare.data && compare.data.runs.length > 0 && <StagesSection compare={compare.data} />}
      {e2e.data && <AnswersSection run={e2e.data} />}
      {e2e.data?.metrics.by_kind && <KindsSection run={e2e.data} />}
      {cache.data && <CacheSection run={cache.data} />}
      {all.length > 0 && <RunsSection runs={all} />}
    </Page>
  );
}

// --- headline numbers ----------------------------------------------------------------

function Headline({ run }: { run: EvalRun }) {
  const m = run.metrics as Record<string, number> & { outcomes?: Record<string, number> };
  const refusals = m.outcomes?.correct_refusal ?? 0;
  const unanswerable = refusals + (m.outcomes?.wrong_answer ?? 0);
  const unanswerableTurns = run.metrics.by_kind?.unanswerable?.turns ?? unanswerable;
  const tiles = [
    { label: "Gold provision in the top 6", value: pct(m["recall@6"]), note: `MRR ${num(m.mrr, 3)}` },
    { label: "Claims supported by the cited text", value: pct(m.faithfulness), note: "faithfulness, LLM judge" },
    {
      label: "Out-of-scope questions refused",
      value: `${refusals} of ${unanswerableTurns}`,
      note: `false refusals ${pct(m.false_refusal_rate)}`,
    },
    { label: "Cost per question", value: usd(m.cost_per_turn_usd), note: `p50 ${ms(run.metrics.latency_ms?.turn?.p50)} end to end` },
  ];
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      {tiles.map((t, i) => (
        <motion.div
          key={t.label}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: DUR.slow, ease: EASE, delay: cascade(i, 0.06) }}
          className="h-full"
        >
          <Stat label={t.label} value={t.value} note={t.note} />
        </motion.div>
      ))}
    </div>
  );
}

// --- the release gate ------------------------------------------------------------------

function GateSection({ run }: { run: EvalRun }) {
  const answer = run.prompts.find((p) => p.name === "answer");
  return (
    <Section
      title="Release gate"
      description="A prompt or retrieval change ships only if an end-to-end run clears every limit in eval/thresholds.yaml and doesn't regress against the last passing run."
      aside={
        <Link
          to={`/evals/runs/${run.id}`}
          className="inline-flex items-center gap-1.5 text-[12.5px] text-ink-2 transition-colors hover:text-ink"
        >
          Run #{run.id} <ArrowRight size={12} weight="bold" />
        </Link>
      }
    >
      <Card className="p-6">
        <div className="mb-6 flex flex-wrap items-center gap-x-6 gap-y-2 text-[12.5px] text-ink-3">
          <GateTag status={run.gate_status} />
          <span>
            {configLabel(run.config_name)} · {run.n_items} reviewed questions, {run.n_turns} turns
          </span>
          {answer && (
            <span>
              answer prompt v{answer.version} ({answer.label})
            </span>
          )}
          <span className="font-mono text-[11.5px]">commit {shortHash(run.git_commit, 13)}</span>
          {run.gate_report?.baseline_run && <span>baseline run #{run.gate_report.baseline_run}</span>}
          <span>{dateTime(run.created_at)}</span>
        </div>
        {run.gate_report && <GateChecks report={run.gate_report} />}
      </Card>
    </Section>
  );
}

// --- retrieval stages ---------------------------------------------------------------------

const GROUPS: { title: string; configs: string[] }[] = [
  { title: "Retrieval stage", configs: ["dense", "sparse", "hybrid", "hybrid_rerank"] },
  { title: "Reranker model", configs: ["hybrid_rerank_base", "hybrid_rerank_minilm"] },
  { title: "Context policy", configs: ["parent_none", "parent_window"] },
];
const COLS = [
  { key: "recall@1", label: "Recall@1" },
  { key: "recall@6", label: "Recall@6" },
  { key: "mrr", label: "MRR" },
  { key: "ndcg@6", label: "nDCG@6" },
  { key: "context_recall", label: "Context recall" },
];

function StagesSection({ compare }: { compare: Compare }) {
  const by = Object.fromEntries(compare.runs.map((r) => [r.config_name, r]));
  const dense = by.dense?.metrics as Record<string, number> | undefined;
  const prod = by.hybrid_rerank?.metrics as Record<string, number> | undefined;
  const best = Object.fromEntries(
    COLS.map((c) => [c.key, Math.max(...compare.runs.map((r) => (r.metrics[c.key] as number) ?? -1))]),
  );
  return (
    <Section
      title="What each retrieval stage adds"
      description="The same reviewed questions run through each configuration, with no answers generated. Each stage is measured rather than assumed to help."
    >
      <div className="grid gap-6 lg:grid-cols-[1.6fr_1fr]">
        <Card className="p-5">
          <div className="mb-4 text-[13px] font-medium text-ink">Recall at k</div>
          <RecallCurve compare={compare} />
        </Card>
        {dense && prod && (
          <div className="grid gap-3">
            <Lift label="Recall@1" from={dense["recall@1"]} to={prod["recall@1"]} />
            <Lift label="MRR" from={dense.mrr} to={prod.mrr} />
            <Lift label="nDCG@6" from={dense["ndcg@6"]} to={prod["ndcg@6"]} />
          </div>
        )}
      </div>
      <div className="mt-6">
        <Card className="overflow-x-auto px-5 py-3">
          <table className="w-full border-collapse">
            <thead>
              <tr>
                <Th>Configuration</Th>
                {COLS.map((c) => (
                  <Th key={c.key} align="right">
                    {c.label}
                  </Th>
                ))}
                <Th align="right">p50</Th>
                <Th align="right">p95</Th>
              </tr>
            </thead>
            {GROUPS.map((g) => {
              const rows = g.configs.map((c) => by[c]).filter(Boolean);
              if (!rows.length) return null;
              return (
                <tbody key={g.title}>
                  <tr>
                    <td colSpan={COLS.length + 3} className="pb-1 pt-4 text-[11px] font-medium uppercase tracking-[0.06em] text-ink-4">
                      {g.title}
                    </td>
                  </tr>
                  {rows.map((r) => (
                    <tr
                      key={r.id}
                      className={cx(r.config_name === "hybrid_rerank" && "bg-surface-2/70")}
                    >
                      <Td className="whitespace-nowrap">
                        <Link to={`/evals/runs/${r.id}`} className="hover:underline">
                          {configLabel(r.config_name)}
                        </Link>
                        {r.config_name === "hybrid_rerank" && (
                          <Tag tone="blue" className="ml-2">
                            prod
                          </Tag>
                        )}
                      </Td>
                      {COLS.map((c) => {
                        const v = r.metrics[c.key] as number | undefined;
                        return (
                          <Td
                            key={c.key}
                            align="right"
                            className={cx(v === best[c.key] ? "font-semibold text-ink" : "text-ink-2")}
                          >
                            {c.key === "mrr" || c.key.startsWith("ndcg") ? num(v, 3) : pct(v, 1)}
                          </Td>
                        );
                      })}
                      <Td align="right" className="text-ink-3">
                        {ms(r.metrics.latency_ms?.retrieve?.p50)}
                      </Td>
                      <Td align="right" className="text-ink-3">
                        {ms(r.metrics.latency_ms?.retrieve?.p95)}
                      </Td>
                    </tr>
                  ))}
                </tbody>
              );
            })}
          </table>
          <div className="pb-1 pt-3 text-[11.5px] text-ink-4">
            Best value per column in bold. Latency is retrieval including reranking, on a local GPU.
            {!compare.golden_is_current && " These runs predate the current golden set."}
          </div>
        </Card>
      </div>
    </Section>
  );
}

function Lift({ label, from, to }: { label: string; from: number; to: number }) {
  const delta = to - from;
  return (
    <div className="card flex items-center justify-between gap-4 px-5 py-4">
      <div>
        <div className="text-[12.5px] text-ink-3">{label}: dense alone vs full pipeline</div>
        <div className="mt-1 flex items-baseline gap-2 text-[22px] font-semibold tracking-[-0.02em] text-ink">
          <span className="text-ink-4">{num(from, 2)}</span>
          <ArrowRight size={14} weight="bold" className="self-center text-ink-4" />
          <span>{num(to, 2)}</span>
        </div>
      </div>
      <Tag tone={delta >= 0 ? "green" : "red"}>
        {delta >= 0 ? "+" : ""}
        {num(delta, 2)}
      </Tag>
    </div>
  );
}

// --- answers ---------------------------------------------------------------------------------

function AnswersSection({ run }: { run: EvalRun }) {
  const m = run.metrics as Record<string, number>;
  const lat = run.metrics.latency_ms ?? {};
  const stages = ["rewrite", "retrieve", "rerank", "answer", "judge", "turn"].filter((s) => lat[s]);
  const cost = run.cost_usd;
  return (
    <Section
      title="Answers"
      description={`End-to-end run #${run.id}: rewrite, retrieve, rerank, answer, then two LLM judges score each answer against the context it was given.`}
    >
      <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        <Card className="p-5">
          <div className="mb-4 text-[13px] font-medium text-ink">Outcome of each turn</div>
          {run.metrics.outcomes && <OutcomeBar outcomes={run.metrics.outcomes} />}
          <div className="mt-6 grid grid-cols-2 gap-x-8 gap-y-3 border-t border-line pt-5 sm:grid-cols-3">
            <Mini label="Citation precision" value={pct(m.citation_precision)} />
            <Mini label="Citation recall" value={pct(m.citation_recall)} />
            <Mini label="Relevance (1–5)" value={num(m.relevance, 2)} />
            <Mini label="Refusal precision" value={pct(m.refusal_precision)} />
            <Mini label="Validator failures" value={pct(m.validator_failure_rate)} />
            <Mini label="Contradicted claims" value={String(m.contradicted_claims ?? 0)} />
          </div>
        </Card>
        <Card className="px-5 py-3">
          <table className="w-full border-collapse">
            <thead>
              <tr>
                <Th>Stage</Th>
                <Th align="right">p50</Th>
                <Th align="right">p95</Th>
                <Th align="right">Cost</Th>
              </tr>
            </thead>
            <tbody>
              {stages.map((s) => (
                <tr key={s}>
                  <Td className={cx(s === "turn" && "font-medium")}>{STAGE_NAMES[s] ?? s}</Td>
                  <Td align="right">{ms(lat[s]?.p50)}</Td>
                  <Td align="right">{ms(lat[s]?.p95)}</Td>
                  <Td align="right" className="text-ink-3">
                    {s === "turn" ? usd(cost.total) : cost[s] != null ? usd(cost[s]) : ""}
                  </Td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="pb-1 pt-3 text-[11.5px] text-ink-4">
            Cost is for the whole run of {run.n_turns} turns.
          </div>
        </Card>
      </div>
    </Section>
  );
}

const STAGE_NAMES: Record<string, string> = {
  rewrite: "Rewrite (follow-ups)",
  retrieve: "Retrieve",
  rerank: "of which rerank",
  answer: "Answer",
  judge: "Judges (eval only)",
  turn: "Whole turn",
};

function Mini({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[12px] text-ink-3">{label}</div>
      <div className="text-[16px] font-semibold text-ink">{value}</div>
    </div>
  );
}

// --- by question kind ---------------------------------------------------------------------------

function KindsSection({ run }: { run: EvalRun }) {
  const kinds = Object.entries(run.metrics.by_kind ?? {}) as [string, KindMetrics][];
  const cols = [
    { key: "recall@6", label: "Recall@6" },
    { key: "mrr", label: "MRR" },
    { key: "faithfulness", label: "Faithfulness" },
    { key: "relevance", label: "Relevance" },
  ];
  return (
    <Section
      title="By kind of question"
      description="Where the pipeline is strong and where it is weakest. Exemptions are the hardest: the answer usually lives in an application rule rather than in the obligation itself."
    >
      <Card className="overflow-x-auto px-5 py-3">
        <table className="w-full border-collapse">
          <thead>
            <tr>
              <Th>Kind</Th>
              <Th align="right">Turns</Th>
              {cols.map((c) => (
                <Th key={c.key} className="w-[16%] pl-6">
                  {c.label}
                </Th>
              ))}
              <Th align="right">Refusals</Th>
            </tr>
          </thead>
          <tbody>
            {kinds.map(([kind, m]) => (
              <tr key={kind}>
                <Td className="whitespace-nowrap">{KIND_LABELS[kind] ?? kind}</Td>
                <Td align="right" className="text-ink-3">
                  {m.turns}
                </Td>
                {cols.map((c) => {
                  const v = m[c.key];
                  const scaled = v == null ? null : c.key === "relevance" ? (v - 1) / 4 : v;
                  return (
                    <Td key={c.key} className="pl-6">
                      {v == null ? (
                        <span className="text-ink-4">–</span>
                      ) : (
                        <div className="flex items-center gap-3">
                          <div className="h-1 flex-1 rounded-full bg-surface-3">
                            <motion.div
                              className={cx("h-1 origin-left rounded-full", (scaled ?? 0) < 0.7 ? "bg-series-4" : "bg-ink-3")}
                              style={{ width: `${(scaled ?? 0) * 100}%` }}
                              initial={{ scaleX: 0 }}
                              whileInView={{ scaleX: 1 }}
                              viewport={{ once: true }}
                              transition={{ duration: 0.6, ease: EASE }}
                            />
                          </div>
                          <span className="w-12 text-right tnum text-[12.5px] text-ink-2">
                            {c.key === "relevance" ? num(v, 2) : c.key === "mrr" ? num(v, 2) : pct(v, 0)}
                          </span>
                        </div>
                      )}
                    </Td>
                  );
                })}
                <Td align="right" className="text-ink-2">
                  {m.refusal_recall != null
                    ? `${pct(m.refusal_recall, 0)} refused`
                    : m.false_refusal_rate
                      ? `${pct(m.false_refusal_rate, 0)} false`
                      : "none"}
                </Td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </Section>
  );
}

// --- answer cache -------------------------------------------------------------------------------

function CacheSection({ run }: { run: EvalRun }) {
  const m = run.metrics as unknown as CacheMetrics;
  if (!m.pairs?.length) return null;
  return (
    <Section
      title="Why the answer cache serves exact repeats only"
      description={`Run #${run.id} scored ${m.n_positive} paraphrase pairs and ${m.n_negative} hard negatives: questions that look alike but need a different answer.`}
    >
      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="p-5">
          <div className="mb-4 text-[13px] font-medium text-ink">Cosine similarity of each pair</div>
          <SimilarityStrip metrics={m} />
        </Card>
        <Card className="p-5">
          <div className="mb-4 text-[13px] font-medium text-ink">Wrong answers a threshold would serve</div>
          <SweepChart metrics={m} />
        </Card>
      </div>
      <div className="mt-4 rounded-[10px] border border-line bg-surface-2 px-5 py-4 text-[13.5px] leading-relaxed text-ink-2">
        <span className="font-medium text-ink">Decision.</span> Paraphrases score between{" "}
        {num(m.positive_similarity.min, 2)} and {num(m.positive_similarity.max, 2)}, but a hard negative
        that passes every guard (<span className="font-mono text-[12.5px]">{m.max_guard_passing_negative?.id}</span>, a
        “major” versus “minor” currency pair) scores {num(m.max_guard_passing_negative?.similarity, 3)}. No
        threshold serves paraphrases without serving a wrong answer, so only exact repeats are served from
        the cache.
      </div>
    </Section>
  );
}

// --- every run ------------------------------------------------------------------------------------

function RunsSection({ runs }: { runs: EvalRunSummary[] }) {
  const [kind, setKind] = useState<"all" | EvalKind>("all");
  const navigate = useNavigate();
  const shown = useMemo(() => runs.filter((r) => kind === "all" || r.kind === kind), [runs, kind]);
  return (
    <Section
      title="Run history"
      description="Every eval run stored in SQLite, newest first."
      aside={
        <Tabs
          layoutId="runs-kind"
          value={kind}
          onChange={setKind}
          options={[
            { value: "all", label: "All" },
            { value: "e2e", label: "End to end" },
            { value: "retrieval", label: "Retrieval" },
            { value: "cache", label: "Cache" },
          ]}
        />
      }
    >
      <div className="overflow-x-auto">
        <table className="w-full border-collapse">
          <thead>
            <tr>
              <Th>Run</Th>
              <Th>Kind</Th>
              <Th>Configuration</Th>
              <Th align="right">Questions</Th>
              <Th align="right">Recall@6</Th>
              <Th align="right">MRR</Th>
              <Th align="right">Faithfulness</Th>
              <Th>Gate</Th>
              <Th align="right">Cost</Th>
              <Th align="right">Date</Th>
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => (
              <tr
                key={r.id}
                onClick={() => navigate(`/evals/runs/${r.id}`)}
                className="cursor-pointer transition-colors duration-150 hover:bg-surface-2"
              >
                <Td mono className="text-ink-3">
                  #{r.id}
                </Td>
                <Td>
                  <Tag tone={r.kind === "e2e" ? "blue" : "gray"}>{r.kind === "e2e" ? "e2e" : r.kind}</Tag>
                </Td>
                <Td className="whitespace-nowrap text-ink">{configLabel(r.config_name)}</Td>
                <Td align="right" className="text-ink-2">
                  {r.n_items}
                  {!r.reviewed_only && <span className="text-ink-4"> draft</span>}
                </Td>
                <Td align="right">{pct(r.headline["recall@6"])}</Td>
                <Td align="right">{num(r.headline.mrr, 3)}</Td>
                <Td align="right">{pct(r.headline.faithfulness)}</Td>
                <Td>{r.gate_status === "not_gated" ? <span className="text-ink-4">–</span> : <GateTag status={r.gate_status} />}</Td>
                <Td align="right" className="text-ink-3">
                  {usd(r.cost_total_usd)}
                </Td>
                <Td align="right" className="whitespace-nowrap text-ink-3">
                  {date(r.created_at)}
                </Td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Section>
  );
}
