import { ArrowLeft, CaretRight, Check, X } from "@phosphor-icons/react";
import { AnimatePresence, motion } from "motion/react";
import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useEvalResults, useEvalRun } from "../../api/hooks";
import type { CacheMetrics, EvalResult, EvalRun } from "../../api/types";
import {
  Card,
  ErrorNote,
  Page,
  PageHeader,
  Section,
  Skeleton,
  Stat,
  Tabs,
  Tag,
  Th,
  cx,
} from "../../components/ui";
import {
  KIND_LABELS,
  OUTCOME_LABELS,
  REFUSAL_LABELS,
  configLabel,
  dateTime,
  num,
  pct,
  shortHash,
  usd,
} from "../../lib/format";
import { DUR, EASE } from "../../theme/motion";
import { SimilarityStrip, SweepChart } from "./charts";
import { Facts, GateChecks, GateTag, OUTCOMES } from "./parts";

export function RunDetailPage() {
  const id = Number(useParams().runId);
  const run = useEvalRun(Number.isFinite(id) ? id : null);
  return (
    <Page wide>
      <Link
        to="/evals"
        className="mb-8 inline-flex items-center gap-1.5 text-[12.5px] text-ink-3 transition-colors hover:text-ink"
      >
        <ArrowLeft size={13} weight="bold" /> Evaluations
      </Link>
      {run.error && <ErrorNote error={run.error} title="Couldn't load this run" />}
      {run.isLoading && <Skeleton className="h-40 w-full" />}
      {run.data && <RunBody run={run.data} />}
    </Page>
  );
}

function RunBody({ run }: { run: EvalRun }) {
  const kindLabel = { e2e: "End-to-end run", retrieval: "Retrieval run", cache: "Cache calibration run" }[run.kind];
  return (
    <>
      <PageHeader
        eyebrow={`${kindLabel} #${run.id}`}
        title={configLabel(run.config_name)}
        actions={run.gate_status !== "not_gated" ? <GateTag status={run.gate_status} /> : undefined}
      >
        {run.kind === "cache"
          ? `${run.n_items} question pairs`
          : `${run.n_items} ${run.reviewed_only ? "reviewed" : "draft and reviewed"} questions, ${run.n_turns} turns`}{" "}
        · {dateTime(run.created_at)}
      </PageHeader>

      <div className="grid gap-6 lg:grid-cols-[1fr_1fr]">
        <Card className="p-5">
          <Facts
            items={[
              ["Commit", <span className="font-mono text-[12.5px]">{run.git_commit ?? "–"}</span>],
              ["Golden set", <span className="font-mono text-[12.5px]">{shortHash(run.golden_hash, 12)}</span>],
              ["Prompt label", run.prompt_label ?? "–"],
              ["Status", run.status === "ok" ? "finished" : run.status],
              ["Total cost", usd(run.cost_usd?.total ?? run.cost_total_usd)],
            ]}
          />
        </Card>
        <Card className="p-5">
          {run.kind === "cache" ? (
            <CacheFacts run={run} />
          ) : run.prompts.length > 0 ? (
            <Facts
              items={run.prompts.map((p) => [
                p.name,
                <span>
                  v{p.version ?? "?"} <span className="text-ink-3">{p.label}</span>{" "}
                  <span className="font-mono text-[11.5px] text-ink-4">{shortHash(p.hash, 8)}</span>
                </span>,
              ])}
            />
          ) : (
            <ConfigFacts run={run} />
          )}
        </Card>
      </div>

      {run.error && (
        <div className="mt-6">
          <ErrorNote error={run.error} title="The run stopped with an error" />
        </div>
      )}

      {run.gate_report && (
        <Section title="Gate checks" className="mt-14">
          <Card className="p-6">
            <GateChecks report={run.gate_report} />
          </Card>
        </Section>
      )}

      {run.kind === "cache" ? <CacheBody run={run} /> : <ResultsBody run={run} />}
    </>
  );
}

function CacheFacts({ run }: { run: EvalRun }) {
  const m = run.metrics as unknown as CacheMetrics;
  return (
    <Facts
      items={[
        ["Similarity threshold", m.threshold == null ? "not set: exact repeats only" : num(m.threshold, 3)],
        ["Recommended by the sweep", num(m.recommended_threshold, 3)],
        ["Paraphrase similarity", `${num(m.positive_similarity?.min, 3)} to ${num(m.positive_similarity?.max, 3)}`],
        ["Hard negative similarity", `${num(m.negative_similarity?.min, 3)} to ${num(m.negative_similarity?.max, 3)}`],
        ["Pairs reviewed", run.reviewed_only ? "yes" : "not yet (draft pairs)"],
      ]}
    />
  );
}

function ConfigFacts({ run }: { run: EvalRun }) {
  const c = run.retrieval_config as Record<string, unknown>;
  const keys = ["mode", "candidate_k", "rerank", "reranker_model", "top_n", "refusal_threshold", "parent_mode"];
  return (
    <Facts
      items={keys
        .filter((k) => c[k] !== undefined)
        .map((k) => [k.replace(/_/g, " "), <span className="font-mono text-[12.5px]">{String(c[k])}</span>])}
    />
  );
}

// --- per-question results ------------------------------------------------------------------

type Filter = "all" | "misses" | "refused" | "multi";

function ResultsBody({ run }: { run: EvalRun }) {
  const { data, isLoading, error } = useEvalResults(run.id);
  const [filter, setFilter] = useState<Filter>("all");
  const m = run.metrics as Record<string, number>;
  const rows = useMemo(() => {
    const all = data ?? [];
    if (filter === "misses")
      return all.filter(
        (r) =>
          (r.expect === "answer" && (r.metrics["recall@6"] as number) < 1) ||
          ["false_refusal", "wrong_answer"].includes(String(r.metrics.outcome)),
      );
    if (filter === "refused") return all.filter((r) => r.expect === "refuse" || r.status === "refused");
    if (filter === "multi") return all.filter((r) => r.item_kind === "multi_turn");
    return all;
  }, [data, filter]);

  return (
    <>
      <div className="mt-10 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Recall@6" value={pct(m["recall@6"])} note={`Recall@1 ${pct(m["recall@1"])}`} />
        <Stat label="MRR" value={num(m.mrr, 3)} note={`nDCG@6 ${num(m["ndcg@6"], 3)}`} />
        {run.kind === "e2e" ? (
          <>
            <Stat label="Faithfulness" value={pct(m.faithfulness)} note={`relevance ${num(m.relevance, 2)} / 5`} />
            <Stat label="False refusals" value={pct(m.false_refusal_rate)} note={`refusal recall ${pct(m.refusal_recall)}`} />
          </>
        ) : (
          <>
            <Stat label="Context recall" value={pct(m.context_recall)} note="gold provision in the LLM context" />
            <Stat label="Truncated by reranker" value={String(m.truncated ?? 0)} note="candidates longer than 512 tokens" />
          </>
        )}
      </div>

      <Section
        title="Every question"
        description="Open a row to see what was retrieved, the answer, and how the judges scored it."
        className="mt-14"
        aside={
          <Tabs
            layoutId="result-filter"
            value={filter}
            onChange={setFilter}
            options={[
              { value: "all", label: `All${data ? ` ${data.length}` : ""}` },
              { value: "misses", label: "Misses" },
              { value: "refused", label: "Refusals" },
              { value: "multi", label: "Follow-ups" },
            ]}
          />
        }
      >
        {error && <ErrorNote error={error} />}
        {isLoading && <Skeleton className="h-64 w-full" />}
        {data && (
          <table className="w-full table-fixed border-collapse">
            <colgroup>
              <col className="w-7" />
              <col />
              <col className="w-36" />
              <col className="w-20" />
              <col className="w-20" />
              <col className="w-20" />
              {run.kind === "e2e" && <col className="w-24" />}
            </colgroup>
            <thead>
              <tr>
                <Th />
                <Th>Question</Th>
                <Th>Outcome</Th>
                <Th align="right">Confidence</Th>
                <Th align="right">Recall@6</Th>
                <Th align="right">MRR</Th>
                {run.kind === "e2e" && <Th align="right">Faithful</Th>}
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <ResultRow key={r.id} r={r} e2e={run.kind === "e2e"} threshold={Number(run.retrieval_config.refusal_threshold ?? 0)} />
              ))}
            </tbody>
          </table>
        )}
        {data && rows.length === 0 && <div className="py-8 text-center text-[13px] text-ink-3">Nothing in this view.</div>}
      </Section>
    </>
  );
}

function Outcome({ r, threshold }: { r: EvalResult; threshold: number }) {
  const o = String(r.metrics.outcome ?? "");
  const spec = OUTCOMES.find((x) => x.key === o);
  if (spec) return <Tag tone={spec.tone}>{OUTCOME_LABELS[o]}</Tag>;
  // Retrieval runs have no answer: say what the confidence gate would have done.
  if (r.confidence == null) return <span className="text-ink-4">–</span>;
  const answer = r.confidence >= threshold;
  return <Tag tone={answer ? "gray" : "yellow"}>{answer ? "would answer" : "would refuse"}</Tag>;
}

function ResultRow({ r, e2e, threshold }: { r: EvalResult; e2e: boolean; threshold: number }) {
  const [open, setOpen] = useState(false);
  const metric = (k: string) => r.metrics[k] as number | undefined;
  const cols = e2e ? 7 : 6;
  return (
    <>
      <tr
        onClick={() => setOpen((o) => !o)}
        className={cx("cursor-pointer transition-colors duration-150 hover:bg-surface-2", open && "bg-surface-2")}
      >
        <td className="h-11 border-b border-line align-middle">
          <motion.span animate={{ rotate: open ? 90 : 0 }} transition={{ duration: DUR.fast, ease: EASE }} className="flex text-ink-4">
            <CaretRight size={12} weight="bold" />
          </motion.span>
        </td>
        <td className="h-11 truncate border-b border-line pr-4 text-[13px]">
          <span className="text-ink">{r.question}</span>
          <span className="ml-2 text-[11.5px] text-ink-4">
            {KIND_LABELS[r.item_kind ?? ""] ?? r.item_kind}
            {r.turn > 0 ? `, turn ${r.turn + 1}` : ""}
          </span>
        </td>
        <td className="h-11 border-b border-line">
          <Outcome r={r} threshold={threshold} />
        </td>
        <td className="h-11 border-b border-line text-right text-[12.5px] tnum text-ink-2">{num(r.confidence, 2)}</td>
        <td className="h-11 border-b border-line text-right text-[12.5px] tnum text-ink-2">
          {r.expect === "refuse" ? <span className="text-ink-4">n/a</span> : pct(metric("recall@6"), 0)}
        </td>
        <td className="h-11 border-b border-line text-right text-[12.5px] tnum text-ink-2">
          {r.expect === "refuse" ? <span className="text-ink-4">n/a</span> : num(metric("mrr"), 2)}
        </td>
        {e2e && (
          <td className="h-11 border-b border-line text-right text-[12.5px] tnum text-ink-2">
            {metric("faithfulness") == null ? <span className="text-ink-4">–</span> : pct(metric("faithfulness"), 0)}
          </td>
        )}
      </tr>
      <AnimatePresence initial={false}>
        {open && (
          <tr>
            <td colSpan={cols} className="border-b border-line p-0">
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: DUR.base, ease: EASE }}
                className="overflow-hidden"
              >
                <ResultDetail r={r} />
              </motion.div>
            </td>
          </tr>
        )}
      </AnimatePresence>
    </>
  );
}

function ResultDetail({ r }: { r: EvalResult }) {
  const gold = new Set(r.gold);
  const cited = new Set(r.cited);
  const claims = r.judges?.faithfulness?.output?.claims ?? [];
  return (
    <div className="grid gap-8 bg-surface-2/50 px-7 py-6 lg:grid-cols-2">
      <div className="min-w-0">
        <div className="eyebrow mb-2">Question</div>
        <div className="text-[13.5px] text-ink">{r.question}</div>
        {r.rewritten_query && r.rewritten_query !== r.question && (
          <div className="mt-1.5 text-[12.5px] text-ink-3">Read as “{r.rewritten_query}”</div>
        )}

        <div className="eyebrow mb-2 mt-5">Expected</div>
        {r.expect === "refuse" ? (
          <div className="text-[13px] text-ink-2">A refusal: the corpus doesn’t cover this.</div>
        ) : (
          <div className="flex flex-wrap gap-1.5">
            {r.gold.map((g) => (
              <span key={g} className="rounded-[5px] border border-line bg-surface px-1.5 py-0.5 font-mono text-[12px] text-ink">
                {g}
              </span>
            ))}
            {r.supporting.map((g) => (
              <span key={g} className="rounded-[5px] border border-dashed border-line-strong px-1.5 py-0.5 font-mono text-[12px] text-ink-3" title="supporting">
                {g}
              </span>
            ))}
          </div>
        )}

        {r.answer != null && (
          <>
            <div className="eyebrow mb-2 mt-5">
              {r.status === "refused" ? `Refused · ${REFUSAL_LABELS[r.refusal_code ?? ""] ?? r.refusal_code}` : "Answer"}
            </div>
            <div className="max-h-56 overflow-y-auto whitespace-pre-line pr-2 text-[13px] leading-relaxed text-ink-2">{r.answer}</div>
            {r.cited.length > 0 && (
              <div className="mt-3 flex flex-wrap items-center gap-1.5">
                <span className="mr-1 text-[11.5px] text-ink-3">Cited</span>
                {r.cited.map((c) => (
                  <span
                    key={c}
                    className={cx(
                      "rounded-[5px] px-1.5 py-0.5 font-mono text-[12px]",
                      gold.has(c) ? "bg-green-bg text-green-fg" : "bg-surface-3 text-ink-2",
                    )}
                  >
                    {c}
                  </span>
                ))}
              </div>
            )}
          </>
        )}

        {claims.length > 0 && (
          <>
            <div className="eyebrow mb-2 mt-5">Faithfulness judge</div>
            <ul className="space-y-1.5">
              {claims.map((c, i) => (
                <li key={i} className="flex gap-2 text-[12.5px] leading-snug">
                  {c.verdict === "supported" ? (
                    <Check size={13} weight="bold" className="mt-0.5 shrink-0 text-good" />
                  ) : (
                    <X size={13} weight="bold" className="mt-0.5 shrink-0 text-critical" />
                  )}
                  <span className={c.verdict === "supported" ? "text-ink-2" : "text-red-fg"}>
                    {c.claim}
                    {c.verdict !== "supported" && <span className="ml-1 text-[11px] uppercase tracking-wide">({c.verdict})</span>}
                  </span>
                </li>
              ))}
            </ul>
          </>
        )}
        {r.judges?.relevance?.output?.reasoning && (
          <>
            <div className="eyebrow mb-2 mt-5">Relevance judge · {num(r.judges.relevance.score, 0)} / 5</div>
            <div className="text-[12.5px] leading-relaxed text-ink-2">{r.judges.relevance.output.reasoning}</div>
          </>
        )}
      </div>

      <div className="min-w-0">
        <div className="eyebrow mb-2">Top of the ranking</div>
        <table className="w-full border-collapse">
          <tbody>
            {r.ranked.slice(0, 10).map((p, i) => (
              <tr key={p.key + i}>
                <td className="h-8 w-6 border-b border-line text-[11.5px] tnum text-ink-4">{i + 1}</td>
                <td className="h-8 border-b border-line">
                  <span className={cx("font-mono text-[12.5px]", gold.has(p.provision) ? "font-medium text-green-fg" : "text-ink-2")}>
                    {p.key}
                  </span>
                  {gold.has(p.provision) && <span className="ml-2 text-[10.5px] uppercase tracking-wide text-green-fg">gold</span>}
                  {cited.has(p.provision) && !gold.has(p.provision) && (
                    <span className="ml-2 text-[10.5px] uppercase tracking-wide text-ink-3">cited</span>
                  )}
                </td>
                <td className="h-8 w-28 border-b border-line">
                  {p.rerank != null && (
                    <div className="flex items-center gap-2">
                      <div className="h-1 flex-1 rounded-full bg-surface-3">
                        <div
                          className={cx("h-1 rounded-full", gold.has(p.provision) ? "bg-good" : "bg-ink-4")}
                          style={{ width: `${Math.max(0, Math.min(1, p.rerank)) * 100}%` }}
                        />
                      </div>
                      <span className="w-9 text-right font-mono text-[11px] text-ink-3">{num(p.rerank, 2)}</span>
                    </div>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-[11.5px] text-ink-4">
          {Object.entries(r.latency_ms)
            .filter(([, v]) => v > 0)
            .map(([k, v]) => (
              <span key={k}>
                {k} {v < 1000 ? `${Math.round(v)} ms` : `${(v / 1000).toFixed(2)} s`}
              </span>
            ))}
          {r.cost_usd > 0 && <span>{usd(r.cost_usd)}</span>}
        </div>
      </div>
    </div>
  );
}

// --- cache calibration run --------------------------------------------------------------------

function CacheBody({ run }: { run: EvalRun }) {
  const m = run.metrics as unknown as CacheMetrics;
  const [kind, setKind] = useState<"negative" | "positive">("negative");
  const pairs = (m.pairs ?? []).filter((p) => p.kind === kind).sort((a, b) => b.similarity - a.similarity);
  return (
    <>
      <div className="mt-10 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="False hits served" value={String(m.false_hits)} note="at the configured threshold" />
        <Stat label="Paraphrases served" value={pct(m.positive_hit_rate, 0)} note="exact repeats only, by design" />
        <Stat label="Negatives stopped by guards" value={`${m.negatives_stopped_by_guards} of ${m.n_negative}`} />
        <Stat label="Closest guard-passing negative" value={num(m.max_guard_passing_negative?.similarity, 3)} note={m.max_guard_passing_negative?.id} />
      </div>
      <div className="mt-10 grid gap-6 lg:grid-cols-2">
        <Card className="p-5">
          <SimilarityStrip metrics={m} />
        </Card>
        <Card className="p-5">
          <SweepChart metrics={m} />
        </Card>
      </div>
      <Section
        title="Pairs"
        className="mt-14"
        aside={
          <Tabs
            layoutId="pair-kind"
            value={kind}
            onChange={setKind}
            options={[
              { value: "negative", label: `Hard negatives ${m.n_negative}` },
              { value: "positive", label: `Paraphrases ${m.n_positive}` },
            ]}
          />
        }
      >
        <table className="w-full border-collapse">
          <thead>
            <tr>
              <Th>Pair</Th>
              <Th align="right">Similarity</Th>
              <Th>Stopped by</Th>
            </tr>
          </thead>
          <tbody>
            {pairs.map((p) => (
              <tr key={p.id}>
                <td className="h-9 border-b border-line font-mono text-[12.5px] text-ink">{p.id}</td>
                <td className="h-9 border-b border-line text-right font-mono text-[12.5px] text-ink-2">{p.similarity.toFixed(3)}</td>
                <td className="h-9 border-b border-line pl-6 text-[12.5px] text-ink-3">{p.guards.length ? p.guards.join(", ") : "–"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Section>
    </>
  );
}
