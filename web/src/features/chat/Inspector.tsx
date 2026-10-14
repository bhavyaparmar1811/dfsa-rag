import { ArrowLeft, Check, Scissors } from "@phosphor-icons/react";
import { AnimatePresence, motion } from "motion/react";
import { useChunk, useTrace } from "../../api/hooks";
import type { Candidate, Trace } from "../../api/types";
import { Drawer } from "../../components/Drawer";
import { ErrorNote, Skeleton, Tabs, Tag, cx } from "../../components/ui";
import { ms, num } from "../../lib/format";
import { DUR, EASE, cascade } from "../../theme/motion";

export type InspectorState = {
  tab: "rule" | "retrieval";
  chunkId: string | null;
  traceId: string | null;
} | null;

export function Inspector({
  state,
  onChange,
}: {
  state: InspectorState;
  onChange: (s: InspectorState) => void;
}) {
  const tabs = [
    ...(state?.chunkId ? [{ value: "rule" as const, label: "Provision" }] : []),
    ...(state?.traceId ? [{ value: "retrieval" as const, label: "Retrieval" }] : []),
  ];
  return (
    <Drawer
      open={!!state}
      onClose={() => onChange(null)}
      title="Inspector"
      subtitle="The exact text an answer rests on, and how it was found"
      width={600}
    >
      {state && (
        <div className="px-6 py-5">
          {tabs.length > 1 && (
            <div className="mb-5">
              <Tabs
                layoutId="inspector-tab"
                value={state.tab}
                onChange={(tab) => onChange({ ...state, tab })}
                options={tabs}
              />
            </div>
          )}
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={`${state.tab}-${state.chunkId}`}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: DUR.base, ease: EASE }}
            >
              {state.tab === "rule" && state.chunkId ? (
                <RuleTab
                  chunkId={state.chunkId}
                  onBack={
                    state.traceId ? () => onChange({ ...state, tab: "retrieval" }) : undefined
                  }
                />
              ) : state.traceId ? (
                <RetrievalTab
                  traceId={state.traceId}
                  onOpen={(chunkId) => onChange({ ...state, tab: "rule", chunkId })}
                />
              ) : null}
            </motion.div>
          </AnimatePresence>
        </div>
      )}
    </Drawer>
  );
}

// --- the provision's text ---------------------------------------------------------

const TYPE_LABEL: Record<string, string> = {
  rule: "Rule",
  guidance: "Guidance",
  definition: "Definition",
  article: "Article",
  text: "Text",
};

function RuleTab({ chunkId, onBack }: { chunkId: string; onBack?: () => void }) {
  const { data: c, isLoading, error } = useChunk(chunkId);
  if (error) return <ErrorNote error={error} />;
  if (isLoading || !c)
    return (
      <div className="space-y-3">
        <Skeleton className="h-7 w-40" />
        <Skeleton className="h-4 w-72" />
        <Skeleton className="mt-6 h-40 w-full" />
      </div>
    );
  const text = c.whole_text ?? c.text;
  const path = c.heading_path.filter((h) => !/^DFSA Rulebook$|^DIFC Laws$/i.test(h));
  return (
    <div>
      {onBack && (
        <button
          onClick={onBack}
          className="mb-4 inline-flex items-center gap-1.5 text-[12.5px] text-ink-3 transition-colors hover:text-ink"
        >
          <ArrowLeft size={13} weight="bold" /> Back to retrieval
        </button>
      )}
      <div className="flex flex-wrap items-center gap-2.5">
        <h3 className="font-mono text-[22px] font-medium tracking-[-0.02em] text-ink">{c.citation}</h3>
        <Tag tone={c.binding ? "green" : "yellow"}>{c.binding ? "Binding" : "Non-binding"}</Tag>
        <Tag>{TYPE_LABEL[c.chunk_type] ?? c.chunk_type}</Tag>
      </div>
      <div className="mt-2 text-[12.5px] text-ink-3">
        {c.module_name} · {c.regulator} · version {c.version} · effective {c.effective_date} · page{" "}
        {c.page_start === c.page_end ? c.page_start : `${c.page_start}–${c.page_end}`}
      </div>

      {path.length > 0 && (
        <ol className="mt-5 space-y-1 border-l border-line pl-4">
          {path.map((h, i) => (
            <li
              key={i}
              className={cx("text-[12.5px] leading-snug", i === path.length - 1 ? "text-ink-2" : "text-ink-3")}
            >
              {h}
            </li>
          ))}
        </ol>
      )}

      {c.part_of && c.whole_text && (
        <div className="mt-5 flex items-center gap-2 rounded-md bg-surface-2 px-3 py-2 text-[12px] text-ink-3">
          <Scissors size={13} weight="bold" />
          Part {c.part} of {c.part_of} was retrieved. The whole provision is shown below.
        </div>
      )}

      <div className="mt-6 whitespace-pre-wrap text-[14px] leading-[1.75] text-ink-2">{text}</div>
    </div>
  );
}

// --- how it was found ---------------------------------------------------------------

function RetrievalTab({ traceId, onOpen }: { traceId: string; onOpen: (chunkId: string) => void }) {
  const { data: trace, isLoading, error } = useTrace(traceId);
  if (error) return <ErrorNote error={error} />;
  if (isLoading || !trace)
    return (
      <div className="space-y-3">
        <Skeleton className="h-20 w-full" />
        {Array.from({ length: 8 }).map((_, i) => (
          <Skeleton key={i} className="h-8 w-full" />
        ))}
      </div>
    );
  const r = trace.retrieval;
  if (!r)
    return (
      <div className="text-[13px] text-ink-3">
        This turn was answered without retrieval (status: {trace.status}).
      </div>
    );
  return (
    <div>
      <Confidence trace={trace} />
      <Pipeline trace={trace} />
      <Candidates trace={trace} onOpen={onOpen} />
    </div>
  );
}

function Confidence({ trace }: { trace: Trace }) {
  const r = trace.retrieval!;
  const threshold = Number(r.config.refusal_threshold ?? 0);
  const conf = r.confidence ?? 0;
  const passed = r.confident;
  return (
    <div className="card p-4">
      <div className="flex items-baseline justify-between gap-4">
        <div>
          <div className="text-[12.5px] text-ink-3">Top reranker score</div>
          <div className="mt-0.5 text-[24px] font-semibold tracking-[-0.02em] text-ink">{num(r.confidence, 3)}</div>
        </div>
        <Tag tone={passed ? "green" : "yellow"}>
          {passed ? "Above threshold: answer" : "Below threshold: refuse"}
        </Tag>
      </div>
      <div className="relative mt-4 h-1.5 rounded-full bg-surface-3">
        <motion.div
          className={cx("absolute inset-y-0 left-0 origin-left rounded-full", passed ? "bg-good" : "bg-series-4")}
          style={{ width: `${Math.min(1, conf) * 100}%` }}
          initial={{ scaleX: 0 }}
          animate={{ scaleX: 1 }}
          transition={{ duration: 0.6, ease: EASE }}
        />
        <div className="absolute -top-1.5 h-[18px] w-px bg-ink" style={{ left: `${threshold * 100}%` }} />
      </div>
      <div className="mt-2.5 flex items-center gap-2 text-[11.5px] text-ink-3">
        <span className="inline-block h-3 w-px bg-ink" aria-hidden />
        Refusal threshold {num(threshold, 2)}, calibrated on the golden eval set
      </div>
    </div>
  );
}

function Pipeline({ trace }: { trace: Trace }) {
  const r = trace.retrieval!;
  const t = r.timings_ms;
  const steps = [
    { label: "Candidates", value: `${r.candidates.length}`, note: `${String(r.config.mode ?? "hybrid")} search, ${ms(t.search)}` },
    { label: "Reranked", value: ms(t.rerank), note: String(r.config.reranker_model ?? "off").replace(/^.*\//, "") },
    { label: "In context", value: `${r.context.length}`, note: `${r.context_tokens.toLocaleString("en")} tokens` },
  ];
  return (
    <div className="mt-4 grid grid-cols-3 gap-px overflow-hidden rounded-[10px] border border-line bg-line">
      {steps.map((s) => (
        <div key={s.label} className="bg-surface px-3.5 py-3">
          <div className="text-[11.5px] text-ink-3">{s.label}</div>
          <div className="mt-0.5 text-[15px] font-semibold text-ink">{s.value}</div>
          <div className="truncate text-[11.5px] text-ink-4" title={s.note}>
            {s.note}
          </div>
        </div>
      ))}
    </div>
  );
}

function Candidates({ trace, onOpen }: { trace: Trace; onOpen: (chunkId: string) => void }) {
  const r = trace.retrieval!;
  const inContext = new Set(r.context.map((c) => c.chunk_id));
  const rows = [...r.candidates].sort((a, b) => (b.rerank_score ?? -1) - (a.rerank_score ?? -1));
  const max = Math.max(...rows.map((c) => c.rerank_score ?? 0), 0.0001);
  return (
    <div className="mt-6">
      <div className="mb-2 flex items-end justify-between">
        <div className="eyebrow">Candidates by reranker score</div>
        <div className="text-[11.5px] text-ink-4">ranks: dense / BM25 before fusion</div>
      </div>
      <table className="w-full border-collapse">
        <thead>
          <tr className="text-left text-[11px] text-ink-3">
            <th className="h-8 w-6 border-b border-line font-medium">#</th>
            <th className="h-8 border-b border-line font-medium">Provision</th>
            <th className="h-8 w-20 border-b border-line text-right font-medium">Ranks</th>
            <th className="h-8 w-40 border-b border-line pl-4 font-medium">Rerank</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((c, i) => (
            <CandidateRow
              key={c.id}
              c={c}
              i={i}
              max={max}
              used={inContext.has(c.id)}
              onOpen={() => onOpen(c.id)}
            />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CandidateRow({
  c,
  i,
  max,
  used,
  onOpen,
}: {
  c: Candidate;
  i: number;
  max: number;
  used: boolean;
  onOpen: () => void;
}) {
  const score = c.rerank_score ?? 0;
  return (
    <motion.tr
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: DUR.slow, ease: EASE, delay: cascade(i, 0.025, 16) }}
      onClick={onOpen}
      className="group cursor-pointer"
    >
      <td className="h-9 border-b border-line text-[11.5px] tnum text-ink-4">{i + 1}</td>
      <td className="h-9 border-b border-line pr-2">
        <div className="flex items-center gap-2">
          <span
            className={cx(
              "truncate font-mono text-[12.5px] transition-colors group-hover:text-ink",
              used ? "text-ink" : "text-ink-3",
            )}
          >
            {c.key}
          </span>
          {used && (
            <span className="inline-flex items-center gap-0.5 text-[10.5px] text-green-fg" title="Passed to the answer as context">
              <Check size={10} weight="bold" />
              context
            </span>
          )}
          {c.source !== "search" && <Tag tone="blue">cited lookup</Tag>}
          {c.truncated && (
            <span className="text-[10.5px] text-ink-4" title="Longer than the reranker's input limit">
              truncated
            </span>
          )}
        </div>
      </td>
      <td className="h-9 border-b border-line text-right font-mono text-[11.5px] text-ink-3">
        {c.dense_rank ?? "–"} / {c.sparse_rank ?? "–"}
      </td>
      <td className="h-9 border-b border-line pl-4">
        <div className="flex items-center gap-2">
          <div className="h-1 flex-1 rounded-full bg-surface-3">
            <div
              className={cx("h-1 rounded-full", used ? "bg-series-1" : "bg-ink-4")}
              style={{ width: `${(score / max) * 100}%` }}
            />
          </div>
          <span className="w-10 text-right font-mono text-[11.5px] tnum text-ink-2">{num(c.rerank_score, 3)}</span>
        </div>
      </td>
    </motion.tr>
  );
}
