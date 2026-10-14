import {
  ArrowClockwise,
  ArrowSquareOut,
  ArrowsClockwise,
  Lightning,
  MagnifyingGlass,
  Prohibit,
  WarningCircle,
} from "@phosphor-icons/react";
import { motion } from "motion/react";
import { useEffect, useState } from "react";
import ReactMarkdown, { defaultUrlTransform } from "react-markdown";
import type { Citation } from "../../api/types";
import { Button, Tag, cx, labelTone } from "../../components/ui";
import { REFUSAL_LABELS, ms, usd } from "../../lib/format";
import { DUR, EASE, cascade } from "../../theme/motion";
import { type TurnView, linkCitations } from "./linkify";

/** The trace whose retrieval an answer rests on: a cached answer's is the turn
 * that first produced it. */
const retrievalTrace = (turn: TurnView) => (turn.cached && turn.sourceTraceId) || turn.traceId;

type Open = {
  chunk: (chunkId: string, traceId: string | null) => void;
  retrieval: (traceId: string) => void;
};

export function TurnCard({
  turn,
  open,
  onRetry,
}: {
  turn: TurnView;
  open: Open;
  onRetry: (turn: TurnView) => void;
}) {
  return (
    <motion.article
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: DUR.slow, ease: EASE }}
      className="py-8 first:pt-2"
    >
      <Question turn={turn} />
      <div className="mt-5">
        {turn.state === "pending" && <Pending startedAt={turn.startedAt} />}
        {turn.state === "answered" && <Answer turn={turn} open={open} />}
        {turn.state === "refused" && <RefusalNote turn={turn} open={open} />}
        {turn.state === "error" && <ErrorNote turn={turn} onRetry={onRetry} />}
      </div>
    </motion.article>
  );
}

function Question({ turn }: { turn: TurnView }) {
  return (
    <div>
      <h3 className="font-serif text-[21px] font-normal leading-snug tracking-[-0.015em] text-ink">
        {turn.question}
      </h3>
      {turn.rewritten && (
        <div className="mt-2 flex items-start gap-2 text-[12.5px] text-ink-3">
          <ArrowsClockwise size={13} weight="bold" className="mt-[3px] shrink-0" />
          <span>
            Read as <span className="text-ink-2">“{turn.rewritten}”</span>
          </span>
        </div>
      )}
    </div>
  );
}

// --- answered ---------------------------------------------------------------------

function Answer({ turn, open }: { turn: TurnView; open: Open }) {
  const linked = linkCitations(turn.text, turn.citations);
  return (
    <div>
      <div className="prose-answer max-w-[68ch] text-[14.5px] leading-[1.7] text-ink-2">
        <ReactMarkdown
          urlTransform={(url) => (url.startsWith("cite:") ? url : defaultUrlTransform(url))}
          components={{
            a: ({ href, children }) => {
              if (href?.startsWith("cite:")) {
                const c = turn.citations[Number(href.slice(5))];
                return c ? (
                  <CitationChip citation={c} onClick={() => open.chunk(c.chunk_id, retrievalTrace(turn))} />
                ) : (
                  <>{children}</>
                );
              }
              return (
                <a href={href} target="_blank" rel="noreferrer" className="underline decoration-line-strong underline-offset-2">
                  {children}
                </a>
              );
            },
          }}
        >
          {linked}
        </ReactMarkdown>
      </div>

      {turn.guidanceOnly && (
        <div className="mt-4 flex items-center gap-2 text-[12.5px] text-yellow-fg">
          <WarningCircle size={14} weight="bold" />
          This answer rests only on non-binding Guidance, not on a binding Rule.
        </div>
      )}

      {turn.citations.length > 0 && <Sources turn={turn} open={open} />}
      <Meta turn={turn} open={open} />
    </div>
  );
}

export function CitationChip({ citation, onClick }: { citation: Citation; onClick: () => void }) {
  const nonBinding = /non-binding/i.test(citation.label);
  return (
    <button
      onClick={onClick}
      title={`${citation.label}: open the text`}
      className={cx(
        "inline-flex translate-y-[-1px] items-center rounded-[5px] border px-1.5 py-[1px] align-baseline font-mono text-[12px] leading-[1.45] transition-[background-color,border-color] duration-150",
        nonBinding
          ? "border-yellow-bg bg-yellow-bg/70 text-yellow-fg hover:border-yellow-fg/30"
          : "border-line bg-surface-2 text-ink hover:border-line-strong hover:bg-surface-3",
      )}
    >
      {citation.citation}
    </button>
  );
}

function Sources({ turn, open }: { turn: TurnView; open: Open }) {
  return (
    <div className="mt-6">
      <div className="eyebrow mb-2">Sources</div>
      <div className="grid gap-px overflow-hidden rounded-lg border border-line bg-line sm:grid-cols-2">
        {turn.citations.map((c, i) => (
          <motion.button
            key={c.chunk_id + i}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: DUR.slow, ease: EASE, delay: 0.1 + cascade(i) }}
            onClick={() => open.chunk(c.chunk_id, retrievalTrace(turn))}
            className="group flex items-center justify-between gap-3 bg-surface px-3.5 py-2.5 text-left transition-colors duration-150 hover:bg-surface-2"
          >
            <span className="truncate font-mono text-[12.5px] text-ink">{c.citation}</span>
            <Tag tone={labelTone(c.label)}>{c.label.replace(/\s*\(.*\)/, "")}</Tag>
          </motion.button>
        ))}
        {turn.citations.length % 2 === 1 && <div className="hidden bg-surface sm:block" />}
      </div>
    </div>
  );
}

const STAGES: { key: string; label: string; color: string }[] = [
  { key: "rewrite", label: "Rewrite", color: "bg-series-4" },
  { key: "cache", label: "Embed + cache check", color: "bg-series-3" },
  { key: "retrieve", label: "Retrieve + rerank", color: "bg-series-1" },
  { key: "answer", label: "Answer", color: "bg-series-2" },
];

function Meta({ turn, open }: { turn: TurnView; open: Open }) {
  const stages = turn.latency ? STAGES.filter((s) => (turn.latency?.[s.key] ?? 0) >= 1) : [];
  const sum = stages.reduce((a, s) => a + (turn.latency?.[s.key] ?? 0), 0);
  return (
    <div className="mt-5 border-t border-line pt-4">
      {stages.length > 0 && (
        <div className="mb-3">
          <div className="flex h-1.5 w-full max-w-[420px] gap-[2px] overflow-hidden rounded-full">
            {stages.map((s, i) => (
              <motion.div
                key={s.key}
                className={cx("h-full origin-left first:rounded-l-full last:rounded-r-full", s.color)}
                style={{ width: `${((turn.latency?.[s.key] ?? 0) / sum) * 100}%` }}
                initial={{ scaleX: 0 }}
                animate={{ scaleX: 1 }}
                transition={{ duration: 0.5, ease: EASE, delay: 0.15 + i * 0.08 }}
                title={`${s.label}: ${ms(turn.latency?.[s.key])}`}
              />
            ))}
          </div>
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11.5px] text-ink-3">
            {stages.map((s) => (
              <span key={s.key} className="inline-flex items-center gap-1.5">
                <span className={cx("h-2 w-2 rounded-[2px]", s.color)} />
                {s.label} <span className="tnum text-ink-2">{ms(turn.latency?.[s.key])}</span>
              </span>
            ))}
          </div>
        </div>
      )}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-[12px] text-ink-3">
        {turn.cached && (
          <Tag tone="blue" icon={<Lightning size={10} weight="fill" />}>
            Cached answer
          </Tag>
        )}
        <span className="tnum">{ms(turn.latencyTotal)}</span>
        <span className="tnum">{usd(turn.costUsd)}</span>
        {turn.traceId && (
          <button
            onClick={() => open.retrieval(retrievalTrace(turn)!)}
            className="inline-flex items-center gap-1.5 text-ink-2 transition-colors hover:text-ink"
          >
            <MagnifyingGlass size={13} weight="bold" />
            {turn.cached ? "Inspect original retrieval" : "Inspect retrieval"}
          </button>
        )}
        {turn.traceUrl && (
          <a
            href={turn.traceUrl}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 text-ink-2 transition-colors hover:text-ink"
          >
            <ArrowSquareOut size={13} weight="bold" />
            Langfuse trace
          </a>
        )}
      </div>
    </div>
  );
}

// --- refused ------------------------------------------------------------------------

function RefusalNote({ turn, open }: { turn: TurnView; open: Open }) {
  const code = turn.refusalCode ?? "";
  return (
    <div>
      <div className="max-w-[68ch] rounded-[10px] border border-yellow-bg bg-yellow-bg/45 px-5 py-4">
        <div className="flex items-center gap-2 text-[13px] font-semibold text-yellow-fg">
          <Prohibit size={15} weight="bold" />
          No confident answer
          {code && <span className="font-normal opacity-80">· {REFUSAL_LABELS[code] ?? code}</span>}
        </div>
        <div className="mt-2 whitespace-pre-line text-[14px] leading-[1.7] text-ink-2">{turn.text}</div>
      </div>
      <p className="mt-3 max-w-[68ch] text-[12.5px] text-ink-3">
        The assistant refuses rather than guess when the rulebook doesn’t clearly cover a question.
      </p>
      <Meta turn={turn} open={open} />
    </div>
  );
}

// --- pending and failed ---------------------------------------------------------------

function Pending({ startedAt }: { startedAt: number }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 100);
    return () => window.clearInterval(id);
  }, []);
  const secs = Math.max(0, (now - startedAt) / 1000);
  return (
    <div aria-live="polite">
      <div className="flex items-center gap-2.5 text-[13px] text-ink-3">
        <span className="relative flex h-2 w-2">
          <span className="absolute inset-0 rounded-full bg-ink-3 [animation:pulse-dot_1.2s_ease-in-out_infinite]" />
        </span>
        Searching the rulebook
        <span className="tnum text-ink-4">{secs.toFixed(1)} s</span>
      </div>
      <div className="mt-4 max-w-[68ch] space-y-2.5">
        <div className="shimmer h-3.5 w-[92%] rounded" />
        <div className="shimmer h-3.5 w-[84%] rounded" />
        <div className="shimmer h-3.5 w-[60%] rounded" />
      </div>
    </div>
  );
}

function ErrorNote({ turn, onRetry }: { turn: TurnView; onRetry: (t: TurnView) => void }) {
  return (
    <div className="max-w-[68ch] rounded-[10px] border border-red-bg bg-red-bg/50 px-5 py-4">
      <div className="flex items-center gap-2 text-[13px] font-semibold text-red-fg">
        <WarningCircle size={15} weight="bold" />
        This question couldn’t be answered
      </div>
      <div className="mt-1.5 text-[13px] text-ink-2">{turn.error}</div>
      {turn.traceId && (
        <div className="mt-1 font-mono text-[11.5px] text-ink-3">trace {turn.traceId}</div>
      )}
      <Button size="sm" className="mt-3" onClick={() => onRetry(turn)}>
        <ArrowClockwise size={13} weight="bold" />
        Try again
      </Button>
    </div>
  );
}
