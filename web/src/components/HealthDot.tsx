import { AnimatePresence, motion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { useHealth } from "../api/hooks";
import type { HealthCheck } from "../api/types";
import { int } from "../lib/format";
import { DUR, EASE } from "../theme/motion";
import { cx } from "./ui";

const NAMES: Record<string, string> = {
  database: "SQLite",
  qdrant: "Qdrant",
  reranker: "Reranker",
  cache: "Answer cache",
  langfuse: "Langfuse",
};

function detail(name: string, c: HealthCheck): string {
  if (!c.ok) return String(c.error ?? "unavailable");
  if (name === "qdrant") return `${int(c.points as number)} points`;
  if (name === "reranker") return c.enabled ? `on ${String(c.device ?? "cpu")}` : "off";
  if (name === "cache")
    return c.enabled === false ? "off" : `exact repeats, ${int(c.points as number)} stored`;
  if (name === "langfuse") return c.enabled === false ? "tracing off" : "tracing";
  return "connected";
}

type State = "ok" | "optional" | "down" | "unknown";

const DOT: Record<State, string> = {
  ok: "bg-good",
  optional: "bg-series-4",
  down: "bg-critical",
  unknown: "bg-ink-4",
};

/** Polls /health; a popover lists each dependency. Langfuse is optional (the API
 * answers without it), so it being down shows amber, not red. */
export function HealthDot() {
  const { data, isLoading } = useHealth();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => !ref.current?.contains(e.target as Node) && setOpen(false);
    window.addEventListener("mousedown", onDown);
    return () => window.removeEventListener("mousedown", onDown);
  }, [open]);

  const checks = data?.checks ?? {};
  const state: State = isLoading
    ? "unknown"
    : !data
      ? "down"
      : data.status === "degraded"
        ? "down"
        : Object.values(checks).some((c) => !c.ok)
          ? "optional"
          : "ok";
  const label = {
    ok: "All systems running",
    optional: "Running, tracing degraded",
    down: data ? "A service is down" : "API not reachable",
    unknown: "Checking services",
  }[state];

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2.5 rounded-md px-2 py-1.5 text-left text-[12.5px] text-ink-3 transition-colors hover:bg-surface-2 hover:text-ink-2"
        aria-expanded={open}
      >
        <span className="relative flex h-2 w-2">
          {state === "ok" && (
            <span className="absolute inset-0 rounded-full bg-good opacity-60 [animation:pulse-dot_2.4s_ease-in-out_infinite]" />
          )}
          <span className={cx("relative h-2 w-2 rounded-full", DOT[state])} />
        </span>
        {label}
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 4 }}
            transition={{ duration: DUR.fast, ease: EASE }}
            className="absolute bottom-full left-0 z-30 mb-2 w-64 rounded-lg border border-line bg-surface p-1.5 shadow-float"
          >
            {Object.keys(checks).length === 0 && (
              <div className="px-2.5 py-2 text-[12.5px] text-ink-3">
                No response from the API on port 8000.
              </div>
            )}
            {Object.entries(checks).map(([name, c]) => {
              const s: State = c.ok ? "ok" : name === "langfuse" ? "optional" : "down";
              return (
                <div key={name} className="flex items-center gap-2.5 rounded-md px-2.5 py-1.5">
                  <span className={cx("h-1.5 w-1.5 shrink-0 rounded-full", DOT[s])} />
                  <span className="text-[12.5px] font-medium text-ink">{NAMES[name] ?? name}</span>
                  <span className="ml-auto truncate text-[12px] text-ink-3" title={detail(name, c)}>
                    {detail(name, c)}
                  </span>
                </div>
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
