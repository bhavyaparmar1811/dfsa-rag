import { ArrowUp, Check, Funnel } from "@phosphor-icons/react";
import { AnimatePresence, motion } from "motion/react";
import { type KeyboardEvent, useEffect, useRef, useState } from "react";
import type { Filters } from "../../api/types";
import { Kbd, cx } from "../../components/ui";
import { DUR, EASE } from "../../theme/motion";

// Each chip is a set of manifest module codes; the selected chips' codes are
// sent as one `modules` filter (an OR), so "COB" + "Laws" means either.
export const SCOPES = [
  { id: "GEN", label: "GEN", modules: ["GEN"], title: "General Module" },
  { id: "COB", label: "COB", modules: ["COB"], title: "Conduct of Business" },
  { id: "AML", label: "AML", modules: ["AML"], title: "Anti-Money Laundering" },
  { id: "MKT", label: "MKT", modules: ["MKT"], title: "Markets Rules" },
  { id: "FPR", label: "FPR", modules: ["FPR"], title: "Fund Protocol Rules" },
  { id: "GLO", label: "Glossary", modules: ["GLO"], title: "Defined Terms" },
  {
    id: "LAWS",
    label: "DIFC Laws",
    modules: ["REG_LAW", "MARKETS_LAW", "COLLECTIVE_INVESTMENT_LAW", "INVESTMENT_LAW", "ISL_LAW"],
    title: "Regulatory, Markets, Collective Investment, Investment Trust and Islamic Finance Laws",
  },
] as const;

export type FilterState = { scopes: string[]; bindingOnly: boolean };

export function toFilters(f: FilterState): Filters | undefined {
  const modules = SCOPES.filter((s) => f.scopes.includes(s.id)).flatMap((s) => [...s.modules]);
  if (!modules.length && !f.bindingOnly) return undefined;
  return { modules: modules.length ? modules : null, binding_only: f.bindingOnly };
}

export function Composer({
  onSend,
  busy,
  filters,
  onFilters,
  autoFocus,
}: {
  onSend: (text: string) => void;
  busy: boolean;
  filters: FilterState;
  onFilters: (f: FilterState) => void;
  autoFocus?: boolean;
}) {
  const [text, setText] = useState("");
  const [showFilters, setShowFilters] = useState(filters.scopes.length > 0 || filters.bindingOnly);
  const ref = useRef<HTMLTextAreaElement>(null);

  // Grow with the text, up to ~8 lines.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [text]);

  useEffect(() => {
    if (autoFocus) ref.current?.focus();
  }, [autoFocus]);

  const submit = () => {
    const q = text.trim();
    if (!q || busy) return;
    onSend(q);
    setText("");
  };

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  const active = filters.scopes.length + (filters.bindingOnly ? 1 : 0);
  const toggleScope = (id: string) =>
    onFilters({
      ...filters,
      scopes: filters.scopes.includes(id)
        ? filters.scopes.filter((s) => s !== id)
        : [...filters.scopes, id],
    });

  return (
    <div className="rounded-xl border border-line bg-surface transition-[border-color,box-shadow] duration-200 focus-within:border-line-strong focus-within:shadow-hover">
      <textarea
        ref={ref}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKey}
        rows={1}
        placeholder="Ask about a DFSA rule, obligation or defined term"
        className="block w-full resize-none bg-transparent px-4 pb-2 pt-3.5 text-[14.5px] leading-[1.6] text-ink outline-none placeholder:text-ink-4"
        aria-label="Your question"
      />

      <AnimatePresence initial={false}>
        {showFilters && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: DUR.base, ease: EASE }}
            className="overflow-hidden"
          >
            <div className="flex flex-wrap items-center gap-1.5 px-3 pb-1 pt-1">
              <span className="mr-1 text-[11.5px] text-ink-4">Search only</span>
              {SCOPES.map((s) => {
                const on = filters.scopes.includes(s.id);
                return (
                  <button
                    key={s.id}
                    title={s.title}
                    onClick={() => toggleScope(s.id)}
                    aria-pressed={on}
                    className={cx(
                      "h-6 rounded-md border px-2 text-[11.5px] font-medium transition-colors duration-150",
                      on
                        ? "border-ink bg-ink text-canvas"
                        : "border-line bg-surface text-ink-2 hover:border-line-strong",
                    )}
                  >
                    {s.label}
                  </button>
                );
              })}
              <span className="mx-1 h-4 w-px bg-line" />
              <button
                onClick={() => onFilters({ ...filters, bindingOnly: !filters.bindingOnly })}
                aria-pressed={filters.bindingOnly}
                className="inline-flex h-6 items-center gap-1.5 rounded-md px-1.5 text-[11.5px] font-medium text-ink-2 transition-colors hover:text-ink"
              >
                <span
                  className={cx(
                    "flex h-3.5 w-3.5 items-center justify-center rounded-[4px] border transition-colors duration-150",
                    filters.bindingOnly ? "border-ink bg-ink text-canvas" : "border-line-strong",
                  )}
                >
                  {filters.bindingOnly && <Check size={9} weight="bold" />}
                </span>
                Binding provisions only
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="flex items-center justify-between gap-3 px-3 pb-3 pt-1.5">
        <button
          onClick={() => setShowFilters((s) => !s)}
          className={cx(
            "inline-flex h-7 items-center gap-1.5 rounded-md px-2 text-[12px] font-medium transition-colors duration-150",
            showFilters || active ? "text-ink" : "text-ink-3 hover:text-ink-2",
          )}
          aria-expanded={showFilters}
        >
          <Funnel size={13} weight={active ? "fill" : "bold"} />
          {active ? `${active} filter${active > 1 ? "s" : ""}` : "Filter sources"}
        </button>
        <div className="flex items-center gap-3">
          <span className="hidden items-center gap-1 text-[11.5px] text-ink-4 sm:inline-flex">
            <Kbd>Enter</Kbd> to send · <Kbd>Shift</Kbd>+<Kbd>Enter</Kbd> new line
          </span>
          <button
            onClick={submit}
            disabled={!text.trim() || busy}
            className="flex h-8 w-8 items-center justify-center rounded-md bg-accent text-accent-ink transition-[background-color,transform,opacity] duration-150 hover:bg-accent-hover active:scale-[0.96] disabled:opacity-30"
            aria-label="Send"
          >
            <ArrowUp size={15} weight="bold" />
          </button>
        </div>
      </div>
    </div>
  );
}
