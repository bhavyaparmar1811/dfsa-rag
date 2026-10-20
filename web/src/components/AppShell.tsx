import {
  ChartLineUp,
  ChatCircleText,
  CurrencyDollarSimple,
  Moon,
  Plus,
  Sun,
} from "@phosphor-icons/react";
import { AnimatePresence, motion } from "motion/react";
import type { ReactNode } from "react";
import { Link, NavLink, useLocation, useNavigate } from "react-router-dom";
import { useConversations } from "../api/hooks";
import { relative } from "../lib/format";
import { DUR, EASE, cascade } from "../theme/motion";
import { HealthDot } from "./HealthDot";
import { useTheme } from "./theme";
import { cx } from "./ui";

const NAV = [
  { to: "/", label: "Ask", icon: ChatCircleText, match: (p: string) => p === "/" || p.startsWith("/c/") },
  { to: "/evals", label: "Evaluations", icon: ChartLineUp, match: (p: string) => p.startsWith("/evals") },
  { to: "/cost", label: "Cost & usage", icon: CurrencyDollarSimple, match: (p: string) => p.startsWith("/cost") },
];

/** Starts a fresh chat: a new key remounts the chat page even when already on "/". */
export function useNewChat() {
  const navigate = useNavigate();
  return () => navigate("/", { state: { chatKey: `new-${Date.now()}` } });
}

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-dvh overflow-hidden bg-canvas">
      <Sidebar />
      <main className="relative min-w-0 flex-1 overflow-y-auto" id="main">
        {children}
      </main>
    </div>
  );
}

function Sidebar() {
  const { pathname } = useLocation();
  const { theme, toggle } = useTheme();
  const newChat = useNewChat();

  return (
    <aside className="flex w-[252px] shrink-0 flex-col border-r border-line bg-canvas">
      <Link to="/" className="flex items-center gap-2.5 px-5 pb-5 pt-6">
        <Mark />
        <div className="leading-tight">
          <div className="text-[14px] font-semibold tracking-[-0.01em] text-ink">Rulebook Assistant</div>
          <div className="text-[11.5px] text-ink-3">DFSA Rulebook and DIFC Laws</div>
        </div>
      </Link>

      <div className="px-3">
        <button
          onClick={newChat}
          className="flex h-9 w-full items-center gap-2 rounded-md border border-line bg-surface px-3 text-[13px] font-medium text-ink transition-[border-color,background-color,transform] duration-150 hover:border-line-strong active:scale-[0.98]"
        >
          <Plus size={14} weight="bold" />
          New question
        </button>
      </div>

      <nav className="mt-4 flex flex-col gap-0.5 px-3">
        {NAV.map(({ to, label, icon: Icon, match }) => {
          const active = match(pathname);
          return (
            <NavLink
              key={to}
              to={to}
              className={cx(
                "relative flex h-8 items-center gap-2.5 rounded-md px-2.5 text-[13px] transition-colors duration-150",
                active ? "font-medium text-ink" : "text-ink-3 hover:text-ink-2",
              )}
            >
              {active && (
                <motion.span
                  layoutId="nav-active"
                  className="absolute inset-0 rounded-md bg-surface-3"
                  transition={{ duration: DUR.base, ease: EASE }}
                />
              )}
              <Icon size={16} weight={active ? "fill" : "bold"} className="relative" />
              <span className="relative">{label}</span>
            </NavLink>
          );
        })}
      </nav>

      <Recent />

      <div className="border-t border-line px-3 py-3">
        <HealthDot />
        <button
          onClick={toggle}
          className="mt-0.5 flex w-full items-center gap-2.5 rounded-md px-2 py-1.5 text-[12.5px] text-ink-3 transition-colors hover:bg-surface-2 hover:text-ink-2"
        >
          <AnimatePresence mode="wait" initial={false}>
            <motion.span
              key={theme}
              initial={{ opacity: 0, rotate: -30 }}
              animate={{ opacity: 1, rotate: 0 }}
              exit={{ opacity: 0, rotate: 30 }}
              transition={{ duration: DUR.fast, ease: EASE }}
              className="flex"
            >
              {theme === "dark" ? <Sun size={14} weight="bold" /> : <Moon size={14} weight="bold" />}
            </motion.span>
          </AnimatePresence>
          {theme === "dark" ? "Light theme" : "Dark theme"}
        </button>
        <p className="mt-2 px-2 text-[11px] leading-snug text-ink-4">
          A personal portfolio project. Not affiliated with or endorsed by the DFSA.
        </p>
      </div>
    </aside>
  );
}

function Recent() {
  const { data } = useConversations();
  const { pathname } = useLocation();
  const activeId = pathname.startsWith("/c/") ? pathname.slice(3) : undefined;

  return (
    <div className="mt-6 flex min-h-0 flex-1 flex-col">
      <div className="eyebrow px-5 pb-2">Recent</div>
      <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-3">
        {data?.length === 0 && (
          <div className="px-2.5 text-[12.5px] text-ink-4">Questions you ask appear here.</div>
        )}
        <AnimatePresence initial={false}>
          {data?.map((c, i) => (
            <motion.div
              key={c.id}
              layout="position"
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0, transition: { duration: DUR.slow, ease: EASE, delay: cascade(i, 0.02) } }}
            >
              <Link
                to={`/c/${c.id}`}
                className={cx(
                  "group block rounded-md px-2.5 py-1.5 transition-colors duration-150",
                  c.id === activeId ? "bg-surface-3" : "hover:bg-surface-2",
                )}
              >
                <div
                  className={cx(
                    "truncate text-[12.5px]",
                    c.id === activeId ? "font-medium text-ink" : "text-ink-2",
                  )}
                >
                  {c.title || "Untitled"}
                </div>
                <div className="text-[11px] text-ink-4">{relative(c.updated_at)}</div>
              </Link>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}

/** The wordmark: a small "D" on an ink square, drawn rather than an icon font. */
function Mark() {
  return (
    <svg width="28" height="28" viewBox="0 0 32 32" aria-hidden className="shrink-0">
      <rect width="32" height="32" rx="7" className="fill-ink" />
      <path
        d="M10.5 9h6a7 7 0 0 1 0 14h-6z"
        fill="none"
        className="stroke-canvas"
        strokeWidth="2.6"
        strokeLinejoin="round"
      />
    </svg>
  );
}
