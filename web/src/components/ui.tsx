import { motion } from "motion/react";
import type { ComponentPropsWithoutRef, ReactNode } from "react";
import { DUR, EASE, fadeUp } from "../theme/motion";

export const cx = (...c: (string | false | null | undefined)[]) => c.filter(Boolean).join(" ");

// --- tags ---------------------------------------------------------------------

export type Tone = "green" | "yellow" | "red" | "blue" | "gray";

const TONE: Record<Tone, string> = {
  green: "bg-green-bg text-green-fg",
  yellow: "bg-yellow-bg text-yellow-fg",
  red: "bg-red-bg text-red-fg",
  blue: "bg-blue-bg text-blue-fg",
  gray: "bg-gray-bg text-gray-fg",
};

/** Small status pill: uppercase, wide tracking, pastel. Never colour alone: the
 * text always says what the colour means. */
export function Tag({
  tone = "gray",
  icon,
  children,
  className,
}: {
  tone?: Tone;
  icon?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <span
      className={cx(
        "inline-flex h-5 shrink-0 items-center gap-1 rounded-full px-2 text-[10.5px] font-medium uppercase leading-none tracking-[0.06em] whitespace-nowrap",
        TONE[tone],
        className,
      )}
    >
      {icon}
      {children}
    </span>
  );
}

/** Rule labels from the API ("Rule (binding)", "Guidance (non-binding)") -> tone. */
export const labelTone = (label: string): Tone =>
  /non-binding/i.test(label) ? "yellow" : /binding/i.test(label) ? "green" : "gray";

// --- buttons ------------------------------------------------------------------

type ButtonProps = ComponentPropsWithoutRef<"button"> & {
  variant?: "primary" | "secondary" | "ghost";
  size?: "sm" | "md";
};

export function Button({ variant = "secondary", size = "md", className, ...rest }: ButtonProps) {
  return (
    <button
      {...rest}
      className={cx(
        "inline-flex items-center justify-center gap-1.5 rounded-md font-medium transition-[background-color,border-color,color,transform] duration-150 ease-out-soft active:scale-[0.98] disabled:pointer-events-none disabled:opacity-45",
        size === "sm" ? "h-7 px-2.5 text-[12.5px]" : "h-9 px-3.5 text-[13.5px]",
        variant === "primary" && "bg-accent text-accent-ink hover:bg-accent-hover",
        variant === "secondary" &&
          "border border-line bg-surface text-ink hover:border-line-strong hover:bg-surface-2",
        variant === "ghost" && "text-ink-2 hover:bg-surface-2 hover:text-ink",
        className,
      )}
    />
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="inline-flex h-[18px] min-w-[18px] items-center justify-center rounded border border-line bg-surface-2 px-1 font-mono text-[10.5px] text-ink-3">
      {children}
    </kbd>
  );
}

// --- layout -------------------------------------------------------------------

export function Page({ children, wide = false }: { children: ReactNode; wide?: boolean }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: DUR.slow, ease: EASE }}
      className={cx("mx-auto w-full px-6 pb-24 pt-12 lg:px-10", wide ? "max-w-6xl" : "max-w-5xl")}
    >
      {children}
    </motion.div>
  );
}

export function PageHeader({
  eyebrow,
  title,
  children,
  actions,
}: {
  eyebrow?: ReactNode;
  title: ReactNode;
  children?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <header className="mb-12 flex flex-wrap items-end justify-between gap-6">
      <div className="max-w-2xl">
        {eyebrow && <div className="eyebrow mb-3">{eyebrow}</div>}
        <h1 className="font-serif text-[34px] font-normal leading-[1.1] tracking-[-0.02em] text-ink">
          {title}
        </h1>
        {children && <p className="mt-3 text-[14.5px] leading-relaxed text-ink-3">{children}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </header>
  );
}

/** A titled block that fades in as it scrolls into view. */
export function Section({
  title,
  description,
  aside,
  children,
  className,
}: {
  title: ReactNode;
  description?: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <motion.section
      variants={fadeUp}
      initial="hidden"
      whileInView="show"
      viewport={{ once: true, margin: "-60px" }}
      className={cx("mt-16 first:mt-0", className)}
    >
      <div className="mb-5 flex flex-wrap items-end justify-between gap-4 border-b border-line pb-3">
        <div>
          <h2 className="text-[15px] font-semibold tracking-[-0.01em] text-ink">{title}</h2>
          {description && <p className="mt-1 max-w-2xl text-[13px] text-ink-3">{description}</p>}
        </div>
        {aside}
      </div>
      {children}
    </motion.section>
  );
}

export function Card({
  children,
  className,
  ...rest
}: ComponentPropsWithoutRef<"div"> & { children: ReactNode }) {
  return (
    <div {...rest} className={cx("card", className)}>
      {children}
    </div>
  );
}

// --- figures ------------------------------------------------------------------

/** Stat tile: label, value (proportional figures), optional note. */
export function Stat({
  label,
  value,
  note,
  tone,
}: {
  label: ReactNode;
  value: ReactNode;
  note?: ReactNode;
  tone?: "good" | "bad";
}) {
  return (
    <div className="card h-full px-5 py-4">
      <div className="text-[12.5px] text-ink-3">{label}</div>
      <div
        className={cx(
          "mt-1.5 text-[26px] font-semibold leading-tight tracking-[-0.02em]",
          tone === "bad" ? "text-red-fg" : "text-ink",
        )}
      >
        {value}
      </div>
      {note && <div className="mt-1 text-[12px] text-ink-3">{note}</div>}
    </div>
  );
}

/** A thin horizontal meter. `value` and `marker` are 0-1. */
export function Meter({
  value,
  marker,
  tone = "ink",
  className,
}: {
  value: number;
  marker?: number | null;
  tone?: "ink" | "good" | "bad" | "series";
  className?: string;
}) {
  const fill =
    tone === "good"
      ? "bg-good"
      : tone === "bad"
        ? "bg-critical"
        : tone === "series"
          ? "bg-series-1"
          : "bg-ink-2";
  return (
    <div className={cx("relative h-1.5 w-full rounded-full bg-surface-3", className)}>
      <motion.div
        className={cx("absolute inset-y-0 left-0 origin-left rounded-full", fill)}
        style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
        initial={{ scaleX: 0 }}
        whileInView={{ scaleX: 1 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6, ease: EASE }}
      />
      {marker != null && (
        <div
          className="absolute -top-1 h-3.5 w-px bg-ink"
          style={{ left: `${Math.max(0, Math.min(1, marker)) * 100}%` }}
          aria-hidden
        />
      )}
    </div>
  );
}

// --- states -------------------------------------------------------------------

export function Skeleton({ className }: { className?: string }) {
  return <div className={cx("shimmer rounded-md", className)} aria-hidden />;
}

export function EmptyState({ title, children }: { title: ReactNode; children?: ReactNode }) {
  return (
    <div className="rounded-[10px] border border-dashed border-line-strong px-6 py-10 text-center">
      <div className="text-[13.5px] font-medium text-ink-2">{title}</div>
      {children && <div className="mt-1 text-[13px] text-ink-3">{children}</div>}
    </div>
  );
}

export function ErrorNote({ error, title = "Couldn't load this" }: { error: unknown; title?: string }) {
  const msg = error instanceof Error ? error.message : String(error);
  return (
    <div className="rounded-lg border border-red-bg bg-red-bg/60 px-4 py-3 text-[13px] text-red-fg">
      <span className="font-medium">{title}.</span> {msg}
    </div>
  );
}

// --- tabs ---------------------------------------------------------------------

export function Tabs<T extends string>({
  value,
  onChange,
  options,
  layoutId,
}: {
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: ReactNode }[];
  layoutId: string;
}) {
  return (
    <div className="inline-flex rounded-lg border border-line bg-surface-2 p-0.5" role="tablist">
      {options.map((o) => (
        <button
          key={o.value}
          role="tab"
          aria-selected={o.value === value}
          onClick={() => onChange(o.value)}
          className={cx(
            "relative h-7 rounded-md px-3 text-[12.5px] font-medium transition-colors duration-150",
            o.value === value ? "text-ink" : "text-ink-3 hover:text-ink-2",
          )}
        >
          {o.value === value && (
            <motion.span
              layoutId={layoutId}
              className="absolute inset-0 rounded-md border border-line bg-surface"
              transition={{ duration: DUR.base, ease: EASE }}
            />
          )}
          <span className="relative">{o.label}</span>
        </button>
      ))}
    </div>
  );
}

// --- tables -------------------------------------------------------------------

export function Th({
  children,
  align = "left",
  className,
}: {
  children?: ReactNode;
  align?: "left" | "right";
  className?: string;
}) {
  return (
    <th
      className={cx(
        "h-9 border-b border-line px-3 text-[11.5px] font-medium text-ink-3 first:pl-0 last:pr-0",
        align === "right" ? "text-right" : "text-left",
        className,
      )}
    >
      {children}
    </th>
  );
}

export function Td({
  children,
  align = "left",
  className,
  mono,
}: {
  children?: ReactNode;
  align?: "left" | "right";
  className?: string;
  mono?: boolean;
}) {
  return (
    <td
      className={cx(
        "h-10 border-b border-line px-3 text-[13px] first:pl-0 last:pr-0",
        align === "right" && "text-right tnum",
        mono && "font-mono text-[12.5px]",
        className,
      )}
    >
      {children}
    </td>
  );
}
