import { X } from "@phosphor-icons/react";
import { AnimatePresence, motion } from "motion/react";
import { type ReactNode, useEffect } from "react";
import { DUR, EASE } from "../theme/motion";

/** A right-hand panel over the page. Escape or the backdrop closes it. */
export function Drawer({
  open,
  onClose,
  title,
  subtitle,
  children,
  width = 560,
}: {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  subtitle?: ReactNode;
  children: ReactNode;
  width?: number;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            className="fixed inset-0 z-40 bg-[rgba(17,17,17,0.18)]"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: DUR.base, ease: EASE }}
            onClick={onClose}
          />
          <motion.aside
            role="dialog"
            aria-modal="true"
            className="fixed inset-y-0 right-0 z-50 flex max-w-full flex-col border-l border-line bg-surface shadow-float"
            style={{ width }}
            initial={{ x: 32, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 32, opacity: 0 }}
            transition={{ duration: DUR.base + 0.05, ease: EASE }}
          >
            <div className="flex items-start justify-between gap-4 border-b border-line px-6 py-4">
              <div className="min-w-0">
                <div className="truncate text-[15px] font-semibold text-ink">{title}</div>
                {subtitle && <div className="mt-0.5 text-[12.5px] text-ink-3">{subtitle}</div>}
              </div>
              <button
                onClick={onClose}
                className="-mr-1.5 rounded-md p-1.5 text-ink-3 transition-colors hover:bg-surface-2 hover:text-ink"
                aria-label="Close"
              >
                <X size={16} weight="bold" />
              </button>
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto">{children}</div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
