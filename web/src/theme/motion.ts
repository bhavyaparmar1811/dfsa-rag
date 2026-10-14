import type { Transition, Variants } from "motion/react";

// One motion vocabulary for the whole app: one easing, three durations. The app
// root wraps everything in <MotionConfig reducedMotion="user">, which drops the
// movement (keeping opacity) for people who ask for reduced motion.
export const EASE = [0.16, 1, 0.3, 1] as const;
export const DUR = { fast: 0.15, base: 0.25, slow: 0.4 } as const;

export const t = (duration: number = DUR.base, delay = 0): Transition => ({
  duration,
  delay,
  ease: EASE,
});

export const fadeUp: Variants = {
  hidden: { opacity: 0, y: 6 },
  show: { opacity: 1, y: 0, transition: t(DUR.slow) },
};

/** Delay for the i-th item of a list: a 40ms cascade, capped so a long list
 * doesn't keep cascading for seconds. */
export const cascade = (i: number, step = 0.04, cap = 12) => Math.min(i, cap) * step;
