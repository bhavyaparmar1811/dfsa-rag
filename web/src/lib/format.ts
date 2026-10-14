// Number and date formatting, in one place so every screen reads the same.

export const pct = (v: number | null | undefined, digits = 1) =>
  v == null ? "–" : `${(v * 100).toFixed(digits)}%`;

export const num = (v: number | null | undefined, digits = 3) =>
  v == null ? "–" : v.toFixed(digits);

export const usd = (v: number | null | undefined) => {
  if (v == null) return "–";
  if (v === 0) return "$0";
  if (Math.abs(v) < 0.0001) return "< $0.0001";
  if (Math.abs(v) < 0.01) return `$${v.toFixed(4)}`;
  if (Math.abs(v) < 1) return `$${v.toFixed(3)}`;
  return `$${v.toFixed(2)}`;
};

export const ms = (v: number | null | undefined) => {
  if (v == null) return "–";
  if (v < 1000) return `${Math.round(v)} ms`;
  return `${(v / 1000).toFixed(v < 10_000 ? 2 : 1)} s`;
};

export const int = (v: number | null | undefined) =>
  v == null ? "–" : new Intl.NumberFormat("en").format(v);

const dateFmt = new Intl.DateTimeFormat("en-GB", {
  day: "numeric",
  month: "short",
  year: "numeric",
});
const dateTimeFmt = new Intl.DateTimeFormat("en-GB", {
  day: "numeric",
  month: "short",
  hour: "2-digit",
  minute: "2-digit",
});
const dayFmt = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short" });

// Some columns are naive UTC timestamps: read a string without an offset as UTC.
export const parseDate = (s: string) => new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(s) ? s : `${s}Z`);
export const date = (s: string) => dateFmt.format(parseDate(s));
export const dateTime = (s: string) => dateTimeFmt.format(parseDate(s));
export const day = (isoDay: string) => dayFmt.format(new Date(`${isoDay}T00:00:00Z`));

export function relative(s: string) {
  const diff = (Date.now() - parseDate(s).getTime()) / 1000;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  if (diff < 86400 * 7) return `${Math.floor(diff / 86400)}d ago`;
  return date(s);
}

export const shortHash = (h: string | null | undefined, n = 7) => (h ? h.slice(0, n) : "–");

export const CONFIG_LABELS: Record<string, string> = {
  dense: "Dense",
  sparse: "Sparse (BM25)",
  hybrid: "Hybrid",
  hybrid_rerank: "Hybrid + rerank",
  hybrid_rerank_base: "Rerank: bge-base",
  hybrid_rerank_minilm: "Rerank: MiniLM",
  parent_none: "No parent context",
  parent_window: "Window context",
  semantic_cache: "Answer cache",
};
export const configLabel = (name: string) => CONFIG_LABELS[name] ?? name;

export const KIND_LABELS: Record<string, string> = {
  single: "Single question",
  multi_turn: "Multi-turn follow-up",
  laws_term: "Term from a Law",
  exemption: "Exemption",
  unanswerable: "Unanswerable",
};

export const REFUSAL_LABELS: Record<string, string> = {
  no_context: "No related provisions",
  low_confidence: "Below confidence threshold",
  invalid_output: "Incomplete answer",
  model_not_confident: "Partial coverage",
  citation_check_failed: "Citation check failed",
};

export const OUTCOME_LABELS: Record<string, string> = {
  correct_answer: "Correct answer",
  correct_refusal: "Correct refusal",
  false_refusal: "False refusal",
  wrong_answer: "Wrong answer",
};

export const METRIC_LABELS: Record<string, string> = {
  "recall@1": "Recall@1",
  "recall@3": "Recall@3",
  "recall@6": "Recall@6",
  "recall@20": "Recall@20",
  "recall@40": "Recall@40",
  mrr: "MRR",
  "ndcg@6": "nDCG@6",
  context_recall: "Context recall",
  citation_precision: "Citation precision",
  citation_recall: "Citation recall",
  faithfulness: "Faithfulness",
  relevance: "Relevance",
  false_refusal_rate: "False refusal rate",
  refusal_recall: "Refusal recall",
  refusal_precision: "Refusal precision",
  validator_failure_rate: "Validator failures",
  contradicted_claims: "Contradicted claims",
  errors: "Errors",
  judge_errors: "Judge errors",
  items: "Reviewed items",
  pairs: "Calibration pairs",
  false_hits: "False cache hits",
  positive_hit_rate: "Paraphrase hit rate",
  threshold: "Similarity threshold",
  reviewed_only: "Reviewed items only",
  kind: "Run kind",
  status: "Run status",
};
export const metricLabel = (name: string) => {
  const vs = name.match(/^(.*) vs #(\d+)$/);
  if (vs) return `${METRIC_LABELS[vs[1]] ?? vs[1]} vs run #${vs[2]}`;
  return METRIC_LABELS[name] ?? name;
};

/** Metrics on a 0-1 scale (bars use it); relevance is a 1-5 judge score. */
export const isUnit = (name: string) =>
  /recall|precision|mrr|ndcg|faithfulness|rate/.test(name) && !name.startsWith("relevance");

/** Unit metrics shown as a percentage; MRR and nDCG read better as decimals. */
export const isRate = (name: string) => isUnit(name) && !/mrr|ndcg/.test(name);
