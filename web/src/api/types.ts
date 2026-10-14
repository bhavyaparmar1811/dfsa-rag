// Mirrors src/api/schemas.py and the JSON the read endpoints return.

export type Citation = {
  citation: string; // "COB 3.3.2", "COB 3.3.2 Guidance"
  label: string; // "Rule (binding)"
  key: string;
  chunk_id: string;
};

export type Refusal = { code: string; reason: string };

export type CacheInfo = {
  kind: "exact" | "semantic";
  similarity: number | null;
  entry_id: string;
  source_trace_id: string;
};

export type Filters = {
  regulators?: string[] | null;
  modules?: string[] | null;
  doc_types?: ("rulebook" | "law")[] | null;
  chunk_types?: ("rule" | "guidance" | "definition" | "article" | "text")[] | null;
  binding_only?: boolean;
};

export type TurnOut = {
  conversation_id: string;
  message_id: number;
  status: "answered" | "refused";
  text: string;
  citations: Citation[];
  guidance_only: boolean;
  refusal: Refusal | null;
  question: string;
  rewritten_query: string;
  trace_id: string;
  trace_url: string | null;
  cost_usd: number;
  latency_ms: Record<string, number>;
  cached: boolean;
  cache: CacheInfo | null;
};

export type Conversation = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
};

export type MessageOut = {
  id: number;
  role: "user" | "assistant";
  content: string;
  rewritten_query: string | null;
  status: string | null;
  refusal_code: string | null;
  citations: Citation[];
  trace_id: string | null;
  cost_usd: number;
  latency_ms: number | null;
  created_at: string;
};

export type ConversationDetail = Conversation & { messages: MessageOut[] };

export type Chunk = {
  id: string;
  key: string;
  citation: string;
  chunk_type: string;
  binding: boolean;
  heading_path: string[];
  text: string;
  page_start: number;
  page_end: number;
  part: number | null;
  part_of: string | null;
  whole_text: string | null;
  regulator: string;
  module_code: string;
  module_name: string;
  doc_type: string;
  version: string;
  effective_date: string;
};

export type Candidate = {
  id: string;
  key: string;
  citation: string;
  source: string; // "search" | "citation"
  dense_rank: number | null;
  dense_score: number | null;
  sparse_rank: number | null;
  sparse_score: number | null;
  fused_score: number | null;
  rerank_score: number | null;
  truncated: boolean;
};

export type ContextBlock = {
  n: number;
  chunk_id: string;
  key: string;
  citation: string;
  chunk_type: string;
  binding: boolean;
  label: string;
  heading: string;
};

export type TraceRetrieval = {
  question: string;
  filters: Filters;
  config: Record<string, unknown> & { refusal_threshold?: number; reranker_model?: string };
  candidates: Candidate[];
  hits: string[];
  context: ContextBlock[];
  context_tokens: number;
  confidence: number | null;
  confident: boolean;
  timings_ms: Record<string, number>;
  truncated: number;
};

export type Trace = {
  id: string;
  conversation_id: string | null;
  created_at: string;
  status: string;
  error: string | null;
  original_query: string;
  rewritten_query: string | null;
  retrieval: TraceRetrieval | null;
  refused: boolean;
  refusal_code: string | null;
  confidence: number | null;
  total_cost_usd: number;
  latency_ms: Record<string, number> | null;
  cache_status: string | null;
  cache: Record<string, unknown> | null;
  langfuse_url: string | null;
};

export type HealthCheck = { ok: boolean; enabled?: boolean; error?: string } & Record<
  string,
  unknown
>;
export type Health = { status: "ok" | "degraded"; checks: Record<string, HealthCheck> };

export type CostSummary = {
  requests: number;
  refused: number;
  errors: number;
  refusal_rate: number | null;
  refusals_by_code: Record<string, number>;
  gate_refusals: number;
  cost_usd: { embedding: number; rewrite: number; answer: number; total: number };
  cost_by_provider_usd: { openai_embeddings: number; azure_generation: number };
  cost_per_request_usd: number | null;
  tokens: Record<string, number>;
  latency_ms: { p50: number | null; p95: number | null };
  cache: {
    lookups: number;
    hits: number;
    exact_hits: number;
    hit_rate: number | null;
    saved_usd: number;
    hit_latency_p50_ms: number | null;
  };
};

export type CostStats = CostSummary & {
  since: string | null;
  until: string | null;
  by_day: (CostSummary & { day: string })[];
};

// --- evals -------------------------------------------------------------------

export type EvalKind = "retrieval" | "e2e" | "cache";

export type Percentiles = { p50: number; p95: number };

export type KindMetrics = {
  turns: number;
  [metric: string]: number | null;
};

export type RunMetrics = {
  turns?: number;
  outcomes?: Record<string, number>;
  latency_ms?: Record<string, Percentiles>;
  by_kind?: Record<string, KindMetrics>;
  [metric: string]: unknown;
};

export type GateCheck = {
  name: string;
  op: string; // ">=" | "<=" | "=="
  value: number | string | null;
  limit: number | string | null;
  passed: boolean;
  note: string;
};

export type GateReport = {
  passed: boolean;
  baseline_run?: number | null;
  checks: GateCheck[];
};

export type EvalRunSummary = {
  id: number;
  kind: EvalKind;
  config_name: string;
  status: string;
  gate_status: "passed" | "failed" | "not_gated";
  created_at: string;
  finished_at: string | null;
  n_items: number;
  n_turns: number;
  reviewed_only: boolean;
  prompt_label: string | null;
  git_commit: string | null;
  cost_total_usd: number | null;
  headline: Record<string, number>;
};

export type PromptRef = {
  name: string;
  label: string | null;
  version: number | null;
  source: string;
  hash: string;
};

export type EvalRun = EvalRunSummary & {
  metrics: RunMetrics;
  gate_report: GateReport | null;
  retrieval_config: Record<string, unknown>;
  prompts: PromptRef[];
  cost_usd: Record<string, number>;
  golden_hash: string;
  langfuse_run: string | null;
  notes: string | null;
  error: string | null;
};

export type Ranked = {
  provision: string;
  key: string;
  source: string;
  hit: boolean;
  rerank: number | null;
  fused: number | null;
  dense_rank: number | null;
  sparse_rank: number | null;
};

export type JudgeClaim = { claim: string; verdict: string; context_ids?: number[] };

export type EvalResult = {
  id: number;
  item_id: string;
  turn: number;
  question: string;
  rewritten_query: string | null;
  expect: "answer" | "refuse";
  ranked: Ranked[];
  confidence: number | null;
  status: string | null;
  refusal_code: string | null;
  answer: string | null;
  cited: string[];
  metrics: Record<string, number | string | null>;
  judges: {
    faithfulness?: { score: number; contradicted: number; output?: { claims?: JudgeClaim[] } };
    relevance?: { score: number; output?: { reasoning?: string; score?: number } };
    [k: string]: unknown;
  } | null;
  cost_usd: number;
  latency_ms: Record<string, number>;
  trace_id: string | null;
  item_kind: string | null;
  module: string | null;
  gold: string[];
  supporting: string[];
};

export type Compare = {
  kind: EvalKind;
  golden_hash: string;
  golden_is_current: boolean;
  runs: (EvalRunSummary & { metrics: RunMetrics })[];
};

export type Thresholds = {
  config: string;
  min_reviewed_items: number;
  max_false_refusal: number;
  floors: Record<string, number | null>;
  ceilings: Record<string, number | null>;
  max_regression: number;
  cache: { min_pairs: number; max_false_hits: number; min_positive_hit_rate: number | null } | null;
};

export type CacheSweepRow = {
  threshold: number;
  false_hits: number;
  false_hit_ids: string[];
  positive_hits: number;
  positive_hit_rate: number;
};

export type CacheMetrics = {
  n_pairs: number;
  n_positive: number;
  n_negative: number;
  threshold: number | null;
  false_hits: number;
  positive_hit_rate: number;
  recommended_threshold: number | null;
  negatives_stopped_by_guards: number;
  positives_stopped_by_guards: string[];
  max_guard_passing_negative: { id: string; similarity: number } | null;
  positive_similarity: { min: number; median: number; max: number };
  negative_similarity: { min: number; median: number; max: number };
  naive: CacheSweepRow[];
  sweep: {
    threshold: number;
    guarded_false_hits: number;
    guarded_hit_rate: number;
    naive_false_hits: number;
    naive_hit_rate: number;
  }[];
  pairs: {
    id: string;
    kind: "positive" | "negative";
    similarity: number;
    exact: boolean;
    guards: string[];
  }[];
};
