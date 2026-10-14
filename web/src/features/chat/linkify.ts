import type { Citation, MessageOut, TurnOut } from "../../api/types";

const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** Turn every mention of a cited provision in the answer into a markdown link
 * `[COB 3.4.2](cite:<index>)`, which the answer renderer shows as a chip. Only
 * citations the answer actually cites are linked; a longer citation wins over its
 * prefix ("COB 3.4.2 Guidance" over "COB 3.4.2"), and "COB 3.4.2" never matches
 * inside "COB 3.4.21". */
export function linkCitations(text: string, citations: Citation[]): string {
  if (!citations.length) return text;
  const order = citations
    .map((c, i) => ({ c, i }))
    .filter(({ c }) => c.citation.trim().length > 0)
    .sort((a, b) => b.c.citation.length - a.c.citation.length);
  if (!order.length) return text;
  const pattern = new RegExp(
    `(?<![\\w.\\[])(${order.map(({ c }) => escape(c.citation)).join("|")})(?![\\w]|\\.\\d)`,
    "g",
  );
  const index = new Map(order.map(({ c, i }) => [c.citation, i]));
  return text.replace(pattern, (m) => `[${m}](cite:${index.get(m)})`);
}

// --- a turn as the thread shows it, from a live answer or a stored message ------

export type TurnView = {
  key: string;
  question: string;
  rewritten: string | null; // set only when the rewrite changed the question
  state: "pending" | "answered" | "refused" | "error";
  text: string;
  citations: Citation[];
  guidanceOnly: boolean;
  refusalCode: string | null;
  refusalReason: string | null;
  traceId: string | null;
  traceUrl: string | null;
  costUsd: number | null;
  latency: Record<string, number> | null; // per stage (live turns only)
  latencyTotal: number | null;
  cached: boolean;
  sourceTraceId: string | null;
  error: string | null;
  startedAt: number;
};

export function fromTurn(turn: TurnOut, key: string, startedAt: number): TurnView {
  const changed = turn.rewritten_query.trim() !== turn.question.trim();
  return {
    key,
    question: turn.question,
    rewritten: changed ? turn.rewritten_query : null,
    state: turn.status,
    text: turn.text,
    citations: turn.citations,
    guidanceOnly: turn.guidance_only,
    refusalCode: turn.refusal?.code ?? null,
    refusalReason: turn.refusal?.reason ?? null,
    traceId: turn.trace_id,
    traceUrl: turn.trace_url,
    costUsd: turn.cost_usd,
    latency: turn.latency_ms,
    latencyTotal: turn.latency_ms.total ?? null,
    cached: turn.cached,
    sourceTraceId: turn.cache?.source_trace_id ?? null,
    error: null,
    startedAt,
  };
}

/** Stored messages come as user/assistant pairs; a user message without an
 * answer (a failed turn is never stored) is skipped. */
export function fromMessages(messages: MessageOut[]): TurnView[] {
  const turns: TurnView[] = [];
  for (let i = 0; i < messages.length - 1; i++) {
    const user = messages[i];
    const reply = messages[i + 1];
    if (user.role !== "user" || reply.role !== "assistant") continue;
    const rewritten =
      user.rewritten_query && user.rewritten_query.trim() !== user.content.trim()
        ? user.rewritten_query
        : null;
    turns.push({
      key: `m${reply.id}`,
      question: user.content,
      rewritten,
      state: reply.status === "refused" ? "refused" : "answered",
      text: reply.content,
      citations: reply.citations,
      guidanceOnly: reply.citations.length > 0 && reply.citations.every((c) => /non-binding/i.test(c.label)),
      refusalCode: reply.refusal_code,
      refusalReason: null,
      traceId: reply.trace_id,
      traceUrl: null,
      costUsd: reply.cost_usd,
      latency: null,
      latencyTotal: reply.latency_ms,
      cached: false,
      sourceTraceId: null,
      error: null,
      startedAt: 0,
    });
    i++;
  }
  return turns;
}
