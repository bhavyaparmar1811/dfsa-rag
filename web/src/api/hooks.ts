import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./client";
import type {
  Chunk,
  Compare,
  Conversation,
  ConversationDetail,
  CostStats,
  EvalKind,
  EvalResult,
  EvalRun,
  EvalRunSummary,
  Filters,
  Health,
  Thresholds,
  Trace,
  TurnOut,
} from "./types";

// Eval runs, chunks and traces never change once written: cache them for the session.
const FOREVER = { staleTime: Infinity, gcTime: Infinity } as const;

export const useHealth = () =>
  useQuery({
    queryKey: ["health"],
    // /health answers 503 with the same body when degraded: read it either way.
    queryFn: async () => {
      const res = await fetch("/api/health").catch(() => null);
      if (!res) return null;
      return ((await res.json().catch(() => null)) as Health | null) ?? null;
    },
    refetchInterval: 30_000,
  });

export const useConversations = () =>
  useQuery({
    queryKey: ["conversations"],
    queryFn: () => api.get<Conversation[]>("/conversations?limit=40"),
  });

export const useConversation = (id: string | undefined) =>
  useQuery({
    queryKey: ["conversation", id],
    queryFn: () => api.get<ConversationDetail>(`/conversations/${id}`),
    enabled: !!id,
  });

export const useChunk = (id: string | null) =>
  useQuery({
    queryKey: ["chunk", id],
    queryFn: () => api.get<Chunk>(`/chunks/${id}`),
    enabled: !!id,
    ...FOREVER,
  });

export const useTrace = (id: string | null) =>
  useQuery({
    queryKey: ["trace", id],
    queryFn: () => api.get<Trace>(`/traces/${id}`),
    enabled: !!id,
    ...FOREVER,
  });

export const useCostStats = (since: string | null, until: string | null) =>
  useQuery({
    queryKey: ["cost", since, until],
    queryFn: () => {
      const q = new URLSearchParams();
      if (since) q.set("since", since);
      if (until) q.set("until", until);
      const qs = q.toString();
      return api.get<CostStats>(`/stats/cost${qs ? `?${qs}` : ""}`);
    },
    placeholderData: keepPreviousData,
  });

export const useEvalRuns = (kind?: EvalKind) =>
  useQuery({
    queryKey: ["eval-runs", kind ?? "all"],
    queryFn: () => api.get<EvalRunSummary[]>(`/eval/runs${kind ? `?kind=${kind}` : ""}`),
  });

export const useEvalRun = (id: number | null | undefined) =>
  useQuery({
    queryKey: ["eval-run", id],
    queryFn: () => api.get<EvalRun>(`/eval/runs/${id}`),
    enabled: id != null,
    ...FOREVER,
  });

export const useEvalResults = (id: number | null | undefined) =>
  useQuery({
    queryKey: ["eval-results", id],
    queryFn: () => api.get<EvalResult[]>(`/eval/runs/${id}/results`),
    enabled: id != null,
    ...FOREVER,
  });

export const useCompare = () =>
  useQuery({
    queryKey: ["eval-compare"],
    queryFn: () => api.get<Compare>("/eval/compare?kind=retrieval"),
  });

export const useThresholds = () =>
  useQuery({
    queryKey: ["eval-thresholds"],
    queryFn: () => api.get<Thresholds>("/eval/thresholds"),
    staleTime: Infinity,
  });

/** Start a conversation if there is none yet, then send one message. */
export function useSendMessage() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: async (vars: { conversationId?: string; content: string; filters?: Filters }) => {
      const id = vars.conversationId ?? (await api.post<Conversation>("/conversations", {})).id;
      return api.post<TurnOut>(`/conversations/${id}/messages`, {
        content: vars.content,
        filters: vars.filters,
      });
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["conversations"] });
      qc.invalidateQueries({ queryKey: ["cost"] });
    },
  });
}
