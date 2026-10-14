import { ArrowRight } from "@phosphor-icons/react";
import { motion } from "motion/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ApiError } from "../../api/client";
import { useConversation, useHealth, useSendMessage } from "../../api/hooks";
import { ErrorNote, Skeleton, Tag } from "../../components/ui";
import { int } from "../../lib/format";
import { DUR, EASE, cascade } from "../../theme/motion";
import { Composer, type FilterState, toFilters } from "./Composer";
import { Inspector, type InspectorState } from "./Inspector";
import { type TurnView, fromMessages, fromTurn } from "./linkify";
import { TurnCard } from "./TurnCard";

// Real questions from the golden eval set: three it answers, one it must refuse.
const EXAMPLES = [
  {
    q: "Do we need to assess whether an investment is suitable for a client before we recommend it to them?",
    tag: "COB",
  },
  { q: "When do we have to carry out customer due diligence on a new customer?", tag: "AML" },
  { q: "How does the Markets Law define inside information?", tag: "DIFC Law" },
  { q: "What does VARA require to license a virtual asset broker-dealer in Dubai?", tag: "Out of scope" },
];

const pendingTurn = (question: string, key: string): TurnView => ({
  key,
  question,
  rewritten: null,
  state: "pending",
  text: "",
  citations: [],
  guidanceOnly: false,
  refusalCode: null,
  refusalReason: null,
  traceId: null,
  traceUrl: null,
  costUsd: null,
  latency: null,
  latencyTotal: null,
  cached: false,
  sourceTraceId: null,
  error: null,
  startedAt: Date.now(),
});

export function ChatPage({ chatKey, conversationId }: { chatKey: string; conversationId?: string }) {
  const navigate = useNavigate();
  const [convId, setConvId] = useState(conversationId);
  const [turns, setTurns] = useState<TurnView[]>([]);
  const [filters, setFilters] = useState<FilterState>({ scopes: [], bindingOnly: false });
  const [inspector, setInspector] = useState<InspectorState>(null);
  const send = useSendMessage();
  const endRef = useRef<HTMLDivElement>(null);

  // An existing conversation: load its stored turns once.
  const history = useConversation(conversationId && turns.length === 0 ? conversationId : undefined);
  const [loaded, setLoaded] = useState(!conversationId);
  useEffect(() => {
    if (history.data && !loaded) {
      setTurns(fromMessages(history.data.messages));
      setLoaded(true);
    }
  }, [history.data, loaded]);

  const scrollToEnd = useCallback(() => {
    requestAnimationFrame(() => endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }));
  }, []);

  const ask = useCallback(
    (question: string, replaceKey?: string) => {
      const key = replaceKey ?? `t${Date.now()}`;
      const pending = pendingTurn(question, key);
      setTurns((ts) => (replaceKey ? ts.map((t) => (t.key === key ? pending : t)) : [...ts, pending]));
      scrollToEnd();
      send.mutate(
        { conversationId: convId, content: question, filters: toFilters(filters) },
        {
          onSuccess: (turn) => {
            setTurns((ts) => ts.map((t) => (t.key === key ? fromTurn(turn, key, pending.startedAt) : t)));
            if (!convId) {
              setConvId(turn.conversation_id);
              navigate(`/c/${turn.conversation_id}`, { replace: true, state: { chatKey } });
            }
          },
          onError: (err) => {
            const e = err instanceof ApiError ? err : null;
            setTurns((ts) =>
              ts.map((t) =>
                t.key === key
                  ? {
                      ...t,
                      state: "error",
                      error:
                        e?.status === 503
                          ? "A service the assistant depends on (Azure OpenAI, OpenAI or Qdrant) didn’t respond. Nothing was stored; you can try again."
                          : (e?.detail ?? String(err)),
                      traceId: e?.traceId ?? null,
                    }
                  : t,
              ),
            );
          },
        },
      );
    },
    [convId, filters, chatKey, navigate, send, scrollToEnd],
  );

  const open = {
    chunk: (chunkId: string, traceId: string | null) => setInspector({ tab: "rule", chunkId, traceId }),
    retrieval: (traceId: string) => setInspector({ tab: "retrieval", chunkId: null, traceId }),
  };

  const empty = loaded && turns.length === 0;
  const busy = send.isPending;

  return (
    <div className="flex min-h-full flex-col">
      {empty ? (
        <Welcome onAsk={(q) => ask(q)} filters={filters} onFilters={setFilters} busy={busy} />
      ) : (
        <>
          <div className="mx-auto w-full max-w-3xl flex-1 px-6 pt-10 lg:px-8">
            {!loaded && <ThreadSkeleton />}
            {history.error && <ErrorNote error={history.error} title="Couldn't load this conversation" />}
            <div className="divide-y divide-line">
              {turns.map((t) => (
                <TurnCard key={t.key} turn={t} open={open} onRetry={(turn) => ask(turn.question, turn.key)} />
              ))}
            </div>
            <div ref={endRef} className="h-6" />
          </div>
          <div className="sticky bottom-0 z-10 bg-canvas/95 pb-6 pt-3 backdrop-blur-[2px]">
            <div className="mx-auto w-full max-w-3xl px-6 lg:px-8">
              <Composer
                onSend={(q) => ask(q)}
                busy={busy}
                filters={filters}
                onFilters={setFilters}
                autoFocus
              />
              <p className="mt-2 text-center text-[11.5px] text-ink-4">
                Answers cite DFSA provisions and are not legal advice. Check the cited text before
                relying on it.
              </p>
            </div>
          </div>
        </>
      )}
      <Inspector state={inspector} onChange={setInspector} />
    </div>
  );
}

function Welcome({
  onAsk,
  filters,
  onFilters,
  busy,
}: {
  onAsk: (q: string) => void;
  filters: FilterState;
  onFilters: (f: FilterState) => void;
  busy: boolean;
}) {
  const { data: health } = useHealth();
  const points = health?.checks?.qdrant?.points as number | undefined;
  return (
    <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col justify-center px-6 py-16 lg:px-8">
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: DUR.slow, ease: EASE }}
      >
        <div className="eyebrow mb-4">
          DFSA Rulebook · DIFC Laws{points ? ` · ${int(points)} provisions indexed` : ""}
        </div>
        <h1 className="font-serif text-[44px] font-normal leading-[1.08] tracking-[-0.03em] text-ink">
          Ask the rulebook.
          <br />
          <span className="text-ink-3">Get the rule, not a paraphrase.</span>
        </h1>
        <p className="mt-5 max-w-[56ch] text-[15px] leading-relaxed text-ink-3">
          Every answer cites the exact provision it rests on, such as COB 3.4.2. When the rulebook
          doesn’t clearly cover a question, the assistant says so instead of guessing.
        </p>
      </motion.div>

      <motion.div
        className="mt-10"
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: DUR.slow, ease: EASE, delay: 0.08 }}
      >
        <Composer onSend={onAsk} busy={busy} filters={filters} onFilters={onFilters} autoFocus />
      </motion.div>

      <div className="mt-8 grid gap-3 sm:grid-cols-2">
        {EXAMPLES.map((e, i) => (
          <motion.button
            key={e.q}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: DUR.slow, ease: EASE, delay: 0.16 + cascade(i, 0.06) }}
            onClick={() => onAsk(e.q)}
            disabled={busy}
            className="group flex flex-col justify-between gap-4 rounded-[10px] border border-line bg-surface p-4 text-left transition-[border-color,box-shadow,transform] duration-200 hover:border-line-strong hover:shadow-hover active:scale-[0.99]"
          >
            <span className="text-[13.5px] leading-snug text-ink-2 group-hover:text-ink">{e.q}</span>
            <span className="flex items-center justify-between">
              <Tag tone={e.tag === "Out of scope" ? "yellow" : "gray"}>{e.tag}</Tag>
              <ArrowRight
                size={14}
                weight="bold"
                className="text-ink-4 transition-transform duration-200 group-hover:translate-x-0.5 group-hover:text-ink-2"
              />
            </span>
          </motion.button>
        ))}
      </div>
    </div>
  );
}

function ThreadSkeleton() {
  return (
    <div className="space-y-4 py-2">
      <Skeleton className="h-6 w-3/4" />
      <Skeleton className="h-4 w-full" />
      <Skeleton className="h-4 w-5/6" />
      <Skeleton className="h-4 w-2/3" />
    </div>
  );
}
