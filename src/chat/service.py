"""ChatService: one conversation turn, end to end, stored and traced.

    send(conversation_id, message)
      history   the last CHAT_HISTORY_TURNS messages, from SQLite (never from the client)
      rewrite   a follow-up -> a standalone question (rewrite.py); none on the first turn
      cache     the standalone question in the semantic answer cache (cache.py): a hit
                skips retrieval and the answer step
      retrieve  the rewritten question (one at a time: a single GPU reranker)
      answer    cited answer or refusal (src/generation/answer.py)
      store     user message + assistant message + trace row, in one transaction;
                then an answered first turn goes into the cache

One Langfuse trace per turn, grouped by conversation (session). Its id is
created up front, so the SQLite trace row and Langfuse share it even when
tracing is off. A turn that fails stores an error trace and no messages, so the
client can simply retry.
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import date

import httpx
import openai
from langfuse import propagate_attributes
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from sqlalchemy import Engine

from src.chat.cache import CacheLookup, SemanticCache
from src.chat.rewrite import QueryRewriter, Rewrite, render_history
from src.config.settings import (
    AzureOpenAISettings,
    CacheSettings,
    ChatSettings,
    EmbeddingSettings,
    LangfuseSettings,
    RetrievalSettings,
)
from src.embeddings.client import EmbeddingResult
from src.generation.answer import Answerer, AnswerResult
from src.generation.llm import LLMClient
from src.observability.langfuse import get_langfuse
from src.prompts.registry import PromptRegistry
from src.retrieval.models import Filters, RetrievalResult
from src.retrieval.pipeline import Retriever
from src.storage.db import assert_migrated, get_engine, session_scope
from src.storage.models import ChunkRow, Conversation, Message, Trace
from src.storage.repositories import (
    ChunkRepository,
    ConversationRepository,
    MessageRepository,
    TraceRepository,
)

logger = logging.getLogger(__name__)

TITLE_CHARS = 80
# Failures of a service we call (Azure OpenAI, OpenAI, Qdrant): worth a retry (HTTP 503).
UPSTREAM_ERRORS = (openai.APIError, httpx.HTTPError, UnexpectedResponse, ResponseHandlingException)


class ConversationNotFound(KeyError):
    pass


class TurnFailed(Exception):
    """A turn failed after its trace id was created; the error trace is stored."""

    def __init__(self, trace_id: str, cause: Exception):
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.trace_id = trace_id
        self.upstream = isinstance(cause, UPSTREAM_ERRORS)


@dataclass
class TurnResult:
    conversation_id: str
    user_message_id: int
    assistant_message_id: int
    trace_id: str
    rewrite: Rewrite
    retrieval: RetrievalResult | None  # None when the answer came from the cache
    answer: AnswerResult
    cost_usd: float
    latency_ms: dict[str, float]
    trace_url: str | None = None
    cache: CacheLookup | None = None
    cache_entry_id: str | None = None  # the entry served, or the one this answer was stored as

    @property
    def cached(self) -> bool:
        return self.cache is not None and self.cache.hit is not None


class ChatService:
    def __init__(
        self,
        engine: Engine,
        retriever: Retriever,
        answerer: Answerer,
        rewriter: QueryRewriter,
        cfg: ChatSettings | None = None,
        cache: SemanticCache | None = None,
    ):
        self.engine = engine
        self.retriever = retriever
        self.answerer = answerer
        self.rewriter = rewriter
        self.cfg = cfg or ChatSettings()
        self.cache = cache
        self._retrieve_lock = threading.Lock()

    @classmethod
    def build(cls) -> "ChatService":
        """The real components: loads the reranker and checks the index matches the
        configured embedding model and BM25 settings."""
        retriever = Retriever.build(RetrievalSettings())
        azure = AzureOpenAISettings()
        llm = LLMClient(azure)
        registry = PromptRegistry.from_settings()
        engine = get_engine()
        cache = None
        if CacheSettings().enabled:
            cache = SemanticCache.build(engine, retriever, registry, azure.deployment)
        return cls(
            engine=engine,
            retriever=retriever,
            answerer=Answerer(llm, registry, retriever.citations.find),
            rewriter=QueryRewriter(llm, registry),
            cache=cache,
        )

    # --- conversations ----------------------------------------------------------

    def create_conversation(self, title: str | None = None) -> Conversation:
        with session_scope(self.engine) as s:
            return ConversationRepository(s).create(str(uuid.uuid4()), title or "")

    def get_conversation(self, conversation_id: str) -> tuple[Conversation, list[Message]]:
        with session_scope(self.engine) as s:
            repo = ConversationRepository(s)
            conv = repo.get(conversation_id)
            if conv is None:
                raise ConversationNotFound(conversation_id)
            return conv, repo.messages(conversation_id)

    def list_conversations(self, limit: int = 30) -> list[Conversation]:
        with session_scope(self.engine) as s:
            return ConversationRepository(s).recent(limit)

    def get_chunk(self, chunk_id: str) -> tuple[ChunkRow, ChunkRow | None] | None:
        """A chunk with its document, and the whole rule when it is one part of one."""
        with session_scope(self.engine) as s:
            repo = ChunkRepository(s)
            chunk = repo.get(chunk_id)
            if chunk is None:
                return None
            return chunk, repo.by_key(chunk.part_of) if chunk.part_of else None

    def get_trace(self, trace_id: str) -> Trace | None:
        with session_scope(self.engine) as s:
            return TraceRepository(s).get(trace_id)

    def cost_stats(self, since: date | None = None, until: date | None = None) -> dict:
        with session_scope(self.engine) as s:
            return TraceRepository(s).cost_stats(since, until)

    def health(self) -> dict[str, dict]:
        """Each dependency: {"ok": bool, ...details}. Cheap enough to poll."""
        checks: dict[str, dict] = {}
        try:
            assert_migrated(self.engine)
            checks["database"] = {"ok": True}
        except Exception as e:
            checks["database"] = {"ok": False, "error": str(e)}
        try:
            store = self.retriever.searcher.store
            checks["qdrant"] = {"ok": True, "collection": store.collection, "points": store.count()}
        except Exception as e:
            checks["qdrant"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        reranker = getattr(self.retriever.reranker, "model", None)
        checks["reranker"] = {
            "ok": True,
            "enabled": reranker is not None,
            "device": getattr(reranker, "device", None),
            "revision": getattr(reranker, "revision", None),
        }
        if self.cache is None:
            checks["cache"] = {"ok": True, "enabled": False}
        else:
            try:
                checks["cache"] = {"ok": True} | self.cache.stats()
            except Exception as e:
                checks["cache"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        cfg = LangfuseSettings()
        if not cfg.active:
            checks["langfuse"] = {"ok": True, "enabled": False}
        else:
            try:
                status = httpx.get(f"{cfg.host}/api/public/health", timeout=2).json()
                checks["langfuse"] = {
                    "ok": status.get("status") == "OK",
                    "enabled": True,
                    "version": status.get("version"),
                }
            except (httpx.HTTPError, ValueError) as e:
                # Not fatal: tracing is lost and prompts fall back to git, answers still work.
                checks["langfuse"] = {"ok": False, "enabled": True, "error": str(e)}
        return checks

    # --- a turn -----------------------------------------------------------------

    def send(
        self,
        conversation_id: str,
        content: str,
        filters: Filters | None = None,
        *,
        label: str | None = None,
    ) -> TurnResult:
        """`label` picks the prompt versions (default production; evals use staging)."""
        filters = filters or Filters()
        with session_scope(self.engine) as s:
            repo = ConversationRepository(s)
            if repo.get(conversation_id) is None:
                raise ConversationNotFound(conversation_id)
            history = render_history(
                repo.recent_messages(conversation_id, self.cfg.history_turns),
                self.cfg.history_max_chars,
            )

        langfuse = get_langfuse()
        trace_id = langfuse.create_trace_id()
        start = time.perf_counter()
        rewrite = lookup = retrieval = answer = None
        try:
            with (
                langfuse.start_as_current_observation(
                    name="chat-turn",
                    as_type="chain",
                    trace_context={"trace_id": trace_id},
                    input={"message": content},
                ) as root,
                propagate_attributes(session_id=conversation_id, trace_name="chat-turn"),
            ):
                rewrite = self.rewriter.rewrite(content, history, label)
                lookup = self._lookup(rewrite.question, filters, label)
                if lookup is not None and lookup.hit is not None:
                    answer = lookup.hit.answer(rewrite.question)
                else:
                    embedding = lookup.embedding if lookup is not None else None
                    retrieval = self._retrieve(rewrite.question, filters, embedding)
                    answer = self.answerer.answer(retrieval, history=history, label=label)
                root.update(
                    output={"status": answer.status, "text": answer.text},
                    metadata={
                        "rewritten_query": rewrite.question,
                        "trace_id": trace_id,
                        "cache": lookup.status if lookup else None,
                    },
                )
        except Exception as e:
            latency = _latency(start, rewrite, retrieval, answer, lookup)
            self._store_error(
                trace_id, conversation_id, content, filters, rewrite, retrieval, lookup, e, latency
            )
            logger.exception("chat turn %s failed", trace_id)
            raise TurnFailed(trace_id, e) from e

        latency = _latency(start, rewrite, retrieval, answer, lookup)
        # The query embedding counts once: retrieval's (it reuses the lookup's vector),
        # or the lookup's alone on a hit.
        embedder = retrieval if retrieval is not None else lookup
        embed_cost = embedder.embed_cost_usd if embedder is not None else 0.0
        cost = embed_cost + rewrite.cost_usd + answer.cost_usd
        with session_scope(self.engine) as s:
            conv = ConversationRepository(s).get(conversation_id)
            ConversationRepository(s).touch(conv, title=_title(content))
            messages = MessageRepository(s)
            user = messages.add(
                Message(
                    conversation_id=conversation_id,
                    role="user",
                    content=content,
                    rewritten_query=rewrite.question,
                    trace_id=trace_id,
                )
            )
            tokens_in = sum(r.input_tokens for r in (rewrite.llm, answer.llm) if r)
            tokens_out = sum(r.output_tokens for r in (rewrite.llm, answer.llm) if r)
            assistant = messages.add(
                Message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=answer.text,
                    status=answer.status,
                    refusal_code=answer.refusal.code if answer.refusal else None,
                    cited=[
                        {
                            "chunk_id": c.chunk_id,
                            "key": c.key,
                            "citation": c.citation,
                            "label": c.label,
                        }
                        for c in answer.cited
                    ],
                    trace_id=trace_id,
                    input_tokens=tokens_in,
                    output_tokens=tokens_out,
                    cost_usd=cost,
                    latency_ms=latency["total"],
                )
            )
            row = _trace_row(
                trace_id, conversation_id, content, filters, rewrite, retrieval, lookup
            )
            TraceRepository(s).add(Trace(**row, **_answer_fields(answer, cost, latency)))
            ids = (user.id, assistant.id)

        entry_id = lookup.hit.entry_id if lookup and lookup.hit else None
        if self.cache is not None and lookup is not None and lookup.hit is None:
            entry_id = self.cache.store(
                lookup,
                question=content,
                query=rewrite.question,
                answer=answer,
                trace_id=trace_id,
                cost_usd=cost,
                first_turn=not history,
            )
            if entry_id is not None:
                with session_scope(self.engine) as s:
                    TraceRepository(s).set_cache_entry(trace_id, entry_id)

        logger.info(
            "chat %s: %s%s%s, $%.6f, %.0f ms",
            conversation_id[:8],
            answer.status,
            " (rewritten)" if rewrite.changed else "",
            f" (cache {lookup.hit.kind} hit)" if lookup and lookup.hit else "",
            cost,
            latency["total"],
        )
        return TurnResult(
            conversation_id=conversation_id,
            user_message_id=ids[0],
            assistant_message_id=ids[1],
            trace_id=trace_id,
            rewrite=rewrite,
            retrieval=retrieval,
            answer=answer,
            cost_usd=cost,
            latency_ms=latency,
            trace_url=trace_url(trace_id),
            cache=lookup,
            cache_entry_id=entry_id,
        )

    def _lookup(self, question: str, filters: Filters, label: str | None) -> CacheLookup | None:
        if self.cache is None:
            return None
        with get_langfuse().start_as_current_observation(
            name="semantic-cache", as_type="retriever", input={"question": question}
        ) as obs:
            lookup = self.cache.lookup(question, filters, label)
            if lookup.hit is not None and lookup.embedding is not None:
                # No retrieval follows, so this is the turn's query embedding.
                _embed_observation(lookup.embed_tokens, lookup.embed_cost_usd)
            obs.update(
                output={"status": lookup.status, "entry_id": lookup.hit and lookup.hit.entry_id},
                metadata={k: v for k, v in lookup.trace().items() if k != "components"},
            )
        return lookup

    def _retrieve(
        self, question: str, filters: Filters, embedding: EmbeddingResult | None = None
    ) -> RetrievalResult:
        """`embedding`: the question's vector from the cache lookup, reused."""
        langfuse = get_langfuse()
        with langfuse.start_as_current_observation(
            name="retrieve", as_type="retriever", input={"question": question}
        ) as obs:
            with self._retrieve_lock:
                retrieval = self.retriever.retrieve(question, filters, embedding)
            _embed_observation(retrieval.embed_tokens, retrieval.embed_cost_usd)
            obs.update(
                output={
                    "hits": [h.key for h in retrieval.hits],
                    "context": [b.key for b in retrieval.context],
                },
                metadata={"confidence": retrieval.confidence, "confident": retrieval.confident},
            )
        return retrieval

    def _store_error(
        self, trace_id, conversation_id, content, filters, rewrite, retrieval, lookup, error,
        latency,
    ) -> None:  # fmt: skip
        row = _trace_row(trace_id, conversation_id, content, filters, rewrite, retrieval, lookup)
        row |= {
            "status": "error",
            "error": f"{type(error).__name__}: {error}"[:2000],
            "latency_ms": latency,
            "total_cost_usd": row["embed_cost_usd"] + row["rewrite_cost_usd"],
        }
        try:
            with session_scope(self.engine) as s:
                TraceRepository(s).add(Trace(**row))
        except Exception:
            logger.exception("could not store the error trace %s", trace_id)


def _embed_observation(tokens: int, cost_usd: float) -> None:
    with get_langfuse().start_as_current_observation(
        name="embed-query",
        as_type="embedding",
        model=EmbeddingSettings().model,
        usage_details={"input": tokens},
        cost_details={"input": cost_usd},
    ):
        pass


def trace_url(trace_id: str) -> str | None:
    """The trace in the Langfuse UI; None when Langfuse is off (the lookup needs the server)."""
    if not LangfuseSettings().active:
        return None
    try:
        return get_langfuse().get_trace_url(trace_id=trace_id)
    except Exception:
        return None


def _title(content: str) -> str:
    text = " ".join(content.split())
    return text if len(text) <= TITLE_CHARS else text[: TITLE_CHARS - 1].rstrip() + "…"


def _latency(start, rewrite, retrieval, answer, lookup) -> dict[str, float]:
    out = {"total": round((time.perf_counter() - start) * 1000, 1)}
    if rewrite is not None and rewrite.llm is not None:
        out["rewrite"] = round(rewrite.llm.latency_ms, 1)
    if lookup is not None and lookup.status != "bypass":
        out["cache"] = round(lookup.latency_ms, 1)
    if retrieval is not None:
        out["retrieve"] = round(retrieval.timings_ms.get("total", 0.0), 1)
    if answer is not None and answer.timings_ms:  # a cached answer has no answer step
        out["answer"] = round(answer.timings_ms.get("total", 0.0), 1)
    return out


def _trace_row(trace_id, conversation_id, content, filters, rewrite, retrieval, lookup) -> dict:
    llm = rewrite.llm if rewrite else None
    hit = lookup.hit if lookup else None
    # The query embedding: retrieval's (it reuses the lookup's vector), else the lookup's.
    embedder = retrieval if retrieval is not None else lookup
    return {
        "id": trace_id,
        "conversation_id": conversation_id,
        "status": "ok",
        "original_query": content,
        "rewritten_query": rewrite.question if rewrite else None,
        "filters": filters.model_dump(exclude_defaults=True),
        "rewrite": rewrite.trace() if rewrite else None,
        "retrieval": retrieval.trace() if retrieval else None,
        "confidence": retrieval.confidence if retrieval else None,
        "embed_tokens": embedder.embed_tokens if embedder else 0,
        "embed_cost_usd": embedder.embed_cost_usd if embedder else 0.0,
        "rewrite_input_tokens": llm.input_tokens if llm else 0,
        "rewrite_output_tokens": llm.output_tokens if llm else 0,
        "rewrite_cost_usd": llm.cost_usd if llm else 0.0,
        "cache_status": lookup.status if lookup else None,
        "cache_entry_id": hit.entry_id if hit else None,
        "cache_similarity": hit.similarity if hit else None,
        "cache": lookup.trace() if lookup else None,
    }


def _answer_fields(answer: AnswerResult, cost: float, latency: dict) -> dict:
    llm = answer.llm
    return {
        "answer": answer.trace(),
        "refused": answer.refused,
        "refusal_code": answer.refusal.code if answer.refusal else None,
        "answer_input_tokens": llm.input_tokens if llm else 0,
        "answer_output_tokens": llm.output_tokens if llm else 0,
        "answer_cost_usd": answer.cost_usd,
        "total_cost_usd": cost,
        "latency_ms": latency,
    }
