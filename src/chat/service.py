"""ChatService: one conversation turn, end to end, stored and traced.

    send(conversation_id, message)
      history   the last CHAT_HISTORY_TURNS messages, from SQLite (never from the client)
      rewrite   a follow-up -> a standalone question (rewrite.py); none on the first turn
      retrieve  the rewritten question (one at a time: a single GPU reranker)
      answer    cited answer or refusal (src/generation/answer.py)
      store     user message + assistant message + trace row, in one transaction

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

import httpx
import openai
from langfuse import propagate_attributes
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from sqlalchemy import Engine

from src.chat.rewrite import QueryRewriter, Rewrite, render_history
from src.config.settings import (
    AzureOpenAISettings,
    ChatSettings,
    EmbeddingSettings,
    LangfuseSettings,
    RetrievalSettings,
)
from src.generation.answer import Answerer, AnswerResult
from src.generation.llm import LLMClient
from src.observability.langfuse import get_langfuse
from src.prompts.registry import PromptRegistry
from src.retrieval.models import Filters, RetrievalResult
from src.retrieval.pipeline import Retriever
from src.storage.db import assert_migrated, get_engine, session_scope
from src.storage.models import Conversation, Message, Trace
from src.storage.repositories import ConversationRepository, MessageRepository, TraceRepository

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
    retrieval: RetrievalResult
    answer: AnswerResult
    cost_usd: float
    latency_ms: dict[str, float]
    trace_url: str | None = None


class ChatService:
    def __init__(
        self,
        engine: Engine,
        retriever: Retriever,
        answerer: Answerer,
        rewriter: QueryRewriter,
        cfg: ChatSettings | None = None,
    ):
        self.engine = engine
        self.retriever = retriever
        self.answerer = answerer
        self.rewriter = rewriter
        self.cfg = cfg or ChatSettings()
        self._retrieve_lock = threading.Lock()

    @classmethod
    def build(cls) -> "ChatService":
        """The real components: loads the reranker and checks the index matches the
        configured embedding model and BM25 settings."""
        retriever = Retriever.build(RetrievalSettings())
        llm = LLMClient(AzureOpenAISettings())
        registry = PromptRegistry.from_settings()
        return cls(
            engine=get_engine(),
            retriever=retriever,
            answerer=Answerer(llm, registry, retriever.citations.find),
            rewriter=QueryRewriter(llm, registry),
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

    def get_trace(self, trace_id: str) -> Trace | None:
        with session_scope(self.engine) as s:
            return TraceRepository(s).get(trace_id)

    def cost_stats(self) -> dict:
        with session_scope(self.engine) as s:
            return TraceRepository(s).cost_stats()

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
        rewrite = retrieval = answer = None
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
                retrieval = self._retrieve(rewrite.question, filters)
                answer = self.answerer.answer(retrieval, history=history, label=label)
                root.update(
                    output={"status": answer.status, "text": answer.text},
                    metadata={"rewritten_query": rewrite.question, "trace_id": trace_id},
                )
        except Exception as e:
            latency = _latency(start, rewrite, retrieval, answer)
            self._store_error(
                trace_id, conversation_id, content, filters, rewrite, retrieval, e, latency
            )
            logger.exception("chat turn %s failed", trace_id)
            raise TurnFailed(trace_id, e) from e

        latency = _latency(start, rewrite, retrieval, answer)
        cost = retrieval.embed_cost_usd + rewrite.cost_usd + answer.cost_usd
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
            row = _trace_row(trace_id, conversation_id, content, filters, rewrite, retrieval)
            TraceRepository(s).add(Trace(**row, **_answer_fields(answer, cost, latency)))
            ids = (user.id, assistant.id)

        logger.info(
            "chat %s: %s%s, $%.6f, %.0f ms",
            conversation_id[:8],
            answer.status,
            " (rewritten)" if rewrite.changed else "",
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
        )

    def _retrieve(self, question: str, filters: Filters) -> RetrievalResult:
        langfuse, emb = get_langfuse(), EmbeddingSettings()
        with langfuse.start_as_current_observation(
            name="retrieve", as_type="retriever", input={"question": question}
        ) as obs:
            with self._retrieve_lock:
                retrieval = self.retriever.retrieve(question, filters)
            with langfuse.start_as_current_observation(
                name="embed-query",
                as_type="embedding",
                model=emb.model,
                usage_details={"input": retrieval.embed_tokens},
                cost_details={"input": retrieval.embed_cost_usd},
            ):
                pass
            obs.update(
                output={
                    "hits": [h.key for h in retrieval.hits],
                    "context": [b.key for b in retrieval.context],
                },
                metadata={"confidence": retrieval.confidence, "confident": retrieval.confident},
            )
        return retrieval

    def _store_error(
        self, trace_id, conversation_id, content, filters, rewrite, retrieval, error, latency
    ) -> None:
        row = _trace_row(trace_id, conversation_id, content, filters, rewrite, retrieval)
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


def _latency(start, rewrite, retrieval, answer) -> dict[str, float]:
    out = {"total": round((time.perf_counter() - start) * 1000, 1)}
    if rewrite is not None and rewrite.llm is not None:
        out["rewrite"] = round(rewrite.llm.latency_ms, 1)
    if retrieval is not None:
        out["retrieve"] = round(retrieval.timings_ms.get("total", 0.0), 1)
    if answer is not None:
        out["answer"] = round(answer.timings_ms.get("total", 0.0), 1)
    return out


def _trace_row(trace_id, conversation_id, content, filters, rewrite, retrieval) -> dict:
    llm = rewrite.llm if rewrite else None
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
        "embed_tokens": retrieval.embed_tokens if retrieval else 0,
        "embed_cost_usd": retrieval.embed_cost_usd if retrieval else 0.0,
        "rewrite_input_tokens": llm.input_tokens if llm else 0,
        "rewrite_output_tokens": llm.output_tokens if llm else 0,
        "rewrite_cost_usd": llm.cost_usd if llm else 0.0,
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
