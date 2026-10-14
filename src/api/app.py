"""The HTTP API over ChatService.

    POST /conversations                  start a conversation
    POST /conversations/{id}/messages    one turn: answer or refusal, with citations
    GET  /conversations/{id}             the conversation, every turn with its trace id
    GET  /traces/{id}                    the stored audit record of one turn
    GET  /stats/cost[?since=&until=]     requests, refusals, tokens, $ by stage and provider
    GET  /health                         database, Qdrant, reranker, cache, Langfuse (no key)

Every endpoint but /health needs `X-API-Key: <API_KEY>`. At startup: the database
must be at the Alembic head, the reranker is loaded once, the index is checked
against the embedding and BM25 settings, and the prompt drift is logged.

    uv run uvicorn src.api.app:app --host 127.0.0.1 --port 8000

Endpoints are plain `def`: retrieval and the LLM calls block, so FastAPI runs
them in its thread pool.
"""

import logging
import secrets
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import date
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from src.api.schemas import (
    CacheInfo,
    Citation,
    ConversationDetail,
    ConversationOut,
    CreateConversation,
    MessageOut,
    Refusal,
    SendMessage,
    TurnOut,
)
from src.chat.service import ChatService, ConversationNotFound, TurnFailed, trace_url
from src.config.settings import ApiSettings
from src.prompts.registry import report_drift
from src.storage.db import assert_migrated
from src.storage.models import Message

logger = logging.getLogger(__name__)


def get_service(request: Request) -> ChatService:
    return request.app.state.service


Service = Annotated[ChatService, Depends(get_service)]


def create_app(
    service_factory: Callable[[], ChatService] = ChatService.build,
    cfg: ApiSettings | None = None,
) -> FastAPI:
    cfg = cfg or ApiSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if cfg.key is None and not cfg.auth_disabled:
            raise RuntimeError("Set API_KEY in .env (or API_AUTH_DISABLED=true for local tests).")
        service = service_factory()
        assert_migrated(service.engine)
        try:
            report_drift(service.answerer.registry)
        except Exception:
            logger.exception("prompt drift check failed")
        app.state.service = service
        yield

    def require_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
        if cfg.auth_disabled:
            return
        if not x_api_key or not secrets.compare_digest(x_api_key, cfg.key.get_secret_value()):
            raise HTTPException(status_code=401, detail="missing or wrong X-API-Key")

    app = FastAPI(title="DFSA RAG", version="0.1.0", lifespan=lifespan)
    keyed = [Depends(require_key)]

    @app.exception_handler(ConversationNotFound)
    def _not_found(_: Request, e: ConversationNotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": f"no conversation {e.args[0]}"})

    @app.exception_handler(TurnFailed)
    def _failed(_: Request, e: TurnFailed) -> JSONResponse:
        # An outage upstream (Azure OpenAI, OpenAI, Qdrant) is worth a retry; anything
        # else is a bug. Either way the error trace is stored under trace_id.
        status, detail = (503, "a service is unavailable") if e.upstream else (500, "internal")
        return JSONResponse(status_code=status, content={"detail": detail, "trace_id": e.trace_id})

    @app.get("/health")
    def health(svc: Service) -> JSONResponse:
        checks = svc.health()
        # Langfuse is optional: answers still work without it.
        ok = all(c["ok"] for name, c in checks.items() if name != "langfuse")
        return JSONResponse(
            status_code=200 if ok else 503,
            content={"status": "ok" if ok else "degraded", "checks": checks},
        )

    @app.post("/conversations", status_code=201, dependencies=keyed)
    def create_conversation(
        svc: Service, body: CreateConversation | None = None
    ) -> ConversationOut:
        conv = svc.create_conversation(body.title if body else None)
        return ConversationOut.model_validate(conv, from_attributes=True)

    @app.post("/conversations/{conversation_id}/messages", dependencies=keyed)
    def send_message(conversation_id: str, body: SendMessage, svc: Service) -> TurnOut:
        filters = body.filters.to_filters() if body.filters else None
        turn = svc.send(conversation_id, body.content, filters)
        answer = turn.answer
        return TurnOut(
            conversation_id=conversation_id,
            message_id=turn.assistant_message_id,
            status=answer.status,
            text=answer.text,
            citations=[
                Citation(citation=c.citation, label=c.label, key=c.key, chunk_id=c.chunk_id)
                for c in answer.cited
            ],
            guidance_only=answer.guidance_only,
            refusal=Refusal(**asdict(answer.refusal)) if answer.refusal else None,
            question=body.content,
            rewritten_query=turn.rewrite.question,
            trace_id=turn.trace_id,
            trace_url=turn.trace_url,
            cost_usd=round(turn.cost_usd, 6),
            latency_ms=turn.latency_ms,
            cached=turn.cached,
            cache=_cache_info(turn.cache.hit) if turn.cached else None,
        )

    @app.get("/conversations/{conversation_id}", dependencies=keyed)
    def get_conversation(conversation_id: str, svc: Service) -> ConversationDetail:
        conv, messages = svc.get_conversation(conversation_id)
        return ConversationDetail(
            **ConversationOut.model_validate(conv, from_attributes=True).model_dump(),
            messages=[_message(m) for m in messages],
        )

    @app.get("/traces/{trace_id}", dependencies=keyed)
    def get_trace(trace_id: str, svc: Service) -> dict:
        trace = svc.get_trace(trace_id)
        if trace is None:
            raise HTTPException(status_code=404, detail=f"no trace {trace_id}")
        row = {c.name: getattr(trace, c.name) for c in trace.__table__.columns}
        return row | {"langfuse_url": trace_url(trace_id)}

    @app.get("/stats/cost", dependencies=keyed)
    def cost_stats(svc: Service, since: date | None = None, until: date | None = None) -> dict:
        """Production requests only (eval runs never write traces); UTC days, inclusive."""
        if since and until and since > until:
            raise HTTPException(status_code=422, detail="since is after until")
        return svc.cost_stats(since, until)

    return app


def _cache_info(hit) -> CacheInfo:
    return CacheInfo(
        kind=hit.kind,
        similarity=None if hit.similarity is None else round(hit.similarity, 4),
        entry_id=hit.entry_id,
        source_trace_id=hit.source_trace_id,
    )


def _message(m: Message) -> MessageOut:
    return MessageOut(
        id=m.id,
        role=m.role,
        content=m.content,
        rewritten_query=m.rewritten_query,
        status=m.status,
        refusal_code=m.refusal_code,
        citations=[Citation(**c) for c in m.cited or []],
        trace_id=m.trace_id,
        cost_usd=round(m.cost_usd, 6),
        latency_ms=m.latency_ms,
        created_at=m.created_at,
    )


app = create_app()
