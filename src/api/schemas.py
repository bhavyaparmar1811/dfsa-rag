"""Request and response bodies. Requests forbid unknown fields: in particular a
client can't send chat history; the server loads it from SQLite."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.config.settings import ChatSettings
from src.retrieval.models import Filters


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateConversation(Request):
    title: str | None = Field(None, max_length=200)


class FiltersIn(Request):
    regulators: list[str] | None = None
    modules: list[str] | None = None  # manifest module codes: ["AML"], ["COB", "REG_LAW"]
    doc_types: list[Literal["rulebook", "law"]] | None = None
    chunk_types: list[Literal["rule", "guidance", "definition", "article", "text"]] | None = None
    binding_only: bool = False

    def to_filters(self) -> Filters:
        return Filters(**self.model_dump())


class SendMessage(Request):
    content: str = Field(min_length=1, max_length=ChatSettings().max_message_chars)
    filters: FiltersIn | None = None


class ConversationOut(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class Citation(BaseModel):
    citation: str  # "COB 3.3.2", "COB 3.3.2 Guidance"
    label: str  # "Rule (binding)"
    key: str
    chunk_id: str


class Refusal(BaseModel):
    code: str
    reason: str


class CacheInfo(BaseModel):
    """Where a cached answer came from: the turn that first produced it is auditable
    through its trace."""

    kind: Literal["exact", "semantic"]
    similarity: float | None  # None for an exact repeat
    entry_id: str
    source_trace_id: str


class TurnOut(BaseModel):
    conversation_id: str
    message_id: int
    status: Literal["answered", "refused"]
    text: str
    citations: list[Citation]
    guidance_only: bool
    refusal: Refusal | None
    question: str
    rewritten_query: str
    trace_id: str
    trace_url: str | None
    cost_usd: float
    latency_ms: dict[str, float]
    cached: bool = False
    cache: CacheInfo | None = None


class MessageOut(BaseModel):
    id: int
    role: Literal["user", "assistant"]
    content: str
    rewritten_query: str | None
    status: str | None
    refusal_code: str | None
    citations: list[Citation]
    trace_id: str | None
    cost_usd: float
    latency_ms: float | None
    created_at: datetime


class ConversationDetail(ConversationOut):
    messages: list[MessageOut]
