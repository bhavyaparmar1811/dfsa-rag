"""Follow-up questions -> standalone questions, before they are embedded.

    "Do we need a Client Agreement with a Market Counterparty?"   (turn 1, as is)
    "what about retail clients?"  ->  "Do we need a Client Agreement with a Retail Client?"

The history comes from SQLite (the last CHAT_HISTORY_TURNS messages), never from
the client. The first message of a conversation is used as is, without an LLM
call. If the rewrite comes back empty or cut off, the original message is used
and the trace says why. The original and the rewritten question are both stored.
"""

from dataclasses import dataclass
from typing import Protocol

from src.generation.llm import LLMClient, LLMResult
from src.prompts.registry import PromptRegistry

REWRITE_PROMPT = "query-rewrite"
ROLES = {"user": "User", "assistant": "Assistant"}


class HistoryMessage(Protocol):
    role: str
    content: str


def render_history(messages: list[HistoryMessage], max_chars: int) -> str:
    """`User: ...` / `Assistant: ...` lines, each message cut to max_chars."""
    lines = []
    for m in messages:
        text = " ".join(m.content.split())
        if len(text) > max_chars:
            text = text[: max_chars - 1].rstrip() + "…"
        lines.append(f"{ROLES.get(m.role, m.role)}: {text}")
    return "\n".join(lines)


@dataclass
class Rewrite:
    original: str
    question: str  # what retrieval and the answer use
    changed: bool = False
    fallback_reason: str | None = None  # set when the rewrite was discarded
    prompt: dict | None = None  # ResolvedPrompt.trace()
    llm: LLMResult | None = None

    @property
    def cost_usd(self) -> float:
        return self.llm.cost_usd if self.llm else 0.0

    def trace(self) -> dict:
        return {
            "original": self.original,
            "question": self.question,
            "changed": self.changed,
            "fallback_reason": self.fallback_reason,
            "prompt": self.prompt,
            "output": self.llm.text if self.llm else None,
            "llm": self.llm.trace() if self.llm else None,
        }


class QueryRewriter:
    def __init__(self, llm: LLMClient, registry: PromptRegistry):
        self.llm = llm
        self.registry = registry

    def rewrite(self, message: str, history: str, label: str | None = None) -> Rewrite:
        if not history:
            return Rewrite(original=message, question=message)
        prompt = self.registry.get(REWRITE_PROMPT, label)
        llm = self.llm.complete(
            prompt, {"history": history, "message": message}, name="rewrite-query"
        )
        result = Rewrite(original=message, question=message, prompt=prompt.trace(), llm=llm)
        text = llm.text.strip().strip("\"'").strip()
        if llm.finish_reason != "stop":
            result.fallback_reason = f"finish_reason {llm.finish_reason}"
        elif not text:
            result.fallback_reason = "empty rewrite"
        else:
            result.question, result.changed = text, text != message
        return result
