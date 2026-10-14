"""A scripted Azure OpenAI client, context blocks, and an Answerer wired to the
prompts in git (Langfuse is off in tests), so generation tests run offline."""

import json
from types import SimpleNamespace

import pytest

from src.config.settings import AzureOpenAISettings, LangfuseSettings, PathSettings
from src.generation.answer import Answerer
from src.generation.llm import LLMClient
from src.prompts.files import load_prompts
from src.prompts.registry import PromptRegistry
from src.retrieval.citations import CitationRouter
from src.retrieval.models import CitationLookup, ContextBlock, Filters, RetrievalResult

# The mini corpus (SQLite + in-memory Qdrant), for the end-to-end test.
from tests.retrieval.conftest import corpus, index, loaded  # noqa: F401

CODES = {"GEN", "COB", "AML", "MKT", "FPR", "GLO"}
LAWS = ["Regulatory Law", "Markets Law"]
LABELS = {
    ("rule", True): "Rule (binding)",
    ("guidance", False): "Guidance (non-binding)",
    ("definition", True): "Definition (binding)",
    ("article", True): "Law article (binding)",
}


class FakeAzure:
    """Answers chat.completions.create with the next scripted reply."""

    def __init__(self, *replies: str | dict | Exception, finish_reason: str = "stop"):
        self.replies = list(replies)
        self.finish_reason = finish_reason
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        content = reply if isinstance(reply, str) else json.dumps(reply)
        return SimpleNamespace(
            model="gpt-4o-2024-11-20",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content), finish_reason=self.finish_reason
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=1000,
                completion_tokens=200,
                prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            ),
        )


def block(n: int, citation: str, kind: str = "rule", text: str = "", binding: bool = True):
    key = f"{citation} Guidance" if kind == "guidance" else citation
    return ContextBlock(
        n=n,
        chunk_id=f"id-{n}",
        key=key,
        citation=citation,
        chunk_type=kind,
        binding=binding,
        label=LABELS[(kind, binding)],
        heading=f"DFSA Rulebook > {citation}",
        text=text or f"{citation} text.",
        tokens=10,
        reason="hit",
        group="g",
        score=0.9,
    )


BLOCKS = [
    block(
        1,
        "COB 3.3.1",
        text="3.3.1 The Rules in this section do not apply to a Market Counterparty.",
    ),
    block(2, "COB 3.3.2", text="3.3.2 A firm must enter into a Client Agreement."),
    block(3, "COB 3.3.2", kind="guidance", binding=False, text="It should be in writing."),
    block(4, "GLO Market Counterparty", kind="definition"),
    block(5, "Regulatory Law Art. 20", kind="article"),
]


def retrieval(
    blocks=None, confident=None, unresolved=(), question="Do we need a Client Agreement?"
) -> RetrievalResult:
    return RetrievalResult(
        question=question,
        filters=Filters(),
        config={},
        citations=CitationLookup(unresolved=list(unresolved)),
        candidates=[],
        hits=[],
        context=BLOCKS if blocks is None else blocks,
        confidence=0.9,
        confident=confident,
        timings_ms={},
    )


def draft(answer: str, *cites: tuple[int, str], confident: bool = True) -> dict:
    return {
        "answer": answer,
        "citations": [{"context_id": n, "citation": c} for n, c in cites],
        "confident": confident,
    }


def azure_settings() -> AzureOpenAISettings:
    return AzureOpenAISettings(
        _env_file=None, endpoint="https://example.openai.azure.com/", api_key="k"
    )


def git_registry() -> PromptRegistry:
    return PromptRegistry(
        load_prompts(PathSettings().prompts_dir), None, LangfuseSettings(enabled=False)
    )


@pytest.fixture
def router():
    return CitationRouter(None, None, CODES, LAWS)


@pytest.fixture
def make_answerer(router):
    def make(*replies, finish_reason="stop") -> tuple[Answerer, FakeAzure]:
        fake = FakeAzure(*replies, finish_reason=finish_reason)
        llm = LLMClient(azure_settings(), client=fake)
        return Answerer(llm, git_registry(), router.find), fake

    return make
