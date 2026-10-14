"""A ChatService over a temporary SQLite database, with a fake retriever and a
scripted Azure client (one reply per LLM call: rewrite, then answer)."""

from types import SimpleNamespace

import pytest

from src.chat.rewrite import QueryRewriter
from src.chat.service import ChatService
from src.config.settings import ChatSettings
from src.generation.answer import Answerer
from src.generation.llm import LLMClient
from src.retrieval.citations import CitationRouter
from tests.generation.conftest import (
    CODES,
    LAWS,
    FakeAzure,
    azure_settings,
    draft,
    git_registry,
    retrieval,
)

ANSWER = draft(
    "Yes, a Client Agreement is required (COB 3.3.2), but not with a Market Counterparty "
    "(COB 3.3.1).",
    (2, "COB 3.3.2"),
    (1, "COB 3.3.1"),
)


class FakeRetriever:
    """Returns the generation tests' context blocks for any question, and records it."""

    def __init__(self, error: Exception | None = None, **kwargs):
        self.questions: list[str] = []
        self.error = error
        self.kwargs = kwargs
        self.citations = CitationRouter(None, None, CODES, LAWS)
        self.reranker = None
        self.searcher = SimpleNamespace(store=SimpleNamespace(collection="test", count=lambda: 5))

    def retrieve(self, question, filters=None):
        self.questions.append(question)
        if self.error:
            raise self.error
        return retrieval(question=question, **self.kwargs)


@pytest.fixture
def make_service(engine):
    def make(*replies, retriever=None, history_turns=6) -> tuple[ChatService, FakeAzure]:
        fake = FakeAzure(*replies)
        retriever = retriever or FakeRetriever()
        llm, registry = LLMClient(azure_settings(), client=fake), git_registry()
        service = ChatService(
            engine=engine,
            retriever=retriever,
            answerer=Answerer(llm, registry, retriever.citations.find),
            rewriter=QueryRewriter(llm, registry),
            cfg=ChatSettings(_env_file=None, history_turns=history_turns),
        )
        return service, fake

    return make
