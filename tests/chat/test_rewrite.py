from types import SimpleNamespace

from src.chat.rewrite import QueryRewriter, render_history
from src.generation.llm import LLMClient
from tests.generation.conftest import FakeAzure, azure_settings, git_registry

HISTORY = "User: Do we need a Client Agreement with a Market Counterparty?\nAssistant: No."


def rewriter(*replies, finish_reason="stop"):
    fake = FakeAzure(*replies, finish_reason=finish_reason)
    return QueryRewriter(LLMClient(azure_settings(), client=fake), git_registry()), fake


def test_render_history_labels_and_cuts_each_message():
    messages = [
        SimpleNamespace(role="user", content="Do we need\na Client Agreement?"),
        SimpleNamespace(role="assistant", content="x" * 50),
    ]
    text = render_history(messages, max_chars=10)
    assert text == "User: Do we nee…\nAssistant: xxxxxxxxx…"  # newlines collapsed, 10 chars


def test_the_first_turn_is_used_as_is_without_an_llm_call():
    rw, fake = rewriter()
    r = rw.rewrite("Do we need a Client Agreement?", history="")
    assert (r.question, r.changed, r.llm) == ("Do we need a Client Agreement?", False, None)
    assert fake.calls == []


def test_a_follow_up_is_rewritten_from_the_history():
    rw, fake = rewriter('"Do we need a Client Agreement with a Retail Client?"')
    r = rw.rewrite("what about retail clients?", HISTORY)

    assert r.question == "Do we need a Client Agreement with a Retail Client?"  # quotes stripped
    assert r.changed and r.cost_usd > 0
    user = fake.calls[0]["messages"][1]["content"]
    assert "Market Counterparty" in user and "Latest message: what about retail clients?" in user
    assert r.trace()["prompt"]["name"] == "query-rewrite"


def test_an_empty_or_cut_off_rewrite_falls_back_to_the_message():
    empty = rewriter("  ")[0].rewrite("what about retail clients?", HISTORY)
    assert (empty.question, empty.fallback_reason) == (
        "what about retail clients?",
        "empty rewrite",
    )

    cut = rewriter("Do we need", finish_reason="length")[0].rewrite("and then?", HISTORY)
    assert cut.question == "and then?" and cut.fallback_reason == "finish_reason length"
