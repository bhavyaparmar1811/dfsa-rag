import json

import httpx
import pytest

from src.chat.service import ConversationNotFound, TurnFailed
from src.storage.db import session_scope
from src.storage.models import Message, Trace
from tests.chat.conftest import ANSWER, FakeRetriever
from tests.generation.conftest import draft


def rows(engine, model):
    with session_scope(engine) as s:
        return s.query(model).order_by(model.created_at).all()


def test_a_two_turn_conversation_is_stored_and_rewritten(make_service, engine):
    retriever = FakeRetriever()
    rewritten = "Do we need a Client Agreement with a Retail Client?"
    service, fake = make_service(ANSWER, rewritten, ANSWER, retriever=retriever)
    conv = service.create_conversation()

    first = service.send(conv.id, "Do we need a Client Agreement with a Market Counterparty?")
    second = service.send(conv.id, "what about retail clients?")

    # Turn 1: no history, so no rewrite call; turn 2: rewrite, then answer.
    assert len(fake.calls) == 3
    assert not first.rewrite.changed and second.rewrite.question == rewritten
    assert retriever.questions == [first.rewrite.question, rewritten]
    # The rewrite and the answer both saw turn 1, loaded from SQLite.
    assert "Market Counterparty" in fake.calls[1]["messages"][1]["content"]
    assert "User: Do we need a Client Agreement" in fake.calls[2]["messages"][1]["content"]

    stored, messages = service.get_conversation(conv.id)
    assert [m.role for m in messages] == ["user", "assistant", "user", "assistant"]
    assert stored.title == "Do we need a Client Agreement with a Market Counterparty?"
    user, assistant = messages[2], messages[3]
    assert user.rewritten_query == rewritten and user.trace_id == second.trace_id
    assert assistant.status == "answered" and assistant.trace_id == second.trace_id
    assert [c["citation"] for c in assistant.cited] == ["COB 3.3.2", "COB 3.3.1"]
    assert assistant.cost_usd == pytest.approx(second.cost_usd) and assistant.cost_usd > 0

    trace = service.get_trace(second.trace_id)
    assert trace.status == "ok" and trace.rewritten_query == rewritten
    assert trace.total_cost_usd == pytest.approx(trace.rewrite_cost_usd + trace.answer_cost_usd)
    assert trace.retrieval["question"] == rewritten and trace.answer["status"] == "answered"
    assert set(trace.latency_ms) == {"total", "rewrite", "retrieve", "answer"}
    json.dumps(trace.retrieval), json.dumps(trace.answer)


def test_history_is_limited_to_the_last_turns(make_service):
    service, fake = make_service(ANSWER, "q2", ANSWER, "q3", ANSWER, history_turns=2)
    conv = service.create_conversation()
    service.send(conv.id, "first question")
    service.send(conv.id, "second question")
    service.send(conv.id, "third question")

    rewrite_input = fake.calls[3]["messages"][1]["content"]  # the third turn's rewrite
    assert "second question" in rewrite_input and "first question" not in rewrite_input


def test_a_refusal_is_stored_with_its_code(make_service):
    retriever = FakeRetriever(confident=False)
    service, fake = make_service(retriever=retriever)
    conv = service.create_conversation()
    turn = service.send(conv.id, "What is the VAT rate?")

    assert turn.answer.refused and fake.calls == []  # the gate: no LLM call
    _, messages = service.get_conversation(conv.id)
    assert (messages[1].status, messages[1].refusal_code) == ("refused", "low_confidence")
    trace = service.get_trace(turn.trace_id)
    assert trace.refused and trace.refusal_code == "low_confidence"


def test_an_upstream_failure_stores_an_error_trace_and_no_messages(make_service, engine):
    retriever = FakeRetriever(error=httpx.ConnectError("qdrant down"))
    service, _ = make_service(retriever=retriever)
    conv = service.create_conversation()

    with pytest.raises(TurnFailed) as failed:
        service.send(conv.id, "Do we need a Client Agreement?")
    assert failed.value.upstream

    assert rows(engine, Message) == []
    trace = service.get_trace(failed.value.trace_id)
    assert trace.status == "error" and "qdrant down" in trace.error
    assert trace.original_query == "Do we need a Client Agreement?"


def test_a_bug_is_not_reported_as_upstream(make_service):
    service, _ = make_service(retriever=FakeRetriever(error=ValueError("bug")))
    conv = service.create_conversation()
    with pytest.raises(TurnFailed) as failed:
        service.send(conv.id, "q")
    assert not failed.value.upstream


def test_an_unknown_conversation_is_not_found(make_service, engine):
    service, _ = make_service(draft("x", (1, "COB 3.3.1")))
    with pytest.raises(ConversationNotFound):
        service.send("no-such-id", "q")
    assert rows(engine, Trace) == []  # nothing ran, nothing stored


def test_cost_stats(make_service):
    service, _ = make_service(ANSWER, retriever=FakeRetriever())
    conv = service.create_conversation()
    service.send(conv.id, "Do we need a Client Agreement?")
    refusing = make_service(retriever=FakeRetriever(confident=False))[0]
    refusing.send(refusing.create_conversation().id, "VAT?")  # a new one: no rewrite call

    stats = service.cost_stats()
    assert (stats["requests"], stats["refused"], stats["errors"]) == (2, 1, 0)
    assert stats["refusal_rate"] == 0.5
    assert stats["cost_usd"]["answer"] > 0 and stats["cost_usd"]["rewrite"] == 0
    assert len(stats["by_day"]) == 1 and stats["by_day"][0]["requests"] == 2
