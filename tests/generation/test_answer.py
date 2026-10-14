import json

import pytest

from src.generation.answer import GUIDANCE_ONLY_NOTE, REASONS, Answerer
from src.generation.llm import LLMClient
from src.retrieval.pipeline import Retriever
from tests.factories import FakeEmbedder, FakeSparse
from tests.generation.conftest import FakeAzure, azure_settings, draft, git_registry, retrieval
from tests.retrieval.conftest import EMB, QD, FakeScorer, settings

GOOD = draft(
    "Yes, a Client Agreement is required (COB 3.3.2), but not with a Market Counterparty "
    "(COB 3.3.1).",
    (2, "COB 3.3.2"), (1, "COB 3.3.1"),
)  # fmt: skip


def test_a_supported_answer_is_returned_with_its_sources(make_answerer):
    answerer, fake = make_answerer(GOOD)
    result = answerer.answer(retrieval())

    assert result.status == "answered" and result.refusal is None
    assert result.text == GOOD["answer"]
    assert [c.citation for c in result.cited] == ["COB 3.3.2", "COB 3.3.1"]
    assert result.gate == "uncalibrated"  # no threshold yet: the LLM still runs
    assert result.llm.cost_usd > 0 and result.prompts[0]["name"] == "answer"
    assert fake.calls[0]["response_format"] == {"type": "json_object"}
    user = fake.calls[0]["messages"][1]["content"]
    assert "[2] COB 3.3.2 | Rule (binding)" in user and "Question: Do we need" in user
    json.dumps(result.trace())  # plain JSON for the trace row


@pytest.mark.parametrize(
    "kwargs, code",
    [({"blocks": []}, "no_context"), ({"confident": False}, "low_confidence")],
)
def test_the_gate_refuses_without_calling_the_llm(make_answerer, kwargs, code):
    answerer, fake = make_answerer()
    result = answerer.answer(retrieval(**kwargs))

    assert result.refused and result.refusal.code == code
    assert fake.calls == [] and result.llm is None
    assert REASONS[code] in result.text  # the refusal prompt, with the reason
    assert result.text.startswith("I don't have a confident answer")


def test_a_passed_gate_is_recorded(make_answerer):
    answerer, _ = make_answerer(GOOD)
    assert answerer.answer(retrieval(confident=True)).gate == "passed"


@pytest.mark.parametrize(
    "reply, finish, code",
    [
        ("not json", "stop", "invalid_output"),
        ({"answer": "Yes"}, "stop", "invalid_output"),  # fields missing
        (GOOD, "length", "invalid_output"),  # cut off
        (GOOD, "content_filter", "invalid_output"),
        (draft("Partly covered.", (2, "COB 3.3.2"), confident=False), "stop",
         "model_not_confident"),
        (draft("Yes (COB 3.3.2), see COB 9.9.9.", (2, "COB 3.3.2")), "stop",
         "citation_check_failed"),  # a fabricated rule
        (draft("Yes (COB 3.3.2).", (2, "COB 3.3.7")), "stop", "citation_check_failed"),
    ],
)  # fmt: skip
def test_failed_checks_refuse_and_never_patch(make_answerer, reply, finish, code):
    answerer, fake = make_answerer(reply, finish_reason=finish)
    result = answerer.answer(retrieval())

    assert result.refused and result.refusal.code == code
    assert len(fake.calls) == 1
    assert result.cited == [] and "COB 9.9.9" not in result.text
    assert result.draft is not None  # kept for the audit trail


def test_a_fabricated_citation_is_reported_in_the_violations(make_answerer):
    answerer, _ = make_answerer(draft("Yes (COB 3.3.2), see COB 9.9.9.", (2, "COB 3.3.2")))
    result = answerer.answer(retrieval())
    assert result.violations == ["COB 9.9.9: mentioned, not in the context"]
    assert result.mentions == ["COB 3.3.2", "COB 9.9.9"]


def test_unresolved_citations_in_the_question_are_named(make_answerer):
    answerer, _ = make_answerer()
    result = answerer.answer(retrieval(blocks=[], unresolved=["COB 99.9.9"]))
    assert "COB 99.9.9 is not in the documents I can search." in result.text


def test_a_guidance_only_answer_is_labelled(make_answerer):
    reply = draft(
        "DFSA Guidance indicates it should be in writing (COB 3.3.2 Guidance), but this is "
        "not a binding Rule.", (3, "COB 3.3.2 Guidance"),
    )  # fmt: skip
    result = make_answerer(reply)[0].answer(retrieval())

    assert result.status == "answered" and result.guidance_only
    assert result.text.startswith(GUIDANCE_ONLY_NOTE)
    assert result.cited[0].label == "Guidance (non-binding)"


def test_a_staging_label_is_passed_to_the_registry(make_answerer):
    answerer, _ = make_answerer(GOOD)
    result = answerer.answer(retrieval(), label="staging")
    assert result.prompts[0]["label"] == "staging"  # Langfuse is off: the git copy, so labelled


def test_end_to_end_on_the_mini_corpus(loaded):
    engine, store = loaded
    retriever = Retriever.build(
        settings(top_n=3), engine=engine, store=store, embedder=FakeEmbedder(EMB),
        sparse=FakeSparse(QD), scorer=FakeScorer(),
    )  # fmt: skip
    r = retriever.retrieve("Does the Client Agreement rule apply to a Market Counterparty?")
    n = {b.citation: b.n for b in r.context if b.chunk_type == "rule"}
    reply = draft(
        "No: the Client Agreement Rules (COB 3.3.2) do not apply to a Market Counterparty "
        "(COB 3.3.1).", (n["COB 3.3.2"], "COB 3.3.2"), (n["COB 3.3.1"], "COB 3.3.1"),
    )  # fmt: skip
    llm = LLMClient(azure_settings(), client=FakeAzure(reply))
    result = Answerer(llm, git_registry(), retriever.citations.find).answer(r)

    assert result.status == "answered", result.violations
    assert {c.key for c in result.cited} == {"COB 3.3.1", "COB 3.3.2"}
