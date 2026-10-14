"""The eval runner on the mini corpus: fake embedder, BM25, cross-encoder and Azure."""

import json
import re
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from src.eval.configs import settings_for
from src.eval.golden import GoldenSet
from src.eval.run import Components, EvalRunner, report
from src.generation.llm import LLMClient
from src.storage.db import session_scope
from src.storage.models import EvalResult, EvalRun, Message, Trace
from tests.factories import FakeEmbedder, FakeSparse
from tests.generation.conftest import azure_settings, git_registry
from tests.retrieval.conftest import EMB, QD, FakeScorer, settings

GOLDEN = {
    "version": 1,
    "items": [
        {"id": "mc", "kind": "exemption", "module": "COB", "reviewed": True, "turns": [
            {"question": "Is a Client Agreement needed with a Market Counterparty?",
             "expect": "answer", "gold": ["COB 3.3.1"], "supporting": ["COB 3.3.2"]}]},
        {"id": "conv", "kind": "multi_turn", "module": "COB", "reviewed": True, "turns": [
            {"question": "Must a firm enter into a Client Agreement?", "expect": "answer",
             "gold": ["COB 3.3.2"]},
            {"question": "And can it be amended for a retail client?", "expect": "answer",
             "gold": ["COB 3.3.3"]}]},
        {"id": "vat", "kind": "unanswerable", "reviewed": True, "turns": [
            {"question": "What is the VAT rate in the UAE?", "expect": "refuse"}]},
    ],
}  # fmt: skip
REWRITTEN = "Can a Client Agreement with a Retail Client be amended?"
BLOCK = re.compile(r"^\[(\d+)\] (.+?) \|", re.M)


class RoutingAzure:
    """Replies by which prompt is calling: rewrite, answer, or one of the judges.
    The answer cites the first context block (or declines for the VAT question)."""

    def __init__(self):
        self.calls: list[str] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        system, user = kwargs["messages"][0]["content"], kwargs["messages"][-1]["content"]
        if system.startswith("You rewrite"):
            kind, reply = "rewrite", REWRITTEN
        elif '"claims"' in system:
            kind, reply = "faithfulness", json.dumps({"claims": [
                {"claim": "a", "verdict": "supported", "context_ids": [1]},
                {"claim": "b", "verdict": "contradicted", "context_ids": [1]}]})  # fmt: skip
        elif '"score"' in system:
            kind, reply = "relevance", json.dumps({"reasoning": "fine", "score": 4})
        else:
            kind = "answer"
            n, citation = BLOCK.search(user.split("Context blocks:")[1]).groups()
            confident = "VAT" not in user
            reply = json.dumps({"answer": f"See {citation}." if confident else "Not covered.",
                                "citations": [{"context_id": int(n), "citation": citation}]
                                if confident else [], "confident": confident})  # fmt: skip
        self.calls.append(kind)
        return SimpleNamespace(
            model="gpt-4o",
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10,
                                  prompt_tokens_details=SimpleNamespace(cached_tokens=0)),
        )  # fmt: skip


@pytest.fixture
def runner(loaded):
    engine, store = loaded
    azure = RoutingAzure()
    parts = Components(
        engine=engine,
        store=store,
        embedder=FakeEmbedder(EMB),
        sparse=FakeSparse(QD),
        llm=LLMClient(azure_settings(), client=azure),
        registry=git_registry(),
        scorer_factory=lambda cfg: FakeScorer(),
    )
    golden = GoldenSet.model_validate(GOLDEN)
    return EvalRunner(parts, golden, golden.select()), azure, engine


def test_retrieval_only_runs_share_the_follow_up_rewrite(runner):
    runner, azure, engine = runner
    first = runner.run("hybrid_rerank", settings_for("hybrid_rerank", settings()), e2e=False)
    second = runner.run("dense", settings_for("dense", settings()), e2e=False)

    assert azure.calls == ["rewrite"]  # once, for the follow-up; reused by the 2nd config
    assert first.kind == "retrieval" and first.status == "ok" and first.n_turns == 4
    assert first.gate_status == "not_gated" and first.prompts == []
    assert first.metrics["turns"] == 4 and first.metrics["errors"] == 0
    assert first.metrics["n_mrr"] == 3  # the refusal turn has no retrieval metrics
    assert first.metrics["by_kind"]["multi_turn"]["turns"] == 2
    assert first.cost_usd["rewrite"] > 0 and second.cost_usd["rewrite"] == 0
    assert second.retrieval_config["mode"] == "dense"

    with session_scope(engine) as s:
        results = list(s.scalars(select(EvalResult).where(EvalResult.run_id == first.id)))
        follow_up = next(r for r in results if r.item_id == "conv" and r.turn == 1)
        assert follow_up.rewritten_query == REWRITTEN
        assert follow_up.ranked and follow_up.ranked[0]["provision"]
        assert all("#p" not in r["provision"] for r in follow_up.ranked)
        assert follow_up.confidence is not None  # reranked
        assert s.scalar(select(func.count()).select_from(Message)) == 0
        assert s.scalar(select(func.count()).select_from(Trace)) == 0


@pytest.mark.parametrize(
    "threshold, vat_answer_calls, vat_code",
    [
        (None, 1, "model_not_confident"),  # uncalibrated: the model declines
        (0.40, 0, "low_confidence"),  # the production threshold refuses before the LLM
    ],
)
def test_e2e_run_answers_judges_and_gates(runner, threshold, vat_answer_calls, vat_code):
    runner, azure, engine = runner
    cfg = settings_for("hybrid_rerank", settings(refusal_threshold=threshold))
    run = runner.run("hybrid_rerank", cfg, e2e=True)

    # mc: answer + 2 judges; conv: answer + 2 judges, rewrite + answer + 2 judges; vat: answer?
    assert azure.calls.count("answer") == 3 + vat_answer_calls
    assert azure.calls.count("rewrite") == 1
    assert azure.calls.count("faithfulness") == azure.calls.count("relevance") == 3

    m = run.metrics
    assert m["outcomes"]["correct_refusal"] == 1 and m["refusal_recall"] == 1.0
    assert m["faithfulness"] == 0.5 and m["contradicted_claims"] == 3
    assert m["relevance"] == 4.0
    assert {p["name"] for p in run.prompts} == {
        "answer",
        "refusal",
        "query-rewrite",
        "judge-faithfulness",
        "judge-relevance",
    }
    assert run.prompt_label == "production"
    assert run.cost_usd["judge"] > 0 and run.cost_usd["answer"] > 0
    # The run is gated against the committed thresholds: 3 items are too few to pass.
    assert run.gate_status == "failed"
    failed = {c["name"] for c in run.gate_report["checks"] if not c["passed"]}
    assert "items" in failed

    with session_scope(engine) as s:
        results = {(r.item_id, r.turn): r for r in s.scalars(select(EvalResult))}
    follow_up = results[("conv", 1)]
    assert follow_up.rewritten_query == REWRITTEN and follow_up.trace_id
    assert results[("vat", 0)].status == "refused"
    assert results[("vat", 0)].refusal_code == vat_code
    assert results[("mc", 0)].judges["faithfulness"]["score"] == 0.5


def test_a_failing_turn_is_recorded_and_the_run_continues(runner):
    runner, _, engine = runner
    cfg = settings_for("hybrid_rerank", settings())
    real = runner._retriever

    def flaky(cfg):
        retriever = real(cfg)
        retrieve = retriever.retrieve

        def maybe_fail(question, filters=None):
            if "VAT" in question:
                raise RuntimeError("qdrant down")
            return retrieve(question, filters)

        retriever.retrieve = maybe_fail
        return retriever

    runner._retriever = flaky
    run = runner.run("hybrid_rerank", cfg, e2e=False)
    assert run.status == "ok" and run.metrics["errors"] == 1
    with session_scope(engine) as s:
        vat = s.scalar(select(EvalResult).where(EvalResult.item_id == "vat"))
        assert vat.status == "error" and "qdrant down" in vat.metrics["error"]


def test_report_renders_both_tables(runner):
    runner, _, engine = runner
    runs = [
        runner.run("hybrid", settings_for("hybrid", settings()), e2e=False),
        runner.run("hybrid_rerank", settings_for("hybrid_rerank", settings()), e2e=True),
    ]
    text = report(runs)
    assert "| r@1 |" in text and "| Faithfulness |" in text and "multi_turn" in text
    with session_scope(engine) as s:
        assert s.scalar(select(func.count()).select_from(EvalRun)) == 2
