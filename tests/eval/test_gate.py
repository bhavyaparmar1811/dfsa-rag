import pytest

from src.eval.gate import Thresholds, apply, evaluate, load_thresholds, promotion_evidence
from src.storage.db import session_scope
from src.storage.models import EvalRun
from src.storage.repositories import EvalRunRepository

GOLDEN = "g" * 64


def thresholds(**kw) -> Thresholds:
    base = {
        "config": "hybrid_rerank",
        "min_reviewed_items": 2,
        "max_false_refusal": 0.05,
        "floors": {"recall@6": 0.8, "faithfulness": 0.9},
        "ceilings": {"false_refusal_rate": 0.1, "errors": 0},
        "max_regression": 0.05,
    }
    return Thresholds.model_validate(base | kw)


def run(id=1, **kw) -> EvalRun:
    fields = {
        "id": id, "kind": "e2e", "status": "ok", "config_name": "hybrid_rerank",
        "golden_hash": GOLDEN, "reviewed_only": True, "n_items": 3, "n_turns": 3,
        "metrics": {"recall@6": 0.9, "faithfulness": 0.95, "false_refusal_rate": 0.0,
                    "errors": 0},
        "prompts": [{"name": "answer", "hash": "h1"}], "gate_status": "not_gated",
    }  # fmt: skip
    return EvalRun(**(fields | kw))


def failed(report) -> set[str]:
    return {c.name for c in report.checks if not c.passed}


def test_a_good_run_passes():
    report = evaluate(run(), thresholds(), None)
    assert report.passed and report.status == "passed"
    assert "PASSED" in report.render()


def test_floors_ceilings_and_preconditions():
    bad = run(
        kind="retrieval", reviewed_only=False, n_items=1,
        metrics={"recall@6": 0.5, "false_refusal_rate": 0.2, "errors": 1},
    )  # fmt: skip
    report = evaluate(bad, thresholds(), None)
    assert failed(report) == {
        "kind", "reviewed_only", "items", "recall@6", "faithfulness",
        "false_refusal_rate", "errors",
    }  # fmt: skip
    missing = next(c for c in report.checks if c.name == "faithfulness")
    assert missing.note == "metric missing from the run"


def test_an_unset_limit_fails():
    report = evaluate(run(), thresholds(floors={"recall@6": None}), None)
    assert failed(report) == {"recall@6"}
    unset = next(c for c in report.checks if c.name == "recall@6")
    assert unset.note == "limit not set in thresholds.yaml"


def test_regression_against_the_baseline():
    baseline = run(id=1, metrics={"recall@6": 1.0, "faithfulness": 0.95})
    current = run(id=2, metrics={"recall@6": 0.9, "faithfulness": 0.94, "false_refusal_rate": 0,
                                 "errors": 0})  # fmt: skip
    report = evaluate(current, thresholds(), baseline)
    assert report.baseline_run == 1
    assert failed(report) == {"recall@6 vs #1"}  # 0.9 < 1.0 * 0.95; 0.94 >= 0.9025


def test_apply_uses_the_last_passing_production_run_as_baseline(engine):
    with session_scope(engine) as s:
        repo = EvalRunRepository(s)
        repo.start(run(id=None, gate_status="passed", metrics={"recall@6": 1.0}))
        repo.start(run(id=None, gate_status="failed", metrics={"recall@6": 0.1}))
        repo.start(run(id=None, config_name="parent_window"))  # a candidate config
    report = apply(engine, 3, thresholds())
    assert report.baseline_run == 1
    assert failed(report) == {"recall@6 vs #1"}  # 0.9 < 0.95
    with session_scope(engine) as s:
        stored = EvalRunRepository(s).get(3)
        assert stored.gate_status == "failed" and stored.gate_report["baseline_run"] == 1


def test_promotion_evidence(engine):
    assert not promotion_evidence(engine, "answer", "h1", GOLDEN).ok
    with session_scope(engine) as s:
        repo = EvalRunRepository(s)
        repo.start(run(id=None, gate_status="passed"))
        repo.start(run(id=None, golden_hash="other", gate_status="passed"))
    ok = promotion_evidence(engine, "answer", "h1", GOLDEN)
    assert ok.ok and ok.run_id == 1
    assert not promotion_evidence(engine, "answer", "h2", GOLDEN).ok  # other content
    assert not promotion_evidence(engine, "refusal", "h1", GOLDEN).ok  # other prompt
    with session_scope(engine) as s:
        EvalRunRepository(s).start(run(id=None, gate_status="failed"))
    newest_failed = promotion_evidence(engine, "answer", "h1", GOLDEN)
    assert not newest_failed.ok and newest_failed.run_id == 3
    assert "failed the gate" in newest_failed.reason


def test_committed_thresholds_parse():
    t = load_thresholds()
    assert t.config == "hybrid_rerank" and 0 < t.max_false_refusal < 1
    with pytest.raises(ValueError):
        Thresholds.model_validate(t.model_dump() | {"typo": 1})
