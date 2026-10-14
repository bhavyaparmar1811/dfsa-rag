"""promote_prompt asks the eval gate before moving `production`."""

from types import SimpleNamespace

import pytest

from scripts import promote_prompt
from src.eval.gate import Evidence
from src.prompts.registry import RemoteVersion


@pytest.fixture
def promote(monkeypatch):
    promoted = []
    target = RemoteVersion(version=3, labels=["staging"], hash="h3", variables=set())
    monkeypatch.setattr(promote_prompt, "LangfuseSettings", lambda: SimpleNamespace(active=True))
    monkeypatch.setattr(promote_prompt, "committed", lambda paths, name: True)
    monkeypatch.setattr(promote_prompt, "make_langfuse", lambda cfg: object())
    monkeypatch.setattr(promote_prompt, "check_promotable", lambda client, local, v: target)
    monkeypatch.setattr(promote_prompt, "fetch_version", lambda client, name, label: None)
    monkeypatch.setattr(promote_prompt, "get_engine", lambda: None)
    monkeypatch.setattr(promote_prompt, "promote", lambda c, name, t: promoted.append(name))

    def run(evidence: Evidence, *args):
        asked = []

        def ask(engine, name, content_hash, golden_hash):
            asked.append((name, content_hash))
            return evidence

        monkeypatch.setattr(promote_prompt, "promotion_evidence", ask)
        return promote_prompt.main(["answer", *args]), promoted, asked

    return run


def test_no_passing_run_blocks_the_promotion(promote, capsys):
    code, promoted, asked = promote(Evidence(False, "no e2e eval run used answer"))
    assert code == 1 and promoted == []
    assert asked == [("answer", "h3")]  # about exactly the version being promoted
    assert "FAIL: eval gate: no e2e eval run" in capsys.readouterr().out


def test_a_passing_run_promotes(promote, capsys):
    code, promoted, _ = promote(Evidence(True, "eval run #9 passed the gate", 9))
    assert code == 0 and promoted == ["answer"]
    assert "GATE PASSED: eval run #9" in capsys.readouterr().out


def test_skip_gate_is_an_explicit_override(promote, capsys):
    code, promoted, asked = promote(Evidence(False, "x"), "--skip-gate", "hotfix")
    assert code == 0 and promoted == ["answer"] and asked == []
    assert "GATE SKIPPED: hotfix" in capsys.readouterr().out
