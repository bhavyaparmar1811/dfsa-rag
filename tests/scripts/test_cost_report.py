"""cost_report adds up production requests, eval runs and ingestion, per provider."""

from datetime import date, datetime

import pytest

from scripts import cost_report
from src.storage.db import session_scope
from src.storage.models import EvalRun, Trace
from src.storage.repositories import IngestionRunRepository, RunStats, TraceRepository


@pytest.fixture
def seeded(engine):
    with session_scope(engine) as s:
        TraceRepository(s).add(
            Trace(
                id="t1", created_at=datetime(2026, 10, 14, 12), status="ok", original_query="q",
                refused=False, embed_tokens=12, embed_cost_usd=0.000001, answer_input_tokens=6000,
                answer_output_tokens=150, answer_cost_usd=0.0165, total_cost_usd=0.016501,
                latency_ms={"total": 4500.0},
            )
        )  # fmt: skip
        s.add(
            EvalRun(
                kind="e2e", config_name="hybrid_rerank", golden_hash="g", n_turns=64,
                created_at=datetime(2026, 10, 13, 9),
                cost_usd={"embed": 0.00002, "rewrite": 0.01, "answer": 0.85, "judge": 0.82,
                          "total": 1.68002},
            )
        )  # fmt: skip
        runs = IngestionRunRepository(s)
        runs.finish(runs.start("fp", "bm25"), RunStats(embed_tokens=587492, embed_cost_usd=0.01175))
    return engine


def test_totals_split_by_provider(seeded):
    report = cost_report.build_report(seeded)
    total = report["total_usd"]
    assert total["production"] == 0.016501 and total["evals"] == 1.68002
    assert total["openai_embeddings"] == round(0.000001 + 0.00002 + 0.01175, 6)
    assert total["azure_generation"] == round(0.0165 + 0.01 + 0.85 + 0.82, 6)
    assert total["total"] == pytest.approx(total["openai_embeddings"] + total["azure_generation"])


def test_markdown_report(seeded):
    text = cost_report.render(cost_report.build_report(seeded))
    assert "| 1 | 0 (0 / 0) | 0 | 0.0% | $0.0165 | $0.0165 | 4.5 s / 4.5 s |" in text
    stages = "answer $0.8500, embed $0.000020, judge $0.8200, rewrite $0.0100"
    assert f"| e2e | 1 | 64 | {stages} | $1.68 |" in text
    assert "| 1 | 0 | 587,492 | $0.0118 |" in text
    assert "**$1.71**" in text


def test_a_window_without_activity(seeded):
    text = cost_report.render(cost_report.build_report(seeded, since=date(2027, 1, 1)))
    assert "No production requests in this window." in text
    assert "No eval runs in this window." in text


def test_cli_arguments(seeded, monkeypatch, capsys):
    monkeypatch.setattr(cost_report, "get_engine", lambda: seeded)
    assert cost_report.main(["--since", "2026-10-14", "--json"]) == 0
    assert '"requests": 1' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cost_report.main(["--days", "7", "--since", "2026-10-01"])
    with pytest.raises(SystemExit):
        cost_report.main(["--since", "2026-10-14", "--until", "2026-10-01"])
