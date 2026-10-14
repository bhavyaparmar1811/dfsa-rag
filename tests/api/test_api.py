import httpx
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config.settings import ApiSettings
from src.eval.golden import load_golden
from src.storage.db import session_scope
from src.storage.models import EvalResult, EvalRun
from src.storage.repositories import ChunkRepository, DocumentRepository, EvalRunRepository
from tests.chat.conftest import ANSWER, FakeRetriever, make_service  # noqa: F401
from tests.factories import chunk, entry

KEY = {"X-API-Key": "test-key"}


@pytest.fixture
def client_for(make_service):  # noqa: F811
    def make(*replies, retriever=None, **cfg) -> TestClient:
        service, _ = make_service(*replies, retriever=retriever)
        settings = ApiSettings(_env_file=None, key="test-key", **cfg)
        return TestClient(create_app(lambda: service, settings))

    return make


def test_every_endpoint_but_health_needs_the_key(client_for):
    with client_for() as client:
        assert client.get("/health").status_code == 200
        assert client.post("/conversations").status_code == 401
        assert client.post("/conversations", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/stats/cost").status_code == 401
        assert client.post("/conversations", headers=KEY).status_code == 201


def test_startup_refuses_without_a_key(make_service):  # noqa: F811
    service, _ = make_service()
    app = create_app(lambda: service, ApiSettings(_env_file=None, key=None))
    with pytest.raises(RuntimeError, match="API_KEY"), TestClient(app):
        pass


def test_a_conversation_with_a_follow_up(client_for):
    rewritten = "Do we need a Client Agreement with a Retail Client?"
    with client_for(ANSWER, rewritten, ANSWER) as client:
        conv = client.post("/conversations", headers=KEY, json={"title": "Client Agreements"})
        cid = conv.json()["id"]

        first = client.post(f"/conversations/{cid}/messages", headers=KEY,
                            json={"content": "Do we need a Client Agreement?"})  # fmt: skip
        assert first.status_code == 200, first.text
        body = first.json()
        assert body["status"] == "answered" and body["refusal"] is None
        assert [c["citation"] for c in body["citations"]] == ["COB 3.3.2", "COB 3.3.1"]
        assert body["rewritten_query"] == "Do we need a Client Agreement?"

        second = client.post(f"/conversations/{cid}/messages", headers=KEY,
                             json={"content": "what about retail clients?",
                                   "filters": {"modules": ["COB"]}}).json()  # fmt: skip
        assert second["rewritten_query"] == rewritten

        detail = client.get(f"/conversations/{cid}", headers=KEY).json()
        assert detail["title"] == "Client Agreements"
        assert [m["role"] for m in detail["messages"]] == ["user", "assistant"] * 2
        assert detail["messages"][3]["trace_id"] == second["trace_id"]

        trace = client.get(f"/traces/{second['trace_id']}", headers=KEY).json()
        assert trace["rewritten_query"] == rewritten and trace["filters"] == {"modules": ["COB"]}

        stats = client.get("/stats/cost", headers=KEY).json()
        assert stats["requests"] == 2 and stats["cost_usd"]["rewrite"] > 0
        assert stats["cost_by_provider_usd"]["azure_generation"] > 0

        today = stats["by_day"][0]["day"]

        def window(query: str):
            return client.get(f"/stats/cost?{query}", headers=KEY)

        assert window(f"since={today}&until={today}").json()["requests"] == 2
        assert window("since=2000-01-01&until=2000-01-31").json()["requests"] == 0
        assert window(f"since={today}&until=2000-01-01").status_code == 422
        assert window("since=yesterday").status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        {"content": ""},
        {"content": "x" * 2001},
        {"content": "q", "history": [{"role": "user", "content": "earlier"}]},  # never trusted
        {"content": "q", "filters": {"chunk_types": ["section"]}},
    ],
)
def test_invalid_messages_are_rejected(client_for, body):
    with client_for() as client:
        cid = client.post("/conversations", headers=KEY).json()["id"]
        r = client.post(f"/conversations/{cid}/messages", headers=KEY, json=body)
        assert r.status_code == 422


def test_unknown_ids_are_404(client_for):
    with client_for() as client:
        assert client.get("/conversations/nope", headers=KEY).status_code == 404
        r = client.post("/conversations/nope/messages", headers=KEY, json={"content": "q"})
        assert r.status_code == 404
        assert client.get("/traces/nope", headers=KEY).status_code == 404


def test_an_upstream_outage_is_503_with_the_trace_id(client_for):
    down = FakeRetriever(error=httpx.ConnectError("qdrant down"))
    with client_for(retriever=down) as client:
        cid = client.post("/conversations", headers=KEY).json()["id"]
        r = client.post(f"/conversations/{cid}/messages", headers=KEY, json={"content": "q"})
        assert r.status_code == 503
        trace = client.get(f"/traces/{r.json()['trace_id']}", headers=KEY).json()
        assert trace["status"] == "error"


def test_health_reports_each_dependency(client_for):
    with client_for() as client:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert set(body["checks"]) == {"database", "qdrant", "reranker", "cache", "langfuse"}
        assert body["checks"]["langfuse"] == {"ok": True, "enabled": False}
        assert body["checks"]["cache"] == {"ok": True, "enabled": False}  # no cache built


# --- read-only views for the web UI --------------------------------------------


def test_the_new_read_endpoints_need_the_key(client_for):
    with client_for() as client:
        for path in ("/conversations", "/chunks/x", "/eval/runs", "/eval/runs/1",
                     "/eval/runs/1/results", "/eval/compare", "/eval/thresholds"):  # fmt: skip
            assert client.get(path).status_code == 401, path


def test_conversations_list_only_those_with_messages(client_for):
    with client_for(ANSWER) as client:
        client.post("/conversations", headers=KEY)  # never used: not listed
        cid = client.post("/conversations", headers=KEY).json()["id"]
        client.post(f"/conversations/{cid}/messages", headers=KEY,
                    json={"content": "Do we need a Client Agreement?"})  # fmt: skip

        listed = client.get("/conversations", headers=KEY).json()
        assert [c["id"] for c in listed] == [cid]
        assert listed[0]["title"] == "Do we need a Client Agreement?"
        assert client.get("/conversations?limit=0", headers=KEY).status_code == 422


def test_a_chunk_part_comes_with_its_whole_rule(client_for, engine):
    cob = [
        chunk("COB 3.3.2", text="The whole rule.", parts=["COB 3.3.2#p1"]),
        chunk("COB 3.3.2#p1", text="Part one.", part_of="COB 3.3.2", part=1, order=1),
    ]
    with session_scope(engine) as s:
        DocumentRepository(s).upsert(entry())
        ChunkRepository(s).upsert(cob, {c.chunk_id: "h" for c in cob})

    with client_for() as client:
        body = client.get(f"/chunks/{cob[1].chunk_id}", headers=KEY).json()
        assert body["citation"] == "COB 3.3.2" and body["part"] == 1
        assert body["text"] == "Part one." and body["whole_text"] == "The whole rule."
        assert body["module_code"] == "COB" and body["binding"] is True

        whole = client.get(f"/chunks/{cob[0].chunk_id}", headers=KEY).json()
        assert whole["whole_text"] is None
        assert client.get("/chunks/nope", headers=KEY).status_code == 404


def _eval_run(engine, **kw) -> int:
    fields = {
        "kind": "retrieval", "status": "ok", "config_name": "hybrid_rerank",
        "golden_hash": load_golden().hash, "reviewed_only": True, "n_items": 1, "n_turns": 1,
        "metrics": {"recall@6": 0.9, "mrr": 0.8}, "cost_usd": {"total": 0.5},
    }  # fmt: skip
    with session_scope(engine) as s:
        repo = EvalRunRepository(s)
        run = repo.start(EvalRun(**(fields | kw)))
        repo.add_result(EvalResult(
            run_id=run.id, item_id="cob-suitability-before-recommending", turn=0,
            question="q", expect="answer", metrics={"recall@6": 1.0},
        ))  # fmt: skip
        return run.id


def test_eval_runs_their_results_and_the_stage_comparison(client_for, engine):
    old = _eval_run(engine, config_name="dense")
    dense = _eval_run(engine, config_name="dense", metrics={"recall@6": 0.7})
    rerank = _eval_run(engine)
    unreviewed = _eval_run(engine, config_name="hybrid", reviewed_only=False)  # not compared
    e2e = _eval_run(engine, kind="e2e", gate_status="passed", gate_report={"passed": True})

    with client_for() as client:
        runs = client.get("/eval/runs", headers=KEY).json()
        assert [r["id"] for r in runs] == [e2e, unreviewed, rerank, dense, old]
        assert runs[2]["headline"] == {"recall@6": 0.9, "mrr": 0.8}
        assert runs[2]["cost_total_usd"] == 0.5
        e2e_only = client.get("/eval/runs?kind=e2e", headers=KEY).json()
        assert [r["id"] for r in e2e_only] == [e2e]
        assert client.get("/eval/runs?kind=nope", headers=KEY).status_code == 422

        run = client.get(f"/eval/runs/{e2e}", headers=KEY).json()
        assert run["gate_status"] == "passed" and run["gate_report"] == {"passed": True}

        results = client.get(f"/eval/runs/{rerank}/results", headers=KEY).json()
        assert results[0]["item_kind"] == "single" and results[0]["gold"] == ["COB 3.4.2"]

        compare = client.get("/eval/compare", headers=KEY).json()
        assert compare["golden_is_current"] is True
        assert [(r["id"], r["config_name"]) for r in compare["runs"]] == [
            (dense, "dense"), (rerank, "hybrid_rerank")
        ]  # fmt: skip
        assert compare["runs"][0]["metrics"] == {"recall@6": 0.7}

        assert client.get("/eval/runs/999", headers=KEY).status_code == 404
        assert client.get("/eval/runs/999/results", headers=KEY).status_code == 404
        assert client.get("/eval/thresholds", headers=KEY).json()["config"] == "hybrid_rerank"
