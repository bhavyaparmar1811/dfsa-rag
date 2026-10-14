import httpx
import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config.settings import ApiSettings
from tests.chat.conftest import ANSWER, FakeRetriever, make_service  # noqa: F401

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
        assert set(body["checks"]) == {"database", "qdrant", "reranker", "langfuse"}
        assert body["checks"]["langfuse"] == {"ok": True, "enabled": False}
