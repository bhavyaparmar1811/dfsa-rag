from types import SimpleNamespace

import pytest

from src.config.settings import EmbeddingSettings
from src.embeddings import EmbeddingClient, EmbeddingModelMismatch, assert_compatible


class FakeEmbeddings:
    """Stands in for `OpenAI().embeddings`: records calls, returns stub vectors."""

    def __init__(self, dims: int, tokens_per_input: int = 5):
        self.dims = dims
        self.tokens_per_input = tokens_per_input
        self.calls: list[dict] = []

    def create(self, *, model, input, dimensions):
        self.calls.append({"model": model, "input": input, "dimensions": dimensions})
        # Returned out of order, to check the client sorts by index.
        data = [
            SimpleNamespace(
                index=i, embedding=[float(len(self.calls)), float(i)] + [0.0] * (self.dims - 2)
            )
            for i in reversed(range(len(input)))
        ]
        usage = SimpleNamespace(prompt_tokens=self.tokens_per_input * len(input))
        return SimpleNamespace(data=data, usage=usage)


@pytest.fixture
def cfg(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    return EmbeddingSettings()


def make_client(cfg, dims=None):
    fake = FakeEmbeddings(dims or cfg.dimensions)
    return EmbeddingClient(cfg, client=SimpleNamespace(embeddings=fake)), fake


def test_batches_and_preserves_order(cfg):
    cfg.batch_size = 100
    client, fake = make_client(cfg)
    texts = [f"rule {i}" for i in range(250)]

    result = client.embed_documents(texts)

    assert [len(c["input"]) for c in fake.calls] == [100, 100, 50]
    assert len(result.vectors) == 250
    # Vector [batch number, position in batch] maps back to input order.
    assert result.vectors[0][:2] == [1.0, 0.0]
    assert result.vectors[99][:2] == [1.0, 99.0]
    assert result.vectors[249][:2] == [3.0, 49.0]


def test_model_and_dimensions_come_from_settings(cfg):
    client, fake = make_client(cfg)
    result = client.embed_query("what is a Client?")

    assert fake.calls[0]["model"] == cfg.model
    assert fake.calls[0]["dimensions"] == cfg.dimensions
    assert result.fingerprint == cfg.fingerprint


def test_wrong_vector_length_raises(cfg):
    client, _ = make_client(cfg, dims=cfg.dimensions - 1)
    with pytest.raises(EmbeddingModelMismatch):
        client.embed_query("what is a Client?")


def test_empty_input_raises(cfg):
    client, fake = make_client(cfg)
    with pytest.raises(ValueError):
        client.embed_documents(["rule text", "   "])
    assert fake.calls == []


def test_over_token_limit_raises(cfg):
    cfg.max_input_tokens = 10
    client, fake = make_client(cfg)
    with pytest.raises(ValueError, match="tokens"):
        client.embed_query("word " * 50)
    assert fake.calls == []


def test_tokens_and_cost(cfg):
    cfg.batch_size = 2
    client, _ = make_client(cfg)
    result = client.embed_documents(["a", "b", "c"])

    assert result.prompt_tokens == 15
    assert result.cost_usd == pytest.approx(15 * cfg.cost_per_1m / 1_000_000)


def test_assert_compatible(cfg):
    assert_compatible(cfg.fingerprint, cfg)
    with pytest.raises(EmbeddingModelMismatch):
        assert_compatible("text-embedding-3-large-3072", cfg)
