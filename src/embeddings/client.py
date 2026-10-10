"""The one embedding client, used by ingestion AND the query path.

Documents and queries go through the same `_embed` call with the same
`EmbeddingSettings`, so they always land in the same vector space. Any vector
of the wrong length, or a collection built with a different model, fails
loudly instead of silently degrading retrieval.
"""

import logging
import time
from dataclasses import dataclass

import tiktoken
from openai import OpenAI

from src.config.settings import EmbeddingSettings

logger = logging.getLogger(__name__)


class EmbeddingModelMismatch(Exception):
    """Stored vectors were built with a different model or dimension than configured."""


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: list[list[float]]
    model: str
    dimensions: int
    fingerprint: str
    prompt_tokens: int
    cost_usd: float
    latency_ms: float


def assert_compatible(stored_fingerprint: str, cfg: EmbeddingSettings) -> None:
    """Raise if `stored_fingerprint` (from a collection or chunk) isn't the configured one."""
    if stored_fingerprint != cfg.fingerprint:
        raise EmbeddingModelMismatch(
            f"Stored vectors use '{stored_fingerprint}' but config is '{cfg.fingerprint}'. "
            "Re-embed the corpus into a new collection or fix EMBEDDING_MODEL/DIMENSIONS."
        )


class EmbeddingClient:
    def __init__(self, cfg: EmbeddingSettings, client: OpenAI | None = None):
        self.cfg = cfg
        self._client = client or OpenAI(api_key=cfg.api_key.get_secret_value())
        try:
            self._encoding = tiktoken.encoding_for_model(cfg.model)
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")

    def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        return self._embed(texts)

    def embed_query(self, text: str) -> EmbeddingResult:
        return self._embed([text])

    def _embed(self, texts: list[str]) -> EmbeddingResult:
        self._validate_inputs(texts)

        start = time.perf_counter()
        vectors: list[list[float]] = []
        prompt_tokens = 0
        for i in range(0, len(texts), self.cfg.batch_size):
            batch = texts[i : i + self.cfg.batch_size]
            resp = self._client.embeddings.create(
                model=self.cfg.model, input=batch, dimensions=self.cfg.dimensions
            )
            for item in sorted(resp.data, key=lambda d: d.index):
                if len(item.embedding) != self.cfg.dimensions:
                    raise EmbeddingModelMismatch(
                        f"Expected {self.cfg.dimensions}-dim vectors from {self.cfg.model}, "
                        f"got {len(item.embedding)}."
                    )
                vectors.append(item.embedding)
            prompt_tokens += resp.usage.prompt_tokens
        latency_ms = (time.perf_counter() - start) * 1000

        if len(vectors) != len(texts):
            raise RuntimeError(f"Sent {len(texts)} inputs but got {len(vectors)} vectors back.")

        cost_usd = prompt_tokens * self.cfg.cost_per_1m / 1_000_000
        logger.info(
            "embedded %d inputs with %s: %d tokens, $%.6f, %.0f ms",
            len(texts),
            self.cfg.fingerprint,
            prompt_tokens,
            cost_usd,
            latency_ms,
        )
        return EmbeddingResult(
            vectors=vectors,
            model=self.cfg.model,
            dimensions=self.cfg.dimensions,
            fingerprint=self.cfg.fingerprint,
            prompt_tokens=prompt_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )

    def _validate_inputs(self, texts: list[str]) -> None:
        if not texts:
            raise ValueError("No texts to embed.")
        for i, text in enumerate(texts):
            if not text.strip():
                raise ValueError(f"Input {i} is empty.")
            n_tokens = len(self._encoding.encode(text))
            if n_tokens > self.cfg.max_input_tokens:
                raise ValueError(
                    f"Input {i} has {n_tokens} tokens, over the {self.cfg.max_input_tokens} "
                    f"limit for {self.cfg.model}. Split it before embedding."
                )
