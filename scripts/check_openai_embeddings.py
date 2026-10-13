"""Smoke test: does the OpenAI embeddings key in `.env` work?

Embeds a sample query and a sample rule-like text with the project's
EmbeddingClient, then reports model, vector size, tokens, cost, latency, and
the cosine similarity between the two. Exits 0 on success, 1 on failure.

    uv run python -m scripts.check_openai_embeddings
"""

import math
import sys

from openai import APIConnectionError, APIStatusError
from pydantic import ValidationError

from src.config.settings import EmbeddingSettings
from src.embeddings import EmbeddingClient

SAMPLE_QUERY = "Do we need suitability assessments before recommending an investment?"
SAMPLE_RULE = (
    "Module: COB > Chapter 3 > Suitability. An Authorised Firm must not recommend a "
    "financial product to a Client unless it has assessed that the product is suitable "
    "for that Client."
)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def main() -> int:
    # Load only the embedding section, so a missing Azure config doesn't block this check.
    try:
        cfg = EmbeddingSettings()
    except ValidationError as e:
        print(f"FAIL: embedding settings missing or invalid in .env:\n{e}")
        return 1
    if cfg.api_key is None:
        print("FAIL: OPENAI_API_KEY is not set in .env (embeddings use the OpenAI API, not Azure).")
        return 1

    print(f"Model:       {cfg.model}")
    print(f"Fingerprint: {cfg.fingerprint}")

    client = EmbeddingClient(cfg)
    try:
        query = client.embed_query(SAMPLE_QUERY)
        docs = client.embed_documents([SAMPLE_RULE])
    except APIStatusError as e:
        print(f"FAIL: HTTP {e.status_code}: {e.message}")
        return 1
    except APIConnectionError as e:
        print(f"FAIL: could not connect to OpenAI: {e}")
        return 1

    tokens = query.prompt_tokens + docs.prompt_tokens
    cost = query.cost_usd + docs.cost_usd
    print(f"Vector size: {len(query.vectors[0])}")
    print(f"Tokens:      {tokens} (${cost:.8f})")
    print(f"Latency:     {query.latency_ms:.0f} ms query / {docs.latency_ms:.0f} ms document")
    print(f"Similarity:  {cosine(query.vectors[0], docs.vectors[0]):.4f} (query vs sample rule)")
    print("OK: OpenAI embeddings are working.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
