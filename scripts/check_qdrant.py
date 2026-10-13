"""Smoke test: is the Qdrant server in `.env` reachable, and does the key work?

Prints the server version, whether the API key is enforced, and each
collection with its point count and the embedding model it was built with.
Exits 0 on success, 1 on failure.

    uv run python -m scripts.check_qdrant
"""

import sys

import httpx
from pydantic import ValidationError
from qdrant_client.http.exceptions import UnexpectedResponse

from src.config.settings import QdrantSettings
from src.storage.qdrant_store import make_client


def main() -> int:
    # Load only the Qdrant section, so missing OpenAI/Azure keys don't block this check.
    try:
        cfg = QdrantSettings()
    except ValidationError as e:
        print(f"FAIL: Qdrant settings missing or invalid in .env:\n{e}")
        return 1

    key = cfg.api_key.get_secret_value() if cfg.api_key else None
    print(f"URL:          {cfg.url}")
    print(f"API key:      {'set' if key else 'NOT SET'}")

    try:
        info = httpx.get(cfg.url, timeout=5).json()
        print(f"Server:       Qdrant {info.get('version', '?')}")
        # Without the key, a protected server must refuse to list collections.
        status = httpx.get(f"{cfg.url}/collections", timeout=5).status_code
        print(f"Without key:  HTTP {status} ({'protected' if status == 401 else 'NOT protected'})")
    except httpx.HTTPError as e:
        print(f"FAIL: cannot reach {cfg.url}: {e}. Is the container running?")
        return 1

    client = make_client(cfg)
    try:
        collections = client.get_collections().collections
    except UnexpectedResponse as e:
        print(f"FAIL: HTTP {e.status_code}. Check QDRANT_API_KEY in .env matches the server.")
        return 1

    print(f"Collections:  {len(collections)}")
    for c in collections:
        detail = client.get_collection(c.name)
        meta = detail.config.metadata or {}
        print(
            f"  - {c.name}: {detail.points_count} points, status {detail.status.value}, "
            f"built with {meta.get('fingerprint', 'unknown model')} + "
            f"{meta.get('sparse_fingerprint', 'no sparse encoder')}, "
            f"schema v{meta.get('schema_version', '?')}"
        )
    if status != 401:
        print("WARN: the server accepts requests without an API key.")
    print("OK: Qdrant is reachable and the API key works.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
