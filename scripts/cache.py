"""Inspect and clean the semantic answer cache (src/chat/cache.py).

    uv run python -m scripts.cache stats     # entries, hits served, points
    uv run python -m scripts.cache purge     # expired + stale-namespace entries, orphan points
    uv run python -m scripts.cache clear     # every entry and point

Stale entries are never served (a lookup only searches the current namespace),
so purging is housekeeping. `purge` builds the retriever (it loads the reranker)
to know the current namespace: the reranker revision is part of it.
"""

import argparse
import json
import sys

from src.chat.cache import SemanticCache, purge
from src.config.settings import AzureOpenAISettings, CacheSettings, EmbeddingSettings
from src.storage.cache_index import CacheIndex
from src.storage.db import assert_migrated, get_engine, session_scope
from src.storage.models import utcnow
from src.storage.repositories import CacheRepository


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["stats", "purge", "clear"])
    args = parser.parse_args(argv)

    engine = get_engine()
    assert_migrated(engine)
    cfg = CacheSettings()
    index = CacheIndex.from_settings(cfg, EmbeddingSettings())
    index.ensure_collection()

    if args.command == "stats":
        with session_scope(engine) as s:
            stats = CacheRepository(s).stats(utcnow())
        stats |= {
            "enabled": cfg.enabled,
            "similarity_threshold": cfg.similarity_threshold,
            "collection": index.collection,
            "points": index.count(),
        }
        print(json.dumps(stats, indent=2))
    elif args.command == "clear":
        print(json.dumps(purge(engine, index, current=None), indent=2))
    else:  # imported here: the retriever pulls in torch and the reranker
        from src.config.settings import RetrievalSettings
        from src.prompts.registry import PromptRegistry
        from src.retrieval.pipeline import Retriever

        retriever = Retriever.build(RetrievalSettings())
        cache = SemanticCache.build(
            engine, retriever, PromptRegistry.from_settings(), AzureOpenAISettings().deployment
        )
        print(json.dumps(cache.purge(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
