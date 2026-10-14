"""Named retrieval configurations the eval compares.

Each is a set of overrides on `RetrievalSettings()` (which reads `.env`), so a
config differs from production only in what it names. `hybrid_rerank` is the
production setup.

    dense / sparse / hybrid      candidate retrieval alone, no reranker (no confidence)
    hybrid_rerank                + bge-reranker-v2-m3 (production)
    hybrid_rerank_base           + bge-reranker-base (the CLAUDE.md CPU fallback)
    hybrid_rerank_minilm         + ms-marco-MiniLM-L6-v2 (the smallest)
    parent_none / parent_window  production retrieval, another context policy:
                                 only context_recall (and e2e answers) can differ
"""

from src.config.settings import RetrievalSettings

PRODUCTION_CONFIG = "hybrid_rerank"

CONFIGS: dict[str, dict] = {
    "dense": {"mode": "dense", "rerank": False},
    "sparse": {"mode": "sparse", "rerank": False},
    "hybrid": {"mode": "hybrid", "rerank": False},
    "hybrid_rerank": {},
    "hybrid_rerank_base": {"reranker_model": "BAAI/bge-reranker-base"},
    "hybrid_rerank_minilm": {"reranker_model": "cross-encoder/ms-marco-MiniLM-L6-v2"},
    "parent_none": {"parent_mode": "none"},
    "parent_window": {"parent_mode": "window"},
}


def settings_for(name: str, base: RetrievalSettings | None = None) -> RetrievalSettings:
    if name not in CONFIGS:
        raise KeyError(f"unknown eval config {name!r}; known: {', '.join(CONFIGS)}")
    base = base or RetrievalSettings()
    return base.model_copy(update=CONFIGS[name])
