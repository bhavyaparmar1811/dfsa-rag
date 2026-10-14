"""Eval tests reuse the retrieval mini corpus and the scripted Azure client."""

# The mini corpus (SQLite + in-memory Qdrant).
from tests.retrieval.conftest import corpus, index, loaded  # noqa: F401
