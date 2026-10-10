"""Deterministic chunk ids.

The same chunk in the same document version always gets the same id, so a
re-run updates points in Qdrant instead of duplicating them. A new document
version gets new ids, and the old version's chunks are removed.
"""

import uuid

# Fixed namespace for this project. Never change it: every stored id derives from it.
NAMESPACE = uuid.UUID("6b1d0c52-6f0e-4c3e-9a51-0d5f3a9c2e7b")


def chunk_id(doc_id: str, version: str, key: str) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{doc_id}|{version}|{key}"))
