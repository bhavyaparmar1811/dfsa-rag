"""The Qdrant collection of cached questions, for the semantic answer cache.

One dense point per cache entry, with the same id as its `cache_entries` row;
SQLite holds the answer, Qdrant only finds the nearest cached questions. The
vectors come from the same EmbeddingClient as the chunks and queries, and the
collection records the embedding fingerprint: a mismatch fails loudly, as it
does for the chunk collection.

Qdrant has no TTL, so `expires_at` (epoch seconds) is a payload field that every
search filters on; `scripts/cache.py purge` deletes expired and stale points.
"""

import logging
from collections.abc import Iterable

from qdrant_client import QdrantClient, models

from src.config.settings import (
    CacheSettings,
    EmbeddingSettings,
    QdrantSettings,
    cache_collection_name,
)
from src.embeddings.client import EmbeddingModelMismatch
from src.storage.qdrant_store import DENSE, make_client

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


class CacheIndex:
    def __init__(self, client: QdrantClient, collection: str, embedding: EmbeddingSettings):
        self.client = client
        self.collection = collection
        self.embedding = embedding

    @classmethod
    def from_settings(
        cls,
        cache: CacheSettings,
        embedding: EmbeddingSettings,
        client: QdrantClient | None = None,
    ) -> "CacheIndex":
        return cls(
            client or make_client(QdrantSettings()),
            cache_collection_name(cache, embedding),
            embedding,
        )

    @property
    def expected_metadata(self) -> dict:
        return {
            "fingerprint": self.embedding.fingerprint,
            "embedding_model": self.embedding.model,
            "dimensions": self.embedding.dimensions,
            "schema_version": SCHEMA_VERSION,
            "purpose": "semantic answer cache",
        }

    def ensure_collection(self) -> bool:
        """Create the collection if it is missing (returns True), otherwise check it."""
        if self.client.collection_exists(self.collection):
            self.check()
            return False
        self.client.create_collection(
            self.collection,
            vectors_config={
                DENSE: models.VectorParams(
                    size=self.embedding.dimensions, distance=models.Distance.COSINE
                )
            },
            metadata=self.expected_metadata,
        )
        self.client.create_payload_index(
            self.collection, "namespace", models.PayloadSchemaType.KEYWORD, wait=True
        )
        self.client.create_payload_index(
            self.collection, "expires_at", models.PayloadSchemaType.INTEGER, wait=True
        )
        logger.info("Created cache collection %s", self.collection)
        return True

    def check(self) -> None:
        meta = self.client.get_collection(self.collection).config.metadata or {}
        if meta.get("fingerprint") != self.embedding.fingerprint:
            raise EmbeddingModelMismatch(
                f"Cache collection {self.collection} was built with "
                f"'{meta.get('fingerprint', 'an unknown model')}' but config is "
                f"'{self.embedding.fingerprint}'."
            )

    def add(self, entry_id: str, vector: list[float], namespace: str, expires_at: int) -> None:
        self.client.upsert(
            self.collection,
            points=[
                models.PointStruct(
                    id=entry_id,
                    vector={DENSE: vector},
                    payload={"namespace": namespace, "expires_at": expires_at},
                )
            ],
            wait=True,
        )

    def search(
        self, vector: list[float], namespace: str, now: int, k: int
    ) -> list[tuple[str, float]]:
        """The k nearest unexpired cached questions in a namespace: [(entry id, cosine)]."""
        flt = models.Filter(
            must=[
                models.FieldCondition(key="namespace", match=models.MatchValue(value=namespace)),
                models.FieldCondition(key="expires_at", range=models.Range(gt=now)),
            ]
        )
        response = self.client.query_points(
            self.collection, query=vector, using=DENSE, query_filter=flt, limit=k
        )
        return [(str(p.id), p.score) for p in response.points]

    def delete(self, ids: Iterable[str]) -> int:
        ids = list(ids)
        if ids:
            self.client.delete(self.collection, models.PointIdsList(points=ids), wait=True)
        return len(ids)

    def ids(self) -> set[str]:
        out, offset = set(), None
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=1000, offset=offset, with_payload=False
            )
            out |= {str(p.id) for p in points}
            if offset is None:
                return out

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count
