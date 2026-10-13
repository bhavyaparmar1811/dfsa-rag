"""The Qdrant collection: the search index over the embedded chunks.

SQLite is the system of record; Qdrant holds one point per embedded chunk,
with the same id as its `chunks` row. Each point has two named vectors:

    dense   text-embedding-3-small, cosine        (meaning)
    bm25    fastembed Qdrant/bm25, IDF in Qdrant   (exact words: "Market Counterparty")

and a payload with the chunk's text and metadata, so a search result can go
straight to the reranker and payload filters ("AML only", "binding only") run
inside Qdrant.

The collection records what built it (embedding fingerprint, sparse
fingerprint, schema version) in its metadata. `ensure_collection()` creates it,
or checks that an existing one matches the configuration and raises if not:
vectors from two models must never share a collection.
"""

import hashlib
import json
import logging
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from urllib.parse import urlparse

from qdrant_client import QdrantClient, models

from src.config.settings import EmbeddingSettings, QdrantSettings, collection_name
from src.embeddings.client import EmbeddingModelMismatch
from src.embeddings.sparse import SparseVector
from src.ingestion.models import Chunk

logger = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "bm25"
# Bump when the payload layout or the vector setup changes in a way that needs a rebuild.
SCHEMA_VERSION = 1

KEYWORD_FIELDS = (
    "regulator", "doc_type", "module", "doc_id", "chunk_type", "citation", "key",
    "parent_key", "term", "defined_terms",
)  # fmt: skip
BOOL_FIELDS = ("binding", "scope_rule")


class CollectionMismatch(Exception):
    """The collection was built with a different configuration than the current one."""


@dataclass(frozen=True)
class PointState:
    """What change detection needs to know about a stored point."""

    content_hash: str | None
    payload_hash: str | None


@dataclass(frozen=True)
class Point:
    id: str
    dense: list[float]
    sparse: SparseVector
    payload: dict


def build_payload(chunk: Chunk, doc_type: str, content_hash: str, fingerprint: str) -> dict:
    """The payload of a chunk's point. `payload_hash` covers everything else in it,
    so a metadata-only change (pages, references, terms) is detected and rewritten
    without re-embedding."""
    payload = {
        # identity and filters
        "key": chunk.key,
        "citation": chunk.citation,
        "chunk_type": chunk.chunk_type.value,
        "binding": chunk.binding,
        "regulator": chunk.regulator,
        "doc_type": doc_type,
        "module": chunk.module,
        "doc_id": chunk.doc_id,
        "version": chunk.version,
        "effective_date": chunk.effective_date,
        # structure
        "section": chunk.section,
        "parent_key": chunk.parent_key,
        "part_of": chunk.part_of,
        "part": chunk.part,
        "guidance_for": chunk.guidance_for,
        "rule_number": chunk.rule_number,
        "term": chunk.term,
        "scope_rule": chunk.scope_rule,
        # links
        "defined_terms": chunk.defined_terms,
        "cross_refs": chunk.cross_refs,
        # content
        "header_text": chunk.header_text,
        "text": chunk.text,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        # provenance
        "content_hash": content_hash,
        "embedding_fingerprint": fingerprint,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    payload["payload_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


def make_client(cfg: QdrantSettings) -> QdrantClient:
    key = cfg.api_key.get_secret_value() if cfg.api_key else None
    with warnings.catch_warnings():
        # The key over plain HTTP is fine on localhost (the port is bound to 127.0.0.1).
        if urlparse(cfg.url).hostname in ("localhost", "127.0.0.1", "::1"):
            warnings.filterwarnings("ignore", message="Api key is used with an insecure connection")
        return QdrantClient(url=cfg.url, api_key=key, timeout=cfg.timeout_s)


class QdrantStore:
    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        embedding: EmbeddingSettings,
        sparse_fingerprint: str,
        upsert_batch_size: int = 100,
    ):
        self.client = client
        self.collection = collection
        self.embedding = embedding
        self.sparse_fingerprint = sparse_fingerprint
        self.upsert_batch_size = upsert_batch_size

    @classmethod
    def from_settings(
        cls,
        qdrant: QdrantSettings,
        embedding: EmbeddingSettings,
        client: QdrantClient | None = None,
    ) -> "QdrantStore":
        return cls(
            client or make_client(qdrant),
            collection_name(qdrant, embedding),
            embedding,
            qdrant.sparse_fingerprint,
            qdrant.upsert_batch_size,
        )

    @property
    def expected_metadata(self) -> dict:
        return {
            "fingerprint": self.embedding.fingerprint,
            "embedding_model": self.embedding.model,
            "dimensions": self.embedding.dimensions,
            "sparse_fingerprint": self.sparse_fingerprint,
            "schema_version": SCHEMA_VERSION,
        }

    # --- collection ----------------------------------------------------------

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def ensure_collection(self) -> bool:
        """Create the collection if it is missing (returns True), otherwise check it
        matches the configuration. Payload indexes are (re)created either way."""
        created = False
        if self.exists():
            self.check()
        else:
            self.client.create_collection(
                self.collection,
                vectors_config={
                    DENSE: models.VectorParams(
                        size=self.embedding.dimensions, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
                metadata=self.expected_metadata,
            )
            logger.info("Created collection %s", self.collection)
            created = True
        self._ensure_payload_indexes()
        return created

    def check(self) -> None:
        """Raise if the existing collection wasn't built with the current configuration."""
        info = self.client.get_collection(self.collection)
        meta = info.config.metadata or {}
        if meta.get("fingerprint") != self.embedding.fingerprint:
            raise EmbeddingModelMismatch(
                f"Collection {self.collection} was built with "
                f"'{meta.get('fingerprint', 'an unknown model')}' but config is "
                f"'{self.embedding.fingerprint}'. Re-embed into a new collection."
            )
        problems = []
        if meta.get("sparse_fingerprint") != self.sparse_fingerprint:
            problems.append(
                f"sparse encoder is '{meta.get('sparse_fingerprint')}', "
                f"config is '{self.sparse_fingerprint}'"
            )
        if meta.get("schema_version") != SCHEMA_VERSION:
            problems.append(
                f"schema version is {meta.get('schema_version')}, code expects {SCHEMA_VERSION}"
            )
        dense = (info.config.params.vectors or {}).get(DENSE)
        if dense is None or dense.size != self.embedding.dimensions:
            problems.append(
                f"dense vector '{DENSE}' missing or not {self.embedding.dimensions}-dim"
            )
        if SPARSE not in (info.config.params.sparse_vectors or {}):
            problems.append(f"sparse vector '{SPARSE}' missing")
        if problems:
            raise CollectionMismatch(
                f"Collection {self.collection} doesn't match the configuration: "
                + "; ".join(problems)
                + ". Delete the collection and re-run ingest to rebuild it."
            )

    def _ensure_payload_indexes(self) -> None:
        existing = set(self.client.get_collection(self.collection).payload_schema or {})
        wanted = {f: models.PayloadSchemaType.KEYWORD for f in KEYWORD_FIELDS}
        wanted |= {f: models.PayloadSchemaType.BOOL for f in BOOL_FIELDS}
        for name, schema in wanted.items():
            if name not in existing:
                self.client.create_payload_index(self.collection, name, schema, wait=True)

    def payload_indexes(self) -> dict[str, str]:
        schema = self.client.get_collection(self.collection).payload_schema or {}
        return {name: info.data_type.value for name, info in schema.items()}

    # --- points --------------------------------------------------------------

    def upsert(self, points: list[Point]) -> None:
        for i in range(0, len(points), self.upsert_batch_size):
            batch = points[i : i + self.upsert_batch_size]
            self.client.upsert(
                self.collection,
                [
                    models.PointStruct(
                        id=p.id,
                        vector={
                            DENSE: p.dense,
                            SPARSE: models.SparseVector(
                                indices=p.sparse.indices, values=p.sparse.values
                            ),
                        },
                        payload=p.payload,
                    )
                    for p in batch
                ],
                wait=True,
            )

    def overwrite_payloads(self, payloads: dict[str, dict]) -> None:
        """Replace the payload of existing points ({id: payload}); vectors are untouched."""
        ops = [
            models.OverwritePayloadOperation(
                overwrite_payload=models.SetPayload(payload=payload, points=[point_id])
            )
            for point_id, payload in payloads.items()
        ]
        for i in range(0, len(ops), self.upsert_batch_size):
            self.client.batch_update_points(
                self.collection, ops[i : i + self.upsert_batch_size], wait=True
            )

    def delete(self, ids: Iterable[str]) -> int:
        ids = list(ids)
        if ids:
            self.client.delete(
                self.collection, models.PointIdsList(points=ids), wait=True
            )  # fmt: skip
        return len(ids)

    def delete_document(self, doc_id: str) -> None:
        if self.exists():
            self.client.delete(self.collection, _match("doc_id", doc_id), wait=True)

    def count(self, doc_id: str | None = None) -> int:
        if not self.exists():
            return 0
        flt = _match("doc_id", doc_id) if doc_id else None
        return self.client.count(self.collection, count_filter=flt, exact=True).count

    def point_states(self, doc_id: str | None = None) -> dict[str, PointState]:
        """{point id: PointState} for one document (or the whole collection)."""
        if not self.exists():
            return {}
        states, offset = {}, None
        flt = _match("doc_id", doc_id) if doc_id else None
        while True:
            records, offset = self.client.scroll(
                self.collection,
                scroll_filter=flt,
                with_payload=["content_hash", "payload_hash"],
                with_vectors=False,
                limit=1000,
                offset=offset,
            )
            for r in records:
                p = r.payload or {}
                states[str(r.id)] = PointState(p.get("content_hash"), p.get("payload_hash"))
            if offset is None:
                return states

    def scroll_hashes(self) -> dict[str, str | None]:
        """{point id: content_hash} for the whole collection, for reconcile."""
        return {pid: s.content_hash for pid, s in self.point_states().items()}

    def get_by_citation(self, citation: str, limit: int = 20) -> list[models.Record]:
        """Exact lookup through the `citation` payload index ("COB 3.3.2"): the rule,
        its parts if it was split, and the Guidance that cites it."""
        records, _ = self.client.scroll(
            self.collection, scroll_filter=_match("citation", citation), limit=limit,
            with_payload=True, with_vectors=False,
        )  # fmt: skip
        return sorted(records, key=lambda r: (r.payload["chunk_type"] != "rule", r.payload["key"]))

    def retrieve(self, ids: list[str], with_vectors: bool = False) -> list[models.Record]:
        return self.client.retrieve(
            self.collection, ids, with_payload=True, with_vectors=with_vectors
        )


def _match(field: str, value: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key=field, match=models.MatchValue(value=value))]
    )
