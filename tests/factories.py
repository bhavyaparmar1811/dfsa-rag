"""Small, valid chunks and manifest entries, and fakes for the index, for
storage and ingestion tests."""

import hashlib

from src.config.settings import EmbeddingSettings, QdrantSettings
from src.embeddings.client import EmbeddingResult
from src.embeddings.sparse import SparseVector
from src.ingestion.ids import chunk_id
from src.ingestion.manifest import ManifestEntry
from src.ingestion.models import Chunk, ChunkType

FINGERPRINT = "text-embedding-3-small-1536"


def entry(doc_id: str = "dfsa-rulebook-cob", module: str = "COB", version: str = "51"):
    law = doc_id.startswith("dfsa-laws-")
    return ManifestEntry(
        id=doc_id, regulator="DFSA", rulebook="DIFC Laws" if law else "DFSA Rulebook",
        module_code=module, module_name=f"{module} Module", version=version,
        effective_date="2026-07", source_url="https://example.com",
        file_path=f"data/raw/{module}.pdf", downloaded_on="2026-10-01", file_hash="a" * 64,
    )  # fmt: skip


def chunk(
    key: str = "COB 3.3.2",
    *,
    text: str | None = None,
    kind: ChunkType = ChunkType.RULE,
    doc_id: str = "dfsa-rulebook-cob",
    version: str = "51",
    order: int = 0,
    parent_key: str | None = "COB 3.3",
    **extra,
) -> Chunk:
    module = key.split()[0]
    return Chunk(
        chunk_id=chunk_id(doc_id, version, key), key=key, citation=key.split("#")[0],
        chunk_type=kind, binding=kind != ChunkType.GUIDANCE, doc_id=doc_id, regulator="DFSA",
        module=module, version=version, effective_date="2026-07", source_file=f"{module}.pdf",
        section="3.3", order=order, parent_key=None if kind == ChunkType.SECTION else parent_key,
        heading_path=[module, "Section 3.3"], header_text=f"{module} > Section 3.3 > {key}",
        text=text if text is not None else f"{key} text", page_start=1, page_end=1,
        token_count=10, **extra,
    )  # fmt: skip


# --- fakes for the index: no network, no model downloads ---------------------

DIMS = 8


def embedding_settings(**kw) -> EmbeddingSettings:
    return EmbeddingSettings(_env_file=None, dimensions=DIMS, **kw)


def qdrant_settings(**kw) -> QdrantSettings:
    return QdrantSettings(_env_file=None, **kw)


def _word_ids(text: str) -> list[int]:
    return sorted({int(hashlib.md5(w.encode()).hexdigest()[:7], 16) for w in text.lower().split()})


class FakeEmbedder:
    """Stands in for EmbeddingClient: deterministic vectors, records every call."""

    def __init__(self, cfg: EmbeddingSettings | None = None, tokens_per_input: int = 7):
        self.cfg = cfg or embedding_settings()
        self.tokens_per_input = tokens_per_input
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vectors.append([b / 255 + 0.01 for b in digest[: self.cfg.dimensions]])
        tokens = self.tokens_per_input * len(texts)
        return EmbeddingResult(
            vectors=vectors, model=self.cfg.model, dimensions=self.cfg.dimensions,
            fingerprint=self.cfg.fingerprint, prompt_tokens=tokens,
            cost_usd=tokens * self.cfg.cost_per_1m / 1_000_000, latency_ms=0.0,
        )  # fmt: skip

    @property
    def embedded_texts(self) -> int:
        return sum(len(c) for c in self.calls)


class FakeSparse:
    """Stands in for SparseEncoder: one term per distinct word, weight 1."""

    def __init__(self, cfg: QdrantSettings | None = None):
        self.cfg = cfg or qdrant_settings()
        self.fingerprint = self.cfg.sparse_fingerprint

    def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        return [self.encode_query(t) for t in texts]

    def encode_query(self, text: str) -> SparseVector:
        ids = _word_ids(text)
        return SparseVector(ids, [1.0] * len(ids))

    def token_count(self, texts) -> int:
        return sum(len(t.split()) for t in texts)
