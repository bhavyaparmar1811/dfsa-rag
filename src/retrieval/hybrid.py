"""Candidate retrieval (recall): dense + BM25 over Qdrant, fused by RRF.

The question goes through the same EmbeddingClient and SparseEncoder as the
documents did, so query and chunk vectors share a space (the collection's
metadata is checked against the configuration when the retriever starts).

Both searches go to Qdrant in one batched request, with the same metadata
filter. Fusion is done here rather than in Qdrant so the trace keeps each
candidate's dense rank and BM25 rank, not just the fused score:

    rrf(c) = sum over retrievers of 1 / (rrf_k + rank(c))      rank is 1-based
"""

import time
from dataclasses import dataclass, field

from qdrant_client import models as qm

from src.config.settings import RetrievalSettings
from src.embeddings.client import EmbeddingClient
from src.embeddings.sparse import SparseEncoder
from src.retrieval.models import Candidate, Filters
from src.storage.qdrant_store import DENSE, SPARSE, QdrantStore


@dataclass
class SearchResult:
    candidates: list[Candidate]
    embed_tokens: int = 0
    embed_cost_usd: float = 0.0
    timings_ms: dict[str, float] = field(default_factory=dict)


class HybridSearcher:
    def __init__(
        self,
        store: QdrantStore,
        embedder: EmbeddingClient | None,
        sparse: SparseEncoder | None,
        cfg: RetrievalSettings,
    ):
        self.store = store
        self.embedder = embedder
        self.sparse = sparse
        self.cfg = cfg

    def search(self, question: str, filters: Filters | None = None) -> SearchResult:
        mode, k = self.cfg.mode, self.cfg.candidate_k
        flt = (filters or Filters()).to_qdrant()
        result = SearchResult(candidates=[])
        requests: list[tuple[str, qm.QueryRequest]] = []

        if mode in ("dense", "hybrid"):
            start = time.perf_counter()
            embedded = self.embedder.embed_query(question)
            result.timings_ms["embed"] = (time.perf_counter() - start) * 1000
            result.embed_tokens, result.embed_cost_usd = embedded.prompt_tokens, embedded.cost_usd
            requests.append(("dense", _request(embedded.vectors[0], DENSE, flt, k)))
        if mode in ("sparse", "hybrid"):
            start = time.perf_counter()
            vector = self.sparse.encode_query(question)
            result.timings_ms["sparse_encode"] = (time.perf_counter() - start) * 1000
            if len(vector):  # a question of stopwords only has no BM25 terms
                sparse = qm.SparseVector(indices=vector.indices, values=vector.values)
                requests.append(("sparse", _request(sparse, SPARSE, flt, k)))

        start = time.perf_counter()
        responses = (
            self.store.client.query_batch_points(self.store.collection, [r for _, r in requests])
            if requests
            else []
        )
        result.timings_ms["search"] = (time.perf_counter() - start) * 1000

        by_id: dict[str, Candidate] = {}
        for (name, _), response in zip(requests, responses, strict=True):
            for rank, point in enumerate(response.points, 1):
                pid = str(point.id)
                c = by_id.setdefault(pid, Candidate(pid, point.payload, source="search"))
                setattr(c, f"{name}_rank", rank)
                setattr(c, f"{name}_score", point.score)
                c.fused_score = (c.fused_score or 0.0) + 1.0 / (self.cfg.rrf_k + rank)

        if mode != "hybrid":  # one retriever: keep its own score, not 1/(k + rank)
            for c in by_id.values():
                c.fused_score = c.dense_score if mode == "dense" else c.sparse_score
        result.candidates = sorted(by_id.values(), key=lambda c: -c.fused_score)[:k]
        return result


def _request(vector, using: str, flt: qm.Filter | None, k: int) -> qm.QueryRequest:
    return qm.QueryRequest(query=vector, using=using, filter=flt, limit=k, with_payload=True)
