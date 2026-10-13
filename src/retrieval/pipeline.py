"""retrieve(question, filters) -> RetrievalResult: the whole retrieval path.

    1. citations   exact provisions named in the question (citations.py)
    2. search      dense + BM25 candidates, fused by RRF (hybrid.py)
    3. rerank      cross-encoder over citation hits + candidates (rerank.py)
    4. select      the top N; exact citation hits stay in, ahead of the rest
    5. context     scope rules, Guidance, sections, definitions (context.py)

Each stage is timed, and everything a trace needs is in the result. The
Retriever is built once (it loads the reranker and checks the collection was
built with the configured embedding model and BM25 settings) and then serves
many questions.
"""

import logging
import time
from dataclasses import dataclass

from sqlalchemy import Engine

from src.config.settings import EmbeddingSettings, QdrantSettings, RetrievalSettings
from src.embeddings.client import EmbeddingClient
from src.embeddings.sparse import SparseEncoder
from src.ingestion.chunkers.law import law_name
from src.ingestion.manifest import load_manifest
from src.retrieval.citations import CitationRouter
from src.retrieval.context import ContextAssembler
from src.retrieval.hybrid import HybridSearcher
from src.retrieval.models import Candidate, CitationLookup, Filters, RetrievalResult
from src.retrieval.rerank import CrossEncoderModel, Reranker, ScoringModel
from src.storage.db import get_engine
from src.storage.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


@dataclass
class Retriever:
    cfg: RetrievalSettings
    citations: CitationRouter
    searcher: HybridSearcher
    reranker: Reranker | None
    assembler: ContextAssembler

    @classmethod
    def build(
        cls,
        cfg: RetrievalSettings | None = None,
        *,
        engine: Engine | None = None,
        store: QdrantStore | None = None,
        embedder: EmbeddingClient | None = None,
        sparse: SparseEncoder | None = None,
        scorer: ScoringModel | None = None,
    ) -> "Retriever":
        """Wire up the real components; tests pass fakes for any of them."""
        cfg = cfg or RetrievalSettings()
        engine = engine or get_engine()
        qdrant, emb = QdrantSettings(), EmbeddingSettings()
        store = store or QdrantStore.from_settings(qdrant, emb)
        store.check()  # fail now, not on the first question, if the index doesn't match
        if cfg.mode in ("dense", "hybrid") and embedder is None:
            embedder = EmbeddingClient(emb)
        if cfg.mode in ("sparse", "hybrid") and sparse is None:
            sparse = SparseEncoder(qdrant)
        if sparse is not None and sparse.fingerprint != store.sparse_fingerprint:
            raise ValueError(f"Sparse encoder {sparse.fingerprint} != {store.sparse_fingerprint}")
        reranker = None
        if cfg.rerank:
            reranker = Reranker(scorer or CrossEncoderModel(cfg), cfg.reranker_max_length)

        manifest = load_manifest()
        codes = {e.module_code for e in manifest if e.doc_type == "rulebook"}
        laws = [law_name(e) for e in manifest if e.doc_type == "law"]
        return cls(
            cfg=cfg,
            citations=CitationRouter(engine, store, codes, laws, cfg.max_citation_hits),
            searcher=HybridSearcher(store, embedder, sparse, cfg),
            reranker=reranker,
            assembler=ContextAssembler(engine, cfg),
        )

    def retrieve(self, question: str, filters: Filters | None = None) -> RetrievalResult:
        cfg, filters = self.cfg, filters or Filters()
        timings: dict[str, float] = {}
        start = time.perf_counter()

        t = time.perf_counter()
        lookup, exact = CitationLookup(), []
        if cfg.citation_lookup:
            lookup, exact = self.citations.lookup(question, filters)
        timings["citations"] = _ms(t)

        searched = self.searcher.search(question, filters)
        timings |= searched.timings_ms

        # Exact hits first, then search results not already among them.
        exact_ids = {c.id for c in exact}
        candidates = exact + [c for c in searched.candidates if c.id not in exact_ids]
        for c in exact:  # a citation hit also found by search keeps its search scores
            found = next((s for s in searched.candidates if s.id == c.id), None)
            if found is not None:
                c.dense_rank, c.dense_score = found.dense_rank, found.dense_score
                c.sparse_rank, c.sparse_score = found.sparse_rank, found.sparse_score
                c.fused_score = found.fused_score

        if self.reranker is not None:
            # The cross-encoder is the slow stage: it scores the exact hits plus the
            # best rerank_k search candidates; the rest stay in the trace, unscored.
            n = len(exact) + cfg.rerank_k
            t = time.perf_counter()
            candidates = self.reranker.rerank(question, candidates[:n]) + candidates[n:]
            timings["rerank"] = _ms(t)
        hits = _select(
            [c for c in candidates if self.reranker is None or c.rerank_score is not None],
            cfg.top_n,
        )

        t = time.perf_counter()
        context, dropped = self.assembler.assemble(question, hits)
        timings["context"] = _ms(t)
        timings["total"] = _ms(start)

        # The best rerank score among the hits (exact hits come first whatever their score).
        confidence = max((h.rerank_score for h in hits), default=None) if self.reranker else None
        confident = None
        if confidence is not None and cfg.refusal_threshold is not None:
            confident = confidence >= cfg.refusal_threshold
        result = RetrievalResult(
            question=question,
            filters=filters,
            config=_config(cfg) | self._runtime(),
            citations=lookup,
            candidates=candidates,
            hits=hits,
            context=context,
            confidence=confidence,
            confident=confident,
            timings_ms=timings,
            embed_tokens=searched.embed_tokens,
            embed_cost_usd=searched.embed_cost_usd,
            truncated=sum(c.truncated for c in candidates),
            dropped=dropped,
        )
        logger.info(
            "retrieve: %d citation + %d search candidates -> %s; confidence %s; %d blocks, "
            "%d tokens; %.0f ms", len(exact), len(searched.candidates),
            [h.key for h in hits], f"{confidence:.3f}" if confidence is not None else "n/a",
            len(context), result.context_tokens, timings["total"],
        )  # fmt: skip
        return result

    def _runtime(self) -> dict:
        """Where the reranker actually runs, for the trace (the config only says "auto")."""
        model = self.reranker.model if self.reranker else None
        return {"reranker_device": getattr(model, "device", None),
                "reranker_dtype": getattr(model, "dtype", None)}  # fmt: skip


def _select(candidates: list[Candidate], top_n: int) -> list[Candidate]:
    """The top N, but a provision the question names is never dropped for a search hit:
    exact citation hits come first (in rerank order), then the best of the rest."""
    exact = [c for c in candidates if c.source == "citation"]
    rest = [c for c in candidates if c.source != "citation"]
    return exact + rest[: max(0, top_n - len(exact))]


def _config(cfg: RetrievalSettings) -> dict:
    keys = (
        "mode", "candidate_k", "rrf_k", "citation_lookup", "rerank", "rerank_k", "reranker_model",
        "top_n",
        "refusal_threshold", "parent_mode", "parent_max_tokens", "include_scope_rules",
        "include_guidance", "expand_refs", "include_definitions", "context_budget_tokens",
    )  # fmt: skip
    return {k: getattr(cfg, k) for k in keys}


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000
