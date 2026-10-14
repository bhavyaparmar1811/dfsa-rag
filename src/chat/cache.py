"""The semantic answer cache: a repeated or equivalent question gets the stored
answer, with no retrieval and no LLM call.

In compliance work two questions can be nearly identical and have opposite
answers ("a Client Agreement with a Market Counterparty?" is no, "... with a
Retail Client?" is yes), so similarity alone never decides a hit:

    namespace  sha256 of everything that shapes an answer: the corpus (every
               embedded chunk's content hash), the embedding and BM25 settings,
               the retrieval config (reranker revision, refusal threshold, ...),
               the answer prompt's content hash, the Azure deployment and the
               filters. An entry is only looked up in its own namespace, so a
               re-ingest, a prompt promotion or a config change makes old entries
               unreachable; `scripts/cache.py purge` deletes them.
    exact      the normalised standalone question, by hash: no embedding call.
    semantic   cosine >= CACHE_SIMILARITY_THRESHOLD (calibrated on hard negatives
               by src/eval/cache_eval.py), AND the same provisions named, the same
               Defined Terms and the same numbers as the cached question.
    verified   every cited chunk still exists with the content hash it was
               answered from.

Only answers that passed the calibrated confidence gate are stored, only on
production prompts, and only from first turns: a follow-up's answer can lean on
the conversation. A follow-up can still hit, through its rewritten question.

On a miss the lookup's embedding is handed to retrieval, so a question is never
embedded twice. Storing is best effort (a failure is logged, the turn goes on);
a failed lookup fails the turn like any other dependency.
"""

import hashlib
import json
import logging
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from typing import Literal

from sqlalchemy import Engine

from src.config.settings import CacheSettings, EmbeddingSettings
from src.embeddings.client import EmbeddingClient, EmbeddingResult
from src.generation.answer import ANSWER_PROMPT, AnswerResult, CitedProvision
from src.prompts.registry import PromptRegistry
from src.retrieval.models import Filters
from src.storage.cache_index import CacheIndex
from src.storage.db import session_scope
from src.storage.models import CacheEntry, utcnow
from src.storage.repositories import CacheRepository, ChunkRepository, IngestionRunRepository

logger = logging.getLogger(__name__)

_SPACES = re.compile(r"\s+")
_NUMBERS = re.compile(r"\d+(?:[.,]\d+)*")


def normalise(question: str) -> str:
    """Case, spacing and the closing punctuation don't change a question."""
    return _SPACES.sub(" ", question).strip().rstrip("?.! ").casefold()


def query_hash(question: str) -> str:
    return hashlib.sha256(normalise(question).encode()).hexdigest()


@dataclass(frozen=True)
class Features:
    """What a semantic hit must share with the cached question, exactly."""

    citations: tuple[str, ...]  # provisions named: "COB 3.3.2"
    terms: tuple[str, ...]  # Defined Terms used: "Retail Client"
    numbers: tuple[str, ...]  # thresholds, amounts, days: "50", "3.3.2"

    @classmethod
    def of_entry(cls, entry: CacheEntry) -> "Features":
        return cls(tuple(entry.citations), tuple(entry.terms), tuple(entry.numbers))

    def differences(self, other: "Features") -> list[str]:
        return [
            name for name in ("citations", "terms", "numbers")
            if getattr(self, name) != getattr(other, name)
        ]  # fmt: skip


def question_features(
    question: str,
    find_citations: Callable[[str], list[str]],
    find_terms: Callable[[str], list[str]],
) -> Features:
    """The guards' view of a question; src/eval/cache_eval.py uses the same function."""
    return Features(
        citations=tuple(sorted(set(find_citations(question)))),
        terms=tuple(sorted(set(find_terms(question)))),
        numbers=tuple(sorted(set(_NUMBERS.findall(question)))),
    )


@dataclass
class CacheHit:
    entry_id: str
    kind: Literal["exact", "semantic"]
    similarity: float | None  # None for an exact repeat
    source_trace_id: str  # the turn that produced the answer: its retrieval, scores, prompt
    source_cost_usd: float
    cached_query: str
    text: str
    guidance_only: bool
    cited: list[CitedProvision]

    def answer(self, question: str) -> AnswerResult:
        return AnswerResult(
            question=question,
            status="answered",
            text=self.text,
            cited=self.cited,
            guidance_only=self.guidance_only,
            gate="cache",
        )


@dataclass
class CacheLookup:
    status: Literal["hit", "miss", "bypass"]
    reason: str = ""
    namespace: str | None = None
    components: dict = field(default_factory=dict)
    features: Features | None = None
    hit: CacheHit | None = None
    # Computed on a miss and reused by retrieval; on a hit it is the turn's only embedding.
    embedding: EmbeddingResult | None = None
    candidates: list[dict] = field(default_factory=list)  # entry_id, similarity, rejected
    latency_ms: float = 0.0

    @property
    def embed_tokens(self) -> int:
        return self.embedding.prompt_tokens if self.embedding else 0

    @property
    def embed_cost_usd(self) -> float:
        return self.embedding.cost_usd if self.embedding else 0.0

    def trace(self) -> dict:
        hit = self.hit
        return {
            "status": self.status,
            "reason": self.reason,
            "namespace": self.namespace,
            "components": self.components,
            "features": asdict(self.features) if self.features else None,
            "kind": hit.kind if hit else None,
            "entry_id": hit.entry_id if hit else None,
            "similarity": hit.similarity if hit else None,
            "source_trace_id": hit.source_trace_id if hit else None,
            "source_cost_usd": hit.source_cost_usd if hit else None,
            "cached_query": hit.cached_query if hit else None,
            "candidates": self.candidates,
            "embedded": self.embedding is not None,
            "latency_ms": round(self.latency_ms, 1),
        }


class SemanticCache:
    def __init__(
        self,
        engine: Engine,
        index: CacheIndex,
        embedder: EmbeddingClient,
        *,
        find_citations: Callable[[str], list[str]],
        find_terms: Callable[[str], list[str]],
        registry: PromptRegistry,
        components: dict,
        cfg: CacheSettings | None = None,
    ):
        """`components`: the fixed part of the namespace (embedding and BM25
        fingerprints, retrieval config, model). The corpus hash, the answer
        prompt's hash and the filters are added per request."""
        self.engine = engine
        self.index = index
        self.embedder = embedder
        self.find_citations = find_citations
        self.find_terms = find_terms
        self.registry = registry
        self.components = _canonical(components)
        self.cfg = cfg or CacheSettings()
        self._corpus: tuple[int, str] | None = None  # (last finished ingestion run, hash)

    @classmethod
    def build(cls, engine: Engine, retriever, registry: PromptRegistry, model: str):
        """The real components, sharing the retriever's Qdrant client and embedder."""
        cfg, emb = CacheSettings(), EmbeddingSettings()
        store = retriever.searcher.store
        index = CacheIndex.from_settings(cfg, emb, client=store.client)
        index.ensure_collection()
        return cls(
            engine,
            index,
            retriever.searcher.embedder or EmbeddingClient(emb),
            find_citations=retriever.citations.find,
            find_terms=retriever.assembler.question_terms,
            registry=registry,
            components={
                "embedding": emb.fingerprint,
                "sparse": store.sparse_fingerprint,
                "retrieval": retriever.config(),
                "model": model,
            },
            cfg=cfg,
        )

    # --- keys ------------------------------------------------------------------

    def namespace(self, filters: Filters) -> tuple[str, dict]:
        components = self.base_components() | {
            "filters": _canonical(filters.model_dump(mode="json", exclude_defaults=True))
        }
        canonical = json.dumps(components, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest(), components

    def base_components(self) -> dict:
        """Everything in the namespace except the filters."""
        return self.components | {
            "corpus": self._corpus_hash(),
            "answer_prompt": self.registry.get(ANSWER_PROMPT).hash,
        }

    def features(self, question: str) -> Features:
        return question_features(question, self.find_citations, self.find_terms)

    def _corpus_hash(self) -> str:
        with session_scope(self.engine) as s:
            run = IngestionRunRepository(s).last_finished_id()
            if self._corpus is None or self._corpus[0] != run:
                self._corpus = (run, ChunkRepository(s).corpus_hash())
        return self._corpus[1]

    # --- lookup ----------------------------------------------------------------

    def lookup(self, question: str, filters: Filters, label: str | None = None) -> CacheLookup:
        """`question` is the standalone (rewritten) question."""
        start = time.perf_counter()
        result = self._lookup(question, filters, label)
        result.latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "cache %s%s (%s), %.0f ms",
            result.status,
            f" {result.hit.kind}" if result.hit else "",
            result.reason or "-",
            result.latency_ms,
        )
        return result

    def _lookup(self, question: str, filters: Filters, label: str | None) -> CacheLookup:
        if not self.cfg.enabled:
            return CacheLookup("bypass", "cache disabled")
        if label is not None:
            return CacheLookup("bypass", f"prompt label {label!r}: only production is cached")
        namespace, components = self.namespace(filters)
        out = CacheLookup(
            "miss", namespace=namespace, components=components, features=self.features(question)
        )
        now = utcnow()

        with session_scope(self.engine) as s:
            entry = CacheRepository(s).by_hash(namespace, query_hash(question), now)
            if entry is not None:
                hit = self._verified(s, entry, "exact", None)
                if hit is not None:
                    CacheRepository(s).record_hit(entry.id, now)
                    out.status, out.hit, out.reason = "hit", hit, "same question"
                    return out
                out.candidates.append(
                    {"entry_id": entry.id, "similarity": None, "rejected": "cited text changed"}
                )

        out.embedding = self.embedder.embed_query(question)
        threshold = self.cfg.similarity_threshold
        if threshold is None:
            out.reason = "exact repeats only (no similarity threshold set)"
            return out
        found = self.index.search(
            out.embedding.vectors[0], namespace, int(now.timestamp()), self.cfg.top_k
        )
        with session_scope(self.engine) as s:
            repo = CacheRepository(s)
            for entry_id, similarity in found:
                entry, hit, rejected = repo.get(entry_id), None, None
                if similarity < threshold:
                    rejected = "below threshold"
                elif entry is None:
                    rejected = "no entry (orphan point)"
                elif entry.expires_at.replace(tzinfo=None) <= now.replace(tzinfo=None):
                    rejected = "expired"  # SQLite decides, whatever the point's payload says
                elif diffs := out.features.differences(Features.of_entry(entry)):
                    rejected = "different " + ", ".join(diffs)
                elif (hit := self._verified(s, entry, "semantic", similarity)) is None:
                    rejected = "cited text changed"
                out.candidates.append(
                    {"entry_id": entry_id, "similarity": round(similarity, 4), "rejected": rejected}
                )
                if hit is not None:
                    repo.record_hit(entry_id, now)
                    out.status, out.hit, out.reason = "hit", hit, "equivalent question"
                    return out
        out.reason = "no cached question passed the checks" if found else "nothing cached nearby"
        return out

    def _verified(self, s, entry: CacheEntry, kind, similarity) -> CacheHit | None:
        """The hit, if every cited chunk is still the text it was answered from."""
        current = ChunkRepository(s).content_hashes(c["chunk_id"] for c in entry.cited)
        if any(current.get(c["chunk_id"]) != c["content_hash"] for c in entry.cited):
            return None
        return CacheHit(
            entry_id=entry.id,
            kind=kind,
            similarity=similarity,
            source_trace_id=entry.source_trace_id,
            source_cost_usd=entry.cost_usd,
            cached_query=entry.query,
            text=entry.answer,
            guidance_only=entry.guidance_only,
            cited=[
                CitedProvision(**{k: v for k, v in c.items() if k != "content_hash"})
                for c in entry.cited
            ],
        )

    # --- store -----------------------------------------------------------------

    def store(
        self,
        lookup: CacheLookup | None,
        *,
        question: str,
        query: str,
        answer: AnswerResult,
        trace_id: str,
        cost_usd: float,
        first_turn: bool,
    ) -> str | None:
        """Cache an answer if it qualifies; returns the new entry's id."""
        if lookup is None or lookup.status != "miss":
            return None  # bypassed, or served from the cache already
        if not first_turn or answer.status != "answered" or answer.gate != "passed":
            return None
        try:
            now = utcnow()
            expires = now + timedelta(days=self.cfg.ttl_days)
            with session_scope(self.engine) as s:
                repo = CacheRepository(s)
                if repo.by_hash(lookup.namespace, query_hash(query), now) is not None:
                    return None
                hashes = ChunkRepository(s).content_hashes(c.chunk_id for c in answer.cited)
                entry = repo.add(
                    CacheEntry(
                        id=str(uuid.uuid4()),
                        namespace=lookup.namespace,
                        components=lookup.components,
                        question=question,
                        query=query,
                        query_hash=query_hash(query),
                        citations=list(lookup.features.citations),
                        terms=list(lookup.features.terms),
                        numbers=list(lookup.features.numbers),
                        answer=answer.text,
                        guidance_only=answer.guidance_only,
                        cited=[
                            asdict(c) | {"content_hash": hashes.get(c.chunk_id)}
                            for c in answer.cited
                        ],
                        source_trace_id=trace_id,
                        cost_usd=cost_usd,
                        created_at=now,
                        expires_at=expires,
                    )
                )
                entry_id = entry.id
            if lookup.embedding is not None:  # exact repeats still work without the point
                vector, until = lookup.embedding.vectors[0], int(expires.timestamp())
                self.index.add(entry_id, vector, lookup.namespace, until)
            return entry_id
        except Exception:
            logger.exception("could not cache the answer of trace %s", trace_id)
            return None

    # --- maintenance -----------------------------------------------------------

    def purge(self) -> dict:
        """Delete expired entries and those from an older namespace, plus orphan points."""
        return purge(self.engine, self.index, self.base_components())

    def stats(self) -> dict:
        with session_scope(self.engine) as s:
            stats = CacheRepository(s).stats(utcnow())
        return stats | {
            "enabled": self.cfg.enabled,
            "similarity_threshold": self.cfg.similarity_threshold,
            "collection": self.index.collection,
            "points": self.index.count(),
        }


def purge(engine: Engine, index: CacheIndex, current: dict | None) -> dict:
    """Delete expired entries, entries whose namespace differs from `current` in
    anything but the filters (another corpus, prompt, config or model), and Qdrant
    points without an entry. `current=None` deletes every entry."""
    now = utcnow().replace(tzinfo=None)
    with session_scope(engine) as s:
        repo = CacheRepository(s)
        stale = [
            entry_id
            for entry_id, components, expires_at in repo.meta()
            if current is None
            or expires_at.replace(tzinfo=None) <= now
            or {k: v for k, v in components.items() if k != "filters"} != current
        ]
        repo.delete(stale)
        remaining = repo.ids()
    orphans = index.ids() - remaining - set(stale)
    index.delete(set(stale) | orphans)
    return {"entries_deleted": len(stale), "orphan_points_deleted": len(orphans)}


def _canonical(value: dict) -> dict:
    """As it comes back from a JSON column (tuples become lists), so a stored
    namespace's components compare equal to freshly computed ones."""
    return json.loads(json.dumps(value, sort_keys=True, default=str))
