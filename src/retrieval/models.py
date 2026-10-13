"""The data that flows through retrieval, and into the trace.

question -> CitationHit (exact) + Candidate (hybrid) -> rerank -> hits
         -> ContextBlock (what the LLM sees) -> RetrievalResult
"""

from dataclasses import asdict, dataclass, field

from pydantic import BaseModel
from qdrant_client import models as qm

from src.ingestion.models import embedding_text


class Filters(BaseModel):
    """Metadata filters, applied inside Qdrant (and to exact citation hits)."""

    regulators: list[str] | None = None  # ["DFSA"]
    modules: list[str] | None = None  # manifest module codes: ["AML"], ["COB", "REG_LAW"]
    doc_types: list[str] | None = None  # ["rulebook"] | ["law"]
    chunk_types: list[str] | None = None  # ["rule", "guidance", "definition", "article", "text"]
    binding_only: bool = False

    def to_qdrant(self) -> qm.Filter | None:
        must: list[qm.Condition] = []
        for key, values in (
            ("regulator", self.regulators),
            ("module", self.modules),
            ("doc_type", self.doc_types),
            ("chunk_type", self.chunk_types),
        ):
            if values:
                must.append(qm.FieldCondition(key=key, match=qm.MatchAny(any=values)))
        if self.binding_only:
            must.append(qm.FieldCondition(key="binding", match=qm.MatchValue(value=True)))
        return qm.Filter(must=must) if must else None

    def accepts(self, payload: dict) -> bool:
        """The same filter in Python, for candidates that didn't come from a search."""
        checks = (
            ("regulator", self.regulators),
            ("module", self.modules),
            ("doc_type", self.doc_types),
            ("chunk_type", self.chunk_types),
        )
        if any(values and payload.get(key) not in values for key, values in checks):
            return False
        return not (self.binding_only and not payload.get("binding"))


@dataclass
class Candidate:
    """A retrieved chunk (one Qdrant point) with every score it collected."""

    id: str
    payload: dict
    source: str  # "citation" (exact lookup) | "search"
    dense_rank: int | None = None  # 1-based
    dense_score: float | None = None  # cosine
    sparse_rank: int | None = None
    sparse_score: float | None = None  # BM25 (with IDF)
    fused_score: float | None = None  # RRF, or the single retriever's score
    rerank_score: float | None = None  # cross-encoder, 0..1
    truncated: bool = False  # the reranker saw only the start of this chunk

    @property
    def key(self) -> str:
        return self.payload["key"]

    @property
    def citation(self) -> str:
        return self.payload["citation"]

    @property
    def text_for_rerank(self) -> str:
        """Header + text: exactly what was embedded."""
        return embedding_text(self.payload["header_text"], self.payload["text"])

    def trace(self) -> dict:
        row = {k: v for k, v in asdict(self).items() if k != "payload"}
        row |= {"key": self.key, "citation": self.citation}
        return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}


@dataclass
class CitationLookup:
    """What the citation router made of the question."""

    found: list[str] = field(default_factory=list)  # normalised: ["COB 3.3.2", "* 2.3.2"]
    resolved: dict[str, list[str]] = field(default_factory=dict)  # citation -> chunk keys
    unresolved: list[str] = field(default_factory=list)  # cited, but not in the corpus
    too_broad: list[str] = field(default_factory=list)  # a chapter or appendix: left to search
    filtered_out: list[str] = field(default_factory=list)  # in the corpus, excluded by filters


@dataclass
class ContextBlock:
    """One numbered unit of LLM context: always exactly one chunk, so every
    statement the LLM makes can be cited back to one provision."""

    n: int
    chunk_id: str
    key: str
    citation: str
    chunk_type: str
    binding: bool
    label: str  # "Rule (binding)", "Guidance (non-binding)", "Definition", "Law article"
    heading: str  # the contextual header
    text: str
    tokens: int
    reason: str  # hit | part | scope | guidance | rule | section | window | definition | ref
    group: str  # the section (or "definitions" / "references") it is shown under
    score: float | None = None  # the rerank score of the hit that brought it in

    def trace(self) -> dict:
        row = asdict(self)
        row.pop("text")
        return row


@dataclass
class RetrievalResult:
    question: str
    filters: Filters
    config: dict
    citations: CitationLookup
    candidates: list[Candidate]  # everything that reached the reranker, best first
    hits: list[Candidate]  # the top N, after rerank
    context: list[ContextBlock]
    confidence: float | None  # the top rerank score (None without a reranker)
    confident: bool | None  # None: no calibrated threshold yet
    timings_ms: dict[str, float]
    embed_tokens: int = 0
    embed_cost_usd: float = 0.0
    truncated: int = 0  # candidates the reranker cut off
    dropped: list[str] = field(default_factory=list)  # context left out by the budget

    @property
    def context_tokens(self) -> int:
        return sum(b.tokens for b in self.context)

    def trace(self) -> dict:
        """Everything Phase G stores per request, as plain JSON."""
        return {
            "question": self.question,
            "filters": self.filters.model_dump(exclude_defaults=True),
            "config": self.config,
            "citations": asdict(self.citations),
            "candidates": [c.trace() for c in self.candidates],
            "hits": [c.key for c in self.hits],
            "context": [b.trace() for b in self.context],
            "context_tokens": self.context_tokens,
            "dropped": self.dropped,
            "confidence": self.confidence,
            "confident": self.confident,
            "timings_ms": {k: round(v, 1) for k, v in self.timings_ms.items()},
            "embed_tokens": self.embed_tokens,
            "embed_cost_usd": self.embed_cost_usd,
            "truncated": self.truncated,
        }
