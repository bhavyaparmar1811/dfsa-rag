"""A miniature corpus with the structure retrieval depends on, ingested for real
(temporary SQLite + in-memory Qdrant, fake embedder and BM25 encoder):

    COB 3.3       section
      3.3.1       scope rule: does not apply to a Market Counterparty
      3.3.2       rule, with Guidance
      3.3.2 Guidance
      3.3.3       rule
      3.3.4       split rule: #p1, #p2
    COB 3.4       section
      3.4.1       references COB 3.3.2
    GLO           definitions: Market Counterparty, Retail Client, Client Agreement,
                  Authorised Firm
"""

import pytest
from qdrant_client import QdrantClient

from src.config.settings import RetrievalSettings
from src.ingestion.ingest import Index, ingest
from src.ingestion.models import ChunkType
from src.storage.qdrant_store import QdrantStore
from tests.factories import (
    FakeEmbedder,
    FakeSparse,
    chunk,
    embedding_settings,
    entry,
    qdrant_settings,
)

GLO = entry("dfsa-rulebook-glo", "GLO", "66")
COB = entry()
EMB = embedding_settings()
QD = qdrant_settings()


def _corpus() -> dict:
    def rule(key, text, order, **kw):
        return chunk(key, text=text, order=order, **kw)

    guidance = chunk(
        "COB 3.3.2 Guidance", kind=ChunkType.GUIDANCE, order=3, guidance_for="COB 3.3.2",
        text="Guidance: a Client Agreement should be in writing and given to the Retail Client.",
        defined_terms=["Client Agreement", "Retail Client"],
    ).model_copy(update={"citation": "COB 3.3.2"})  # fmt: skip
    cob = [
        chunk("COB 3.3", kind=ChunkType.SECTION, order=0, text="Section 3.3 Client Agreement"),
        rule("COB 3.3.1", "3.3.1 The Rules in this section do not apply to an Authorised Firm "
             "dealing with a Market Counterparty.", 1, scope_rule=True,
             defined_terms=["Authorised Firm", "Market Counterparty"]),
        rule("COB 3.3.2", "3.3.2 An Authorised Firm must not carry on a Financial Service with a "
             "Person unless a Client Agreement has been entered into.", 2,
             guidance_keys=["COB 3.3.2 Guidance"],
             defined_terms=["Authorised Firm", "Client Agreement"]),
        guidance,
        rule("COB 3.3.3", "3.3.3 A Client Agreement with a Retail Client may be amended only "
             "with notice.", 4, defined_terms=["Client Agreement", "Retail Client"]),
        rule("COB 3.3.4", "3.3.4 (1) reliance on an agreement executed elsewhere. (2) records of "
             "the agreement.", 5, parts=["COB 3.3.4#p1", "COB 3.3.4#p2"]),
        rule("COB 3.3.4#p1", "3.3.4 (1) reliance on an agreement executed elsewhere.", 6,
             part_of="COB 3.3.4", part=1),
        rule("COB 3.3.4#p2", "(2) records of the agreement.", 7, part_of="COB 3.3.4", part=2),
        chunk("COB 3.4", kind=ChunkType.SECTION, order=8, text="Section 3.4 Suitability"),
        rule("COB 3.4.1", "3.4.1 An Authorised Firm must assess suitability before recommending "
             "an Investment, subject to Rule 3.3.2.", 9, parent_key="COB 3.4",
             cross_refs=["COB 3.3.2"], defined_terms=["Authorised Firm"]),
    ]  # fmt: skip

    def definition(term, text, order):
        return chunk(f"GLO {term}", kind=ChunkType.DEFINITION, doc_id=GLO.id, version="66",
                     parent_key=None, term=term, text=f"{term}: {text}", order=order)  # fmt: skip

    glo = [
        definition("Market Counterparty", "a Person classified under COB Rule 2.3.4.", 0),
        definition("Retail Client", "a Client who is not a Professional Client.", 1),
        definition("Client Agreement", "the agreement under COB Rule 3.3.2.", 2),
        definition("Authorised Firm", "a Person holding a Licence.", 3),
    ]
    return {COB.id: cob, GLO.id: glo}


@pytest.fixture
def corpus():
    return _corpus()


@pytest.fixture
def index():
    store = QdrantStore.from_settings(QD, EMB, client=QdrantClient(":memory:"))
    return Index(store, FakeEmbedder(EMB), FakeSparse(QD))


@pytest.fixture
def loaded(engine, index, corpus):
    """(engine, store): the miniature corpus in SQLite and Qdrant."""
    ingest([GLO, COB], lambda e: corpus[e.id], engine, fingerprint=EMB.fingerprint,
           sparse_model=QD.sparse_fingerprint, index=index)  # fmt: skip
    return engine, index.store


def settings(**kw) -> RetrievalSettings:
    """Hermetic settings (no .env); terms are never 'too common' in a tiny corpus."""
    return RetrievalSettings(_env_file=None, **({"term_max_share": 1.0} | kw))


class FakeScorer:
    """Stands in for the cross-encoder: the share of question words in the text."""

    def __init__(self):
        self.calls = 0

    def score(self, question: str, texts: list[str]) -> list[float]:
        self.calls += 1
        q = {w.strip("?.,").lower() for w in question.split() if len(w) > 3}
        return [len(q & {w.strip("?.,").lower() for w in t.split()}) / (len(q) or 1) for t in texts]

    def count_tokens(self, question: str, text: str) -> int:
        return len(question.split()) + len(text.split())
