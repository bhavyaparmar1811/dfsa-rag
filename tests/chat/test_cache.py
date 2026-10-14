"""The semantic answer cache: what may be served, what may be stored, and what
invalidates it. A temporary SQLite database, an in-memory Qdrant cache collection,
an embedder whose vectors the tests choose, and the scripted Azure client."""

import dataclasses
import math
from datetime import timedelta

import pytest
from qdrant_client import QdrantClient
from sqlalchemy import select, update

from src.chat.cache import SemanticCache, normalise
from src.chat.rewrite import QueryRewriter
from src.chat.service import ChatService
from src.config.settings import CacheSettings, ChatSettings
from src.embeddings.client import EmbeddingModelMismatch
from src.generation.answer import Answerer
from src.generation.llm import LLMClient
from src.ingestion.diff import diff_chunks
from src.retrieval.citations import CitationRouter
from src.retrieval.models import Filters
from src.storage.cache_index import CacheIndex
from src.storage.db import session_scope
from src.storage.models import CacheEntry, ChunkRow, Trace, utcnow
from src.storage.repositories import (
    ChunkRepository,
    DocumentRepository,
    IngestionRunRepository,
    RunStats,
    TraceRepository,
)
from tests.chat.conftest import ANSWER, FakeRetriever
from tests.factories import DIMS, FINGERPRINT, FakeEmbedder, chunk, embedding_settings, entry
from tests.generation.conftest import BLOCKS, CODES, LAWS, FakeAzure, azure_settings, draft
from tests.generation.conftest import git_registry as make_registry

MC = "Do we need a Client Agreement with a Market Counterparty?"
MC_PARAPHRASE = "Is a Client Agreement required when dealing with a Market Counterparty?"
RC = "Do we need a Client Agreement with a Retail Client?"  # nearly the same words, other answer
TERMS = ("Market Counterparty", "Retail Client", "Client Agreement")


def unit(*xs: float) -> list[float]:
    v = list(xs) + [0.0] * (DIMS - len(xs))
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


def cos_to_mc(similarity: float) -> list[float]:
    """A vector with exactly this cosine to MC's."""
    return unit(similarity, math.sqrt(1 - similarity**2))


VECTORS = {
    MC: unit(1.0),
    MC_PARAPHRASE: cos_to_mc(0.97),
    RC: cos_to_mc(0.99),  # closer than the paraphrase: only the term guard stops it
    "What does COB 3.3.2 require?": unit(0.0, 0.0, 1.0),
    "What does COB 3.3.1 require?": unit(0.0, 0.05, 1.0),
    "Is an offer to 50 investors exempt?": unit(0.0, 0.0, 0.0, 1.0),
    "Is an offer to 150 investors exempt?": unit(0.0, 0.0, 0.0, 1.0, 0.02),
    "A distant question?": unit(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
}


class ScriptedEmbedder(FakeEmbedder):
    """FakeEmbedder with chosen vectors for the questions above."""

    def embed_documents(self, texts):
        result = super().embed_documents(texts)
        vectors = [VECTORS.get(t, v) for t, v in zip(texts, result.vectors, strict=True)]
        return dataclasses.replace(result, vectors=vectors)


@pytest.fixture
def ids(engine):
    """COB 3.3.1 and 3.3.2 as embedded chunks (what the answers cite), and one
    finished ingestion run."""
    chunks = [chunk("COB 3.3.1", order=1), chunk("COB 3.3.2", order=2)]
    with session_scope(engine) as s:
        DocumentRepository(s).upsert(entry())
        ChunkRepository(s).upsert(chunks, diff_chunks(chunks, {}, FINGERPRINT).hashes)
        ChunkRepository(s).mark_embedded([c.chunk_id for c in chunks], "fake", DIMS)
        runs = IngestionRunRepository(s)
        runs.finish(runs.start(FINGERPRINT, "bm25"), RunStats())
    return {c.key: c.chunk_id for c in chunks}


@pytest.fixture
def setup(engine, ids):
    """make(*replies, threshold=..., confident=...) -> (service, cache, retriever, azure)."""
    client = QdrantClient(":memory:")
    blocks = [
        dataclasses.replace(BLOCKS[0], chunk_id=ids["COB 3.3.1"]),
        dataclasses.replace(BLOCKS[1], chunk_id=ids["COB 3.3.2"]),
    ]
    router = CitationRouter(None, None, CODES, LAWS)

    def make(*replies, threshold=0.95, confident=True, components=None, retriever=None):
        index = CacheIndex(client, "cache_test", embedding_settings())
        index.ensure_collection()
        cache = SemanticCache(
            engine,
            index,
            ScriptedEmbedder(),
            find_citations=router.find,
            find_terms=lambda q: [t for t in TERMS if t.lower() in q.lower()],
            registry=make_registry(),
            components=components or {"model": "gpt-4o-test", "retrieval": {"top_n": 6}},
            cfg=CacheSettings(_env_file=None, similarity_threshold=threshold),
        )
        azure = FakeAzure(*replies)
        llm, registry = LLMClient(azure_settings(), client=azure), make_registry()
        retriever = retriever or FakeRetriever(blocks=blocks, confident=confident)
        service = ChatService(
            engine=engine,
            retriever=retriever,
            answerer=Answerer(llm, registry, retriever.citations.find),
            rewriter=QueryRewriter(llm, registry),
            cfg=ChatSettings(_env_file=None),
            cache=cache,
        )
        return service, cache, retriever, azure

    return make


def ask(service, question, conversation=None, **kw):
    conversation = conversation or service.create_conversation().id
    return service.send(conversation, question, **kw)


def entries(engine) -> list[CacheEntry]:
    with session_scope(engine) as s:
        return list(s.scalars(select(CacheEntry)))


def trace(engine, trace_id) -> Trace:
    with session_scope(engine) as s:
        return TraceRepository(s).get(trace_id)


# --- hits ---------------------------------------------------------------------


def test_an_exact_repeat_is_served_without_embedding_retrieval_or_llm(setup, engine):
    service, cache, retriever, azure = setup(ANSWER)
    first = ask(service, MC)
    assert first.cache.status == "miss" and first.cache_entry_id is not None
    embeds = len(cache.embedder.calls)

    again = ask(service, "  do we need a client agreement with a MARKET counterparty ")
    assert again.cached and again.cache.hit.kind == "exact"
    assert again.answer.text == first.answer.text
    assert [c.citation for c in again.answer.cited] == ["COB 3.3.2", "COB 3.3.1"]
    assert len(cache.embedder.calls) == embeds  # no embedding for an exact repeat
    assert len(retriever.questions) == 1 and len(azure.calls) == 1
    assert again.cost_usd == 0.0 and again.retrieval is None

    row = trace(engine, again.trace_id)
    assert (row.cache_status, row.cache_entry_id) == ("hit", first.cache_entry_id)
    assert row.cache["source_trace_id"] == first.trace_id  # the audit chain
    assert row.cache_similarity is None and row.refused is False
    assert trace(engine, first.trace_id).cache["stored_entry_id"] == first.cache_entry_id
    assert entries(engine)[0].hit_count == 1


def test_a_paraphrase_is_served_semantically(setup, engine):
    service, cache, retriever, azure = setup(ANSWER)
    first = ask(service, MC)
    hit = ask(service, MC_PARAPHRASE)
    assert hit.cached and hit.cache.hit.kind == "semantic"
    assert hit.cache.hit.similarity == pytest.approx(0.97, abs=1e-4)
    assert hit.cache.hit.source_trace_id == first.trace_id
    assert hit.cost_usd == pytest.approx(hit.cache.embed_cost_usd) and hit.cost_usd > 0
    assert len(retriever.questions) == 1 and len(azure.calls) == 1
    assert trace(engine, hit.trace_id).embed_tokens == hit.cache.embed_tokens


def test_a_miss_embeds_the_question_once(setup):
    service, cache, retriever, _ = setup(ANSWER)
    turn = ask(service, MC)
    assert len(cache.embedder.calls) == 1
    assert retriever.embeddings == [turn.cache.embedding]  # handed to retrieval, reused


# --- the guards -----------------------------------------------------------------


def test_a_different_defined_term_is_never_served(setup):
    """Retail Client is closer to the cached Market Counterparty question than the
    paraphrase is, and the answer is the opposite."""
    rc_answer = draft("Yes, a Client Agreement is required (COB 3.3.2).", (2, "COB 3.3.2"))
    service, _, retriever, azure = setup(ANSWER, rc_answer)
    ask(service, MC)
    turn = ask(service, RC)
    assert not turn.cached and turn.answer.text.startswith("Yes")
    [candidate] = turn.cache.candidates
    assert candidate["similarity"] == pytest.approx(0.99, abs=1e-4)
    assert candidate["rejected"] == "different terms"
    assert len(retriever.questions) == 2 and len(azure.calls) == 2


@pytest.mark.parametrize(
    ("cached", "asked", "guard"),
    [
        ("What does COB 3.3.2 require?", "What does COB 3.3.1 require?", "citations, numbers"),
        ("Is an offer to 50 investors exempt?", "Is an offer to 150 investors exempt?", "numbers"),
    ],
)
def test_other_provisions_or_numbers_are_never_served(setup, cached, asked, guard):
    service, *_ = setup(ANSWER, ANSWER)
    ask(service, cached)
    turn = ask(service, asked)
    assert not turn.cached
    assert turn.cache.candidates[0]["rejected"] == f"different {guard}"


def test_below_the_threshold_is_a_miss(setup):
    service, *_ = setup(ANSWER, ANSWER, threshold=0.98)
    ask(service, MC)
    turn = ask(service, MC_PARAPHRASE)  # 0.97
    assert not turn.cached and turn.cache.candidates[0]["rejected"] == "below threshold"


def test_without_a_threshold_only_exact_repeats_hit(setup):
    service, *_ = setup(ANSWER, ANSWER, threshold=None)
    ask(service, MC)
    assert not ask(service, MC_PARAPHRASE).cached
    assert ask(service, MC + "  ").cache.hit.kind == "exact"


def test_changed_cited_text_is_not_served(setup, engine, ids):
    service, *_ = setup(ANSWER, ANSWER)
    ask(service, MC)
    with session_scope(engine) as s:  # the chunk changed, the corpus hash not yet noticed
        s.execute(
            update(ChunkRow).where(ChunkRow.id == ids["COB 3.3.2"]).values(content_hash="new")
        )
    turn = ask(service, MC)
    assert not turn.cached and turn.cache.candidates[0]["rejected"] == "cited text changed"


def test_an_expired_entry_is_not_served(setup, engine):
    service, *_ = setup(ANSWER, ANSWER)
    stored = ask(service, MC).cache_entry_id
    with session_scope(engine) as s:  # expired in SQLite; the Qdrant payload still says not
        s.execute(update(CacheEntry).values(expires_at=utcnow() - timedelta(seconds=1)))
    again = ask(service, MC)  # neither as an exact repeat nor through its point
    assert not again.cached
    assert again.cache.candidates == [
        {"entry_id": stored, "similarity": pytest.approx(1.0), "rejected": "expired"}
    ]
    assert again.cache_entry_id != stored  # answered afresh, and cached afresh


# --- the namespace ---------------------------------------------------------------


def test_other_filters_are_another_namespace(setup):
    service, *_ = setup(ANSWER, ANSWER)
    ask(service, MC)
    turn = ask(service, MC, filters=Filters(modules=["COB"]))
    assert not turn.cached and turn.cache.candidates == []


def test_a_new_ingest_that_changes_the_corpus_invalidates(setup, engine, ids):
    service, cache, *_ = setup(ANSWER, ANSWER)
    ask(service, MC)
    with session_scope(engine) as s:
        s.execute(update(ChunkRow).where(ChunkRow.id == ids["COB 3.3.1"]).values(content_hash="v2"))
        runs = IngestionRunRepository(s)
        runs.finish(runs.start(FINGERPRINT, "bm25"), RunStats())
    turn = ask(service, MC)
    assert not turn.cached and turn.cache.namespace != entries(engine)[0].namespace


def test_another_config_or_prompt_is_another_namespace(setup, engine, monkeypatch):
    service, *_ = setup(ANSWER)
    ask(service, MC)
    other_config, *_ = setup(ANSWER, components={"model": "gpt-4o-test", "retrieval": {"top_n": 8}})
    assert not ask(other_config, MC).cached

    promoted, cache, *_ = setup(ANSWER)
    real = cache.registry.get
    monkeypatch.setattr(
        cache.registry, "get", lambda name, label=None: dataclasses.replace(real(name), hash="v3")
    )
    assert not ask(promoted, MC).cached


# --- what is stored ---------------------------------------------------------------


def test_refusals_are_not_stored(setup, engine):
    service, *_ = setup(confident=False)  # refused at the gate, no LLM call
    turn = ask(service, MC)
    assert turn.answer.refused and turn.cache_entry_id is None and entries(engine) == []


def test_staging_prompts_bypass_the_cache(setup, engine):
    service, *_ = setup(ANSWER)
    turn = ask(service, MC, label="staging")
    assert turn.cache.status == "bypass" and entries(engine) == []
    assert trace(engine, turn.trace_id).cache_status == "bypass"


def test_follow_ups_hit_but_are_not_stored(setup, engine):
    """A follow-up's answer can lean on the conversation, so it isn't cached, but
    its standalone rewrite can be served from a first turn's entry."""
    other = draft("A Retail Client needs one (COB 3.3.2).", (2, "COB 3.3.2"))
    service, *_ = setup(ANSWER, ANSWER, MC, RC, other)
    ask(service, MC)  # stored
    conversation = ask(service, "Tell me about Client Agreements").conversation_id  # stored
    assert len(entries(engine)) == 2

    follow_up = ask(service, "what about market counterparties?", conversation)  # rewritten: MC
    assert follow_up.cached and follow_up.rewrite.question == MC
    missed = ask(service, "and retail clients?", conversation)  # rewritten: RC, answered
    assert not missed.cached and missed.answer.status == "answered"
    assert missed.cache_entry_id is None and len(entries(engine)) == 2


# --- maintenance and reporting ----------------------------------------------------


def test_purge_deletes_stale_and_expired_entries_and_orphan_points(setup, engine, ids):
    service, cache, *_ = setup(ANSWER, ANSWER, ANSWER)
    ask(service, MC)
    ask(service, "Is an offer to 50 investors exempt?")
    with session_scope(engine) as s:  # the second one expires
        s.execute(
            update(CacheEntry)
            .where(CacheEntry.query == "Is an offer to 50 investors exempt?")
            .values(expires_at=utcnow() - timedelta(days=1))
        )
    cache.index.add("00000000-0000-0000-0000-000000000001", unit(0.5, 0.5), "x", 2**40)
    assert cache.purge() == {"entries_deleted": 1, "orphan_points_deleted": 1}
    assert [e.query for e in entries(engine)] == [MC] and cache.index.count() == 1

    with session_scope(engine) as s:  # a re-ingest makes the rest stale
        s.execute(update(ChunkRow).values(content_hash="v2"))
        runs = IngestionRunRepository(s)
        runs.finish(runs.start(FINGERPRINT, "bm25"), RunStats())
    assert cache.purge()["entries_deleted"] == 1 and cache.index.count() == 0


def test_cost_stats_report_the_hit_rate_and_savings(setup):
    service, *_ = setup(ANSWER)
    first = ask(service, MC)
    ask(service, MC_PARAPHRASE)
    stats = service.cost_stats()["cache"]
    assert (stats["lookups"], stats["hits"], stats["exact_hits"]) == (2, 1, 0)
    assert stats["hit_rate"] == 0.5
    assert stats["saved_usd"] == pytest.approx(first.cost_usd, abs=1e-5)


def test_the_cache_collection_refuses_another_embedding_model():
    client = QdrantClient(":memory:")
    CacheIndex(client, "c", embedding_settings()).ensure_collection()
    other = CacheIndex(client, "c", embedding_settings(model="text-embedding-3-large"))
    with pytest.raises(EmbeddingModelMismatch):
        other.ensure_collection()


def test_normalise():
    assert normalise("  Do we need A  Client Agreement?? ") == "do we need a client agreement"
