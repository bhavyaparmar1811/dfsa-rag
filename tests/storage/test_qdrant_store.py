"""QdrantStore against Qdrant's in-memory local mode (no server needed)."""

import pytest
from qdrant_client import QdrantClient

from src.embeddings.client import EmbeddingModelMismatch
from src.ingestion.models import ChunkType, content_hash
from src.storage.qdrant_store import (
    BOOL_FIELDS,
    DENSE,
    KEYWORD_FIELDS,
    SCHEMA_VERSION,
    SPARSE,
    CollectionMismatch,
    Point,
    QdrantStore,
    build_payload,
)
from tests.factories import (
    DIMS,
    FINGERPRINT,
    FakeEmbedder,
    FakeSparse,
    chunk,
    embedding_settings,
    qdrant_settings,
)


@pytest.fixture
def client():
    return QdrantClient(":memory:")


@pytest.fixture
def store(client):
    s = QdrantStore.from_settings(qdrant_settings(), embedding_settings(), client=client)
    s.ensure_collection()
    return s


def points(chunks, doc_type="rulebook"):
    texts = [c.embedding_text() for c in chunks]
    dense = FakeEmbedder().embed_documents(texts).vectors
    sparse = FakeSparse().encode_documents(texts)
    payloads = [
        build_payload(c, doc_type, content_hash(c, FINGERPRINT), FINGERPRINT) for c in chunks
    ]
    return [
        Point(c.chunk_id, d, s, p)
        for c, d, s, p in zip(chunks, dense, sparse, payloads, strict=True)
    ]


def test_collection_is_created_with_both_vectors_and_its_metadata(client):
    store = QdrantStore.from_settings(qdrant_settings(), embedding_settings(), client=client)
    assert store.collection == f"dfsa_rules__text-embedding-3-small-{DIMS}"
    assert store.ensure_collection() is True
    assert store.ensure_collection() is False  # second call only checks

    config = client.get_collection(store.collection).config
    assert config.params.vectors[DENSE].size == DIMS
    assert config.params.sparse_vectors[SPARSE].modifier.value == "idf"
    assert config.metadata == {
        "fingerprint": f"text-embedding-3-small-{DIMS}",
        "embedding_model": "text-embedding-3-small",
        "dimensions": DIMS,
        "sparse_fingerprint": "Qdrant/bm25-k1.2-b0.75-avg124",
        "schema_version": SCHEMA_VERSION,
    }


def test_payload_indexes_are_requested(client, monkeypatch):
    """Local mode doesn't keep payload indexes, so check what was asked for."""
    created = {}
    monkeypatch.setattr(
        client, "create_payload_index",
        lambda collection, name, schema, wait: created.__setitem__(name, schema.value),
    )  # fmt: skip
    QdrantStore.from_settings(
        qdrant_settings(), embedding_settings(), client=client
    ).ensure_collection()
    assert created == {f: "keyword" for f in KEYWORD_FIELDS} | {f: "bool" for f in BOOL_FIELDS}


def test_a_different_embedding_model_is_refused(client, store):
    other = QdrantStore(client, store.collection, embedding_settings(model="other-model"),
                        store.sparse_fingerprint)  # fmt: skip
    with pytest.raises(EmbeddingModelMismatch, match="Re-embed"):
        other.ensure_collection()


def test_a_different_sparse_encoder_is_refused(client, store):
    other_cfg = qdrant_settings(sparse_avg_len=256)
    other = QdrantStore(client, store.collection, store.embedding, other_cfg.sparse_fingerprint)
    with pytest.raises(CollectionMismatch, match="sparse encoder"):
        other.check()


def test_upsert_count_states_and_delete(store):
    cob = [chunk("COB 3.3.1"), chunk("COB 3.3.2")]
    gen = [chunk("GEN 2.26.1", doc_id="dfsa-rulebook-gen")]
    store.upsert(points(cob + gen))
    assert store.count() == 3 and store.count("dfsa-rulebook-cob") == 2

    states = store.point_states("dfsa-rulebook-cob")
    assert set(states) == {c.chunk_id for c in cob}
    assert states[cob[0].chunk_id].content_hash == content_hash(cob[0], FINGERPRINT)
    assert store.scroll_hashes().keys() == {c.chunk_id for c in cob + gen}

    assert store.delete([cob[0].chunk_id]) == 1
    assert store.count("dfsa-rulebook-cob") == 1
    store.delete_document("dfsa-rulebook-gen")
    assert store.count() == 1


def test_overwrite_payloads_keeps_vectors(store):
    c = chunk()
    store.upsert(points([c]))
    before = store.retrieve([c.chunk_id], with_vectors=True)[0]
    moved = c.model_copy(update={"page_start": 40, "page_end": 41})
    new_payload = build_payload(moved, "rulebook", content_hash(moved, FINGERPRINT), FINGERPRINT)
    store.overwrite_payloads({c.chunk_id: new_payload})
    after = store.retrieve([c.chunk_id], with_vectors=True)[0]
    assert after.payload["page_start"] == 40
    assert after.payload["payload_hash"] != before.payload["payload_hash"]
    assert after.vector == before.vector


def test_get_by_citation_finds_a_rule_its_parts_and_its_guidance(store):
    chunks = [
        chunk("COB 3.3.2#p1", part_of="COB 3.3.2", part=1),
        chunk("COB 3.3.2#p2", part_of="COB 3.3.2", part=2),
        chunk("COB 3.3.2 Guidance", kind=ChunkType.GUIDANCE, guidance_for="COB 3.3.2"),
        chunk("COB 3.3.3"),
    ]
    # The factory cites a Guidance block under its own key; real Guidance cites its rule.
    chunks[2] = chunks[2].model_copy(update={"citation": "COB 3.3.2"})
    store.upsert(points(chunks))
    found = [r.payload["key"] for r in store.get_by_citation("COB 3.3.2")]
    assert found == ["COB 3.3.2#p1", "COB 3.3.2#p2", "COB 3.3.2 Guidance"]
    assert store.get_by_citation("COB 9.9.9") == []


def test_payload_hash_tracks_metadata_but_not_the_hash_inputs_alone():
    c = chunk(cross_refs=["GEN 2.26.1"])
    h = content_hash(c, FINGERPRINT)
    same = build_payload(c, "rulebook", h, FINGERPRINT)
    assert same == build_payload(c.model_copy(), "rulebook", h, FINGERPRINT)
    changed_refs = c.model_copy(update={"cross_refs": ["GEN 2.26.2"]})
    assert (
        build_payload(changed_refs, "rulebook", h, FINGERPRINT)["payload_hash"]
        != (same["payload_hash"])
    )
    assert same["binding"] is True and same["doc_type"] == "rulebook"
    assert same["embedding_fingerprint"] == FINGERPRINT and same["content_hash"] == h


def test_a_missing_collection_reads_as_empty(client):
    store = QdrantStore.from_settings(qdrant_settings(), embedding_settings(), client=client)
    assert not store.exists()
    assert store.point_states("dfsa-rulebook-cob") == {} and store.count() == 0
