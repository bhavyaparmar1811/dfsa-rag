import pytest

from src.ingestion.diff import StoredChunk, diff_chunks, plan_index
from src.ingestion.models import ChunkType, content_hash
from src.storage.qdrant_store import PointState
from tests.factories import FINGERPRINT, chunk


def stored(c, *, embedded=True, fingerprint=FINGERPRINT, id=None):
    return StoredChunk(id or c.chunk_id, c.key, content_hash(c, fingerprint), embedded)


def test_content_hash_covers_text_header_and_model():
    c = chunk()
    assert content_hash(c, FINGERPRINT) == content_hash(chunk(), FINGERPRINT)
    assert content_hash(c, FINGERPRINT) != content_hash(chunk(text="other"), FINGERPRINT)
    assert content_hash(c, FINGERPRINT) != content_hash(c, "text-embedding-3-large-3072")
    header_changed = c.model_copy(update={"header_text": "COB > Section 3.3 > renamed"})
    assert content_hash(c, FINGERPRINT) != content_hash(header_changed, FINGERPRINT)
    # Metadata the vector doesn't depend on doesn't change the hash.
    pages_changed = c.model_copy(update={"page_start": 9, "cross_refs": ["GEN 2.2.1"]})
    assert content_hash(c, FINGERPRINT) == content_hash(pages_changed, FINGERPRINT)


def test_first_ingest_is_all_new_and_embeds_only_embeddable_chunks():
    chunks = [
        chunk("COB 3.3", kind=ChunkType.SECTION),
        chunk("COB 3.3.1"),
        chunk("COB 3.3.2", parts=["COB 3.3.2#p1", "COB 3.3.2#p2"]),  # whole: not embedded
        chunk("COB 3.3.2#p1", part_of="COB 3.3.2", part=1),
        chunk("COB 3.3.2#p2", part_of="COB 3.3.2", part=2),
        chunk("COB 3.3.3", deleted=True),
    ]
    diff = diff_chunks(chunks, {}, FINGERPRINT)
    assert len(diff.new) == 6 and not (diff.changed or diff.unchanged or diff.removed)
    assert [c.key for c in diff.to_embed] == ["COB 3.3.1", "COB 3.3.2#p1", "COB 3.3.2#p2"]
    assert diff.embed_tokens == 30
    assert set(diff.hashes) == {c.chunk_id for c in chunks}


def test_unchanged_reingest_embeds_nothing():
    chunks = [chunk("COB 3.3.1"), chunk("COB 3.3.2")]
    diff = diff_chunks(chunks, {c.key: stored(c) for c in chunks}, FINGERPRINT)
    assert len(diff.unchanged) == 2 and not (diff.new or diff.changed or diff.removed)
    assert diff.to_embed == []


def test_unchanged_but_never_embedded_still_needs_a_vector():
    """A chunk loaded with --no-embed is embedded by the next full ingest."""
    c = chunk()
    diff = diff_chunks([c], {c.key: stored(c, embedded=False)}, FINGERPRINT)
    assert diff.unchanged == [c] and diff.to_embed == [c]


def test_editing_one_chunk_reembeds_only_that_chunk():
    before = [chunk("COB 3.3.1"), chunk("COB 3.3.2")]
    after = [chunk("COB 3.3.1"), chunk("COB 3.3.2", text="amended text")]
    diff = diff_chunks(after, {c.key: stored(c) for c in before}, FINGERPRINT)
    assert [c.key for c in diff.changed] == ["COB 3.3.2"]
    assert [c.key for c in diff.to_embed] == ["COB 3.3.2"]


def test_a_new_embedding_model_reembeds_everything():
    chunks = [chunk("COB 3.3.1"), chunk("COB 3.3.2")]
    diff = diff_chunks(chunks, {c.key: stored(c) for c in chunks}, "text-embedding-3-large-3072")
    assert len(diff.changed) == 2 and len(diff.to_embed) == 2


def test_a_new_document_version_changes_every_id():
    old = chunk(version="50")
    new = chunk(version="51")
    assert old.chunk_id != new.chunk_id
    diff = diff_chunks([new], {old.key: stored(old)}, FINGERPRINT)
    assert diff.changed == [new] and diff.to_embed == [new]


def test_removed_chunks_are_reported():
    kept, gone = chunk("COB 3.3.1"), chunk("COB 3.3.2")
    diff = diff_chunks([kept], {kept.key: stored(kept), gone.key: stored(gone)}, FINGERPRINT)
    assert [r.key for r in diff.removed] == ["COB 3.3.2"]


# --- plan_index: the embeddable chunks against the points in Qdrant -----------


def plan(chunks, points, payload_hashes=None):
    hashes = {c.chunk_id: content_hash(c, FINGERPRINT) for c in chunks}
    payload_hashes = payload_hashes or {c.chunk_id: "p" for c in chunks}
    return plan_index(chunks, hashes, payload_hashes, points)


def point(c, payload_hash="p", fingerprint=FINGERPRINT):
    return {c.chunk_id: PointState(content_hash(c, fingerprint), payload_hash)}


def test_plan_with_an_empty_collection_embeds_every_embeddable_chunk():
    chunks = [chunk("COB 3.3", kind=ChunkType.SECTION), chunk("COB 3.3.1"), chunk("COB 3.3.2")]
    p = plan(chunks, {})
    assert [c.key for c in p.to_embed] == ["COB 3.3.1", "COB 3.3.2"]
    assert p.payload_only == p.stale_ids == p.current == []
    assert p.embed_tokens == 20


def test_plan_with_current_points_does_nothing():
    chunks = [chunk("COB 3.3.1"), chunk("COB 3.3.2")]
    p = plan(chunks, point(chunks[0]) | point(chunks[1]))
    assert p.to_embed == [] and p.stale_ids == [] and p.payload_only == []
    assert p.current == [c.chunk_id for c in chunks] and p.indexed_ids == p.current


def test_plan_reembeds_a_changed_chunk_and_a_missing_point():
    """A missing point is re-embedded even if SQLite says it is embedded: the index heals."""
    before = chunk("COB 3.3.1")
    after = [chunk("COB 3.3.1", text="amended"), chunk("COB 3.3.2")]
    p = plan(after, point(before))
    assert [c.key for c in p.to_embed] == ["COB 3.3.1", "COB 3.3.2"]
    assert p.stale_ids == []  # same id: the upsert overwrites the point


def test_plan_rewrites_only_the_payload_when_only_metadata_changed():
    c = chunk()
    p = plan([c], point(c, payload_hash="old"), {c.chunk_id: "new"})
    assert p.to_embed == [] and p.payload_only == [c.chunk_id]
    assert p.indexed_ids == [c.chunk_id]


def test_plan_deletes_points_nothing_maps_to():
    """Removed chunks, old-version ids, and chunks that stopped being embeddable."""
    removed = chunk("COB 3.3.9")
    old_version = chunk("COB 3.3.1", version="50")
    grew_parts = chunk("COB 3.3.2")
    now = [chunk("COB 3.3.1"), chunk("COB 3.3.2", parts=["COB 3.3.2#p1"])]
    p = plan(now, point(removed) | point(old_version) | point(grew_parts))
    assert set(p.stale_ids) == {removed.chunk_id, old_version.chunk_id, grew_parts.chunk_id}
    assert [c.key for c in p.to_embed] == ["COB 3.3.1"]


def test_plan_reembeds_points_from_another_model():
    c = chunk()
    p = plan([c], point(c, fingerprint="text-embedding-3-large-3072"))
    assert p.to_embed == [c]


def test_duplicate_keys_are_rejected():
    with pytest.raises(ValueError, match="Duplicate chunk key"):
        diff_chunks([chunk(), chunk()], {}, FINGERPRINT)
