import pytest

from src.ingestion.diff import StoredChunk, diff_chunks
from src.ingestion.models import ChunkType, content_hash
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
    assert diff.stale_point_ids == []
    assert set(diff.hashes) == {c.chunk_id for c in chunks}


def test_unchanged_reingest_embeds_nothing():
    chunks = [chunk("COB 3.3.1"), chunk("COB 3.3.2")]
    diff = diff_chunks(chunks, {c.key: stored(c) for c in chunks}, FINGERPRINT)
    assert len(diff.unchanged) == 2 and not (diff.new or diff.changed or diff.removed)
    assert diff.to_embed == [] and diff.stale_point_ids == []


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
    assert diff.stale_point_ids == []  # same id: the upsert overwrites the point


def test_a_new_embedding_model_reembeds_everything():
    chunks = [chunk("COB 3.3.1"), chunk("COB 3.3.2")]
    diff = diff_chunks(chunks, {c.key: stored(c) for c in chunks}, "text-embedding-3-large-3072")
    assert len(diff.changed) == 2 and len(diff.to_embed) == 2


def test_a_new_document_version_replaces_every_point():
    old = chunk(version="50")
    new = chunk(version="51")
    assert old.chunk_id != new.chunk_id
    diff = diff_chunks([new], {old.key: stored(old)}, FINGERPRINT)
    assert diff.changed == [new] and diff.to_embed == [new]
    assert diff.stale_point_ids == [old.chunk_id]


def test_removed_chunks_are_reported_and_their_points_are_stale():
    kept, gone, never_embedded = chunk("COB 3.3.1"), chunk("COB 3.3.2"), chunk("COB 3.3.3")
    store = {
        kept.key: stored(kept),
        gone.key: stored(gone),
        never_embedded.key: stored(never_embedded, embedded=False),
    }
    diff = diff_chunks([kept], store, FINGERPRINT)
    assert {r.key for r in diff.removed} == {"COB 3.3.2", "COB 3.3.3"}
    assert diff.stale_point_ids == [gone.chunk_id]


def test_a_chunk_that_stops_being_embeddable_leaves_a_stale_point():
    """E.g. a rule that grew past the limit is now a whole with parts."""
    before = chunk()
    after = chunk(parts=["COB 3.3.2#p1"])
    diff = diff_chunks([after], {before.key: stored(before)}, FINGERPRINT)
    assert diff.unchanged == [after]  # same text, same hash
    assert diff.to_embed == [] and diff.stale_point_ids == [after.chunk_id]


def test_duplicate_keys_are_rejected():
    with pytest.raises(ValueError, match="Duplicate chunk key"):
        diff_chunks([chunk(), chunk()], {}, FINGERPRINT)
