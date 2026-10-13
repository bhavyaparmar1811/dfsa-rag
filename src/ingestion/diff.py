"""What changed in a document since it was last ingested.

A fresh chunking of a document is compared, chunk by chunk, with what SQLite
holds for it, by `key` and `content_hash`:

    new         the key isn't stored yet
    changed     the key is stored, but its text, header, embedding model or id
                (a new document version) differs
    unchanged   same key, same id, same content_hash
    removed     stored, but no longer produced by the chunker

Only the embedding depends on `content_hash`, so an unchanged chunk is never
re-embedded; its row is still rewritten, because metadata the vector doesn't
depend on (pages, cross-references, Defined Terms) may have changed.
"""

from dataclasses import dataclass, field

from src.ingestion.models import Chunk, content_hash


@dataclass(frozen=True)
class StoredChunk:
    """The part of a stored row that change detection needs."""

    id: str
    key: str
    content_hash: str
    is_embedded: bool


@dataclass
class ChunkDiff:
    new: list[Chunk] = field(default_factory=list)
    changed: list[Chunk] = field(default_factory=list)
    unchanged: list[Chunk] = field(default_factory=list)
    removed: list[StoredChunk] = field(default_factory=list)
    # Embeddable chunks that need a vector: new, changed, or unchanged but never
    # embedded (e.g. loaded with --no-embed).
    to_embed: list[Chunk] = field(default_factory=list)
    # Points to delete from the index because nothing will overwrite them: removed
    # chunks, chunks whose id changed, chunks that are no longer embeddable.
    stale_point_ids: list[str] = field(default_factory=list)
    hashes: dict[str, str] = field(default_factory=dict)  # chunk_id -> content_hash

    @property
    def embed_tokens(self) -> int:
        """Estimated from the chunker's tiktoken counts; the API reports the real figure."""
        return sum(c.token_count for c in self.to_embed)


def diff_chunks(chunks: list[Chunk], stored: dict[str, StoredChunk], fingerprint: str) -> ChunkDiff:
    """Classify `chunks` against `stored` ({key: StoredChunk}) for one document."""
    diff = ChunkDiff()
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.key in seen:
            raise ValueError(f"Duplicate chunk key {chunk.key!r} in {chunk.doc_id}.")
        seen.add(chunk.key)

        h = content_hash(chunk, fingerprint)
        diff.hashes[chunk.chunk_id] = h
        old = stored.get(chunk.key)
        if old is None:
            diff.new.append(chunk)
            needs_vector = True
        else:
            same_id = old.id == chunk.chunk_id
            if same_id and old.content_hash == h:
                diff.unchanged.append(chunk)
            else:
                diff.changed.append(chunk)
            needs_vector = not same_id or old.content_hash != h or not old.is_embedded
            # An upsert under the same id overwrites the old point; anything else leaves it behind.
            if old.is_embedded and (not same_id or not chunk.embedded):
                diff.stale_point_ids.append(old.id)

        if chunk.embedded and needs_vector:
            diff.to_embed.append(chunk)

    for key, old in stored.items():
        if key not in seen:
            diff.removed.append(old)
            if old.is_embedded:
                diff.stale_point_ids.append(old.id)
    return diff
