"""What changed in a document since it was last ingested.

Two comparisons, both pure functions:

`diff_chunks()`: a fresh chunking against what SQLite holds, by `key` and
`content_hash`. This drives the SQLite writes and the run's counts.

    new         the key isn't stored yet
    changed     the key is stored, but its text, header, embedding model or id
                (a new document version) differs
    unchanged   same key, same id, same content_hash
    removed     stored, but no longer produced by the chunker

`plan_index()`: the embeddable chunks against the points actually in Qdrant.
This decides what is embedded, so the index heals itself: a dropped collection,
or a crash between the Qdrant write and the SQLite commit, is repaired by the
next run without re-embedding anything that is already there.

    to_embed      no point, or a point whose content_hash differs: embed + upsert
    payload_only  the vector is current but the payload isn't (pages, references,
                  terms changed): rewrite the payload, no embedding
    stale_ids     points of this document that no chunk maps to any more: delete
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
    # Embeddable chunks SQLite doesn't record as embedded with their current hash.
    # Used for estimates when Qdrant isn't consulted (--no-embed).
    to_embed: list[Chunk] = field(default_factory=list)
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
        elif old.id == chunk.chunk_id and old.content_hash == h:
            diff.unchanged.append(chunk)
        else:
            diff.changed.append(chunk)

        current = old is not None and old.id == chunk.chunk_id and old.content_hash == h
        if chunk.embedded and not (current and old.is_embedded):
            diff.to_embed.append(chunk)

    diff.removed = [old for key, old in stored.items() if key not in seen]
    return diff


@dataclass
class IndexPlan:
    to_embed: list[Chunk] = field(default_factory=list)
    payload_only: list[str] = field(default_factory=list)  # point ids
    stale_ids: list[str] = field(default_factory=list)
    current: list[str] = field(default_factory=list)  # point ids already up to date

    @property
    def embed_tokens(self) -> int:
        return sum(c.token_count for c in self.to_embed)

    @property
    def indexed_ids(self) -> list[str]:
        """Every embeddable chunk's id: after the plan runs, each has a current point."""
        return [c.chunk_id for c in self.to_embed] + self.payload_only + self.current


def plan_index(
    chunks: list[Chunk],
    hashes: dict[str, str],
    payload_hashes: dict[str, str],
    points: dict,
) -> IndexPlan:
    """Plan one document's index update.

    `hashes` and `payload_hashes` are {chunk_id: hash} for the new chunks;
    `points` is {point id: PointState} for the document's points in Qdrant.
    """
    plan = IndexPlan()
    wanted = set()
    for chunk in chunks:
        if not chunk.embedded:
            continue
        wanted.add(chunk.chunk_id)
        point = points.get(chunk.chunk_id)
        if point is None or point.content_hash != hashes[chunk.chunk_id]:
            plan.to_embed.append(chunk)
        elif point.payload_hash != payload_hashes[chunk.chunk_id]:
            plan.payload_only.append(chunk.chunk_id)
        else:
            plan.current.append(chunk.chunk_id)
    plan.stale_ids = sorted(pid for pid in points if pid not in wanted)
    return plan
