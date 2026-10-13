"""Load the chunked corpus into SQLite (and, from step C4, into Qdrant).

    uv run python -m src.ingestion.ingest --dry-run      # what would change, and the cost
    uv run python -m src.ingestion.ingest --no-embed     # write SQLite only
    uv run python -m src.ingestion.ingest --doc dfsa-rulebook-glo --no-embed

Each document is re-chunked (from the cached Docling parse) and diffed against
SQLite by key + content_hash (see diff.py): only new and changed chunks need
embedding, so re-running on an unchanged corpus costs nothing. The Glossary goes
first, because the other documents are tagged with its Defined Terms.

Every document is written in its own transaction: rows, references, terms and
deletions land together or not at all. Each run (except --dry-run) is recorded
in `ingestion_runs`, including failed ones.

A full run (no --doc) also removes documents that are no longer in the manifest.
"""

import argparse
import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from sqlalchemy import Engine

from src.config.settings import ChunkingSettings, EmbeddingSettings, PathSettings, QdrantSettings
from src.ingestion.chunk_corpus import chunk_document, is_glossary, rulebook_codes, supported
from src.ingestion.diff import ChunkDiff, StoredChunk, diff_chunks
from src.ingestion.manifest import ManifestEntry, load_manifest
from src.ingestion.models import Chunk
from src.storage.db import assert_migrated, get_engine, session_scope
from src.storage.repositories import (
    ChunkRepository,
    DocumentRepository,
    IngestionRunRepository,
    RunStats,
)

logger = logging.getLogger(__name__)

Chunker = Callable[[ManifestEntry], list[Chunk]]


@dataclass
class DocResult:
    doc_id: str
    chunks: list[Chunk]
    diff: ChunkDiff


@dataclass
class IngestResult:
    docs: list[DocResult] = field(default_factory=list)
    # Documents in SQLite but no longer in the manifest, with the chunks removed.
    orphans: dict[str, list[StoredChunk]] = field(default_factory=dict)
    stats: RunStats = field(default_factory=RunStats)
    run_id: int | None = None

    @property
    def embed_tokens(self) -> int:
        return sum(d.diff.embed_tokens for d in self.docs)


def ingest(
    entries: list[ManifestEntry],
    chunker: Chunker,
    engine: Engine,
    *,
    fingerprint: str,
    sparse_model: str,
    dry_run: bool = False,
    prune_orphans: bool = False,
) -> IngestResult:
    """Chunk `entries` and write them to SQLite. Embedding is not done here yet (C4)."""
    assert_migrated(engine)
    entries = sorted(entries, key=lambda e: not is_glossary(e))  # GLO first
    result = IngestResult()

    if not dry_run:
        with session_scope(engine) as s:
            result.run_id = IngestionRunRepository(s).start(fingerprint, sparse_model)
    try:
        for entry in entries:
            result.docs.append(_ingest_document(entry, chunker, engine, fingerprint, dry_run))
        if prune_orphans:
            result.orphans = _prune_orphans({e.id for e in entries}, engine, dry_run)
    except Exception as e:
        if result.run_id is not None:
            with session_scope(engine) as s:
                IngestionRunRepository(s).finish(result.run_id, _stats(result), f"{e!r}")
        raise

    result.stats = _stats(result)
    if result.run_id is not None:
        with session_scope(engine) as s:
            IngestionRunRepository(s).finish(result.run_id, result.stats)
    return result


def _ingest_document(
    entry: ManifestEntry, chunker: Chunker, engine: Engine, fingerprint: str, dry_run: bool
) -> DocResult:
    chunks = chunker(entry)
    with session_scope(engine) as s:
        chunk_repo = ChunkRepository(s)
        diff = diff_chunks(chunks, chunk_repo.existing(entry.id), fingerprint)
        if dry_run:
            s.rollback()
        else:
            docs = DocumentRepository(s)
            docs.upsert(entry)
            chunk_repo.delete(old.key for old in diff.removed)
            chunk_repo.upsert(chunks, diff.hashes)
            docs.update_counts(entry.id)
    logger.info(
        "%s: %d chunks (%d new, %d changed, %d unchanged, %d removed)%s",
        entry.id, len(chunks), len(diff.new), len(diff.changed), len(diff.unchanged),
        len(diff.removed), " [dry run]" if dry_run else "",
    )  # fmt: skip
    return DocResult(entry.id, chunks, diff)


def _prune_orphans(
    manifest_ids: set[str], engine: Engine, dry_run: bool
) -> dict[str, list[StoredChunk]]:
    orphans = {}
    with session_scope(engine) as s:
        docs = DocumentRepository(s)
        for doc_id in sorted(docs.ids() - manifest_ids):
            if dry_run:
                orphans[doc_id] = list(ChunkRepository(s).existing(doc_id).values())
            else:
                orphans[doc_id] = docs.delete(doc_id)
            logger.info("%s: not in the manifest, %d chunks removed", doc_id, len(orphans[doc_id]))
        if dry_run:
            s.rollback()
    return orphans


def _stats(result: IngestResult) -> RunStats:
    return RunStats(
        docs_processed=len(result.docs),
        chunks_new=sum(len(d.diff.new) for d in result.docs),
        chunks_changed=sum(len(d.diff.changed) for d in result.docs),
        chunks_unchanged=sum(len(d.diff.unchanged) for d in result.docs),
        chunks_removed=sum(len(d.diff.removed) for d in result.docs)
        + sum(len(v) for v in result.orphans.values()),
        # Nothing is embedded until C4; the dry-run estimate is printed, not recorded.
        embed_tokens=0,
        embed_cost_usd=0.0,
    )


def print_summary(result: IngestResult, cost_per_1m: float, dry_run: bool) -> None:
    cols = (
        f"{'document':<52}{'chunks':>7}{'embed':>7}{'new':>6}{'chg':>6}{'same':>6}"
        f"{'del':>5}{'to embed':>9}{'tokens':>9}"
    )
    print("\n" + cols + "\n" + "-" * len(cols))
    for d in result.docs:
        embeddable = sum(c.embedded for c in d.chunks)
        print(
            f"{d.doc_id:<52}{len(d.chunks):>7}{embeddable:>7}{len(d.diff.new):>6}"
            f"{len(d.diff.changed):>6}{len(d.diff.unchanged):>6}{len(d.diff.removed):>5}"
            f"{len(d.diff.to_embed):>9}{d.diff.embed_tokens:>9,}"
        )
    for doc_id, removed in result.orphans.items():
        print(f"{doc_id + ' (not in manifest)':<52}{0:>7}{0:>7}{0:>6}{0:>6}{0:>6}{len(removed):>5}")
    print("-" * len(cols))

    to_embed = sum(len(d.diff.to_embed) for d in result.docs)
    tokens = result.embed_tokens
    cost = tokens * cost_per_1m / 1_000_000
    print(
        f"{'total':<52}{sum(len(d.chunks) for d in result.docs):>7}"
        f"{sum(sum(c.embedded for c in d.chunks) for d in result.docs):>7}"
        f"{result.stats.chunks_new:>6}{result.stats.chunks_changed:>6}"
        f"{result.stats.chunks_unchanged:>6}{result.stats.chunks_removed:>5}"
        f"{to_embed:>9}{tokens:>9,}"
    )
    verb = "would need" if dry_run else "still need"
    print(f"\n{to_embed} chunks {verb} embedding: ~{tokens:,} tokens, ~${cost:.4f} (estimate)")
    if result.run_id is not None:
        print(f"Recorded as ingestion_runs.id = {result.run_id}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Load the chunked corpus into SQLite.")
    ap.add_argument("--doc", help="manifest id; default is every supported document")
    ap.add_argument("--dry-run", action="store_true", help="show what would change; write nothing")
    ap.add_argument("--no-embed", action="store_true", help="write SQLite only, embed nothing")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("docling", "httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if not (args.dry_run or args.no_embed):
        logger.error("Embedding into Qdrant is step C4 and isn't built yet: use --no-embed.")
        return 2

    manifest = load_manifest()
    entries = [e for e in manifest if args.doc in (None, e.id) and supported(e)]
    if not entries:
        logger.error("No supported manifest entry with id %r.", args.doc)
        return 1

    cfg, paths, emb = ChunkingSettings(), PathSettings(), EmbeddingSettings()
    codes = rulebook_codes(manifest)
    start = time.perf_counter()
    result = ingest(
        entries,
        lambda e: chunk_document(e, cfg, paths, codes).chunks,
        get_engine(),
        fingerprint=emb.fingerprint,
        sparse_model=QdrantSettings().sparse_model,
        dry_run=args.dry_run,
        prune_orphans=args.doc is None,
    )
    print_summary(result, emb.cost_per_1m, args.dry_run)
    print(f"Done in {time.perf_counter() - start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
