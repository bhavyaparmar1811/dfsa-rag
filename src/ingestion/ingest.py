"""Ingest the corpus: SQLite (system of record) and Qdrant (search index).

    uv run python -m src.ingestion.ingest --dry-run      # what would change, and the cost
    uv run python -m src.ingestion.ingest                # embed + index + SQLite
    uv run python -m src.ingestion.ingest --doc dfsa-rulebook-glo
    uv run python -m src.ingestion.ingest --no-embed     # SQLite only, Qdrant untouched
    uv run python -m src.ingestion.ingest --reconcile    # do SQLite and Qdrant agree?

Each document is re-chunked (from the cached Docling parse), then:

    1. diffed against SQLite by key + content_hash (diff_chunks), and its
       embeddable chunks planned against the points in Qdrant (plan_index);
    2. new or changed chunks are embedded (OpenAI, dense) and BM25-encoded
       (fastembed, locally), then upserted; payload-only changes are rewritten;
       stale points are deleted, all with wait=True;
    3. SQLite is updated in one transaction: rows, references, terms,
       deletions, and the embedding state of every chunk with a current point.

Because step 1 reads Qdrant itself, a crash between steps 2 and 3, or a lost
collection, is repaired by the next run, and only what is missing is embedded.
Re-running on an unchanged corpus embeds nothing and costs $0.

The Glossary goes first (the others are tagged with its Defined Terms). Each
run, except --dry-run, is recorded in `ingestion_runs`, failed ones included.
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
from src.embeddings.client import EmbeddingClient, EmbeddingModelMismatch
from src.embeddings.sparse import SparseEncoder
from src.ingestion.chunk_corpus import chunk_document, is_glossary, rulebook_codes, supported
from src.ingestion.diff import ChunkDiff, IndexPlan, StoredChunk, diff_chunks, plan_index
from src.ingestion.manifest import ManifestEntry, load_manifest
from src.ingestion.models import Chunk
from src.ingestion.reconcile import fix, print_report, reconcile
from src.storage.db import assert_migrated, get_engine, session_scope
from src.storage.qdrant_store import Point, QdrantStore, build_payload
from src.storage.repositories import (
    ChunkRepository,
    DocumentRepository,
    IngestionRunRepository,
    RunStats,
)

logger = logging.getLogger(__name__)

Chunker = Callable[[ManifestEntry], list[Chunk]]
# Warn when the corpus's mean BM25 length drifts this far from QdrantSettings.sparse_avg_len.
AVG_LEN_TOLERANCE = 0.15


@dataclass
class Index:
    """Everything embedding needs: the dense client, the BM25 encoder, the store."""

    store: QdrantStore
    embedder: EmbeddingClient
    sparse: SparseEncoder


@dataclass
class DocResult:
    doc_id: str
    chunks: list[Chunk]
    diff: ChunkDiff
    plan: IndexPlan | None = None  # None with --no-embed
    embed_tokens: int = 0  # as reported by the API
    embed_cost_usd: float = 0.0

    @property
    def to_embed(self) -> list[Chunk]:
        return self.plan.to_embed if self.plan else self.diff.to_embed


@dataclass
class IngestResult:
    docs: list[DocResult] = field(default_factory=list)
    # Documents in SQLite but no longer in the manifest, with the chunks removed.
    orphans: dict[str, list[StoredChunk]] = field(default_factory=dict)
    stats: RunStats = field(default_factory=RunStats)
    run_id: int | None = None
    collection_created: bool = False

    @property
    def estimated_tokens(self) -> int:
        return sum(c.token_count for d in self.docs for c in d.to_embed)


def ingest(
    entries: list[ManifestEntry],
    chunker: Chunker,
    engine: Engine,
    *,
    fingerprint: str,
    sparse_model: str,
    index: Index | None = None,
    dry_run: bool = False,
    prune_orphans: bool = False,
) -> IngestResult:
    """Chunk `entries` and write them to SQLite and, given an `index`, to Qdrant."""
    assert_migrated(engine)
    if index is not None:
        if index.store.embedding.fingerprint != fingerprint:
            raise EmbeddingModelMismatch(
                f"The store is configured for {index.store.embedding.fingerprint}, "
                f"ingest for {fingerprint}."
            )
        if index.sparse.fingerprint != sparse_model:
            raise ValueError(f"Sparse encoder {index.sparse.fingerprint} != {sparse_model}.")

    entries = sorted(entries, key=lambda e: not is_glossary(e))  # GLO first
    result = IngestResult()
    if not dry_run:
        with session_scope(engine) as s:
            result.run_id = IngestionRunRepository(s).start(fingerprint, sparse_model)
    try:
        if index is not None:
            if not dry_run:
                result.collection_created = index.store.ensure_collection()
            elif index.store.exists():
                index.store.check()
        for entry in entries:
            result.docs.append(
                _ingest_document(entry, chunker, engine, fingerprint, index, dry_run)
            )
        if prune_orphans:
            result.orphans = _prune_orphans({e.id for e in entries}, engine, index, dry_run)
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
    entry: ManifestEntry,
    chunker: Chunker,
    engine: Engine,
    fingerprint: str,
    index: Index | None,
    dry_run: bool,
) -> DocResult:
    chunks = chunker(entry)
    with session_scope(engine) as s:
        diff = diff_chunks(chunks, ChunkRepository(s).existing(entry.id), fingerprint)
    doc = DocResult(entry.id, chunks, diff)

    if index is not None:
        payloads = {
            c.chunk_id: build_payload(c, entry.doc_type, diff.hashes[c.chunk_id], fingerprint)
            for c in chunks
            if c.embedded
        }
        doc.plan = plan_index(
            chunks,
            diff.hashes,
            {pid: p["payload_hash"] for pid, p in payloads.items()},
            index.store.point_states(entry.id),
        )
        if not dry_run:
            _write_index(index, doc, payloads)

    if not dry_run:
        with session_scope(engine) as s:
            docs, chunk_repo = DocumentRepository(s), ChunkRepository(s)
            docs.upsert(entry)
            chunk_repo.delete(old.key for old in diff.removed)
            chunk_repo.upsert(chunks, diff.hashes)
            if doc.plan is not None:
                cfg = index.store.embedding
                chunk_repo.mark_embedded(
                    [c.chunk_id for c in doc.plan.to_embed], cfg.model, cfg.dimensions
                )
                # Points that were already current (e.g. SQLite was reloaded with --no-embed).
                chunk_repo.mark_embedded(
                    doc.plan.indexed_ids, cfg.model, cfg.dimensions, only_unmarked=True
                )
            docs.update_counts(entry.id)

    plan = doc.plan
    logger.info(
        "%s: %d chunks (%d new, %d changed, %d unchanged, %d removed)%s%s",
        entry.id, len(chunks), len(diff.new), len(diff.changed), len(diff.unchanged),
        len(diff.removed),
        f"; index: {len(plan.to_embed)} to embed, {len(plan.payload_only)} payload updates, "
        f"{len(plan.stale_ids)} stale" if plan else "",
        " [dry run]" if dry_run else "",
    )  # fmt: skip
    return doc


def _write_index(index: Index, doc: DocResult, payloads: dict[str, dict]) -> None:
    plan = doc.plan
    if plan.to_embed:
        texts = [c.embedding_text() for c in plan.to_embed]
        dense = index.embedder.embed_documents(texts)
        if dense.fingerprint != index.store.embedding.fingerprint:
            raise EmbeddingModelMismatch(
                f"Embedded with {dense.fingerprint}, collection is "
                f"{index.store.embedding.fingerprint}."
            )
        sparse = index.sparse.encode_documents(texts)
        index.store.upsert(
            [
                Point(c.chunk_id, vector, sv, payloads[c.chunk_id])
                for c, vector, sv in zip(plan.to_embed, dense.vectors, sparse, strict=True)
            ]
        )
        doc.embed_tokens, doc.embed_cost_usd = dense.prompt_tokens, dense.cost_usd
    if plan.payload_only:
        index.store.overwrite_payloads({pid: payloads[pid] for pid in plan.payload_only})
    index.store.delete(plan.stale_ids)


def _prune_orphans(
    manifest_ids: set[str], engine: Engine, index: Index | None, dry_run: bool
) -> dict[str, list[StoredChunk]]:
    orphans = {}
    with session_scope(engine) as s:
        docs = DocumentRepository(s)
        for doc_id in sorted(docs.ids() - manifest_ids):
            if dry_run:
                orphans[doc_id] = list(ChunkRepository(s).existing(doc_id).values())
            else:
                if index is not None:
                    index.store.delete_document(doc_id)
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
        embed_tokens=sum(d.embed_tokens for d in result.docs),
        embed_cost_usd=sum(d.embed_cost_usd for d in result.docs),
    )


def print_summary(result: IngestResult, cost_per_1m: float, dry_run: bool) -> None:
    indexed = any(d.plan for d in result.docs)
    cols = (
        f"{'document':<52}{'chunks':>7}{'embed':>7}{'new':>6}{'chg':>6}{'same':>6}"
        f"{'del':>5}{'to embed':>9}{'tokens':>9}"
    )
    if indexed:
        cols += f"{'payload':>8}{'stale':>6}"
    print("\n" + cols + "\n" + "-" * len(cols))

    def row(name, chunks, embeddable, new, chg, same, removed, to_embed, tokens, payload, stale):
        line = (
            f"{name:<52}{chunks:>7}{embeddable:>7}{new:>6}{chg:>6}{same:>6}{removed:>5}"
            f"{to_embed:>9}{tokens:>9,}"
        )
        print(line + (f"{payload:>8}{stale:>6}" if indexed else ""))

    totals = [0] * 10
    for d in result.docs:
        values = [
            len(d.chunks), sum(c.embedded for c in d.chunks), len(d.diff.new),
            len(d.diff.changed), len(d.diff.unchanged), len(d.diff.removed), len(d.to_embed),
            sum(c.token_count for c in d.to_embed),
            len(d.plan.payload_only) if d.plan else 0, len(d.plan.stale_ids) if d.plan else 0,
        ]  # fmt: skip
        totals = [t + v for t, v in zip(totals, values, strict=True)]
        row(d.doc_id, *values)
    for doc_id, removed in result.orphans.items():
        row(f"{doc_id} (not in manifest)", 0, 0, 0, 0, 0, len(removed), 0, 0, 0, 0)
        totals[5] += len(removed)
    print("-" * len(cols))
    row("total", *totals)

    to_embed, tokens = totals[6], totals[7]
    estimate = tokens * cost_per_1m / 1_000_000
    if not indexed:
        verb = "would need" if dry_run else "still need"
        print(
            f"\n{to_embed} chunks {verb} embedding: ~{tokens:,} tokens, ~${estimate:.4f} (estimate)"
        )
    elif dry_run:
        print(
            f"\nWould embed {to_embed} chunks: ~{tokens:,} tokens, ~${estimate:.4f} (estimate); "
            f"rewrite {totals[8]} payloads; delete {totals[9]} stale points"
        )
    else:
        print(
            f"\nEmbedded {to_embed} chunks: {result.stats.embed_tokens:,} tokens, "
            f"${result.stats.embed_cost_usd:.6f} (from the API; estimate was ~{tokens:,}); "
            f"rewrote {totals[8]} payloads; deleted {totals[9]} stale points"
        )
    if result.run_id is not None:
        print(f"Recorded as ingestion_runs.id = {result.run_id}")


def check_avg_len(result: IngestResult, sparse: SparseEncoder) -> None:
    """Warn if the corpus's mean BM25 length has drifted from the configured avg_len."""
    texts = [c.embedding_text() for d in result.docs for c in d.chunks if c.embedded]
    if not texts:
        return
    measured = sparse.token_count(texts) / len(texts)
    configured = sparse.cfg.sparse_avg_len
    if abs(measured - configured) / configured > AVG_LEN_TOLERANCE:
        logger.warning(
            "BM25: the corpus's mean chunk length is %.1f tokens, QDRANT_SPARSE_AVG_LEN is %g. "
            "Update it and rebuild the collection.", measured, configured,
        )  # fmt: skip
    else:
        print(f"BM25 avg_len: measured {measured:.1f}, configured {configured:g} (ok)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest the corpus into SQLite and Qdrant.")
    ap.add_argument("--doc", help="manifest id; default is every supported document")
    ap.add_argument("--dry-run", action="store_true", help="show what would change; write nothing")
    ap.add_argument("--no-embed", action="store_true", help="write SQLite only; skip Qdrant")
    ap.add_argument(
        "--reconcile", action="store_true", help="only check that SQLite and Qdrant agree"
    )
    ap.add_argument("--fix", action="store_true", help="with --reconcile: repair what it finds")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("docling", "httpx", "huggingface_hub", "alembic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.fix and not args.reconcile:
        ap.error("--fix only works with --reconcile")
    if args.reconcile:
        return run_reconcile(args.fix)

    manifest = load_manifest()
    entries = [e for e in manifest if args.doc in (None, e.id) and supported(e)]
    if not entries:
        logger.error("No supported manifest entry with id %r.", args.doc)
        return 1

    cfg, paths, emb, qdrant = (
        ChunkingSettings(),
        PathSettings(),
        EmbeddingSettings(),
        QdrantSettings(),
    )
    index = None
    if not args.no_embed:
        if emb.api_key is None and not args.dry_run:
            logger.error("OPENAI_API_KEY is not set: add it to .env, or use --no-embed.")
            return 1
        sparse = SparseEncoder(qdrant)
        embedder = EmbeddingClient(emb) if emb.api_key else None  # a dry run embeds nothing
        index = Index(QdrantStore.from_settings(qdrant, emb), embedder, sparse)

    codes = rulebook_codes(manifest)
    start = time.perf_counter()
    result = ingest(
        entries,
        lambda e: chunk_document(e, cfg, paths, codes).chunks,
        get_engine(),
        fingerprint=emb.fingerprint,
        sparse_model=qdrant.sparse_fingerprint,
        index=index,
        dry_run=args.dry_run,
        prune_orphans=args.doc is None,
    )
    print_summary(result, emb.cost_per_1m, args.dry_run)
    if index is not None:
        if args.doc is None:
            check_avg_len(result, index.sparse)
        if not args.dry_run:
            store = index.store
            created = " (created)" if result.collection_created else ""
            print(f"Collection {store.collection}{created}: {store.count():,} points")
    print(f"Done in {time.perf_counter() - start:.0f} s")
    return 0


def run_reconcile(apply_fix: bool) -> int:
    emb = EmbeddingSettings()
    store = QdrantStore.from_settings(QdrantSettings(), emb)
    engine = get_engine()
    assert_migrated(engine)
    if not store.exists():
        logger.error("Collection %s doesn't exist yet: run ingest first.", store.collection)
        return 1
    report = reconcile(engine, store, emb)
    print_report(report, store)
    if report.clean:
        return 0
    if apply_fix:
        fix(engine, store, report)
        print(
            f"\nFixed: deleted {len(report.orphan_points)} orphan points, un-marked "
            f"{len(report.rows_to_unmark)} rows. Run ingest to embed what is missing."
        )
    else:
        print("\nRun ingest (it re-embeds what is missing), or --reconcile --fix for orphans.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
