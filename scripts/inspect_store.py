"""Show one chunk as stored in SQLite and in Qdrant, side by side.

    uv run python -m scripts.inspect_store "GLO Retail Client"
    uv run python -m scripts.inspect_store "COB 3.3.2"          # a split rule: shows its parts

SQLite: the row, its references and Defined Terms, its embedding state.
Qdrant: whether the point exists, its vector sizes, its strongest BM25 terms,
and whether its content_hash agrees with SQLite's.
"""

import argparse
import math
import sys
import textwrap

from src.config.settings import EmbeddingSettings, QdrantSettings
from src.embeddings.sparse import SparseEncoder
from src.storage.db import session_scope
from src.storage.qdrant_store import DENSE, SPARSE, QdrantStore
from src.storage.repositories import ChunkRepository

PAYLOAD_SHOWN = (
    "citation", "chunk_type", "binding", "regulator", "doc_type", "module", "doc_id", "version",
    "effective_date", "section", "parent_key", "part_of", "guidance_for", "rule_number", "term",
    "scope_rule", "page_start", "page_end", "embedding_fingerprint",
)  # fmt: skip


def show_row(row, repo: ChunkRepository) -> None:
    print(f"== SQLite: {row.key}")
    print(f"  id            {row.id}")
    print(f"  document      {row.document_id}  (order {row.order_idx}, pages {row.page_start}-"
          f"{row.page_end})")  # fmt: skip
    print(f"  type          {row.chunk_type}, binding={row.binding}, scope_rule={row.scope_rule}, "
          f"deleted={row.deleted}")  # fmt: skip
    print(f"  citation      {row.citation}   parent {row.parent_key}   part_of {row.part_of}")
    if row.parts:
        print(f"  parts         {', '.join(row.parts)}  (the whole is not embedded)")
    print(f"  tokens        {row.token_count}")
    print(f"  content_hash  {row.content_hash}")
    state = "no"
    if row.is_embedded:
        when = f"{row.embedded_at:%Y-%m-%d %H:%M}"
        state = f"yes, {row.embedding_model} ({row.embedding_dimensions}) at {when}"
    print(f"  embedded      {state}")
    print(f"  {'cross_refs':<22}{', '.join(repo.refs_of(row.id)) or '-'}")
    print(f"  terms         {', '.join(repo.terms_of(row.id)) or '-'}")
    print(f"  header        {row.header_text}")
    print(textwrap.indent(textwrap.shorten(row.text, 400), "  text          ").rstrip())


def show_point(row, store: QdrantStore, sparse: SparseEncoder) -> None:
    print(f"\n== Qdrant: {store.collection}")
    found = store.retrieve([row.id], with_vectors=True) if store.exists() else []
    if not found:
        print(
            "  no point"
            + ("" if row.parts or row.chunk_type == "section" else " (not embedded yet?)")
        )
        return
    point = found[0]
    dense, bm25 = point.vector[DENSE], point.vector[SPARSE]
    payload = point.payload
    print(f"  dense         {len(dense)} dims, norm {math.sqrt(sum(x * x for x in dense)):.4f}")
    vocab = sparse.vocabulary(f"{row.header_text}\n\n{row.text}")
    top = sorted(zip(bm25.indices, bm25.values, strict=True), key=lambda iv: -iv[1])[:12]
    terms = ", ".join(f"{vocab.get(i, i)} {v:.2f}" for i, v in top)
    print(f"  bm25          {len(bm25.indices)} terms; strongest (TF part, before IDF): {terms}")
    match = "matches SQLite" if payload.get("content_hash") == row.content_hash else "DIFFERS"
    print(f"  content_hash  {payload.get('content_hash')}  ({match})")
    print(f"  payload_hash  {payload.get('payload_hash')}")
    for name in PAYLOAD_SHOWN:
        print(f"  {name:<22}{payload.get(name)}")
    print(f"  {'defined_terms':<22}{', '.join(payload.get('defined_terms') or []) or '-'}")
    print(f"  {'cross_refs':<22}{', '.join(payload.get('cross_refs') or []) or '-'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Show a chunk in SQLite and Qdrant.")
    ap.add_argument("key", help='chunk key, e.g. "GLO Retail Client" or "COB 3.3.2"')
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    qdrant = QdrantSettings()
    store = QdrantStore.from_settings(qdrant, EmbeddingSettings())
    sparse = SparseEncoder(qdrant)
    with session_scope() as s:
        repo = ChunkRepository(s)
        row = repo.by_key(args.key)
        if row is None:
            print(f"No chunk with key {args.key!r} in SQLite.")
            return 1
        rows = [row, *repo.parts_of(row.key)]
        for i, r in enumerate(rows):
            if i:
                print("\n" + "=" * 80)
            show_row(r, repo)
            show_point(r, store, sparse)
    return 0


if __name__ == "__main__":
    sys.exit(main())
