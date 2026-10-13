"""Check that SQLite and Qdrant agree, chunk by chunk.

    uv run python -m src.ingestion.ingest --reconcile          # report; exit 1 if not clean
    uv run python -m src.ingestion.ingest --reconcile --fix    # then repair what ingest can't

Clean means:
    - the collection matches the configuration (embedding model, BM25 settings, schema);
    - every embeddable row is marked embedded, with the configured model and dimension;
    - every row marked embedded has a point with the same content_hash;
    - every point belongs to an embeddable row (no orphans), so points == embeddable rows.

`--fix` deletes orphan points and un-marks rows whose point is missing or stale.
Re-running `ingest` then re-embeds exactly those rows (it plans against Qdrant).
"""

from dataclasses import dataclass, field

from sqlalchemy import Engine

from src.config.settings import EmbeddingSettings
from src.ingestion.models import EMBEDDED_TYPES
from src.storage.db import session_scope
from src.storage.qdrant_store import QdrantStore
from src.storage.repositories import ChunkRepository


@dataclass
class ReconcileReport:
    rows: int = 0
    embeddable: int = 0
    points: int = 0
    # chunk keys, except orphan_points (point ids: there may be no row to name them by)
    unembedded: list[str] = field(default_factory=list)
    missing_points: list[str] = field(default_factory=list)
    hash_mismatch: list[str] = field(default_factory=list)
    wrong_model: list[str] = field(default_factory=list)
    marked_not_embeddable: list[str] = field(default_factory=list)
    orphan_points: list[str] = field(default_factory=list)
    # ids of rows whose embedding state is wrong, for --fix
    rows_to_unmark: list[str] = field(default_factory=list)

    PROBLEMS = (
        "unembedded", "missing_points", "hash_mismatch", "wrong_model",
        "marked_not_embeddable", "orphan_points",
    )  # fmt: skip

    @property
    def clean(self) -> bool:
        return not any(getattr(self, name) for name in self.PROBLEMS)


def _embeddable(row) -> bool:
    return row.chunk_type in EMBEDDED_TYPES and not row.deleted and not row.parts


def reconcile(engine: Engine, store: QdrantStore, embedding: EmbeddingSettings) -> ReconcileReport:
    store.check()  # raises if the collection was built differently
    points = store.scroll_hashes()
    with session_scope(engine) as s:
        rows = ChunkRepository(s).index_states()

    report = ReconcileReport(rows=len(rows), points=len(points))
    expected_model = (embedding.model, embedding.dimensions)
    embeddable_ids = set()
    for row in rows:
        embeddable = _embeddable(row)
        point_hash = points.get(row.id, ...)  # ... = no point
        if embeddable:
            report.embeddable += 1
            embeddable_ids.add(row.id)
            if not row.is_embedded:
                report.unembedded.append(row.key)
        if not row.is_embedded:
            continue
        if not embeddable:
            report.marked_not_embeddable.append(row.key)
        elif point_hash is ...:
            report.missing_points.append(row.key)
        elif point_hash != row.content_hash:
            report.hash_mismatch.append(row.key)
        elif (row.embedding_model, row.embedding_dimensions) != expected_model:
            report.wrong_model.append(row.key)
        else:
            continue
        report.rows_to_unmark.append(row.id)

    report.orphan_points = sorted(pid for pid in points if pid not in embeddable_ids)
    return report


def fix(engine: Engine, store: QdrantStore, report: ReconcileReport) -> None:
    store.delete(report.orphan_points)
    with session_scope(engine) as s:
        ChunkRepository(s).clear_embedded(report.rows_to_unmark)


def print_report(report: ReconcileReport, store: QdrantStore) -> None:
    print(f"\nReconcile: SQLite vs {store.collection}")
    print(f"  rows {report.rows:,}, embeddable {report.embeddable:,}, points {report.points:,}")
    for name in ReconcileReport.PROBLEMS:
        items = getattr(report, name)
        shown = ", ".join(items[:5]) + (f", ... (+{len(items) - 5})" if len(items) > 5 else "")
        print(f"  {name:<22}{len(items):>6}" + (f"   {shown}" if items else ""))
    print("  CLEAN" if report.clean else "  NOT CLEAN")
