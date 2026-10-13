"""Small, valid chunks and manifest entries for storage and ingestion tests."""

from src.ingestion.ids import chunk_id
from src.ingestion.manifest import ManifestEntry
from src.ingestion.models import Chunk, ChunkType

FINGERPRINT = "text-embedding-3-small-1536"


def entry(doc_id: str = "dfsa-rulebook-cob", module: str = "COB", version: str = "51"):
    law = doc_id.startswith("dfsa-laws-")
    return ManifestEntry(
        id=doc_id, regulator="DFSA", rulebook="DIFC Laws" if law else "DFSA Rulebook",
        module_code=module, module_name=f"{module} Module", version=version,
        effective_date="2026-07", source_url="https://example.com",
        file_path=f"data/raw/{module}.pdf", downloaded_on="2026-10-01", file_hash="a" * 64,
    )  # fmt: skip


def chunk(
    key: str = "COB 3.3.2",
    *,
    text: str | None = None,
    kind: ChunkType = ChunkType.RULE,
    doc_id: str = "dfsa-rulebook-cob",
    version: str = "51",
    order: int = 0,
    parent_key: str | None = "COB 3.3",
    **extra,
) -> Chunk:
    module = key.split()[0]
    return Chunk(
        chunk_id=chunk_id(doc_id, version, key), key=key, citation=key.split("#")[0],
        chunk_type=kind, binding=kind != ChunkType.GUIDANCE, doc_id=doc_id, regulator="DFSA",
        module=module, version=version, effective_date="2026-07", source_file=f"{module}.pdf",
        section="3.3", order=order, parent_key=None if kind == ChunkType.SECTION else parent_key,
        heading_path=[module, "Section 3.3"], header_text=f"{module} > Section 3.3 > {key}",
        text=text if text is not None else f"{key} text", page_start=1, page_end=1,
        token_count=10, **extra,
    )  # fmt: skip
