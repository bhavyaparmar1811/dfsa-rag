"""PDF -> DoclingDocument, with an on-disk cache.

Docling is layout-aware: it labels every element (section_header, text,
list_item, table, page_header, ...) and orders them in reading order. We use
it for that page-level cleanup only; rule structure is reconstructed later by
our own chunkers, because DFSA rule numbers are plain text to Docling.

Parsing is slow on CPU, so the result is cached under
`data/parsed/docling/<sha256>.json`. The cache key includes the Docling
version and our pipeline options, so upgrading Docling or changing an option
re-parses instead of silently serving stale output.
"""

import hashlib
import json
import logging
import time
from importlib.metadata import version
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument

from src.config.settings import PathSettings
from src.ingestion.manifest import ManifestEntry, verify_file

logger = logging.getLogger(__name__)

# The PDFs have a real text layer, so OCR would only add errors and time.
# Images are out of scope (logos and a few AML flowcharts), so we don't render
# or describe them; Docling still detects picture regions, which lets us drop
# any text that sits inside a figure.
PIPELINE_OPTIONS = {
    "do_ocr": False,
    "do_table_structure": True,
    "generate_page_images": False,
    "generate_picture_images": False,
    "do_picture_classification": False,
    "do_picture_description": False,
}


def cache_key(file_hash: str) -> str:
    """Unique per (document, Docling version, pipeline options)."""
    fingerprint = json.dumps(
        {"docling": version("docling"), "options": PIPELINE_OPTIONS}, sort_keys=True
    )
    return f"{file_hash[:16]}-{hashlib.sha256(fingerprint.encode()).hexdigest()[:8]}"


def _converter() -> DocumentConverter:
    options = PdfPipelineOptions(**PIPELINE_OPTIONS)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def parse(
    entry: ManifestEntry, *, force: bool = False, paths: PathSettings | None = None
) -> DoclingDocument:
    paths = paths or PathSettings()
    pdf = verify_file(entry, paths.root)

    cache_dir = paths.parsed_dir / "docling"
    key = cache_key(entry.file_hash)
    cached = cache_dir / f"{key}.json"
    meta_path = cache_dir / f"{key}.meta.json"

    if cached.is_file() and not force:
        logger.info("%s: using cached parse %s", entry.id, cached.name)
        return DoclingDocument.load_from_json(cached)

    logger.info("%s: parsing %s with Docling (this can take minutes)", entry.id, pdf.name)
    start = time.perf_counter()
    doc = _converter().convert(pdf).document
    seconds = time.perf_counter() - start

    cache_dir.mkdir(parents=True, exist_ok=True)
    doc.save_as_json(cached)
    _write_meta(meta_path, entry, pdf, seconds, doc)
    logger.info("%s: parsed %d pages in %.0f s", entry.id, len(doc.pages), seconds)
    return doc


def main() -> int:
    """Parse every manifest document (or one, with --doc), skipping cached ones.

    uv run python -m src.ingestion.parse
    uv run python -m src.ingestion.parse --doc dfsa-rulebook-aml --force
    """
    import argparse

    from src.ingestion.manifest import load_manifest

    ap = argparse.ArgumentParser(description="Parse manifest PDFs with Docling (cached).")
    ap.add_argument("--doc", help="manifest id; default is every document")
    ap.add_argument("--force", action="store_true", help="re-parse even if cached")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    # Docling logs every HTTP request and plugin load at INFO; keep our own lines readable.
    for noisy in ("docling", "httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    entries = [e for e in load_manifest() if args.doc in (None, e.id)]
    if not entries:
        logger.error("No manifest entry with id %r.", args.doc)
        return 1
    for i, entry in enumerate(entries, 1):
        logger.info("[%d/%d] %s", i, len(entries), entry.id)
        parse(entry, force=args.force)
    return 0


def _write_meta(
    path: Path, entry: ManifestEntry, pdf: Path, seconds: float, doc: DoclingDocument
) -> None:
    meta = {
        "doc_id": entry.id,
        "source_file": pdf.name,
        "file_hash": entry.file_hash,
        "docling_version": version("docling"),
        "pipeline_options": PIPELINE_OPTIONS,
        "pages": len(doc.pages),
        "parse_seconds": round(seconds, 1),
    }
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
