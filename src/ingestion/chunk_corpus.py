"""Chunk manifest documents into JSONL files you can read before embedding.

    uv run python -m src.ingestion.chunk_corpus --doc dfsa-rulebook-cob
    uv run python -m src.ingestion.chunk_corpus            # every supported document

Writes, per document:
    data/parsed/chunks/<doc_id>.jsonl    one chunk per line (children, then sections)
    data/parsed/reports/<doc_id>.json    the quality report from validate.py

Uses the cached Docling parse (see parse.py); a document that hasn't been
parsed yet is parsed first.
"""

import argparse
import json
import logging
import sys
from functools import lru_cache

import tiktoken

from src.config.settings import ChunkingSettings, PathSettings
from src.ingestion.chunkers.rulebook import RulebookChunker
from src.ingestion.elements import to_elements
from src.ingestion.manifest import ManifestEntry, load_manifest
from src.ingestion.parse import parse
from src.ingestion.refs import annotate
from src.ingestion.split import Splitter
from src.ingestion.validate import report, toc_sections

logger = logging.getLogger(__name__)

# Laws and the Glossary table get their own chunkers in later steps.
UNSUPPORTED_MODULES = {"GLO"}


def supported(entry: ManifestEntry) -> bool:
    return entry.doc_type == "rulebook" and entry.module_code not in UNSUPPORTED_MODULES


@lru_cache
def _encoding(name: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


def chunk_document(
    entry: ManifestEntry,
    cfg: ChunkingSettings,
    paths: PathSettings,
    module_codes: set[str] | None = None,
) -> dict:
    doc = parse(entry, paths=paths)
    encoding = _encoding(cfg.tokenizer)

    def count(text: str) -> int:
        return len(encoding.encode(text))

    result = RulebookChunker(entry, count).chunk(to_elements(doc))
    chunks = Splitter(count, cfg.max_chunk_tokens).split_chunks(result.chunks)
    chunks = annotate(chunks, module_codes or {entry.module_code})

    chunks_dir = paths.parsed_dir / "chunks"
    reports_dir = paths.parsed_dir / "reports"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    with (chunks_dir / f"{entry.id}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")

    rep = report(
        chunks,
        result.issues,
        toc=toc_sections(doc),
        rerank_warn_tokens=cfg.rerank_warn_tokens,
        max_chunk_tokens=cfg.max_chunk_tokens,
    )
    rep = {"doc_id": entry.id, **rep}
    (reports_dir / f"{entry.id}.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return rep


def print_summary(reports: list[dict], cfg: ChunkingSettings) -> None:
    over_rr, over_max = f"over_{cfg.rerank_warn_tokens}", f"over_{cfg.max_chunk_tokens}"
    cols = (
        f"{'document':<22}{'rules':>6}{'del':>5}{'guid':>6}{'sect':>6}{'toc miss':>9}"
        f"{'gaps':>6}{'reject':>7}{'loose':>6}{'leaks':>6}{'med tok':>8}{'>' + over_rr[5:]:>6}"
        f"{'>' + over_max[5:]:>7}{'max':>7}"
    )
    print("\n" + cols + "\n" + "-" * len(cols))
    for r in reports:
        c, t = r["counts"], r["tokens"]
        print(
            f"{r['doc_id'].removeprefix('dfsa-rulebook-'):<22}{c['rules']:>6}{c['deleted_rules']:>5}"
            f"{c['guidance']:>6}{c['sections']:>6}{len(r['toc_sections_missing']):>9}"
            f"{len(r['rule_gaps']):>6}{len(r['rejected_numbers']):>7}{len(r['loose_text']):>6}"
            f"{len(r['furniture_leaks']):>6}{t['median']:>8}{t[over_rr]:>6}{t[over_max]:>7}"
            f"{t['max']:>7}"
        )
    print("\nDetails: data/parsed/reports/<doc_id>.json")


def main() -> int:
    ap = argparse.ArgumentParser(description="Chunk manifest documents into JSONL.")
    ap.add_argument("--doc", help="manifest id; default is every supported document")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("docling", "httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cfg, paths = ChunkingSettings(), PathSettings()
    entries = [e for e in load_manifest() if args.doc in (None, e.id)]
    if not entries:
        logger.error("No manifest entry with id %r.", args.doc)
        return 1

    codes = {e.module_code for e in load_manifest() if e.doc_type == "rulebook"}
    reports = []
    for entry in entries:
        if not supported(entry):
            logger.info("%s: no chunker yet for this document type, skipped", entry.id)
            continue
        reports.append(chunk_document(entry, cfg, paths, codes))
    if reports:
        print_summary(reports, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
