"""Chunk manifest documents into JSONL files you can read before embedding.

    uv run python -m src.ingestion.chunk_corpus --doc dfsa-rulebook-cob
    uv run python -m src.ingestion.chunk_corpus            # every supported document

Writes, per document:
    data/parsed/chunks/<doc_id>.jsonl    one chunk per line (children, then sections)
    data/parsed/reports/<doc_id>.json    the quality report from validate.py

Uses the cached Docling parse (see parse.py); a document that hasn't been
parsed yet is parsed first.

The Glossary (GLO) is chunked first: its terms are used to tag every other
document's chunks with the Defined Terms they use. Chunking one document on
its own reuses GLO's saved output (and chunks GLO first if there is none).
"""

import argparse
import json
import logging
import sys
from functools import lru_cache

import tiktoken

from src.config.settings import ChunkingSettings, PathSettings
from src.ingestion.chunkers.glossary import GlossaryChunker
from src.ingestion.chunkers.law import LawChunker, law_name
from src.ingestion.chunkers.rulebook import RulebookChunker
from src.ingestion.elements import to_elements
from src.ingestion.manifest import ManifestEntry, load_manifest
from src.ingestion.parse import parse
from src.ingestion.refs import annotate
from src.ingestion.split import Splitter
from src.ingestion.terms import TermMatcher, tag, term_frequencies
from src.ingestion.validate import report, toc_sections

logger = logging.getLogger(__name__)

GLOSSARY_MODULE = "GLO"


def supported(entry: ManifestEntry) -> bool:
    return entry.doc_type in ("rulebook", "law")


def is_glossary(entry: ManifestEntry) -> bool:
    return entry.module_code == GLOSSARY_MODULE


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
    codes = module_codes or {entry.module_code}

    def count(text: str) -> int:
        return len(encoding.encode(text))

    elements = to_elements(doc)
    law = None
    if is_glossary(entry):
        result = GlossaryChunker(entry, count).chunk(doc, elements)
        matcher = TermMatcher([c.term for c in result.chunks if c.term])
    elif entry.doc_type == "law":
        law = law_name(entry)
        result = LawChunker(entry, count).chunk(elements)
        # A Law's own Schedule definitions, on top of the Glossary's terms.
        own_terms = [c.term for c in result.chunks if c.term]
        matcher = TermMatcher(glossary_matcher(cfg, paths, codes).terms + own_terms)
    else:
        result = RulebookChunker(entry, count).chunk(elements)
        matcher = glossary_matcher(cfg, paths, codes)
    chunks = Splitter(count, cfg.max_chunk_tokens).split_chunks(result.chunks)
    chunks = annotate(chunks, codes, law)
    chunks = tag(chunks, matcher)

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
    frequent = term_frequencies(chunks).most_common(15)
    rep = {"doc_id": entry.id, **rep, "most_used_terms": dict(frequent)}
    (reports_dir / f"{entry.id}.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return rep


_matchers: dict[str, TermMatcher] = {}


def glossary_matcher(cfg: ChunkingSettings, paths: PathSettings, codes: set[str]) -> TermMatcher:
    """Matcher over the Glossary's terms, from GLO's saved chunks."""
    glo = next(e for e in load_manifest() if is_glossary(e))
    path = paths.parsed_dir / "chunks" / f"{glo.id}.jsonl"
    key = f"{path}:{path.stat().st_mtime_ns}" if path.is_file() else ""
    if key not in _matchers:
        if not path.is_file():
            logger.info("Chunking %s first: its terms are needed for tagging", glo.id)
            chunk_document(glo, cfg, paths, codes)
            key = f"{path}:{path.stat().st_mtime_ns}"
        terms = [
            row["term"]
            for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())
            if row.get("term")
        ]
        _matchers[key] = TermMatcher(terms)
    return _matchers[key]


def print_law_summary(reports: list[dict], cfg: ChunkingSettings) -> None:
    over = f"over_{cfg.max_chunk_tokens}"
    cols = (
        f"{'law':<44}{'articles':>9}{'del':>5}{'sch para':>9}{'defs':>6}{'text':>6}"
        f"{'gaps':>6}{'reject':>7}{'loose':>6}{'med tok':>8}{'>' + over[5:]:>6}{'max':>6}"
    )
    print("\n" + cols + "\n" + "-" * len(cols))
    for r in reports:
        c, t = r["counts"], r["tokens"]
        print(
            f"{r['doc_id'].removeprefix('dfsa-laws-'):<44}{c['articles']:>9}"
            f"{c['deleted_articles']:>5}{c['schedule_paragraphs']:>9}{c['definitions']:>6}"
            f"{c['text']:>6}{len(r['article_gaps']):>6}{len(r['rejected_numbers']):>7}"
            f"{len(r['loose_text']):>6}{t['median']:>8}{t[over]:>6}{t['max']:>6}"
        )


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
    entries.sort(key=lambda e: not is_glossary(e))  # GLO first: the others use its terms
    reports = []
    for entry in entries:
        if not supported(entry):
            logger.info("%s: no chunker yet for this document type, skipped", entry.id)
            continue
        reports.append(chunk_document(entry, cfg, paths, codes))
    rulebook = [r for r in reports if not r["doc_id"].startswith("dfsa-laws-")]
    laws = [r for r in reports if r["doc_id"].startswith("dfsa-laws-")]
    if rulebook:
        print_summary(rulebook, cfg)
    if laws:
        print_law_summary(laws, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
