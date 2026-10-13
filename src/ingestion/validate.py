"""Quality report for one document's chunks, checked before anything is embedded.

Nothing here changes the chunks. It answers: did we find every section the
table of contents lists, are rule numbers complete and in order, did any page
furniture leak in, and how big are the chunks?
"""

import re
import statistics

from docling_core.types.doc import DocItemLabel, DoclingDocument

from src.ingestion import patterns as P
from src.ingestion.chunkers.rulebook import ChunkIssues
from src.ingestion.elements import TOC_LINE, is_toc
from src.ingestion.models import Chunk, ChunkType

FURNITURE_LEAK = re.compile(r"/VER\d+/\d{2}-\d{2}")
# The TOC is a table whose number cells hold section numbers ("3.3", "A6.1").
# Docling sometimes merges two rows into one cell ("8.5 8.6"), so a cell may
# hold several numbers separated by spaces.
TOC_SECTION = re.compile(r"^A?\d+[A-Z]?\.\d+[A-Z]?$")
TOKEN_BUCKETS = (100, 250, 500, 750, 1000, 1500, 2000, 4000)


def toc_sections(doc: DoclingDocument) -> set[str]:
    """Section numbers listed in the document's table of contents."""
    found: set[str] = set()
    for item, _level in doc.iterate_items():
        if item.label == DocItemLabel.DOCUMENT_INDEX and is_toc(item):
            for cell in getattr(getattr(item, "data", None), "table_cells", []):
                tokens = cell.text.split()
                if tokens and all(TOC_SECTION.match(t) for t in tokens):
                    found.update(tokens)
                elif tokens and TOC_SECTION.match(tokens[0]):
                    found.add(tokens[0])  # number and title merged: "2.3 By way of ..."
        # TOC lines Docling didn't put in its TOC table: "9.3 Circumstances ... 22".
        elif (text := getattr(item, "text", "")) and TOC_LINE.search(text):
            number = text.split(" ", 1)[0]
            if TOC_SECTION.match(number):
                found.add(number)
    return found


def sequence_gaps(rule_numbers: list[str]) -> list[str]:
    """Missing rule numbers inside a section, e.g. 3.3.2 -> 3.3.4 reports 3.3.3.

    Only plain consecutive integers are checked; lettered rules (3.3.2A) are
    insertions and never count as gaps.
    """
    gaps = []
    for prev, cur in zip(rule_numbers, rule_numbers[1:], strict=False):
        p_sec, _, p_last = prev.rpartition(".")
        c_sec, _, c_last = cur.rpartition(".")
        if p_sec == c_sec and p_last.isdigit() and c_last.isdigit():
            gaps.extend(f"{p_sec}.{n}" for n in range(int(p_last) + 1, int(c_last)))
    return gaps


def report(
    chunks: list[Chunk],
    issues: ChunkIssues,
    *,
    toc: set[str] | None = None,
    rerank_warn_tokens: int = 512,
    max_chunk_tokens: int = 1000,
) -> dict:
    children = [c for c in chunks if c.chunk_type != ChunkType.SECTION]
    embedded = [c for c in children if c.embedded]
    whole = [c for c in children if c.part is None]  # parts of a split chunk count once
    rules = [c for c in whole if c.chunk_type == ChunkType.RULE]
    # Real sections only; chapter-level containers ("3", "APP2") are not TOC entries.
    sections = {
        c.section
        for c in chunks
        if c.chunk_type == ChunkType.SECTION and TOC_SECTION.match(c.section)
    }
    tokens = sorted(c.token_count for c in embedded)

    def pct(q: float) -> int:
        return tokens[min(len(tokens) - 1, int(len(tokens) * q))] if tokens else 0

    histogram, lower = {}, 0
    for upper in (*TOKEN_BUCKETS, None):
        n = sum(1 for t in tokens if t > lower and (upper is None or t <= upper))
        histogram[f"{lower + 1}-{upper}" if upper else f">{lower}"] = n
        lower = upper or lower

    return {
        "counts": {
            "rules": len(rules),
            "deleted_rules": sum(c.deleted for c in rules),
            "guidance": sum(c.chunk_type == ChunkType.GUIDANCE for c in whole),
            "text": sum(c.chunk_type == ChunkType.TEXT for c in whole),
            "section_guidance": sum(
                c.chunk_type == ChunkType.GUIDANCE and c.rule_number is None for c in whole
            ),
            "sections": len(sections),
            "embedded_chunks": len(embedded),
            "scope_rules": sum(c.scope_rule for c in rules),
        },
        # Deleted sections ("5.4 Deleted") are listed in the TOC but have no content.
        "toc_sections_missing": (
            sorted(toc - sections - set(issues.deleted_sections), key=_sort_key) if toc else []
        ),
        "deleted_sections": issues.deleted_sections,
        "sections_not_in_toc": sorted(sections - toc, key=_sort_key) if toc else [],
        "rule_gaps": sequence_gaps([c.rule_number for c in rules]),
        "rejected_numbers": issues.rejected_numbers,
        "implicit_sections": issues.implicit_sections,
        "loose_text": issues.loose_text,
        "unnumbered_text": issues.unnumbered_text,
        "repeated_numbers": issues.repeated_numbers,
        "skipped": {
            "front_matter": issues.skipped_front_matter,
            "running_headers": issues.skipped_running_headers,
            "figure_text": issues.skipped_figure_text,
        },
        "split": {
            "chunks_split": sum(bool(c.parts) for c in children),
            "parts": sum(c.part is not None for c in children),
            "still_over_max": [c.key for c in embedded if c.token_count > max_chunk_tokens],
        },
        "refs": _refs_report(chunks),
        "furniture_leaks": [c.key for c in chunks if FURNITURE_LEAK.search(c.text)],
        "empty_chunks": [c.key for c in children if len(c.text.strip()) < 20 and not c.deleted],
        "tokens": {
            "median": statistics.median(tokens) if tokens else 0,
            "p90": pct(0.9),
            "p99": pct(0.99),
            "max": tokens[-1] if tokens else 0,
            f"over_{rerank_warn_tokens}": sum(t > rerank_warn_tokens for t in tokens),
            f"over_{max_chunk_tokens}": sum(t > max_chunk_tokens for t in tokens),
            "largest": [
                f"{c.key} ({c.token_count})"
                for c in sorted(embedded, key=lambda c: -c.token_count)[:10]
            ],
            "histogram": histogram,
        },
    }


def _refs_report(chunks: list[Chunk]) -> dict:
    """Do in-module references point at something that exists?"""
    module = chunks[0].module if chunks else ""
    whole = [c for c in chunks if c.part is None]
    known = {c.key for c in whole} | {c.citation for c in whole}
    sections = {c.section for c in chunks if c.chunk_type == ChunkType.SECTION}
    known |= {f"{module} {s.split('.')[0]}" for s in sections if not s.startswith("A")}
    known |= {f"{module} App{s[1:].split('.')[0]}" for s in sections if s[1:2].isdigit()}
    # An appendix holding only text has a container ("APP4") and no A4.x sections.
    known |= {f"{module} App{s[3:]}" for s in sections if s.startswith("APP")}

    unresolved, external = [], {}
    for c in whole:
        for ref in c.cross_refs:
            if ref.startswith(f"{module} "):
                if ref not in known:
                    unresolved.append(f"{c.key} -> {ref}")
            else:
                target = ref.rsplit(" ", 1)[0] if " Art. " not in ref else ref.split(" Art. ")[0]
                external[target] = external.get(target, 0) + 1
    return {
        "chunks_with_refs": sum(bool(c.cross_refs) for c in whole),
        "total_refs": sum(len(c.cross_refs) for c in whole),
        "unresolved_in_module": unresolved,
        "external_targets": dict(sorted(external.items(), key=lambda kv: -kv[1])),
    }


def _sort_key(number: str):
    app = number.startswith("A")
    try:
        return (app, P.number_key(number.removeprefix("A")))
    except ValueError:
        return (app, ((0, number),))
