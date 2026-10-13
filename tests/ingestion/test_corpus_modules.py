"""Facts about GEN, AML, MKT and FPR, each checked by hand against the PDF.

Skipped when the PDFs aren't present. Runs on the cached Docling parse.
"""

import json

import pytest

from src.config.settings import ChunkingSettings, PathSettings
from src.ingestion.chunk_corpus import chunk_document
from src.ingestion.manifest import load_manifest

pytestmark = pytest.mark.corpus

PATHS = PathSettings()
ENTRIES = {e.id: e for e in load_manifest()}
CODES = {e.module_code for e in ENTRIES.values() if e.doc_type == "rulebook"}
_cache: dict[str, tuple[dict, dict]] = {}


def chunked(doc_id: str) -> tuple[dict, dict]:
    if doc_id not in _cache:
        entry = ENTRIES[doc_id]
        if not entry.path(PATHS.root).is_file():
            pytest.skip(f"{doc_id} PDF not present in data/raw")
        report = chunk_document(entry, ChunkingSettings(), PATHS, CODES)
        path = PATHS.parsed_dir / "chunks" / f"{doc_id}.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        _cache[doc_id] = ({c["key"]: c for c in map(json.loads, lines)}, report)
    return _cache[doc_id]


@pytest.mark.parametrize("module", ["gen", "aml", "mkt", "fpr"])
def test_report_is_clean(module):
    _, report = chunked(f"dfsa-rulebook-{module}")
    assert report["toc_sections_missing"] == []
    assert report["rejected_numbers"] == [] and report["loose_text"] == []
    assert report["furniture_leaks"] == [] and report["split"]["still_over_max"] == []


def test_aml_margin_numbers_and_figures():
    chunks, report = chunked("dfsa-rulebook-aml")
    assert chunks["AML 8.1.1"]["text"].startswith("8.1.1 (1) A Relevant Person may rely")
    assert report["deleted_sections"] == ["10.1", "11.1"] and chunks["AML 11.2.2"]["deleted"]
    text = "\n".join(c["text"] for c in chunks.values())
    for figure_text in ("Objective and", "Assign risk rating", "Can I rely", "Source of wealth"):
        assert figure_text not in text


def test_fpr_pages_docling_labelled_as_a_table_of_contents():
    chunks, _ = chunked("dfsa-rulebook-fpr")
    assert chunks["FPR 6.1.1"]["text"].startswith("6.1.1 This section applies")
    assert "FPR 9.4.3" in chunks and "FPR 6.1" in chunks
    rule = chunks["FPR 3.2.3"]["text"]
    assert rule.startswith("3.2.3 The notice") and "\n(a) be in such form" in rule


def test_mkt_appendix_tables_belong_to_the_rule_that_names_them():
    chunks, _ = chunked("dfsa-rulebook-mkt")
    assert chunks["MKT A1.1.1"]["page_end"] > 130  # the table runs on for ~26 pages
    assert chunks["MKT A1.1.2"]["page_end"] == chunks["MKT A1.1.2"]["page_start"]
    app4 = chunks["MKT APP4 Text"]
    assert app4["chunk_type"] == "text" and not app4["binding"]


def test_gen_lettered_chapter_and_source_typo():
    chunks, report = chunked("dfsa-rulebook-gen")
    assert chunks["GEN 3A.1.1"]["text"].startswith("3A.1.1 In this chapter:")
    # GEN App6 Guidance cites "Rule A2.3.2(b)(i)"; GEN has no A2.3.2 (elsewhere
    # the same Guidance cites A2.3.1(b)(i)). A typo in the source PDF.
    assert report["refs"]["unresolved_in_module"] == ["GEN APP6 Guidance -> GEN A2.3.2"]
