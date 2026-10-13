"""Facts about the real COB module, checked by hand against the PDF.

Skipped when the PDFs aren't present (they're never committed). Runs on the
cached Docling parse, so it's fast after the first parse.
"""

import pytest

from src.config.settings import ChunkingSettings, PathSettings
from src.ingestion.chunk_corpus import chunk_document
from src.ingestion.manifest import load_manifest

pytestmark = pytest.mark.corpus

PATHS = PathSettings()


@pytest.fixture(scope="module")
def cob():
    entry = next(e for e in load_manifest() if e.id == "dfsa-rulebook-cob")
    if not entry.path(PATHS.root).is_file():
        pytest.skip("COB PDF not present in data/raw")
    report = chunk_document(entry, ChunkingSettings(), PATHS)
    import json

    path = PATHS.parsed_dir / "chunks" / f"{entry.id}.jsonl"
    chunks = {c["key"]: c for c in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
    return chunks, report


def test_section_3_3(cob):
    chunks, _ = cob
    assert chunks["COB 3.3.2"]["text"].startswith("3.3.2 (1) Subject to (2)")
    assert chunks["COB 3.3.1"]["scope_rule"] and not chunks["COB 3.3.2"]["scope_rule"]
    guidance = chunks["COB 3.3.2 Guidance"]["text"]
    assert (
        guidance.startswith("1. App 2") and "\n3. For the purposes of Rule 3.3.2(2)(a)" in guidance
    )
    assert chunks["COB 3.2.6"]["subheading"] == "Past performance and forecasts"


def test_rule_continuing_from_previous_page_stays_with_its_rule(cob):
    # p.42: "(2) The following provisions ... prevail" belongs to 4.4.1, not to Guidance.
    chunks, _ = cob
    assert "(2) The following provisions of laws" in chunks["COB 4.4.1"]["text"]


def test_rules_in_appendix_tables_are_binding_rules(cob):
    chunks, _ = cob
    assert chunks["COB A7.2.1"]["chunk_type"] == "rule" and chunks["COB A7.2.1"]["binding"]


def test_report_is_clean(cob):
    _, report = cob
    assert report["toc_sections_missing"] == [] and report["sections_not_in_toc"] == []
    assert report["deleted_sections"] == ["5.4"]
    assert report["rule_gaps"] == ["6.10.3", "7.12.3", "7.12.4"]  # absent from the PDF itself
    assert report["furniture_leaks"] == [] and report["loose_text"] == []
    assert report["rejected_numbers"] == []


def test_no_embedded_chunk_exceeds_the_limit(cob):
    _, report = cob
    assert report["split"]["still_over_max"] == []
    assert report["split"]["chunks_split"] > 0


def test_cross_references_resolve(cob):
    chunks, report = cob
    assert "COB 3.3.4" in chunks["COB 3.3.2"]["cross_refs"]
    assert "GEN 2.26.1" in chunks["COB 3.3.1"]["cross_refs"]
    # The only dangling in-module reference is a typo in the source PDF itself:
    # Guidance on 6.16.4 cites "Rule 16.6.4(1)", and COB has no chapter 16.
    assert report["refs"]["unresolved_in_module"] == ["COB 6.16.4 Guidance -> COB 16.6.4"]
