"""The Law chunker on short invented element streams, each modelled on a real Law."""

from src.ingestion.chunkers.law import LawChunker, law_name
from src.ingestion.elements import Element
from src.ingestion.manifest import ManifestEntry

ENTRY = ManifestEntry(
    id="dfsa-laws-markets_law",
    regulator="DFSA",
    rulebook="DIFC Laws",
    module_code="MARKETS_LAW",
    module_name="MARKETS LAW",
    version="71",
    effective_date="2025-12",
    source_url="https://example.com/markets-law.pdf",
    file_path="data/raw/Laws/markets.pdf",
    downloaded_on="2026-09-26",
    file_hash="0" * 64,
)


def h(text, page=1):
    return Element("section_header", text, page, 0.0, 85.0)


def t(text, page=1, label="list_item"):
    return Element(label, text, page, 0.0, 121.0)


def run(*elements):
    result = LawChunker(ENTRY, lambda s: len(s.split())).chunk(list(elements))
    return {c.key: c for c in result.chunks}, result.issues


def test_law_name_matches_how_the_rulebook_cites_it():
    assert law_name(ENTRY) == "Markets Law"
    entry = ENTRY.model_copy(update={"module_name": "LAW REGULATING ISLAMIC FINANCIAL BUSINESS"})
    assert law_name(entry) == "Law Regulating Islamic Financial Business"


def test_front_matter_is_skipped_and_articles_are_chunks():
    chunks, issues = run(
        h("LIST OF AMENDMENTS MADE TO THIS LAW SINCE ENACTMENT"),
        t("(This list does not form part of the Law)", label="text"),
        h("PART 1: GENERAL"),
        h("1. Title"),
        t('This Law may be cited as the "Markets Law".', label="text"),
        h("PART 2: OFFER OF SECURITIES"),
        h("CHAPTER 1: APPLICATION"),
        h("2. Application"),
        t("(1) This Part applies to an offer of Securities in the DIFC."),
        t("(2) Article 13 does not apply to a Crypto Token."),
    )
    assert issues.skipped_front_matter == 2
    art = chunks["Markets Law Art. 2"]
    assert art.chunk_type == "article" and art.binding and art.rule_number == "2"
    assert art.text.startswith("2. Application\n(1) This Part applies")
    assert art.parent_key == "Markets Law Part 2 Ch. 1"
    assert art.heading_path[-2:] == ["Part 2: OFFER OF SECURITIES", "Chapter 1: APPLICATION"]
    assert chunks["Markets Law Art. 1"].parent_key == "Markets Law Part 1"


def test_numbered_list_inside_an_article_is_not_an_article():
    chunks, issues = run(
        h("PART 1: GENERAL"),
        h("7. The DFSA"),
        t("(1) The DFSA may publish:"),
        t("1. a statement of its policy;"),  # sentence-like: a list item
        t("2. Guidance."),
        h("8. Powers of the DFSA"),
    )
    assert "Markets Law Art. 1" not in chunks and "Markets Law Art. 2" not in chunks
    assert "1. a statement of its policy;" in chunks["Markets Law Art. 7"].text
    assert issues.rejected_numbers == []


def test_deleted_articles_including_a_merged_line():
    chunks, issues = run(
        h("PART 6: NOT IN USE"),
        t("42. Deleted", label="text"),
        t("43. Deleted 44. Deleted 45. Deleted", label="text"),
        h("PART 7: GOVERNANCE"),
        h("46. Fund governance"),
    )
    assert all(chunks[f"Markets Law Art. {n}"].deleted for n in (42, 43, 44, 45))
    assert not chunks["Markets Law Art. 46"].deleted
    assert issues.deleted_sections == ["Part 6"]


def test_title_case_chapter():
    # The Collective Investment Law writes "Chapter 1: General".
    chunks, _ = run(
        h("PART 2: DEFINITIONS"),
        t("Chapter 1: Collective Investment Funds", label="text"),
        h("11. Meaning of a Fund"),
        t("(1) A Fund is an arrangement ..."),
        t("Chapter 2: Other definitions", label="text"),
        h("12. Meaning of a Unit"),
    )
    assert "Chapter 2" not in chunks["Markets Law Art. 11"].text
    assert chunks["Markets Law Art. 12"].parent_key == "Markets Law Part 2 Ch. 2"


def test_schedule_paragraphs_restart_and_definitions_table():
    chunks, _ = run(
        h("PART 1: GENERAL"),
        h("70. Final provision"),
        h("SCHEDULE 1"),
        h("INTERPRETATION"),  # the title as a separate heading
        h("1. Rules of interpretation"),
        t("(1) In the Law, a reference to a person includes a body corporate."),
        h("2. Defined Terms"),
        Element(
            "table",
            "| Term | Definition |\n|---|---|\n| Board Member | a person appointed to the Board |",
            2,
            0.0,
            85.0,
        ),
        Element(
            "table",
            "| Term | Definition |\n|---|---|\n"
            "|  | of Directors of the DFSA. |\n"  # continues the last definition
            "| Offer | an offer of Securities |",
            3,
            0.0,
            85.0,
        ),
    )
    para = chunks["Markets Law Sch. 1 para. 1"]
    assert para.chunk_type == "article" and para.parent_key == "Markets Law Sch. 1"
    assert para.heading_path[-1] == "Schedule 1: INTERPRETATION"
    member = chunks["Markets Law Sch. 1 'Board Member'"]
    assert member.chunk_type == "definition" and member.term == "Board Member"
    assert member.text == "Board Member: a person appointed to the Board\nof Directors of the DFSA."
    assert member.page_end == 3
    assert "Markets Law Sch. 1 'Offer'" in chunks
