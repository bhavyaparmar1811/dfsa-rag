"""The Rulebook chunker on short invented element streams.

Each case recreates something found in the real COB/FPR output.
"""

import pytest

from src.ingestion.chunkers.rulebook import RulebookChunker
from src.ingestion.elements import Element
from src.ingestion.manifest import ManifestEntry

ENTRY = ManifestEntry(
    id="dfsa-rulebook-cob",
    regulator="DFSA",
    rulebook="DFSA Rulebook",
    module_code="COB",
    module_name="Conduct of Business Module",
    version="51",
    effective_date="2026-07",
    source_url="https://example.com/cob.pdf",
    file_path="data/raw/cob.pdf",
    downloaded_on="2026-09-26",
    file_hash="0" * 64,
)


def h(text, page=1):
    return Element("section_header", text, page, 0.0, 72.0)


def t(text, page=1):
    return Element("list_item", text, page, 0.0, 72.0)


CHAPTER_3 = h("3 CORE RULES - INVESTMENT BUSINESS")


def run(*elements):
    result = RulebookChunker(ENTRY, lambda s: len(s.split())).chunk(list(elements))
    return {c.key: c for c in result.chunks}, result.issues


SECTION_3_3 = [
    h("3 CORE RULES - INVESTMENT BUSINESS"),
    h("3.3 Key information and Client Agreement"),
    h("Application"),
    t("3.3.1 The Rules in this section do not apply to an Authorised Firm when it is:"),
    t("(a) carrying on a Financial Service with or for a Market Counterparty;"),
    t("3.3.2 (1) Subject to (2), an Authorised Firm must not carry on a Financial Service"),
    t("(2) An Authorised Firm may provide a Financial Service without a Client Agreement"),
    h("Guidance"),
    t("1. App 2 sets out the core information."),
    t("2. For the purposes of Rule 3.3.2(1)(b), a firm may provide a copy."),
    h("Changes to the client agreement"),
    t("3.3.3 If the Client Agreement allows the firm to amend it"),
]


def test_one_chunk_per_rule_with_sub_rules_kept_together():
    chunks, _ = run(*SECTION_3_3)
    rule = chunks["COB 3.3.2"]
    assert rule.chunk_type == "rule" and rule.binding
    assert rule.text.startswith("3.3.2 (1) Subject to (2)")
    assert "(2) An Authorised Firm may provide" in rule.text


def test_guidance_is_its_own_non_binding_chunk_linked_both_ways():
    chunks, _ = run(*SECTION_3_3)
    guidance = chunks["COB 3.3.2 Guidance"]
    assert guidance.chunk_type == "guidance" and not guidance.binding
    assert guidance.citation == "COB 3.3.2" and guidance.guidance_for == "COB 3.3.2"
    assert chunks["COB 3.3.2"].guidance_keys == ["COB 3.3.2 Guidance"]
    assert guidance.text.startswith("1. App 2")


def test_scope_rule_and_application_block_ends_at_first_obligation():
    chunks, _ = run(*SECTION_3_3)
    assert chunks["COB 3.3.1"].scope_rule and chunks["COB 3.3.1"].subheading == "Application"
    assert not chunks["COB 3.3.2"].scope_rule and chunks["COB 3.3.2"].subheading is None
    assert chunks["COB 3.3.3"].subheading == "Changes to the client agreement"


def test_contextual_header_and_parent_section():
    chunks, _ = run(*SECTION_3_3)
    header = chunks["COB 3.3.1"].header_text
    assert header == (
        "DFSA Rulebook > Conduct of Business Module (COB) > Chapter 3: CORE RULES - "
        "INVESTMENT BUSINESS > Section 3.3: Key information and Client Agreement > "
        "Application > Rule COB 3.3.1"
    )
    section = chunks["COB 3.3"]
    assert section.chunk_type == "section" and not section.embedded
    assert all(c.parent_key == "COB 3.3" for c in chunks.values() if c.chunk_type != "section")
    # The parent holds every rule in order, with sub-headings and Guidance marked.
    assert section.text.index("3.3.1") < section.text.index("3.3.2") < section.text.index("3.3.3")
    assert (
        "Guidance\n1. App 2" in section.text and "Changes to the client agreement" in section.text
    )


def test_chunk_ids_are_deterministic():
    first, _ = run(*SECTION_3_3)
    second, _ = run(*SECTION_3_3)
    assert {k: c.chunk_id for k, c in first.items()} == {k: c.chunk_id for k, c in second.items()}


def test_wrapped_reference_cannot_move_the_sequence_backwards():
    chunks, issues = run(
        CHAPTER_3,
        h("3.3 Key information"),
        t("3.3.2 The firm must comply with the requirements in Rule"),
        t("3.3.1 and record it."),  # a wrapped reference, not a rule
        t("3.3.3 Next rule"),
    )
    assert "3.3.1 and record it." in chunks["COB 3.3.2"].text
    assert "COB 3.3.1" not in chunks and "COB 3.3.3" in chunks
    assert any("not after previous rule" in r for r in issues.rejected_numbers)


def test_rule_cannot_jump_to_a_section_mid_way():
    chunks, issues = run(
        CHAPTER_3, h("3.3 Key information"), t("3.3.1 A rule"), t("3.5.4 a stray line")
    )
    assert "COB 3.5.4" not in chunks
    assert any("without its heading" in r for r in issues.rejected_numbers)


def test_first_rule_of_unannounced_section_opens_it():
    chunks, issues = run(
        CHAPTER_3, h("3.3 Key information"), t("3.3.1 A rule"), t("3.4.1 Next section")
    )
    assert chunks["COB 3.4.1"].parent_key == "COB 3.4"
    assert issues.implicit_sections == ["COB 3.4 (p.1)"]


def test_stray_toc_like_line_before_first_chapter_is_ignored():
    # FPR: a TOC line "9.3 Circumstances ..." survived before chapter 1.
    chunks, _ = run(
        t("9.3 Circumstances in which the DFSA may decline"),
        h("1. INTRODUCTION"),
        h("1.1 Application"),
        t("1.1.1 The Rules in this module apply"),
    )
    assert "COB 1.1.1" in chunks and "COB 9.3" not in chunks


def test_deleted_rule_and_section():
    chunks, issues = run(
        h("5 ADDITIONAL RULES - PROVIDING TRUST SERVICES"),
        h("5.3 Reviews"),
        t("5.3.1 A Trust Service Provider must"),
        h("5.4 Deleted"),
        h("5.5 Dual control"),
        t("5.5.1 [Deleted]"),
        t("5.5.2 The firm must"),
    )
    assert chunks["COB 5.5.1"].deleted and not chunks["COB 5.5.1"].embedded
    assert issues.deleted_sections == ["5.4"]
    assert "COB 5.4" not in chunks  # no empty parent


def test_chapter_level_guidance_gets_a_container():
    # COB chapter 3 opens with Guidance before its first section.
    chunks, issues = run(
        h("2 CLIENT CLASSIFICATION"),
        h("2.1 Application"),
        t("2.1.1 This chapter applies"),
        t("3 CORE RULES - INVESTMENT BUSINESS"),  # chapter title as plain text
        h("Guidance"),
        t("1. The Rules in this chapter give support to the Principles."),
        h("3.1 Application"),
        t("3.1.1 This chapter applies"),
    )
    guidance = chunks["COB 3 Guidance"]
    assert guidance.parent_key == "COB 3" and guidance.citation == "COB 3"
    assert "Section" not in guidance.header_text
    assert "COB 2.1.1 Guidance" not in chunks  # not glued onto the previous chapter
    assert issues.loose_text == []


def test_heading_inside_guidance_stays_in_its_text():
    chunks, _ = run(
        h("2 CLIENT CLASSIFICATION"),
        h("2.2 Types of Clients"),
        t("2.2.1 A firm must classify"),
        h("Guidance"),
        t("1. This chapter sets out"),
        h("Risk based approach"),  # followed by body text: a heading inside the Guidance
        t("2. The Rules reflect the risk based approach"),
        t("2.2.2 Next rule"),
    )
    assert "Risk based approach" in chunks["COB 2.2.1 Guidance"].text
    assert chunks["COB 2.2.2"].subheading is None


@pytest.mark.parametrize("running_header", ["CONDUCT OF BUSINESS (COB)"])
def test_running_header_is_dropped(running_header):
    chunks, issues = run(
        CHAPTER_3, h("3.3 Key"), t("3.3.1 A rule"), h(running_header), t("(a) continues")
    )
    assert running_header not in chunks["COB 3.3.1"].text
    assert issues.skipped_running_headers == 1


def test_text_drawn_figure_is_skipped_until_the_next_structure():
    # AML Figure 4 is drawn with real text: box titles come through as headings.
    chunks, issues = run(
        h("7 CUSTOMER DUE DILIGENCE"),
        t("Figure 4. CDD"),
        h("Enhanced CDD 7.4.1"),
        t("· Obtain and verify:"),
        t("- Source of wealth"),
        h("7.1 Application"),
        t("7.1.1 This chapter applies to every Relevant Person."),
    )
    assert all("Source of wealth" not in c.text for c in chunks.values())
    assert "COB 7.1.1" in chunks and chunks["COB 7.1.1"].subheading is None
    assert len(issues.skipped_figure_text) == 4


def test_section_with_deleted_content():
    chunks, issues = run(
        h("10 SANCTIONS"),
        h("10.1 Application"),
        t("[deleted]"),
        h("10.2 Relevant resolutions"),
        t("10.2.1 A Relevant Person must"),
    )
    assert issues.deleted_sections == ["10.1"] and "COB 10.1" not in chunks


def test_unnumbered_text_becomes_a_non_binding_text_chunk():
    # MKT App4: a whole appendix of best-practice standards with no rule numbers.
    chunks, _ = run(
        h("9 LAST CHAPTER"),
        h("9.1 Application"),
        t("9.1.1 A rule"),
        h("APP4 CORPORATE GOVERNANCE BEST PRACTICE"),
        t("1. This Appendix sets out, by way of Guidance, best practice standards."),
        t("2. A Reporting Entity must explain any departure."),
    )
    text = chunks["COB APP4 Text"]
    assert text.chunk_type == "text" and not text.binding and text.embedded
    assert text.citation == "COB APP4" and "2. A Reporting Entity" in text.text
    assert "not a numbered Rule" in text.header_text


def test_dotted_appendix_numbers_and_repeated_table_number():
    # MKT App1: "A.1.1.1" in the text; the table it describes is headed "A1.1.1"
    # and the number is repeated above the table on each later page.
    table = Element("table", "| Item | Shares |\n|---|---|\n| 1.1 General | yes |", 1, 0.0, 47.0)
    more = Element("table", "| Item | Shares |\n|---|---|\n| 2.1 Business | yes |", 2, 0.0, 47.0)
    chunks, issues = run(
        h("APP 1 CONTENT OF A PROSPECTUS"),
        h("A1.1 Registration statement"),
        t("A.1.1.1 This table forms part of Rule 2.5.1(3)(b)."),
        t("A.1.1.2 (1) The reference to an 'issuer' is a reference to the Person."),
        t("A1.1.1"),
        table,
        t("1 Applicable laws include any laws in the jurisdiction."),  # a table footnote
        t("A1.1.1", page=2),
        more,
        h("A1.2 Securities note", page=3),
        t("A.1.2.1 This table forms part of Rule 2.5.1(3)(c).", page=3),
    )
    assert (
        "1.1 General" in chunks["COB A1.1.1"].text and "2.1 Business" in chunks["COB A1.1.1"].text
    )
    assert "Applicable laws" in chunks["COB A1.1.1"].text
    assert "General" not in chunks["COB A1.1.2"].text
    assert issues.repeated_numbers == 2 and issues.rejected_numbers == []
    assert "COB A1.2.1" in chunks
