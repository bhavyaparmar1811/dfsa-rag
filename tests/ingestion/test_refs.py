import pytest

from src.ingestion.refs import extract_refs


@pytest.mark.parametrize(
    "text, refs",
    [
        ("in accordance with the requirements in Rule 3.3.4; and", ["COB 3.3.4"]),
        ("described in GEN Rule 2.26.1 that constitutes marketing", ["GEN 2.26.1"]),
        ("For the purposes of Rule 3.3.2(1)(b), a firm", ["COB 3.3.2"]),
        ("Rules 3.3.2, 3.3.3 and 3.3.5 apply", ["COB 3.3.2", "COB 3.3.3", "COB 3.3.5"]),
        ("a trading system specified under GEN 2.22.1.", ["GEN 2.22.1"]),
        ("see GEN section 2.11 and AML Rule 7.3.1(1)", ["GEN 2.11", "AML 7.3.1"]),
        ("Section 15.5 sets out further Guidance", ["COB 15.5"]),
        ("the key information specified in App2 which", ["COB App2"]),
        ("including REP Rule 4.10.1, which relates", ["REP 4.10.1"]),  # module outside the list
        ("under Article 76 of the Regulatory Law", ["Regulatory Law Art. 76"]),
    ],
)
def test_reference_forms(text, refs):
    assert extract_refs(text, "COB") == refs


@pytest.mark.parametrize(
    "text",
    [
        "CDD 7.1.1 Reliance 8.1.1",  # AML figure labels, not module references
        "Applying a risk-based approach",  # "App" inside a word
        "subject to (2), where it is impracticable",  # an internal sub-rule reference
    ],
)
def test_not_references(text):
    assert extract_refs(text, "COB") == []


@pytest.mark.parametrize(
    "text, module, refs",
    [
        ("only defined under Rule 1.2.1 of PIB rather than", "GEN", ["PIB 1.2.1"]),
        ("see section 3.2.15 of the RPP Sourcebook", "GEN", ["RPP 3.2.15"]),
        ("referred to in Rule 3A.3.4 of the Previous Regime", "GEN", ["Previous Regime 3A.3.4"]),
        ("section 6.2, of the General (GEN) module", "AML", ["GEN 6.2"]),
        ("in accordance with the Rules in AUD App8, if", "GEN", ["AUD App8"]),
        # A capitalised Defined Term after "of the" is not a document.
        ("the requirements of Rule 6.1.2 of the Fund Manager", "FPR", ["FPR 6.1.2"]),
    ],
)
def test_document_named_after_the_reference(text, module, refs):
    assert extract_refs(text, module) == refs


@pytest.mark.parametrize(
    "text, refs",
    [
        ("appointed under Article 19(1) in the exercise", ["Regulatory Law Art. 19"]),
        ("Articles 37 and 38 of the Markets Law", ["Markets Law Art. 37", "Markets Law Art. 38"]),
        ("Schedule 1 contains:", ["Regulatory Law Sch. 1"]),
        ("see GEN Rule 2.26.1", ["GEN 2.26.1"]),
        ("as defined in Rule 2.3", []),  # Laws have no local Rules
    ],
)
def test_references_inside_a_law(text, refs):
    assert extract_refs(text, "REG_LAW", law="Regulatory Law") == refs


def test_difc_prefix_is_dropped_from_law_names():
    assert extract_refs("Article 76 of the DIFC Regulatory Law", "COB") == [
        "Regulatory Law Art. 76"
    ]
