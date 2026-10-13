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
