import pytest

from src.ingestion.terms import TermMatcher, surface_forms

TERMS = [
    "Advising",
    "Advising on Financial Products",
    "Authorised Firm",
    "Client",
    "Client Assets Crisis Preparedness Pack or Client Assets Pack",
    "Designated Non-Financial Business or Profession (DNFBP)",
    "Financial Service",
    "Person",
    "Retail Client",
]
MATCHER = TermMatcher(TERMS)


def test_aliases():
    assert surface_forms("Designated Non-Financial Business or Profession (DNFBP)") == [
        "Designated Non-Financial Business or Profession (DNFBP)",
        "Designated Non-Financial Business or Profession",
        "DNFBP",
    ]
    assert "Client Assets Pack" in surface_forms(TERMS[4])


@pytest.mark.parametrize(
    "text, terms",
    [
        (
            "An Authorised Firm must not provide a Financial Service to Retail Clients.",
            ["Authorised Firm", "Financial Service", "Retail Client"],
        ),
        ("Advising on Financial Products is regulated", ["Advising on Financial Products"]),
        ("the Client's money", ["Client"]),
        ("a DNFBP must", ["Designated Non-Financial Business or Profession (DNFBP)"]),
        ("keep the Client Assets Pack up to date", [TERMS[4]]),
        ("non-Retail Client business", ["Retail Client"]),
        ("clients and firms", []),  # lower case: an ordinary word, not the Defined Term
        ("Person means any natural person.", []),  # capital only because it starts a sentence
        ("(a) Person who is a Client; or", ["Client"]),
    ],
)
def test_find(text, terms):
    assert MATCHER.find(text) == terms
