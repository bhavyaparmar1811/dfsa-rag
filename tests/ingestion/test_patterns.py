import pytest

from src.ingestion import patterns as P


@pytest.mark.parametrize(
    "text, number",
    [
        ("3.3.2 (1) Subject to (2), an Authorised Firm", "3.3.2"),
        ("3A.1.1 An Algorithmic Token", "3A.1.1"),
        ("7.12.5A The firm must", "7.12.5A"),
        ("A2.1.1 The key information", "2.1.1"),
        ("8.1.1", "8.1.1"),
    ],
)
def test_rule_numbers(text, number):
    assert P.RULE.match(text).group("num") == number


@pytest.mark.parametrize(
    "text",
    [
        "Rule 3.3.4 applies",  # a reference, not a rule start
        "3.3.2(1)(b) refers to",  # no space after the number
        "3.3 Key information",  # a section
        "1. App 2 sets out the core information",  # a Guidance paragraph
    ],
)
def test_not_rule_numbers(text):
    assert P.RULE.match(text) is None


def test_section_needs_capitalised_title():
    assert P.SECTION.match("3.3 Key information and Client Agreement").group("num") == "3.3"
    assert P.SECTION.match("2.5 per cent of the capital") is None


def test_chapter_headings():
    assert P.CHAPTER.match("3 CORE RULES - INVESTMENT BUSINESS").group("num") == "3"
    assert P.CHAPTER.match("2A. DEFINITION OF FINANCIAL PRODUCT").group("num") == "2A"
    assert P.CHAPTER.match("1. App 2 sets out the core information") is None


@pytest.mark.parametrize(
    "rest, expected",
    [
        ("The Rules in this section do not apply to an Authorised Firm when it is:", True),
        ("This section applies to an ATS Operator.", True),
        ("(1) Subject to (2), this section applies to an Authorised Firm", True),
        ("(1) Subject to (2), an Authorised Firm must not carry on", False),
        (
            "(1) An Authorised Firm must not provide a Financial Service to which this section "
            "applies",
            False,
        ),
    ],
)
def test_scope_rules(rest, expected):
    assert bool(P.SCOPE_RULE.match(rest)) is expected


def test_number_key_orders_lettered_insertions():
    keys = ["3.3.2", "3.3.2A", "3.3.10", "3A.1.1"]
    assert sorted(keys, key=P.number_key) == keys


def test_deleted_and_running_header():
    assert P.DELETED.match("[Deleted]") and P.DELETED.match("Deleted")
    assert P.RUNNING_HEADER.match("CONDUCT OF BUSINESS (COB)").group("code") == "COB"
