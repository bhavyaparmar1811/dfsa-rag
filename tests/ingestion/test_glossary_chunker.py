from src.ingestion.chunkers.glossary import is_navigation, rows_to_entries
from src.ingestion.elements import Element

HEADER = ["Defined Term", "Definition"]


def test_one_entry_per_term_and_header_rows_skipped():
    rows = [
        (3, HEADER),
        (3, ["Accepting Deposits", "Has the meaning given in GEN section 2.4."]),
        (3, ["Advising on Credit", "Has the meaning given in GEN Rule 2.28.1(1)(b)."]),
        (4, HEADER),  # repeated on every page
        (4, ["Retail Client", "A Client specified under COB Rule 2.3.2."]),
    ]
    assert [term for term, _, _ in rows_to_entries(rows)] == [
        "Accepting Deposits",
        "Advising on Credit",
        "Retail Client",
    ]


def test_definition_continued_over_a_page_break():
    rows = [
        (5, ["Associate", "In relation to a Person A: (a) a spouse of A;"]),
        (6, HEADER),
        (6, ["", "(b) a child or stepchild of A;"]),  # empty term cell: a continuation
    ]
    [(term, text, pages)] = rows_to_entries(rows)
    assert term == "Associate" and text.endswith("\n(b) a child or stepchild of A;")
    assert pages == [5, 6]


def test_spanning_definition_cell_is_kept_once():
    rows = [(9, ["Fund", "A collective investment fund.", "A collective investment fund."])]
    assert rows_to_entries(rows)[0][1] == "A collective investment fund."


def test_navigation_is_dropped():
    def el(text, label="section_header"):
        return Element(label, text, 1, 0.0, 72.0)

    assert is_navigation(el("Back to top")) and is_navigation(el("[no definitions]", "text"))
    assert is_navigation(el("I", "text"))  # a letter heading read as plain text
    assert not is_navigation(el("1.2 Defined Terms"))
