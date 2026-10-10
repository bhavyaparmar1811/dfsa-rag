from src.ingestion.elements import (
    Element,
    clean,
    collapse_row,
    is_layout_table,
    merge_margin_numbers,
    sort_reading_order,
)


def el(text, page=1, top=100.0, left=72.0, label="list_item"):
    return Element(label=label, text=text, page=page, top=top, left=left)


def test_clean_normalises_ligatures_and_whitespace():
    assert clean("  the ﬁrm   must­\n comply ") == "the firm must comply"


def test_sorts_by_page_then_position_not_input_order():
    # Docling's order for COB page 32: Guidance and 3.3.3 came before 3.3.2.
    docling_order = [
        el("Guidance", page=32, top=475, label="section_header"),
        el("3.3.3 If the Client Agreement ...", page=32, top=712),
        el("3.3.2 (1) Subject to (2), ...", page=32, top=101),
        el("(2) An Authorised Firm may ...", page=32, top=283),
        el("end of 3.3.1 from previous page", page=31, top=700),
    ]
    texts = [e.text for e in sort_reading_order(docling_order)]
    assert texts == [
        "end of 3.3.1 from previous page",
        "3.3.2 (1) Subject to (2), ...",
        "(2) An Authorised Firm may ...",
        "Guidance",
        "3.3.3 If the Client Agreement ...",
    ]


def test_same_line_is_read_left_to_right():
    # A margin number whose top is a fraction lower than its text still comes first.
    line = [
        el("(1) A Relevant Person may ...", top=100.0, left=115),
        el("8.1.1", top=100.6, left=72),
    ]
    assert [e.text for e in sort_reading_order(line)] == ["8.1.1", "(1) A Relevant Person may ..."]


def test_merges_margin_rule_number_with_following_text():
    merged = merge_margin_numbers([el("8.1.1", left=72), el("(1) A Relevant Person may ...")])
    assert [e.text for e in merged] == ["8.1.1 (1) A Relevant Person may ..."]
    assert merged[0].left == 72


def test_does_not_merge_number_into_a_heading_or_across_pages():
    heading = el("8.2 Outsourcing", label="section_header")
    assert len(merge_margin_numbers([el("8.1.1"), heading])) == 2
    assert len(merge_margin_numbers([el("8.1.1", page=1), el("(1) ...", page=2)])) == 2


def test_layout_table_is_rule_text_read_as_a_table():
    # FPR p.14: rules and even a section heading laid out in columns.
    rows = [
        ["3.1.8", "If the DFSA receives notice", "If the DFSA receives notice"],
        ["3.2", "Notification for Public Funds", "Notification for Public Funds"],
        ["(c)", "appropriate investor redress", "appropriate investor redress"],
        ["", "(ii)", "the duties owed by the member"],
    ]
    assert is_layout_table(rows)
    assert collapse_row(rows[0]) == ["3.1.8", "If the DFSA receives notice"]


def test_data_table_is_kept_as_a_table():
    # MKT App1: a matrix with a tick per product column.
    rows = [
        ["A1.1.1"] * 5,
        ["1.1", "General information", "Shares", "Warrants", "Debentures"],
    ]
    assert not is_layout_table(rows)
    # GLO-style two-column table: first column is a term, not a marker.
    assert not is_layout_table([["Retail Client", "A Client who is not ..."]])
