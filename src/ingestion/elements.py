"""DoclingDocument -> a flat, cleaned, correctly ordered list of `Element`s.

This is the only module the chunkers depend on for document content, so they
can be unit-tested with hand-written `Element` lists and never touch Docling.

Two things Docling gets wrong for DFSA PDFs are corrected here (measured on
COB, see the step-2 checkpoint):

1. **Rule numbers live in `orig`, not `text`.** Docling reads "3.3.2 (1) ..."
   as a numbered list item and moves "3.3.2" into `marker`, leaving `text`
   without it. We use `orig`, the line as printed.
2. **Docling's reading order is unreliable.** It groups numbered items into
   lists, and when a list continues onto the next page or across a heading,
   the whole group moves. On COB that put rule 3.3.2 after 3.3.3, and put the
   end of a rule that continues from the previous page after the Guidance at
   the bottom of the page. These rulebooks are single-column, so we order
   elements by their position on the page instead.

Dropped entirely: running headers/footers (Docling's FURNITURE layer), the
table of contents, pictures, captions, and any text inside a picture.
"""

import re
import unicodedata
from dataclasses import dataclass

from docling_core.types.doc import (
    BoundingBox,
    ContentLayer,
    DocItemLabel,
    DoclingDocument,
    PictureItem,
    TableItem,
)

# Elements with these labels never carry rule text.
SKIP_LABELS = {
    DocItemLabel.PICTURE,
    DocItemLabel.CAPTION,
    DocItemLabel.PAGE_HEADER,
    DocItemLabel.PAGE_FOOTER,
}

# A table-of-contents line that Docling didn't label as document_index:
# "5.8 Qualification and experience ..........39", "9.3 Circumstances ... 22"
TOC_LINE = re.compile(r"(\.{3,}|…)\s*\d{1,3}\s*$")
PAGE_NUMBER = re.compile(r"^\d{1,4}$")
# A rule number standing alone, e.g. printed in the left margin of AML.
BARE_RULE_NUMBER = re.compile(r"^A?\d+[A-Z]?\.\d+[A-Z]?\.\d+[A-Z]?$")
# What the first column of a "layout table" holds: nothing, a chapter, section
# or rule number ("6", "3.2", "3.1.8", "14.2.3 (1)", "A7.2.1"), or a sub-item
# marker ("(1)", "(c)", "(ii)", "1.").
LAYOUT_MARKER = re.compile(
    r"^(|A?\d+[A-Z]?(\.\d+[A-Z]?){0,2}(\s*\(\d+\))?|\(\d+\)|\(?[a-z]{1,5}\)|\d+\.)$"
)
# Pictures smaller than this (in points) are logos or decorations, too small
# to contain text worth excluding by position.
MIN_FIGURE_AREA = 150 * 100
# Two elements whose tops are this close (points) are on the same line.
SAME_LINE_TOLERANCE = 4.0


@dataclass(frozen=True)
class Element:
    label: str  # Docling's label, e.g. "section_header", "list_item", "text", "table"
    text: str  # cleaned text as printed; Markdown for tables
    page: int  # PDF page number, 1-based
    top: float  # distance from the top of the page, in points
    left: float  # distance from the left edge of the page, in points

    @property
    def is_heading(self) -> bool:
        return self.label in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE)

    @property
    def is_table(self) -> bool:
        return self.label == DocItemLabel.TABLE


def clean(text: str) -> str:
    """Normalise Unicode (ligatures, compatibility forms) and whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")  # soft hyphen
    return re.sub(r"\s+", " ", text).strip()


def to_elements(doc: DoclingDocument) -> list[Element]:
    figures = _figure_boxes(doc)
    elements: list[Element] = []

    # traverse_pictures=False: Docling skips text it attached to a picture.
    for item, _level in doc.iterate_items(
        traverse_pictures=False, included_content_layers={ContentLayer.BODY}
    ):
        if item.label in SKIP_LABELS or not item.prov:
            continue
        if item.label == DocItemLabel.DOCUMENT_INDEX and is_toc(item):
            continue  # the real table of contents
        prov = item.prov[0]
        box = prov.bbox.to_top_left_origin(doc.pages[prov.page_no].size.height)

        # Figure text that Docling didn't attach to the picture.
        if any(_inside(box, fig) for fig in figures.get(prov.page_no, ())):
            continue

        if isinstance(item, TableItem):
            elements.extend(table_elements(item, doc, prov.page_no, box))
            continue

        text = clean(getattr(item, "orig", None) or getattr(item, "text", ""))
        if not text or PAGE_NUMBER.match(text) or TOC_LINE.search(text):
            continue

        elements.append(
            Element(
                label=item.label.value,
                text=text,
                page=prov.page_no,
                top=round(box.t, 1),
                left=round(box.l, 1),
            )
        )

    return merge_margin_numbers(sort_reading_order(elements))


def sort_reading_order(elements: list[Element]) -> list[Element]:
    """Top-to-bottom per page; elements on the same line go left to right."""
    ordered = sorted(elements, key=lambda e: (e.page, e.top, e.left))
    # Re-sort within each line, so a margin number whose top is a fraction of
    # a point lower than its text still comes first.
    result: list[Element] = []
    line: list[Element] = []
    for el in ordered:
        if line and (el.page != line[0].page or el.top - line[0].top > SAME_LINE_TOLERANCE):
            result.extend(sorted(line, key=lambda e: e.left))
            line = []
        line.append(el)
    result.extend(sorted(line, key=lambda e: e.left))
    return result


def merge_margin_numbers(elements: list[Element]) -> list[Element]:
    """Join a rule number printed on its own ("8.1.1") with the text after it."""
    result: list[Element] = []
    i = 0
    while i < len(elements):
        el = elements[i]
        nxt = elements[i + 1] if i + 1 < len(elements) else None
        if (
            BARE_RULE_NUMBER.match(el.text)
            and nxt is not None
            and nxt.page == el.page
            and not nxt.is_heading
            and not nxt.is_table  # a table under a rule number is that rule's body
        ):
            result.append(Element(nxt.label, f"{el.text} {nxt.text}", el.page, el.top, el.left))
            i += 2
            continue
        result.append(el)
        i += 1
    return result


def compact_markdown(markdown: str) -> str:
    """Docling pads table cells with spaces to align columns; that costs tokens
    and carries no meaning, so collapse it."""
    lines = []
    for line in markdown.strip().splitlines():
        line = re.sub(r"-{3,}", "---", line)
        lines.append(re.sub(r"[ \t]{2,}", " ", line).strip())
    return "\n".join(lines)


def is_toc(item: TableItem) -> bool:
    """Is a `document_index` item really the table of contents?

    Docling sometimes labels pages of rules as a TOC because of their number
    column (FPR pp.19 and 25: chapter 6 and sections 9.1-9.4). Every real TOC
    in the corpus has dot leaders ("Application ......... 3"); those pages
    have none, so they are kept as content.
    """
    cells = getattr(getattr(item, "data", None), "table_cells", [])
    return any(re.search(r"\.{3,}|…", c.text) for c in cells)


def collapse_row(cells: list[str]) -> list[str]:
    """Drop empty cells and the copies Docling makes of a cell spanning columns."""
    out: list[str] = []
    for cell in cells:
        if cell and (not out or out[-1] != cell):
            out.append(cell)
    return out


def is_layout_table(rows: list[list[str]]) -> bool:
    """True if a "table" is really rule text that Docling read as a table.

    DFSA pages print rule numbers in a left column, so Docling sometimes
    detects whole blocks of rules (even section headings) as a table. Such a
    table has at most 3 distinct cells per row (number, text, and sometimes a
    sub-item marker) and only structure markers in its first column. A real
    data table (e.g. MKT's prospectus matrix with a tick per product column)
    fails one of these tests.
    """
    if not rows:
        return False
    return all(len(collapse_row(r)) <= 3 for r in rows) and all(
        LAYOUT_MARKER.match(r[0] if r else "") for r in rows
    )


def layout_row_texts(rows: list[list[str]]) -> list[str]:
    """One line of text per row of a layout table.

    A rule number spanning several rows is repeated by Docling in each of them
    ("3.2.3 (a)", "3.2.3 (b)"), so it's kept on the first row only.
    """
    out = []
    for i, row in enumerate(rows):
        if i and row and row[0] and row[0] == rows[i - 1][0]:
            row = ["", *row[1:]]
        out.append(" ".join(collapse_row(row)))
    return out


def table_elements(
    item: TableItem, doc: DoclingDocument, page: int, box: BoundingBox
) -> list[Element]:
    grid = item.data.grid
    rows = [[clean(c.text) for c in row] for row in grid]
    height = doc.pages[page].size.height

    if is_layout_table(rows):
        out = []
        for i, (text, cells) in enumerate(zip(layout_row_texts(rows), grid, strict=True)):
            if not text:
                continue
            # Each row becomes an ordinary text element at its own position.
            placed = [
                c.bbox.to_top_left_origin(height)
                for c in cells
                if c.bbox is not None and c.text.strip()
            ]
            top = min(b.t for b in placed) if placed else box.t + i * 0.01
            left = min(b.l for b in placed) if placed else box.l
            out.append(Element("text", text, page, round(top, 1), round(left, 1)))
        return out

    first = collapse_row(rows[0]) if rows else []
    numbered = len(first) == 1 and BARE_RULE_NUMBER.match(first[0])
    # A data table headed by a rule number (MKT "A1.1.1") is that rule's body:
    # emit the number first so the chunker opens the rule, and drop that row so
    # the real column headings become the table's header row.
    markdown = to_markdown(rows[1:] if numbered else rows)
    if not markdown:
        return []
    table = Element(DocItemLabel.TABLE.value, markdown, page, round(box.t, 1), round(box.l, 1))
    if numbered:
        number = Element("text", first[0], page, round(box.t - 0.1, 1), round(box.l, 1))
        return [number, table]
    return [table]


def to_markdown(rows: list[list[str]]) -> str:
    """A compact Markdown table; the first non-empty row is the header row."""
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""
    cells = [[c.replace("|", "/") for c in r] for r in rows]
    lines = ["| " + " | ".join(cells[0]) + " |", "|" + "---|" * len(cells[0])]
    lines += ["| " + " | ".join(r) + " |" for r in cells[1:]]
    return "\n".join(lines)


def _figure_boxes(doc: DoclingDocument) -> dict[int, list[BoundingBox]]:
    boxes: dict[int, list[BoundingBox]] = {}
    for item, _level in doc.iterate_items(traverse_pictures=False):
        if isinstance(item, PictureItem) and item.prov:
            prov = item.prov[0]
            box = prov.bbox.to_top_left_origin(doc.pages[prov.page_no].size.height)
            if box.width * box.height >= MIN_FIGURE_AREA:
                boxes.setdefault(prov.page_no, []).append(box)
    return boxes


def _inside(inner: BoundingBox, outer: BoundingBox, slack: float = 2.0) -> bool:
    return (
        inner.l >= outer.l - slack
        and inner.r <= outer.r + slack
        and inner.t >= outer.t - slack
        and inner.b <= outer.b + slack
    )
