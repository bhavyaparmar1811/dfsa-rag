"""Chunker for the Glossary module (GLO).

GLO has two parts:

    1 INTRODUCTION   ordinary rules (1.1.1, 1.2.1) and Guidance
    2 GLOSSARY       77 two-column tables, "Defined Term | Definition",
                     one row per term, under letter headings A-Z

Chapter 1 goes through the Rulebook chunker. Each table row of chapter 2
becomes one `definition` chunk. A row with an empty term cell continues the
previous definition across a page break, and each page's repeated header row
is skipped.
"""

from collections.abc import Callable

from docling_core.types.doc import DocItemLabel, DoclingDocument, TableItem

from src.ingestion.chunkers.rulebook import ChunkResult, RulebookChunker
from src.ingestion.elements import Element, clean, collapse_row
from src.ingestion.header import build_header
from src.ingestion.ids import chunk_id
from src.ingestion.manifest import ManifestEntry
from src.ingestion.models import Chunk, ChunkType, embedding_text

HEADER_ROW = ("Defined Term", "Definition")
# Navigation on the Glossary pages: the A-Z index, letter headings (sometimes
# read as plain text, e.g. "I"), "Back to top", and "[no definitions]" under X and Y.
NAVIGATION = {
    "Back to top",
    "[no definitions]",
    "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z",
}


def is_navigation(el: Element) -> bool:
    return el.text in NAVIGATION or (len(el.text) == 1 and el.text.isupper())


class GlossaryChunker:
    def __init__(self, entry: ManifestEntry, count_tokens: Callable[[str], int]):
        self.entry = entry
        self.count_tokens = count_tokens

    def chunk(self, doc: DoclingDocument, elements: list[Element]) -> ChunkResult:
        intro = [e for e in elements if not e.is_table and not is_navigation(e)]
        result = RulebookChunker(self.entry, self.count_tokens).chunk(intro)
        start = max((c.order for c in result.chunks), default=-1) + 1
        definitions = [
            self._definition(term, text, pages, start + i)
            for i, (term, text, pages) in enumerate(self.entries(doc))
        ]
        return ChunkResult(chunks=result.chunks + definitions, issues=result.issues)

    @staticmethod
    def entries(doc: DoclingDocument) -> list[tuple[str, str, list[int]]]:
        """(term, definition, pages) for every Glossary row, in document order."""
        tables = [
            item
            for item, _level in doc.iterate_items()
            if isinstance(item, TableItem) and item.label == DocItemLabel.TABLE and item.prov
        ]
        tables.sort(key=lambda t: (t.prov[0].page_no, _top(doc, t)))
        rows = [
            (table.prov[0].page_no, [clean(c.text) for c in row])
            for table in tables
            for row in table.data.grid
        ]
        return rows_to_entries(rows)

    def _definition(self, term: str, definition: str, pages: list[int], order: int) -> Chunk:
        e = self.entry
        key = f"{e.module_code} {term}"
        heading_path = [e.rulebook, f"{e.module_name} ({e.module_code})", f"Definition of '{term}'"]
        header = build_header(heading_path, ChunkType.DEFINITION, key)
        text = f"{term}: {definition}"
        return Chunk(
            chunk_id=chunk_id(e.id, e.version, key),
            key=key,
            citation=key,
            chunk_type=ChunkType.DEFINITION,
            binding=True,
            doc_id=e.id,
            regulator=e.regulator,
            module=e.module_code,
            version=e.version,
            effective_date=e.effective_date,
            source_file=e.file_path.rsplit("/", 1)[-1],
            section="2",
            order=order,
            heading_path=heading_path,
            header_text=header,
            text=text,
            page_start=min(pages),
            page_end=max(pages),
            token_count=self.count_tokens(embedding_text(header, text)),
            term=term,
        )


def rows_to_entries(rows: list[tuple[int, list[str]]]) -> list[tuple[str, str, list[int]]]:
    """(term, definition, pages) from (page, cells) table rows, in order.

    Each page's repeated "Defined Term | Definition" header row is skipped, and
    a row with an empty term cell continues the previous definition (it ran on
    over a page break).
    """
    out: list[tuple[str, str, list[int]]] = []
    for page, cells in rows:
        if tuple(cells[:2]) == HEADER_ROW or not any(cells):
            continue
        term, definition = cells[0], " ".join(collapse_row(cells[1:]))
        if not term and out:
            prev_term, prev_text, prev_pages = out[-1]
            out[-1] = (prev_term, f"{prev_text}\n{definition}", [*prev_pages, page])
        elif term:
            out.append((term, definition, [page]))
    return out


def _top(doc: DoclingDocument, table: TableItem) -> float:
    prov = table.prov[0]
    return prov.bbox.to_top_left_origin(doc.pages[prov.page_no].size.height).t
