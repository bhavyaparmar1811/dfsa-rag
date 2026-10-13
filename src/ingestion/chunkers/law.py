"""Structure-aware chunker for the DIFC Laws.

A Law is structured differently from a Rulebook module:

    PART 2: THE DFSA
      CHAPTER 1 - THE STRUCTURE OF THE DFSA     ("CHAPTER 1: ..." in the Markets Law)
        7. The DFSA                              an Article: heading, then (1), (a) ...
    SCHEDULE 1 INTERPRETATION                    numbering restarts at 1
      1. Rules of interpretation                 a numbered Schedule paragraph
      3. Defined Terms                           followed by a "Term | Definition" table

Each Article and each numbered Schedule paragraph becomes an `article` chunk,
each row of a definitions table a `definition` chunk, and unnumbered Schedule
text a `text` chunk (binding: in a Law it is still law). The parent is the
Chapter, or the Part if it has no Chapters, or the Schedule.

Everything before PART 1 (title page, contents, the list of amendments, which
"does not form part of the Law") is skipped. As in the Rulebook chunker, the
numbers decide the structure: an "N. Title" line only opens an Article if it
comes after the previous one, so a numbered list item inside an Article can't.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from src.ingestion import patterns as P
from src.ingestion.chunkers.glossary import HEADER_ROWS
from src.ingestion.chunkers.rulebook import ChunkIssues, ChunkResult
from src.ingestion.elements import Element, collapse_row
from src.ingestion.header import build_header
from src.ingestion.ids import chunk_id
from src.ingestion.manifest import ManifestEntry
from src.ingestion.models import Chunk, ChunkType, embedding_text

# "PART 2: THE DFSA"; "CHAPTER 1 - ..." or "CHAPTER 1: ..."; the Collective
# Investment Law writes its Chapters in title case ("Chapter 1: General").
PART = re.compile(r"^(?i:PART)\s+(?P<num>\d+[A-Z]?)\s*[:\-–]\s*(?P<title>.+)$")
CHAPTER = re.compile(r"^(?i:CHAPTER)\s+(?P<num>\d+[A-Z]?)\s*[:\-–]\s*(?P<title>.+)$")
SCHEDULE = re.compile(r"^SCHEDULE(?:\s+(?P<num>\d+))?(?:\s+(?P<title>[A-Z][A-Z ,&'()-]+))?$")
ARTICLE = re.compile(r"^(?P<num>\d+[A-Z]?)\.\s+(?P<title>[A-Z].*)$")
ARTICLE_DELETED = re.compile(r"^(?P<num>\d+[A-Z]?)\.\s+(?:Deleted|not in use)\.?$", re.I)
# The Collective Investment Law prints six deleted Articles on one line.
DELETED_RUN = re.compile(r"^(?:\d+[A-Z]?\.\s+Deleted\s*){2,}$", re.I)
DELETED_TITLE = re.compile(r"^(DELETED|NOT IN USE)$", re.I)
MAX_TITLE_CHARS = 160
# The furthest an Article number may jump forward. Deleted Articles are usually
# printed ("55. Deleted"), so real gaps are small; a larger jump is a stray line.
MAX_ARTICLE_JUMP = 10

SMALL_WORDS = {"a", "an", "and", "of", "the", "on", "in", "for", "to", "or"}


def law_name(entry: ManifestEntry) -> str:
    """The Law's name as the Rulebook cites it: "MARKETS LAW" -> "Markets Law"."""
    words = entry.module_name.split()
    if not any(w.isupper() and len(w) > 1 for w in words):
        return entry.module_name
    return " ".join(
        w.lower() if i and w.lower() in SMALL_WORDS else w.capitalize() for i, w in enumerate(words)
    )


@dataclass
class _Container:
    key: str  # "Regulatory Law Part 2 Ch. 1", "Regulatory Law Sch. 1"
    section: str  # "Part 2 Ch. 1", "Sch. 1"
    heading_path: list[str]


@dataclass
class _Unit:
    kind: ChunkType
    key: str
    citation: str
    container: _Container
    number: str | None = None
    term: str | None = None
    lines: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    deleted: bool = False

    def add(self, el: Element) -> None:
        self.lines.append(el.text)
        self.pages.append(el.page)


class LawChunker:
    def __init__(self, entry: ManifestEntry, count_tokens: Callable[[str], int]):
        self.entry = entry
        self.law = law_name(entry)
        self.count_tokens = count_tokens

    def chunk(self, elements: list[Element]) -> ChunkResult:
        self.issues = ChunkIssues()
        self.started = False
        self.part: tuple[str, str] | None = None
        self.chapter: tuple[str, str] | None = None
        self.schedule: str | None = None
        self.schedule_title = ""
        self.container: _Container | None = None
        self.unit: _Unit | None = None
        self.units: list[_Unit] = []
        self.last_article: tuple[tuple[int, str], ...] | None = None
        self.in_definitions = False
        self.containers: dict[str, _Container] = {}

        for el in elements:
            self._step(el)
        self._close()
        return ChunkResult(chunks=self._build(), issues=self.issues)

    # ---- state machine ------------------------------------------------------------

    def _step(self, el: Element) -> None:
        text = el.text
        if not el.is_table:
            if m := PART.match(text):
                self.started = True
                self.part, self.chapter, self.schedule = (m["num"], m["title"].strip()), None, None
                self._enter_container()
                return
            if self.started and (m := CHAPTER.match(text)):
                self.chapter = (m["num"], m["title"].strip())
                self._enter_container()
                return
            if self.started and el.is_heading and (m := SCHEDULE.match(text)):
                self.schedule = m["num"] or "1"
                self.schedule_title = (m["title"] or "").strip()
                self.last_article = None  # Schedule paragraphs restart at 1
                self._enter_container()
                return
            if self._awaiting_schedule_title(el):
                # "SCHEDULE 1" and its title "INTERPRETATION" as two headings.
                self.schedule_title = text
                self.container.heading_path[-1] = f"Schedule {self.schedule}: {text}"
                return
            if self.started and self._try_deleted_run(el):
                return
            if self.started and self._try_article(el):
                return
        if not self.started:
            self.issues.skipped_front_matter += 1
        elif el.is_table and self.in_definitions or el.is_table and _is_definitions(text):
            self._definitions(el)
        elif self.unit is not None:
            self.unit.add(el)
        elif self.container is not None:
            self._open(ChunkType.TEXT, f"{self.container.key} Text", self.container.key)
            self.issues.unnumbered_text.append(f"p.{el.page} {self.unit.key}: {text[:50]!r}")
            self.unit.add(el)
        else:
            self.issues.loose_text.append(f"p.{el.page} {text[:60]!r}")

    def _awaiting_schedule_title(self, el: Element) -> bool:
        return (
            self.schedule is not None
            and not self.schedule_title
            and self.unit is None
            and el.is_heading
            and el.text.isupper()
            and len(el.text) <= MAX_TITLE_CHARS
        )

    def _try_article(self, el: Element) -> bool:
        m = ARTICLE_DELETED.match(el.text) or ARTICLE.match(el.text)
        if not m or len(el.text) > MAX_TITLE_CHARS:
            return False
        key = P.number_key(m["num"])
        if self.last_article is not None and (
            key <= self.last_article or key[0][0] - self.last_article[0][0] > MAX_ARTICLE_JUMP
        ):
            if el.is_heading:
                self.issues.rejected_numbers.append(f"p.{el.page} {el.text[:60]!r}")
            return False
        # Outside a heading, "N. Title" needs no full stop at the end: a numbered
        # list item ("1. a statement of ...;") is sentence-like.
        sentence_like = el.text.rstrip().endswith((".", ";", ":", ","))
        if not el.is_heading and sentence_like and not ARTICLE_DELETED.match(el.text):
            return False
        self._open_article(m["num"], el, deleted=bool(ARTICLE_DELETED.match(el.text)))
        return True

    def _try_deleted_run(self, el: Element) -> bool:
        if not DELETED_RUN.match(el.text):
            return False
        for num in re.findall(r"(\d+[A-Z]?)\.", el.text):
            self._open_article(
                num, Element(el.label, f"{num}. Deleted", el.page, el.top, el.left), True
            )
        return True

    def _open_article(self, num: str, el: Element, deleted: bool) -> None:
        if self.container is None:
            self._enter_container()
        prefix = f"{self.law} Sch. {self.schedule} para." if self.schedule else f"{self.law} Art."
        self._open(ChunkType.ARTICLE, f"{prefix} {num}", f"{prefix} {num}", number=num)
        self.unit.deleted = deleted
        self.unit.add(el)
        self.last_article = P.number_key(num)
        self.in_definitions = False

    def _definitions(self, el: Element) -> None:
        """Each row of a definitions table becomes its own `definition` unit. A row
        with an empty term cell continues the previous definition, including one
        that started in the table on the previous page."""
        self._close()
        self.in_definitions = True
        for cells in _markdown_rows(el.text):
            if tuple(cells[:2]) in HEADER_ROWS or not any(cells):
                continue
            term, definition = cells[0], " ".join(collapse_row(cells[1:]))
            if not term:
                last = next(
                    (u for u in reversed(self.units) if u.kind == ChunkType.DEFINITION), None
                )
                if last is not None:
                    last.lines.append(definition)
                    last.pages.append(el.page)
                continue
            key = f"{self.container.key} '{term}'"
            self._open(ChunkType.DEFINITION, key, key, term=term)
            self.unit.lines.append(f"{term}: {definition}")
            self.unit.pages.append(el.page)
            self._close()

    # ---- helpers ------------------------------------------------------------------

    def _enter_container(self) -> None:
        self._close()
        self.in_definitions = False
        path = ["DIFC Laws", self.law]
        if self.schedule:
            title = self.schedule_title
            path.append(f"Schedule {self.schedule}" + (f": {title}" if title else ""))
            section = f"Sch. {self.schedule}"
        else:
            num, title = self.part or ("0", "")
            path.append(f"Part {num}: {title}")
            section = f"Part {num}"
            if self.chapter:
                path.append(f"Chapter {self.chapter[0]}: {self.chapter[1]}")
                section += f" Ch. {self.chapter[0]}"
            title = self.chapter[1] if self.chapter else title
            if DELETED_TITLE.match(title):
                self.issues.deleted_sections.append(section)
        key = f"{self.law} {section}"
        self.container = self.containers.setdefault(key, _Container(key, section, path))

    def _open(self, kind: ChunkType, key: str, citation: str, **extra) -> None:
        self._close()
        n = sum(1 for u in self.units if u.key == key or u.key.startswith(f"{key} ("))
        if n:
            key = f"{key} ({n + 1})"
        self.unit = _Unit(kind=kind, key=key, citation=citation, container=self.container, **extra)

    def _close(self) -> None:
        if self.unit is not None and self.unit.lines:
            self.units.append(self.unit)
        self.unit = None

    # ---- output -------------------------------------------------------------------

    def _chunk(self, *, kind, key, citation, container, text, pages, order, **extra) -> Chunk:
        e = self.entry
        path = list(container.heading_path)
        if kind == ChunkType.DEFINITION:
            path.append(f"Definition of '{extra['term']}'")
        header = build_header(path, kind, citation)
        return Chunk(
            chunk_id=chunk_id(e.id, e.version, key),
            key=key,
            citation=citation,
            chunk_type=kind,
            binding=kind != ChunkType.GUIDANCE,  # all of a Law is law
            doc_id=e.id,
            regulator=e.regulator,
            module=e.module_code,
            version=e.version,
            effective_date=e.effective_date,
            source_file=e.file_path.rsplit("/", 1)[-1],
            section=container.section,
            order=order,
            heading_path=path,
            header_text=header,
            text=text,
            page_start=min(pages),
            page_end=max(pages),
            token_count=self.count_tokens(embedding_text(header, text)),
            **extra,
        )

    def _build(self) -> list[Chunk]:
        children = [
            self._chunk(
                kind=u.kind,
                key=u.key,
                citation=u.citation,
                container=u.container,
                text="\n".join(u.lines),
                pages=u.pages,
                order=order,
                rule_number=u.number,
                parent_key=u.container.key,
                term=u.term,
                deleted=u.deleted,
            )
            for order, u in enumerate(self.units)
        ]
        parents = []
        for container in self.containers.values():
            kids = [c for c in children if c.parent_key == container.key and not c.deleted]
            if not kids:
                continue
            parents.append(
                self._chunk(
                    kind=ChunkType.SECTION,
                    key=container.key,
                    citation=container.key,
                    container=container,
                    text="\n".join(c.text for c in kids),
                    pages=[p for c in kids for p in (c.page_start, c.page_end)],
                    order=kids[0].order,
                )
            )
        return children + parents


def _is_definitions(markdown: str) -> bool:
    rows = _markdown_rows(markdown)
    return bool(rows) and tuple(rows[0][:2]) in HEADER_ROWS


def _markdown_rows(markdown: str) -> list[list[str]]:
    """Rows of a Markdown table (written by elements.to_markdown, which never
    leaves a literal "|" inside a cell)."""
    rows = []
    for line in markdown.splitlines():
        line = line.strip()
        if not line.startswith("|") or re.fullmatch(r"\|[\s\-:|]+\|", line):
            continue
        rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows
