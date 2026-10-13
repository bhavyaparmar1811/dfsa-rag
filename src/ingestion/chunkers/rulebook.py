"""Structure-aware chunker for DFSA Rulebook modules (GEN, COB, AML, MKT, FPR).

Walks the cleaned elements in reading order as a state machine:

    chapter -> section -> sub-heading -> rule | Guidance

and emits one child chunk per rule and per Guidance block, plus one parent
chunk per section (stored for LLM context, never embedded).

The hierarchy comes from the numbers, not from Docling's labels: rule 3.3.2
belongs to section 3.3 and chapter 3 even if a heading was mislabelled or
missed. Docling labels are only used as hints (e.g. to accept a section
heading). Sequence guards reject number-like text that would move the
structure backwards or jump ahead implausibly; rejected lines stay in the text
of the current chunk and are listed in the issues report.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from src.ingestion import patterns as P
from src.ingestion.elements import Element
from src.ingestion.header import build_header
from src.ingestion.ids import chunk_id
from src.ingestion.manifest import ManifestEntry
from src.ingestion.models import Chunk, ChunkType, embedding_text

NumKey = tuple[bool, tuple[tuple[int, str], ...]]


@dataclass
class _Section:
    number: str  # "3.3" or "A2.1"; "3" or "APP2" for a chapter-level container
    title: str | None
    chapter_label: str
    loose_text: list[str] = field(default_factory=list)  # text outside any rule/Guidance
    # Holds Guidance/text that sits directly under a chapter or appendix heading,
    # before its first section (e.g. COB chapter 3's introductory Guidance).
    is_container: bool = False


@dataclass
class _Unit:
    kind: ChunkType
    key: str
    citation: str
    section: _Section
    rule_number: str | None
    guidance_for: str | None
    subheading: str | None
    scope_rule: bool
    lines: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    deleted: bool = False

    def add(self, el: Element) -> None:
        self.lines.append(el.text)
        self.pages.append(el.page)


@dataclass
class ChunkIssues:
    """Things the validator reports so a human can check them."""

    rejected_numbers: list[str] = field(default_factory=list)  # "p.42 3.5.4 ...: reason"
    implicit_sections: list[str] = field(default_factory=list)  # sections with no heading seen
    deleted_sections: list[str] = field(default_factory=list)  # "5.4 Deleted": no parent chunk
    loose_text: list[str] = field(default_factory=list)  # text outside rules/Guidance
    skipped_front_matter: int = 0
    skipped_running_headers: int = 0


@dataclass
class ChunkResult:
    chunks: list[Chunk]
    issues: ChunkIssues


class RulebookChunker:
    def __init__(self, entry: ManifestEntry, count_tokens: Callable[[str], int]):
        self.entry = entry
        self.module = entry.module_code
        self.count_tokens = count_tokens

    # ---- public -------------------------------------------------------------------

    def chunk(self, elements: list[Element]) -> ChunkResult:
        self._reset()
        for i, el in enumerate(elements):
            nxt = elements[i + 1] if i + 1 < len(elements) else None
            self._step(el, nxt)
        self._close_unit()
        return ChunkResult(chunks=self._build_chunks(), issues=self.issues)

    # ---- state machine ------------------------------------------------------------

    def _reset(self) -> None:
        self.issues = ChunkIssues()
        self.started = False
        self.chapter: tuple[int, str] | None = None  # number key of the current chapter
        self.chapter_titles: dict[str, str] = {}
        self.chapter_label = ""
        self.container_id: str | None = None  # "3" or "APP2": id for chapter-level content
        self.appendix: str | None = None
        self.section: _Section | None = None
        self.sections: dict[str, _Section] = {}
        self.subheading: str | None = None
        self.in_application = False
        self.unit: _Unit | None = None
        self.units: list[_Unit] = []
        self.last_rule: NumKey | None = None
        self.last_section: NumKey | None = None

    def _step(self, el: Element, nxt: Element | None) -> None:
        text = el.text

        m = P.RUNNING_HEADER.match(text)
        if m and m.group("code") == self.module:
            self.issues.skipped_running_headers += 1
            return

        if el.is_heading and P.GUIDANCE.match(text):
            if self.started:
                self._open_guidance(el)
            return

        if not el.is_table and self._try_chapter(el):
            return
        if not el.is_table and self._try_appendix(el):
            return
        if not el.is_table and self._try_rule(el):
            return
        if not el.is_table and self._try_section(el, nxt):
            return
        if el.is_heading and self.started:
            self._subheading(el, nxt)
            return
        self._body(el)

    def _try_chapter(self, el: Element) -> bool:
        m = P.CHAPTER.match(el.text)
        if not m or len(el.text) > P.MAX_CHAPTER_CHARS or self.appendix:
            return False
        key = P.number_key(m.group("num"))[0]
        if self.chapter is not None and key <= self.chapter:
            return False
        self._close_unit()
        self.started = True
        self.chapter = key
        num = m.group("num")
        self.chapter_titles[num] = m.group("title").strip()
        self.chapter_label = f"Chapter {num}: {m.group('title').strip()}"
        self.container_id = num
        self.section = None
        self._set_subheading(None)
        return True

    def _try_appendix(self, el: Element) -> bool:
        m = P.APPENDIX.match(el.text)
        if not m or len(el.text) > P.MAX_HEADING_CHARS:
            return False
        if self.appendix is not None and int(m.group("num")) <= int(self.appendix):
            return False
        self._close_unit()
        self.started = True
        self.appendix = m.group("num")
        self.chapter_label = f"Appendix {self.appendix}: {m.group('title').strip()}"
        self.container_id = f"APP{self.appendix}"
        self.section = None
        self._set_subheading(None)
        return True

    def _try_rule(self, el: Element) -> bool:
        m = P.RULE.match(el.text)
        if not m:
            return False
        is_app = bool(m.group("app"))
        num = m.group("num")
        key: NumKey = (is_app, P.number_key(num))
        section_num = num.rsplit(".", 1)[0]
        section_id = ("A" if is_app else "") + section_num
        label = ("A" if is_app else "") + num

        reason = None
        if is_app != bool(self.appendix):
            reason = "appendix numbering outside an appendix" if is_app else "inside an appendix"
        elif self.last_rule is not None and key <= self.last_rule:
            reason = f"not after previous rule {self._last_rule_label}"
        elif self.section is None or self.section.number != section_id:
            # A rule opening a section whose heading we didn't see must be its first rule.
            if key[1][-1][0] != 1:
                reason = f"jumps to section {section_id} without its heading"
            elif self.last_section is not None and (is_app, key[1][:-1]) <= self.last_section:
                reason = f"section {section_id} is not after the current section"
        if reason:
            self.issues.rejected_numbers.append(f"p.{el.page} {el.text[:60]!r}: {reason}")
            return False

        self._close_unit()
        self.started = True
        if self.section is None or self.section.number != section_id:
            self.issues.implicit_sections.append(f"{self.module} {section_id} (p.{el.page})")
            self._enter_section(section_id, None, (is_app, key[1][:-1]))
            if not is_app:
                self._sync_chapter(num.split(".")[0])

        # An "Application" sub-heading covers the section's scope rules only. The
        # first rule that isn't one (e.g. COB 3.3.2, the main obligation) ends it.
        scope_rule = bool(P.SCOPE_RULE.match(m.group("rest")))
        if self.in_application and not scope_rule:
            self._set_subheading(None)

        citation = f"{self.module} {label}"
        self.unit = _Unit(
            kind=ChunkType.RULE,
            key=citation,
            citation=citation,
            section=self.section,
            rule_number=label,
            guidance_for=None,
            subheading=self.subheading,
            scope_rule=scope_rule,
            deleted=bool(P.DELETED.match(m.group("rest").strip())),
        )
        self.unit.add(el)
        self.last_rule = key
        self._last_rule_label = label
        return True

    def _try_section(self, el: Element, nxt: Element | None) -> bool:
        m = P.SECTION.match(el.text)
        if not m or P.RULE.match(el.text) or len(el.text) > P.MAX_HEADING_CHARS:
            return False
        is_app = bool(m.group("app"))
        if is_app != bool(self.appendix):
            return False
        key: NumKey = (is_app, P.number_key(m.group("num")))
        if self.last_section is not None and key <= self.last_section:
            return False
        # A plain-text match (not labelled a heading) needs structure right after it.
        if not el.is_heading and not (nxt and (nxt.is_heading or P.RULE.match(nxt.text))):
            return False
        if not is_app and self.chapter is not None and key[1][0] < self.chapter:
            return False
        # Before the first chapter only a chapter's first section can start the
        # structure, so a stray number-like line can't shift everything after it.
        if not self.started and key[1][-1][0] != 1:
            return False

        self._close_unit()
        self.started = True
        section_id = ("A" if is_app else "") + m.group("num")
        if not is_app:
            self._sync_chapter(m.group("num").split(".")[0])
        title = m.group("title").strip()
        if P.DELETED.match(title):
            self.issues.deleted_sections.append(section_id)
        self._enter_section(section_id, title, key)
        return True

    def _subheading(self, el: Element, nxt: Element | None) -> None:
        # A heading followed by more structure (a rule, another heading) labels what
        # comes next. A heading followed by body text while a rule/Guidance is open is
        # a heading *inside* that rule or Guidance, so it stays in its text.
        structural = (
            self.unit is None or nxt is None or nxt.is_heading or bool(P.RULE.match(nxt.text))
        )
        if structural:
            self._close_unit()
            self._set_subheading(el.text)
        else:
            self.unit.add(el)

    def _body(self, el: Element) -> None:
        if not self.started:
            self.issues.skipped_front_matter += 1
        elif self.unit is not None:
            self.unit.add(el)
        elif (section := self._current_section()) is not None:
            section.loose_text.append(el.text)
            self.issues.loose_text.append(f"p.{el.page} {section.number}: {el.text[:60]!r}")
        else:
            self.issues.loose_text.append(f"p.{el.page} (no section) {el.text[:60]!r}")

    def _open_guidance(self, el: Element) -> None:
        if self.unit is not None and self.unit.kind == ChunkType.GUIDANCE:
            return  # a repeated "Guidance" label continues the same block
        rule = self.unit if self.unit is not None and self.unit.kind == ChunkType.RULE else None
        self._close_unit()
        section = self._current_section()
        if section is None:
            self.issues.loose_text.append(f"p.{el.page} Guidance outside any chapter")
            return
        target = rule.key if rule else f"{self.module} {section.number}"
        key = f"{target} Guidance"
        n = sum(1 for u in self.units if u.key == key or u.key.startswith(f"{key} ("))
        if n:
            key = f"{key} ({n + 1})"
        self.unit = _Unit(
            kind=ChunkType.GUIDANCE,
            key=key,
            citation=target,
            section=section,
            rule_number=rule.rule_number if rule else None,
            guidance_for=target,
            subheading=self.subheading,
            scope_rule=False,
        )

    # ---- helpers ------------------------------------------------------------------

    def _enter_section(self, number: str, title: str | None, key: NumKey) -> None:
        self.section = _Section(number=number, title=title, chapter_label=self.chapter_label)
        self.sections[number] = self.section
        self.last_section = key
        self._set_subheading(None)

    def _current_section(self) -> _Section | None:
        """The current section, or a chapter-level container if we're before the first one."""
        if self.section is None and self.container_id is not None:
            self.section = self.sections.get(self.container_id) or _Section(
                number=self.container_id,
                title=None,
                chapter_label=self.chapter_label,
                is_container=True,
            )
            self.sections[self.container_id] = self.section
        return self.section

    def _sync_chapter(self, num: str) -> None:
        """Keep the chapter in step with section/rule numbers if its heading was missed."""
        key = P.number_key(num)[0]
        if self.chapter != key:
            self.chapter = key
            self.container_id = num
            title = self.chapter_titles.get(num)
            self.chapter_label = f"Chapter {num}: {title}" if title else f"Chapter {num}"

    def _set_subheading(self, text: str | None) -> None:
        self.subheading = text
        self.in_application = bool(text and P.APPLICATION_HEADING.match(text))

    def _close_unit(self) -> None:
        if self.unit is not None and self.unit.lines:
            self.units.append(self.unit)
        self.unit = None

    # ---- output -------------------------------------------------------------------

    def _heading_path(self, section: _Section, subheading: str | None) -> list[str]:
        e = self.entry
        path = [e.rulebook, f"{e.module_name} ({self.module})", section.chapter_label]
        if not section.is_container:
            title = f": {section.title}" if section.title else ""
            path.append(f"Section {section.number}{title}")
        if subheading:
            path.append(subheading)
        return [p for p in path if p]

    def _make_chunk(self, *, kind, key, citation, section, text, pages, order, **extra) -> Chunk:
        e = self.entry
        heading_path = self._heading_path(section, extra.get("subheading"))
        header = build_header(heading_path, kind, citation)
        return Chunk(
            chunk_id=chunk_id(e.id, e.version, key),
            key=key,
            citation=citation,
            chunk_type=kind,
            binding=kind != ChunkType.GUIDANCE,
            doc_id=e.id,
            regulator=e.regulator,
            module=self.module,
            version=e.version,
            effective_date=e.effective_date,
            source_file=e.file_path.rsplit("/", 1)[-1],
            section=section.number,
            order=order,
            heading_path=heading_path,
            header_text=header,
            text=text,
            page_start=min(pages),
            page_end=max(pages),
            token_count=self.count_tokens(embedding_text(header, text)),
            **extra,
        )

    def _build_chunks(self) -> list[Chunk]:
        guidance_by_target: dict[str, list[str]] = {}
        for u in self.units:
            if u.kind == ChunkType.GUIDANCE:
                guidance_by_target.setdefault(u.guidance_for, []).append(u.key)

        children: list[Chunk] = []
        for order, u in enumerate(self.units):
            children.append(
                self._make_chunk(
                    kind=u.kind,
                    key=u.key,
                    citation=u.citation,
                    section=u.section,
                    text="\n".join(u.lines),
                    pages=u.pages,
                    order=order,
                    rule_number=u.rule_number,
                    parent_key=f"{self.module} {u.section.number}",
                    guidance_for=u.guidance_for,
                    guidance_keys=guidance_by_target.get(u.key, []) if u.kind == "rule" else [],
                    subheading=u.subheading,
                    scope_rule=u.scope_rule,
                    deleted=u.deleted,
                )
            )

        return children + self._build_sections(children)

    def _build_sections(self, children: list[Chunk]) -> list[Chunk]:
        """Parent chunks: each section's children (and loose text) in document order."""
        by_section: dict[str, list[Chunk]] = {}
        for c in children:
            by_section.setdefault(c.section, []).append(c)

        parents = []
        for number, section in self.sections.items():
            kids = by_section.get(number, [])
            lines = list(section.loose_text)
            subheading = None
            for c in kids:
                if c.deleted:
                    continue
                if c.subheading != subheading and c.subheading:
                    lines.append(c.subheading)
                subheading = c.subheading
                lines.append(
                    f"Guidance\n{c.text}" if c.chunk_type == ChunkType.GUIDANCE else c.text
                )
            if not lines:
                continue
            key = f"{self.module} {number}"
            pages = [p for c in kids for p in (c.page_start, c.page_end)] or [0]
            parents.append(
                self._make_chunk(
                    kind=ChunkType.SECTION,
                    key=key,
                    citation=key,
                    section=section,
                    text="\n".join(lines),
                    pages=pages,
                    order=kids[0].order if kids else -1,
                )
            )
        return parents
