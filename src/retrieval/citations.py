"""The citation router: exact provisions named in the question, looked up directly.

"What does COB 3.3.2 require?" must return COB 3.3.2 itself, not whatever a
search finds near it: BM25 splits "3.3.2" into "3", "3", "2", and dense
embeddings blur numbers. So citations are parsed out of the question (with
the same patterns that extract cross-references from the corpus) and resolved
against SQLite, which knows every provision, embedded or not:

    COB 3.3.2, COB Rule 3.3.2(1)(b), Rule 3.3.2 of COB   -> the rule, its parts, its Guidance
    COB 3.3, COB section 3.3                            -> the section's children
    Rule 3.3.2 (no module)                              -> that number in every module
    Article 20 of the Regulatory Law, Regulatory Law Art. 20, Markets Law Article 2

A citation the corpus doesn't have (a typo, a deleted rule, another module) is
reported as unresolved, so the answer can say so instead of guessing.
"""

import re

from sqlalchemy import Engine

from src.ingestion.models import EMBEDDED_TYPES
from src.ingestion.refs import ART_LIST, KNOWN_MODULES, LOCAL_ARTICLE, _numbers, extract_refs
from src.retrieval.models import Candidate, CitationLookup, Filters
from src.storage.db import session_scope
from src.storage.qdrant_store import QdrantStore
from src.storage.repositories import ChunkRepository

ANY_MODULE = "*"  # "Rule 3.3.2" with no module named
RULE_NUMBER = re.compile(r"^(?P<mod>\S+) (?P<num>A?\d+[A-Z]?(?:\.\d+[A-Z]?)*)$")


def _normalise(question: str, codes: set[str]) -> str:
    """Make a typed question look like Rulebook text, so refs.py's patterns apply:
    "cob rule 3.3.2" -> "COB Rule 3.3.2", "art. 20" -> "Article 20"."""
    known = "|".join(sorted(codes | KNOWN_MODULES))
    text = re.sub(
        rf"\b({known})(?=\s+(?:rules?\s+|sections?\s+|chapters?\s+|app(?:endix)?\s?)?A?\d)",
        lambda m: m.group(1).upper(), question, flags=re.IGNORECASE,
    )  # fmt: skip
    text = re.sub(r"\brule(s?)\b(?=\s+A?\d)", r"Rule\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsection(s?)\b(?=\s+A?\d)", r"section\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:article|art\.?)(s?)(?=\s+\d)", r"Article\1", text, flags=re.IGNORECASE)
    return text


class CitationRouter:
    def __init__(
        self,
        engine: Engine,
        store: QdrantStore,
        module_codes: set[str],
        law_names: list[str],
        max_hits: int = 8,
    ):
        self.engine = engine
        self.store = store
        self.codes = set(module_codes)
        self.laws = sorted(law_names, key=len, reverse=True)  # longest name first
        self.max_hits = max_hits
        law_alt = "|".join(re.escape(name) for name in self.laws)
        # "Regulatory Law Art. 20", "Markets Law Article 2", "the Markets Law, Article 2"
        self._law_first = re.compile(
            rf"\b(?P<law>{law_alt}),?\s+Articles?\s+{ART_LIST}", re.IGNORECASE
        )
        # "Article 20 of the Regulatory Law" (refs.ARTICLE needs names ending in "Law")
        self._article_first = re.compile(
            rf"\bArticles?\s+{ART_LIST}\s+of\s+(?:the\s+)?(?:DIFC\s+)?(?P<law>{law_alt})\b",
            re.IGNORECASE,
        )

    # --- parsing -------------------------------------------------------------

    def find(self, question: str) -> list[str]:
        """Normalised citations in the question, in order of appearance."""
        text = _normalise(question, self.codes)
        found: list[str] = []
        spans: list[tuple[int, int]] = []

        def add(citation: str) -> None:
            if citation not in found:
                found.append(citation)

        named: list[tuple[int, str]] = []  # (position, law) of every Law citation
        for pattern in (self._law_first, self._article_first):
            for m in pattern.finditer(text):
                spans.append(m.span())
                law = self._canonical_law(m.group("law"))
                named.append((m.start(), law))
                for num in _numbers(m.group("nums")):
                    add(f"{law} Art. {num.upper()}")
        # "Regulatory Law Art. 76 and Article 77": a bare Article belongs to the
        # Law cited last before it.
        for m in LOCAL_ARTICLE.finditer(text):
            earlier = [law for pos, law in sorted(named) if pos < m.start()]
            if earlier and not any(s <= m.start() < e for s, e in spans):
                spans.append(m.span())
                for num in _numbers(m.group("nums")):
                    add(f"{earlier[-1]} Art. {num.upper()}")

        # Blank out the Law citations, then let refs.py read the Rulebook ones.
        rest = "".join(
            " " if any(s <= i < e for s, e in spans) else ch for i, ch in enumerate(text)
        )
        for ref in extract_refs(rest, ANY_MODULE, self.codes):
            if " Art. " in ref:
                law = self._canonical_law(ref.split(" Art. ")[0])
                if law in self.laws:
                    add(f"{law} Art. {ref.split(' Art. ')[1]}")
            elif self._is_module(ref):
                add(ref)
        return found

    def _is_module(self, ref: str) -> bool:
        """A Rulebook citation: a known module code (in the corpus or not, so "PIB
        1.2.1" is reported as not in the corpus) or a bare rule number. refs.py also
        reads "Chapter 7 of the Rules" as a reference into a document named "Rules";
        that is prose, not a provision, so it is dropped here."""
        m = RULE_NUMBER.match(ref)
        mod = m.group("mod") if m else ref.split(" App")[0] if " App" in ref else None
        return mod is not None and (mod == ANY_MODULE or mod in self.codes | KNOWN_MODULES)

    def _canonical_law(self, name: str) -> str:
        return next((law for law in self.laws if law.lower() == name.lower()), name)

    # --- lookup --------------------------------------------------------------

    def lookup(
        self, question: str, filters: Filters | None = None
    ) -> tuple[CitationLookup, list[Candidate]]:
        filters = filters or Filters()
        result = CitationLookup(found=self.find(question))
        if not result.found:
            return result, []

        keys_by_citation: dict[str, list[str]] = {}
        ids: dict[str, str] = {}  # key -> point id
        with session_scope(self.engine) as s:
            repo = ChunkRepository(s)
            for citation in result.found:
                rows = self._resolve(repo, citation)
                if rows:
                    keys_by_citation[citation] = [r.key for r in rows]
                    ids |= {r.key: r.id for r in rows}
                elif _broad(citation):
                    result.too_broad.append(citation)
                else:
                    result.unresolved.append(citation)

        records = {str(r.id): r for r in self.store.retrieve(list(ids.values()))} if ids else {}
        candidates: list[Candidate] = []
        seen: set[str] = set()
        for citation, keys in keys_by_citation.items():
            kept = []
            for key in keys:
                record = records.get(ids[key])
                if record is None or ids[key] in seen:
                    continue  # not indexed (yet), or already found via an earlier citation
                if not filters.accepts(record.payload):
                    result.filtered_out.append(key)
                    continue
                seen.add(ids[key])
                kept.append(key)
                candidates.append(Candidate(ids[key], record.payload, source="citation"))
            if kept:
                result.resolved[citation] = kept
        return result, candidates[: self.max_hits]

    def _resolve(self, repo: ChunkRepository, citation: str) -> list:
        """The embedded chunks a citation stands for, in document order."""
        mod, _, num = citation.partition(" ")
        targets = (
            [f"{code} {num}" for code in sorted(self.codes)] if mod == ANY_MODULE else [citation]
        )
        rows = []
        for target in targets:
            section = repo.by_key(target)
            if section is not None and section.chunk_type == "section":
                # "COB 3.3": every child, and the parts of split children.
                children = repo.children_of(target)
                rows += children + [p for c in children if c.parts for p in repo.parts_of(c.key)]
            else:
                # A rule (its parts and its Guidance share its citation), an article, a
                # definition; or an appendix of free text ("MKT App4" is stored "MKT APP4").
                rows += repo.by_citations([target, target.replace(" App", " APP")])
        return _embeddable(rows)


def _broad(citation: str) -> bool:
    """A chapter or an appendix ("COB 3", "COB App2"): real, but too broad to look up."""
    num = citation.partition(" ")[2]
    return "App" in num or (" Art. " not in citation and "." not in num)


def _embeddable(rows: list) -> list:
    rows = [r for r in rows if r.chunk_type in EMBEDDED_TYPES and not r.deleted and not r.parts]
    return sorted(rows, key=lambda r: (r.document_id, r.order_idx))
