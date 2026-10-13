"""Tag chunks with the Glossary (GLO) terms they use.

DFSA marks a Defined Term by capitalising it ("a Retail Client", "the
Authorised Firm"), so matching is case-sensitive:

- longest term first, so "Advising on Financial Products" wins over "Advising";
- plurals and possessives count: "Retail Clients", "Client's";
- a term's aliases are matched too: "Designated Non-Financial Business or
  Profession (DNFBP)" also matches "DNFBP", and "Client Assets Crisis
  Preparedness Pack or Client Assets Pack" matches either name;
- a one-word term at the start of a sentence or list item is skipped, because
  there the capital letter says nothing ("Person" in "Person means ...").

Every term is tagged, however common. Which terms are worth adding to an
answer's context ("Person" appears almost everywhere) is decided at
generation time, not here, so ingestion never throws information away.
"""

import re
from collections import Counter

from src.ingestion.models import Chunk, ChunkType

ALIAS_IN_BRACKETS = re.compile(r"^(?P<name>.+?)\s*\((?P<alias>[A-Za-z][\w&-]{1,15})\)$")
SUFFIX = r"(?:s|es|'s|’s)?"
SENTENCE_START = re.compile(r"(^|[.;:]\s*|\n|\(\w{1,5}\)\s*|^\d+\.\s*)$")


def surface_forms(term: str) -> list[str]:
    """The ways a term can appear in text."""
    forms = [term]
    m = ALIAS_IN_BRACKETS.match(term)
    if m:
        forms += [m.group("name"), m.group("alias")]
    for form in [f for f in forms if "(" not in f]:
        sides = form.split(" or ")
        # Only split real alternatives, both at least two words:
        # "Client Assets Crisis Preparedness Pack or Client Assets Pack" splits,
        # "Designated Non-Financial Business or Profession" doesn't.
        if len(sides) == 2 and all(len(s.split()) >= 2 for s in sides):
            forms += sides
    return list(dict.fromkeys(forms))


class TermMatcher:
    def __init__(self, terms: list[str]):
        self.canonical: dict[str, str] = {}
        for term in terms:
            for form in surface_forms(term):
                self.canonical.setdefault(form, term)
        alternatives = "|".join(re.escape(f) for f in sorted(self.canonical, key=len, reverse=True))
        # A hyphen may precede a term ("non-Retail Client" uses "Retail Client").
        self.pattern = re.compile(rf"(?<!\w)(?:{alternatives}){SUFFIX}(?!\w)")

    def find(self, text: str) -> list[str]:
        """Canonical terms used in `text`, in order of first appearance."""
        found: list[str] = []
        for m in self.pattern.finditer(text):
            form = self._form(m.group(0))
            if form is None:
                continue
            if " " not in form and SENTENCE_START.search(text[: m.start()][-12:]):
                continue
            term = self.canonical[form]
            if term not in found:
                found.append(term)
        return found

    def _form(self, matched: str) -> str | None:
        if matched in self.canonical:
            return matched
        for suffix in ("'s", "’s", "es", "s"):
            if matched.endswith(suffix) and matched[: -len(suffix)] in self.canonical:
                return matched[: -len(suffix)]
        return None


def tag(chunks: list[Chunk], matcher: TermMatcher) -> list[Chunk]:
    """Set `defined_terms` on every child chunk (a definition doesn't list itself)."""
    out = []
    for c in chunks:
        if c.chunk_type == ChunkType.SECTION:
            out.append(c)
            continue
        terms = [t for t in matcher.find(c.text) if t != c.term]
        out.append(c.model_copy(update={"defined_terms": terms}))
    return out


def term_frequencies(chunks: list[Chunk]) -> Counter:
    return Counter(t for c in chunks if c.embedded for t in c.defined_terms)
