"""Cross-references between provisions, extracted from chunk text.

DFSA text refers to other provisions in several forms:

    GEN Rule 2.26.1            another module's rule        -> "GEN 2.26.1"
    AML Rule 7.3.1(1)(b)       ... with a sub-paragraph     -> "AML 7.3.1"
    GEN section 2.11           another module's section     -> "GEN 2.11"
    GEN 2.22.1                 module + number, no keyword  -> "GEN 2.22.1"
    Rule 3.3.4                 this module                  -> "COB 3.3.4"
    Rules 3.3.2 and 3.3.3      a list                       -> "COB 3.3.2", "COB 3.3.3"
    section 15.5 / chapter 7   this module                  -> "COB 15.5", "COB 7"
    App 2 / App2               this module's appendix       -> "COB App2"
    Article 76 of the Regulatory Law                        -> "Regulatory Law Art. 76"

References are normalised to citation keys, so retrieval can later follow
them ("in accordance with Rule 3.3.4" -> fetch COB 3.3.4) and the validator
can check that in-module references resolve to a real chunk.
"""

import re

from src.ingestion.models import Chunk

# DFSA Rulebook module codes. Those outside the corpus are still recognised so
# their references are recorded (and reported as external), and so acronyms
# that aren't modules (e.g. "CDD 7.1.1") are never mistaken for one.
KNOWN_MODULES = {
    "GEN", "COB", "AML", "MKT", "FPR", "GLO", "PIB", "PIN", "IFR", "CIR", "AMI",
    "AUD", "REC", "ISF", "DEP", "CRF", "SRI", "QFP", "PRU",
}  # fmt: skip

NUM = r"A?\d+[A-Z]?(?:\.\d+[A-Z]?){0,2}"
KIND = r"(?:Rules?|[Ss]ections?|[Cc]hapters?)"
# "3.3.2", "3.3.2(1)(b)", then optionally ", 3.3.3", " and 3.3.4", " to 3.3.6".
NUM_LIST = rf"(?P<nums>{NUM}(?:\(\w+\))*(?:\s*(?:,|and|or|to)\s*{NUM}(?:\(\w+\))*)*)"
SINGLE_NUM = re.compile(rf"{NUM}")
SUB_PARAGRAPH = re.compile(r"\(\w+\)")  # "(1)", "(b)": dropped, refs point at the rule
ARTICLE = re.compile(
    r"\bArticles?\s+(?P<num>\d+[A-Z]?)(?:\(\w+\))*\s+of\s+the\s+(?P<law>(?:[A-Z][\w-]*\s+)+Law)\b"
)
APPENDIX = re.compile(r"\bApp(?:endix)?\s?(?P<num>\d+)\b")


# Capitalised words that can precede "Rule"/"section" but aren't module codes.
NOT_MODULES = {"DFSA", "DIFC", "UAE", "FATF"}


def _module_ref(codes: set[str]) -> re.Pattern:
    """A known code with or without a keyword ("GEN 2.22.1", "GEN Rule 2.26.1"), or
    any 3-4 capital letters followed by a keyword ("REP Rule 4.10.1"), which
    covers modules outside the corpus and outside KNOWN_MODULES."""
    known = "|".join(sorted(codes))
    excluded = "|".join(sorted(NOT_MODULES))
    return re.compile(
        rf"\b(?:(?P<mod>{known})\s+(?:{KIND}\s+)?"
        rf"|(?!(?:{excluded})\b)(?P<anymod>[A-Z]{{3,4}})\s+{KIND}\s+){NUM_LIST}"
    )


LOCAL_REF = re.compile(rf"\b{KIND}\s+{NUM_LIST}")


def extract_refs(text: str, module: str, codes: set[str] | None = None) -> list[str]:
    """Normalised references found in `text`, in order of first appearance."""
    codes = (codes or set()) | KNOWN_MODULES
    refs: list[str] = []
    taken: list[tuple[int, int]] = []

    def add(ref: str) -> None:
        if ref not in refs:
            refs.append(ref)

    for m in _module_ref(codes).finditer(text):
        taken.append(m.span())
        mod = m.group("mod") or m.group("anymod")
        for num in _numbers(m.group("nums")):
            add(f"{mod} {num}")

    for m in LOCAL_REF.finditer(text):
        if any(s <= m.start() < e for s, e in taken):
            continue  # already read as "GEN Rule ..."
        for num in _numbers(m.group("nums")):
            add(f"{module} {num}")

    for m in APPENDIX.finditer(text):
        add(f"{module} App{m.group('num')}")

    for m in ARTICLE.finditer(text):
        add(f"{m.group('law').strip()} Art. {m.group('num')}")

    return refs


def _numbers(nums: str) -> list[str]:
    return SINGLE_NUM.findall(SUB_PARAGRAPH.sub("", nums))


def annotate(chunks: list[Chunk], codes: set[str]) -> list[Chunk]:
    """Set `cross_refs` on every child chunk (not on section parents)."""
    out = []
    for c in chunks:
        if c.chunk_type == "section":
            out.append(c)
            continue
        own = {c.citation, f"{c.module} {c.section}"}
        refs = [r for r in extract_refs(c.text, c.module, codes) if r not in own]
        out.append(c.model_copy(update={"cross_refs": refs}))
    return out
