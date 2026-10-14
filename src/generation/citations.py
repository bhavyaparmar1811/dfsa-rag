"""The citation validator: nothing reaches the user unless every provision it
names is one the model was given and listed.

    1. the answer lists at least one citation
    2. every listed context_id is a block in the context
    3. every listed citation is that block's citation
    4. every provision the answer text mentions ("COB 3.3.2(1)", "Rule 3.3.4",
       "section 3.3", "Article 20 of the Regulatory Law") is covered by a
       listed block, or is referred to in a listed block's own text ("the
       information specified in App2" may be relayed as such); mentions are
       parsed with the same router that reads citations in questions

So no provision number in an answer ever comes from the model itself: each one
is a provision it was given and listed, or a reference quoted from one.

Any violation means refusal, never a patched answer. An answer whose listed
blocks are all non-binding is not a violation: it is flagged `guidance_only`
and the answerer labels it.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from src.generation.context import cite_as
from src.retrieval.citations import ANY_MODULE
from src.retrieval.models import ContextBlock

FindCitations = Callable[[str], list[str]]
LAW_ARTICLE = " Art. "


class DraftCitation(BaseModel):
    context_id: int
    citation: str


class AnswerDraft(BaseModel):
    """The JSON the answer prompt asks for."""

    answer: str
    citations: list[DraftCitation] = Field(default_factory=list)
    confident: bool


@dataclass
class Validation:
    violations: list[str] = field(default_factory=list)
    cited: list[ContextBlock] = field(default_factory=list)  # in the order listed, once each
    mentions: list[str] = field(default_factory=list)  # provisions named in the answer text
    relayed: list[str] = field(default_factory=list)  # of those, references a listed block makes
    guidance_only: bool = False

    @property
    def ok(self) -> bool:
        return not self.violations


def _split(citation: str) -> tuple[str, str]:
    """ "COB 3.3.2" -> ("COB", "3.3.2");
    "Regulatory Law Art. 20" -> ("Regulatory Law", "Art. 20")."""
    if LAW_ARTICLE in citation:
        law, num = citation.split(LAW_ARTICLE, 1)
        return law, f"Art. {num}"
    module, _, num = citation.partition(" ")
    return module, num


def covers(mention: str, citation: str) -> bool:
    """Does a listed block's citation cover a provision the answer mentions?

    "COB 3.3.2" covers "COB 3.3.2"; a section "COB 3.3" and a chapter "COB 3" are
    covered by any block inside them; "COB App2" by "COB APP2" or "COB A2.x.y";
    "* 3.3.2" (a Rule number without a module) by that number in any module.
    """
    m_mod, m_num = _split(mention)
    c_mod, c_num = _split(citation)
    if m_mod == ANY_MODULE:
        if c_num.startswith("Art. "):
            return False
    elif m_mod.lower() != c_mod.lower():
        return False
    if m_num.lower().startswith("app"):
        app = m_num[3:]
        return c_num.upper() == f"APP{app}" or c_num.startswith(f"A{app}.")
    return c_num == m_num or c_num.startswith(m_num + ".")


def references(block: ContextBlock, find_citations: FindCitations) -> list[str]:
    """Provisions a block's text refers to, with "Rule 3.3.4" resolved to the
    block's own module ("COB 3.3.4")."""
    module = None if LAW_ARTICLE in block.citation else block.citation.split(" ")[0]
    prefix = f"{ANY_MODULE} "
    return [
        f"{module} {r.removeprefix(prefix)}" if module and r.startswith(prefix) else r
        for r in find_citations(block.text)
    ]


def check_citations(
    draft: AnswerDraft, blocks: list[ContextBlock], find_citations: FindCitations
) -> Validation:
    v = Validation()
    by_n = {b.n: b for b in blocks}
    if not draft.citations:
        v.violations.append("no citations listed")

    for c in draft.citations:
        block = by_n.get(c.context_id)
        if block is None:
            v.violations.append(f"[{c.context_id}] {c.citation}: no such context block")
            continue
        if c.citation.strip() not in (cite_as(block), block.citation):
            v.violations.append(
                f"[{c.context_id}] {c.citation!r} is not that block's citation {cite_as(block)!r}"
            )
            continue
        if block not in v.cited:
            v.cited.append(block)

    listed = [b.citation for b in v.cited]
    referenced = [r for b in v.cited for r in references(b, find_citations)]
    v.mentions = find_citations(draft.answer)
    for mention in v.mentions:
        if any(covers(mention, c) for c in listed):
            continue
        if any(mention == r or covers(mention, r) for r in referenced):
            v.relayed.append(mention)
            continue
        in_context = any(covers(mention, b.citation) for b in blocks)
        where = "in the context but not listed" if in_context else "not in the context"
        v.violations.append(f"{mention.replace(ANY_MODULE + ' ', '')}: mentioned, {where}")

    v.guidance_only = bool(v.cited) and not any(b.binding for b in v.cited)
    return v
