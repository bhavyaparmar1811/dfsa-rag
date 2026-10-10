"""The `Chunk`: the unit of retrieval and of citation.

Children (rules, Guidance, definitions, articles) are embedded and searched.
Parents (sections) are stored as text only and sent to the LLM for context
after retrieval. Links between chunks use `key`, the human-readable citation
("COB 3.3.2"), which is unique within a document version; `chunk_id` is a
deterministic uuid derived from it, used as the Qdrant point id.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class ChunkType(StrEnum):
    RULE = "rule"
    GUIDANCE = "guidance"
    SECTION = "section"  # parent: stored, never embedded


EMBEDDED_TYPES = {ChunkType.RULE, ChunkType.GUIDANCE}


class Chunk(BaseModel):
    chunk_id: str
    key: str  # unique within a document version: "COB 3.3.2", "COB 3.3.2 Guidance", "COB 3.3"
    citation: str  # what an answer cites: "COB 3.3.2" (Guidance cites the rule it explains)
    chunk_type: ChunkType
    binding: bool  # Rules bind; Guidance does not

    doc_id: str
    regulator: str
    module: str
    version: str
    effective_date: str
    source_file: str

    rule_number: str | None = None  # "3.3.2", "A2.1.1"; None for sections and section Guidance
    section: str  # "3.3", "A2.1"
    order: int  # position in the document, for reassembling sections in order

    parent_key: str | None = None  # the section this child belongs to
    guidance_for: str | None = None  # Guidance -> the rule (or section) it explains
    guidance_keys: list[str] = Field(default_factory=list)  # rule -> its Guidance

    heading_path: list[str]  # module, chapter, section, sub-heading
    subheading: str | None = None
    # "The Rules in this section do not apply to ...": always sent along with its
    # section, whatever the parent-size policy, because exemptions change answers.
    scope_rule: bool = False
    header_text: str  # contextual header, prepended to `text` at embedding time
    text: str

    deleted: bool = False
    page_start: int
    page_end: int
    token_count: int

    @property
    def embedded(self) -> bool:
        return self.chunk_type in EMBEDDED_TYPES and not self.deleted
