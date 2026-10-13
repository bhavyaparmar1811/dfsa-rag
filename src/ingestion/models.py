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
    # Text that is neither a numbered Rule nor labelled Guidance, e.g. an
    # appendix of best-practice standards. Embedded, but never presented as binding.
    TEXT = "text"
    # One Glossary (GLO) entry. Binding: GLO 1.2.1 gives a defined term the
    # meaning set out in the Glossary.
    DEFINITION = "definition"
    SECTION = "section"  # parent: stored, never embedded


EMBEDDED_TYPES = {ChunkType.RULE, ChunkType.GUIDANCE, ChunkType.TEXT, ChunkType.DEFINITION}


class Chunk(BaseModel):
    chunk_id: str
    key: str  # unique within a document version: "COB 3.3.2", "COB 3.3.2 Guidance", "COB 3.3"
    citation: str  # what an answer cites: "COB 3.3.2" (Guidance cites the rule it explains)
    chunk_type: ChunkType
    binding: bool  # Rules and Glossary definitions bind; Guidance and unnumbered text don't

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
    token_count: int  # tokens of embedding_text(): what the embedder and reranker see

    # A chunk over max_chunk_tokens is split into parts. The whole chunk is kept
    # (for LLM context, not embedded) and lists its parts; each part points back.
    parts: list[str] = Field(default_factory=list)  # whole -> ["COB 3.3.2#p1", ...]
    part_of: str | None = None  # part -> "COB 3.3.2"
    part: int | None = None  # 1-based part number

    cross_refs: list[str] = Field(default_factory=list)  # ["GEN 2.26.1", "COB 3.3.4", ...]
    # Glossary terms used in the text ("Retail Client", ...), so answers can be
    # given with the regulator's own definitions.
    defined_terms: list[str] = Field(default_factory=list)
    term: str | None = None  # the term a `definition` chunk defines

    @property
    def embedded(self) -> bool:
        return self.chunk_type in EMBEDDED_TYPES and not self.deleted and not self.parts

    def embedding_text(self) -> str:
        return embedding_text(self.header_text, self.text)


def embedding_text(header: str, text: str) -> str:
    """The exact string sent to the embedding model (and the reranker).

    Defined once here so token counting, splitting, embedding and reranking
    all agree on it.
    """
    return f"{header}\n\n{text}"
