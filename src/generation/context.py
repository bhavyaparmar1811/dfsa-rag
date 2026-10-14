"""What the answer prompt's {{context}} looks like: one numbered block per chunk.

    [3] COB 3.3.2 | Rule (binding)
    DFSA Rulebook > Conduct of Business (COB) > ... > Rule COB 3.3.2
    3.3.2 An Authorised Firm must not ...

The model cites a block by its number and its display citation. A rule and its
Guidance share a citation ("COB 3.3.2"), so Guidance is shown as
"COB 3.3.2 Guidance": an inline "(COB 3.3.2)" then always means the binding
rule.
"""

from src.retrieval.models import ContextBlock


def cite_as(block: ContextBlock) -> str:
    """The citation the model must use for this block."""
    return f"{block.citation} Guidance" if block.chunk_type == "guidance" else block.citation


def render_block(block: ContextBlock) -> str:
    return f"[{block.n}] {cite_as(block)} | {block.label}\n{block.heading}\n{block.text.strip()}"


def render_context(blocks: list[ContextBlock]) -> str:
    return "\n\n".join(render_block(b) for b in blocks)
