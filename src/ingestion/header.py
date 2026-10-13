"""Contextual headers, prepended to each chunk's text before embedding.

A rule on its own often doesn't say what it's about ("The Rules in this
section do not apply to ..."). The header carries the path down to it, so the
dense vector knows the topic and BM25 can match the module code and number:

    DFSA Rulebook > Conduct of Business Module (COB) > Chapter 3: CORE RULES - ...
    > Section 3.3: Key information and Client Agreement > Application > Rule COB 3.3.1
"""

from src.ingestion.models import ChunkType

# Some titles are long lists (COB chapter 3's is ~330 characters). Repeated in
# every chunk of the chapter, the full title would dilute each vector and eat
# into the 512-token budget, so headers keep only its start. The full title
# stays in `heading_path`.
MAX_TITLE_CHARS = 90


def shorten(title: str, limit: int = MAX_TITLE_CHARS) -> str:
    if len(title) <= limit:
        return title
    cut = title[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return f"{cut} ..."


def build_header(
    heading_path: list[str],
    chunk_type: ChunkType,
    citation: str,
    part: int | None = None,
    total_parts: int | None = None,
) -> str:
    parts = [shorten(p) for p in heading_path]
    if chunk_type == ChunkType.RULE:
        parts.append(f"Rule {citation}")
    elif chunk_type == ChunkType.GUIDANCE:
        parts.append(f"Guidance on {citation} (non-binding)")
    header = " > ".join(parts)
    if part is not None:
        header += f" (part {part} of {total_parts})"
    return header
