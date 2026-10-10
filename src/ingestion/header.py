"""Contextual headers, prepended to each chunk's text before embedding.

A rule on its own often doesn't say what it's about ("The Rules in this
section do not apply to ..."). The header carries the path down to it, so the
dense vector knows the topic and BM25 can match the module code and number:

    DFSA Rulebook > Conduct of Business Module (COB) > Chapter 3: CORE RULES - ...
    > Section 3.3: Key information and Client Agreement > Application > Rule COB 3.3.1
"""

from src.ingestion.models import ChunkType


def build_header(heading_path: list[str], chunk_type: ChunkType, citation: str) -> str:
    parts = list(heading_path)
    if chunk_type == ChunkType.RULE:
        parts.append(f"Rule {citation}")
    elif chunk_type == ChunkType.GUIDANCE:
        parts.append(f"Guidance on {citation} (non-binding)")
    return " > ".join(parts)
