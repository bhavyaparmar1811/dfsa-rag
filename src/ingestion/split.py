"""Split chunks that are too long to embed and rerank well.

The budget is `max_chunk_tokens` for the whole embedding input (header +
text), since that is what the embedder and the reranker read. A chunk over it
is cut at the most meaningful boundary available, coarsest first:

    1. top-level sub-rules        (1) (2) (3)
    2. items, Guidance paragraphs, headings inside a Guidance block
                                  (a) (i) / 1. 2. / a. b. / "Risk based approach"
    3. lines (e.g. rows of a table)
    4. sentences
    5. words (last resort, logged)

then consecutive pieces are packed back together up to the budget, so parts
are as few and as large as the budget allows. Never mid-sentence unless a
single sentence exceeds the budget.

The whole chunk is kept (not embedded, used for LLM context) with `parts`
listing its parts; each part keeps the citation and points back via
`part_of`. A part that starts inside a Markdown table repeats the table's
header row, so its columns stay readable.
"""

import logging
import re
from collections.abc import Callable

from src.ingestion.header import build_header
from src.ingestion.ids import chunk_id
from src.ingestion.models import Chunk, embedding_text

logger = logging.getLogger(__name__)

SUB_RULE = re.compile(r"^\(\d+\)\s")  # "(1) ", "(2) "
PARAGRAPH = re.compile(r"^\d+\.\s")  # Guidance paragraphs: "1. ", "12. "
SUB_ITEM = re.compile(r"^(\([a-z]{1,5}\)|[a-z]\.)\s")  # "(a) ", "(ii) ", "a. "
ITEM = re.compile(r"^(\([a-z]{1,5}\)|\d+\.|[a-z]\.)\s")  # a paragraph or a sub-item
TABLE_ROW = re.compile(r"^\|")
TABLE_RULE = re.compile(r"^\|[\s\-:|]+\|$")  # the |---|---| line under a header row
SENTENCE_END = re.compile(r"(?<=[.;:])\s+(?=[A-Z(])")


def is_heading_line(line: str) -> bool:
    """A short line inside a Guidance block that titles what follows it."""
    return (
        len(line) <= 80
        and not line.endswith((".", ";", ":", ","))
        and not ITEM.match(line)
        and not SUB_RULE.match(line)
        and not TABLE_ROW.match(line)
    )


# Coarsest first. Guidance paragraphs ("1.") and headings come before their
# sub-items ("a.", "(i)"), so a list stays with the paragraph that introduces it.
LEVELS: list[Callable[[str], bool]] = [
    lambda line: bool(SUB_RULE.match(line)),
    lambda line: bool(PARAGRAPH.match(line)) or is_heading_line(line),
    lambda line: bool(SUB_ITEM.match(line)),
    lambda line: True,
]


class Splitter:
    def __init__(self, count_tokens: Callable[[str], int], max_tokens: int):
        self.count = count_tokens
        self.max_tokens = max_tokens

    def split_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        out: list[Chunk] = []
        for chunk in chunks:
            if not chunk.embedded or chunk.token_count <= self.max_tokens:
                out.append(chunk)
                continue
            out.extend(self._split(chunk))
        return out

    # ---- one chunk ----------------------------------------------------------------

    def _split(self, chunk: Chunk) -> list[Chunk]:
        # Reserve room for the longest header a part will carry.
        widest = build_header(chunk.heading_path, chunk.chunk_type, chunk.citation, 99, 99)
        budget = self.max_tokens - self.count(embedding_text(widest, ""))
        texts = ["\n".join(lines) for lines in self.split_lines(chunk.text.split("\n"), budget)]
        if len(texts) < 2:
            logger.warning("%s: %d tokens but no split point found", chunk.key, chunk.token_count)
            return [chunk]

        keys = [f"{chunk.key}#p{i}" for i in range(1, len(texts) + 1)]
        parts = []
        for i, (key, text) in enumerate(zip(keys, texts, strict=True), 1):
            header = build_header(
                chunk.heading_path, chunk.chunk_type, chunk.citation, i, len(texts)
            )
            parts.append(
                chunk.model_copy(
                    update={
                        "chunk_id": chunk_id(chunk.doc_id, chunk.version, key),
                        "key": key,
                        "header_text": header,
                        "text": text,
                        "token_count": self.count(embedding_text(header, text)),
                        "part": i,
                        "part_of": chunk.key,
                    }
                )
            )
        return [chunk.model_copy(update={"parts": keys}), *parts]

    def split_lines(self, lines: list[str], budget: int) -> list[list[str]]:
        return self._pack(self._pieces(lines, budget, level=0), budget)

    def _pieces(self, lines: list[str], budget: int, level: int) -> list[list[str]]:
        if self._tokens(lines) <= budget:
            return [lines]
        if level >= len(LEVELS):
            return self._sentences(lines, budget)
        blocks = _blocks(lines, LEVELS[level])
        if len(blocks) == 1:
            return self._pieces(lines, budget, level + 1)
        out: list[list[str]] = []
        for block in blocks:
            out.extend(self._pieces(block, budget, level + 1))
        return out

    def _sentences(self, lines: list[str], budget: int) -> list[list[str]]:
        sentences = [s for line in lines for s in SENTENCE_END.split(line) if s]
        out: list[list[str]] = []
        for sentence in sentences:
            if self._tokens([sentence]) <= budget:
                out.append([sentence])
            else:
                out.extend([w] for w in self._words(sentence, budget))
        return out

    def _words(self, sentence: str, budget: int) -> list[str]:
        logger.warning("Splitting a single %d-token sentence by words", self.count(sentence))
        words, out, current = sentence.split(" "), [], []
        for word in words:
            if current and self.count(" ".join([*current, word])) > budget:
                out.append(" ".join(current))
                current = []
            current.append(word)
        if current:
            out.append(" ".join(current))
        return out

    def _pack(self, pieces: list[list[str]], budget: int) -> list[list[str]]:
        """Join consecutive pieces up to the budget. A part that starts mid-table
        begins with the table's header rows, counted inside the budget."""
        packed: list[list[str]] = []
        tables = _TableTracker()
        for piece in pieces:
            if packed and self._tokens(packed[-1] + piece) <= budget:
                packed[-1] = packed[-1] + piece
            else:
                start = list(piece)
                header = tables.header
                if header and TABLE_ROW.match(start[0]) and start[:2] != header:
                    with_header = header + start
                    if self._tokens(with_header) <= budget:
                        start = with_header
                packed.append(start)
            tables.feed(piece)
        return packed

    def _tokens(self, lines: list[str]) -> int:
        return self.count("\n".join(lines))


def _blocks(lines: list[str], starts_block: Callable[[str], bool]) -> list[list[str]]:
    """Group lines into blocks, each starting at a line where `starts_block` is true.

    A block holding only heading lines is never closed, so a heading stays
    with the paragraph it introduces.
    """
    blocks: list[list[str]] = []
    for line in lines:
        if blocks and starts_block(line) and not all(is_heading_line(x) for x in blocks[-1]):
            blocks.append([line])
        elif blocks:
            blocks[-1].append(line)
        else:
            blocks.append([line])
    return blocks


class _TableTracker:
    """Follows the line stream and remembers the header rows of the open table.

    Lines arrive in pieces that may separate a header row from its |---| line,
    so the state carries across pieces.
    """

    def __init__(self) -> None:
        self.header: list[str] | None = None
        self._previous: str | None = None

    def feed(self, lines: list[str]) -> None:
        for line in lines:
            if not TABLE_ROW.match(line):
                self.header = None
            elif TABLE_RULE.match(line) and self._previous and TABLE_ROW.match(self._previous):
                self.header = [self._previous, line]
            self._previous = line
