from src.ingestion.models import Chunk, ChunkType
from src.ingestion.split import Splitter


def words(text: str) -> int:
    """Stand-in tokenizer: one token per word, so budgets are easy to reason about."""
    return len(text.split())


def chunk(text: str, kind=ChunkType.RULE, key="COB 3.3.2") -> Chunk:
    header = f"Section 3.3 > Rule {key}"
    return Chunk(
        chunk_id="x",
        key=key,
        citation="COB 3.3.2",
        chunk_type=kind,
        binding=kind == ChunkType.RULE,
        doc_id="dfsa-rulebook-cob",
        regulator="DFSA",
        module="COB",
        version="51",
        effective_date="2026-07",
        source_file="cob.pdf",
        rule_number="3.3.2",
        section="3.3",
        order=0,
        parent_key="COB 3.3",
        heading_path=["Section 3.3"],
        header_text=header,
        text=text,
        page_start=1,
        page_end=1,
        token_count=words(f"{header}\n\n{text}"),
    )


def split(c: Chunk, max_tokens: int) -> list[Chunk]:
    return Splitter(words, max_tokens).split_chunks([c])


def test_short_chunk_is_untouched():
    c = chunk("3.3.2 A firm must comply.")
    assert split(c, 100) == [c]


def test_splits_at_top_level_sub_rules_and_keeps_items_with_them():
    text = "\n".join(
        [
            "3.3.2 (1) Subject to (2), a firm must not act unless:",
            "(a) there is a Client Agreement; and",
            "(b) it has given the key information.",
            "(2) A firm may act without a Client Agreement where impracticable.",
            "(3) When (2) applies, the firm must explain why.",
        ]
    )
    out = split(chunk(text), 40)  # budget for text: 40 - 10 header words = 30
    whole, parts = out[0], out[1:]

    assert not whole.embedded and whole.parts == [p.key for p in parts]
    assert parts[0].text.startswith("3.3.2 (1)") and "(b) it has given" in parts[0].text
    assert parts[1].text.startswith("(2)")
    assert all(p.part_of == "COB 3.3.2" and p.citation == "COB 3.3.2" for p in parts)
    assert all(p.embedded and p.token_count <= 40 for p in parts)
    assert parts[0].header_text.endswith("(part 1 of 2)")
    assert len({p.chunk_id for p in parts}) == len(parts)


def test_guidance_list_stays_with_its_paragraph():
    text = "\n".join(
        [
            "1. A firm must classify each Person as:",
            "a. a Retail Client;",
            "b. a Professional Client; or",
            "c. a Market Counterparty.",
            "Risk based approach",
            "2. The Rules reflect the DFSA's risk based approach to regulation.",
        ]
    )
    out = split(chunk(text, ChunkType.GUIDANCE, "COB 2.2 Guidance"), 36)
    parts = [c for c in out if c.part]
    assert "c. a Market Counterparty." in parts[0].text  # not cut off from a. and b.
    assert parts[1].text.startswith("Risk based approach\n2.")  # heading kept with paragraph


def test_table_continuation_repeats_the_header_row():
    rows = [f"| {i}.1 | item {i} | yes |" for i in range(1, 9)]
    text = "\n".join(["| No | Item | Shares |", "|---|---|---|", *rows])
    parts = [c for c in split(chunk(text), 40) if c.part]
    assert len(parts) > 1
    assert all(p.text.startswith("| No | Item | Shares |\n|---|---|---|") for p in parts)


def test_single_long_sentence_falls_back_to_words():
    text = "3.3.2 " + " ".join(["word"] * 60) + "."
    parts = [c for c in split(chunk(text), 30) if c.part]
    assert len(parts) > 1 and all(p.token_count <= 30 for p in parts)
