import pytest

from src.retrieval.context import ContextAssembler
from src.retrieval.models import Candidate
from src.storage.db import session_scope
from src.storage.repositories import ChunkRepository
from tests.retrieval.conftest import settings


def hits(engine, *keys, scores=None):
    """Candidates for these chunk keys, best first."""
    with session_scope(engine) as s:
        rows = {r.key: r for r in ChunkRepository(s).by_keys(keys)}
    scores = scores or [1.0 - i / 10 for i in range(len(keys))]
    return [
        Candidate(rows[k].id, {"key": k, "citation": rows[k].citation}, "search",
                  rerank_score=score)
        for k, score in zip(keys, scores, strict=True)
    ]  # fmt: skip


def assemble(engine, question, keys, **cfg):
    assembler = ContextAssembler(engine, settings(**cfg))
    blocks, dropped = assembler.assemble(question, hits(engine, *keys))
    return blocks, dropped


def reasons(blocks):
    return {b.key: b.reason for b in blocks}


@pytest.fixture
def engine(loaded):
    return loaded[0]


def test_a_hit_brings_its_scope_rule_and_guidance_and_small_section(engine):
    blocks, dropped = assemble(engine, "client agreement", ["COB 3.3.2"], include_definitions=False)
    assert reasons(blocks) == {
        "COB 3.3.1": "scope", "COB 3.3.2": "hit", "COB 3.3.2 Guidance": "guidance",
        "COB 3.3.3": "section", "COB 3.3.4": "section",
    }  # fmt: skip
    assert [b.key for b in blocks] == [
        "COB 3.3.1", "COB 3.3.2", "COB 3.3.2 Guidance", "COB 3.3.3", "COB 3.3.4",
    ]  # document order within the section  # fmt: skip
    assert [b.n for b in blocks] == [1, 2, 3, 4, 5] and dropped == []
    by_key = {b.key: b for b in blocks}
    assert by_key["COB 3.3.2"].label == "Rule (binding)" and by_key["COB 3.3.2"].score == 1.0
    assert by_key["COB 3.3.2 Guidance"].label == "Guidance (non-binding)"
    assert by_key["COB 3.3.2"].group == "COB 3.3"


def test_parent_mode_none_keeps_scope_rules_but_not_the_section(engine):
    blocks, _ = assemble(engine, "q", ["COB 3.3.3"], parent_mode="none", include_definitions=False)
    assert reasons(blocks) == {"COB 3.3.1": "scope", "COB 3.3.3": "hit"}


def test_scope_rules_can_be_switched_off(engine):
    blocks, _ = assemble(engine, "q", ["COB 3.3.3"], parent_mode="none",
                         include_scope_rules=False, include_definitions=False)  # fmt: skip
    assert reasons(blocks) == {"COB 3.3.3": "hit"}


def test_a_big_section_gets_a_window(engine):
    blocks, _ = assemble(engine, "q", ["COB 3.3.3"], parent_max_tokens=20, parent_window=1,
                         include_definitions=False)  # fmt: skip
    # 3.3.3's neighbours are the Guidance block and 3.3.4; 3.3.1 is there as the scope rule.
    assert reasons(blocks) == {
        "COB 3.3.1": "scope", "COB 3.3.2 Guidance": "window", "COB 3.3.3": "hit",
        "COB 3.3.4": "window",
    }  # fmt: skip


def test_many_hits_in_one_section_bring_the_whole_section(engine):
    keys = ["COB 3.3.2", "COB 3.3.3", "COB 3.3.4#p1"]
    blocks, _ = assemble(engine, "q", keys, parent_max_tokens=20, section_hit_min=3,
                         include_definitions=False)  # fmt: skip
    assert set(reasons(blocks)) == {
        "COB 3.3.1", "COB 3.3.2", "COB 3.3.2 Guidance", "COB 3.3.3", "COB 3.3.4",
    }  # fmt: skip


def test_a_part_is_shown_as_its_whole_rule(engine):
    blocks, _ = assemble(engine, "q", ["COB 3.3.4#p2"], parent_mode="none",
                         include_definitions=False)  # fmt: skip
    rule = next(b for b in blocks if b.key == "COB 3.3.4")
    assert rule.reason == "part" and "(2) records" in rule.text and "(1) reliance" in rule.text
    assert not any("#p" in b.key for b in blocks)


def test_a_guidance_hit_brings_its_rule(engine):
    blocks, _ = assemble(engine, "q", ["COB 3.3.2 Guidance"], parent_mode="none",
                         include_definitions=False)  # fmt: skip
    assert reasons(blocks)["COB 3.3.2"] == "rule"


def test_definitions_of_question_terms_come_first_and_lower_case_matches(engine):
    blocks, _ = assemble(engine, "is a market counterparty exempt?", ["COB 3.4.1"],
                         parent_mode="none", max_definitions=2)  # fmt: skip
    defs = [b for b in blocks if b.group == "definitions"]
    assert defs[0].key == "GLO Market Counterparty" and defs[0].reason == "question_term"
    assert defs[1].key == "GLO Authorised Firm" and defs[1].reason == "definition"
    assert defs[0].label == "Definition (binding)"
    assert blocks[-1].group == "definitions"  # after the sections


def test_common_and_skipped_terms_are_not_defined(engine):
    blocks, _ = assemble(engine, "q", ["COB 3.3.2"], parent_mode="none", include_guidance=False,
                         term_max_share=0.0, skip_terms=["Client Agreement"])  # fmt: skip
    assert [b for b in blocks if b.group == "definitions"] == []


def test_a_definition_hit_is_not_repeated_as_a_definition(engine):
    blocks, _ = assemble(engine, "what is a Market Counterparty", ["GLO Market Counterparty"])
    assert [b.key for b in blocks].count("GLO Market Counterparty") == 1


def test_references_are_followed_one_hop(engine):
    blocks, _ = assemble(engine, "q", ["COB 3.4.1"], parent_mode="none",
                         include_definitions=False, expand_refs=True)  # fmt: skip
    assert reasons(blocks) == {"COB 3.4.1": "hit", "COB 3.3.2": "ref"}
    assert blocks[-1].group == "references"


def test_the_budget_drops_the_least_important_blocks_never_the_hits(engine):
    # Every chunk is 10 tokens in the fixture: room for the hit + 2 more.
    blocks, dropped = assemble(engine, "client agreement", ["COB 3.3.2"],
                               context_budget_tokens=30)  # fmt: skip
    assert reasons(blocks) == {"COB 3.3.2": "hit", "COB 3.3.1": "scope",
                               "COB 3.3.2 Guidance": "guidance"}  # fmt: skip
    assert "COB 3.3.3" in dropped and "GLO Client Agreement" in dropped
    blocks, _ = assemble(engine, "q", ["COB 3.3.2", "COB 3.3.3"], context_budget_tokens=0)
    assert {b.key for b in blocks} == {"COB 3.3.2", "COB 3.3.3"}


def test_sections_are_ordered_by_their_best_hit(engine):
    blocks, _ = assemble(engine, "q", ["COB 3.4.1", "COB 3.3.3"], parent_mode="none",
                         include_definitions=False)  # fmt: skip
    assert [b.group for b in blocks] == ["COB 3.4", "COB 3.3", "COB 3.3"]
