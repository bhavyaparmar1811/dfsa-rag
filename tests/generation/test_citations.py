import pytest

from src.generation.citations import AnswerDraft, check_citations, covers
from src.generation.context import cite_as, render_context
from tests.generation.conftest import BLOCKS, block, draft


def check(router, answer, *cites, blocks=BLOCKS):
    return check_citations(AnswerDraft(**draft(answer, *cites)), blocks, router.find)


@pytest.mark.parametrize(
    "mention, citation, ok",
    [
        ("COB 3.3.2", "COB 3.3.2", True),
        ("COB 3.3.2", "COB 3.3.20", False),
        ("COB 3.3", "COB 3.3.2", True),  # a section: any block inside it
        ("COB 3", "COB 3.3.2", True),  # a chapter
        ("GEN 3.3.2", "COB 3.3.2", False),
        ("* 3.3.2", "COB 3.3.2", True),  # "Rule 3.3.2" without a module
        ("* 20", "Regulatory Law Art. 20", False),
        ("COB App2", "COB A2.1.1", True),
        ("COB App2", "COB APP2", True),
        ("Regulatory Law Art. 20", "Regulatory Law Art. 20", True),
        ("Regulatory Law Art. 2", "Regulatory Law Art. 20", False),
        ("Markets Law Art. 20", "Regulatory Law Art. 20", False),
    ],
)
def test_coverage(mention, citation, ok):
    assert covers(mention, citation) is ok


def test_a_well_cited_answer_passes(router):
    v = check(
        router,
        "Yes (COB 3.3.2), unless the client is a Market Counterparty (COB 3.3.1). "
        "Under Rule 3.3.2(1) it should be in writing (COB 3.3.2 Guidance).",
        (2, "COB 3.3.2"), (1, "COB 3.3.1"), (3, "COB 3.3.2 Guidance"),
    )  # fmt: skip
    assert v.ok, v.violations
    assert [b.n for b in v.cited] == [2, 1, 3]
    assert not v.guidance_only  # a binding rule is among them


@pytest.mark.parametrize(
    "answer, cites, violation",
    [
        ("Yes.", [], "no citations listed"),
        ("Yes (COB 3.3.2).", [(9, "COB 3.3.2")], "no such context block"),
        ("Yes (COB 3.3.2).", [(1, "COB 3.3.2")], "is not that block's citation"),
        # A fabricated rule: named in the text, not in the context.
        ("Yes (COB 3.3.2), see also COB 9.9.9.", [(2, "COB 3.3.2")],
         "COB 9.9.9: mentioned, not in the context"),
        # In the context, but not listed: the citation list is the audit record.
        ("Yes (COB 3.3.2), except under COB 3.3.1.", [(2, "COB 3.3.2")],
         "COB 3.3.1: mentioned, in the context but not listed"),
        ("Article 21 of the Regulatory Law applies.", [(5, "Regulatory Law Art. 20")],
         "Regulatory Law Art. 21: mentioned, not in the context"),
    ],
)  # fmt: skip
def test_violations(router, answer, cites, violation):
    v = check(router, answer, *cites)
    assert not v.ok
    assert any(violation in x for x in v.violations), v.violations


def test_a_reference_made_by_a_cited_block_may_be_relayed(router):
    rule = block(2, "COB 3.3.2", text="3.3.2 The agreement must contain the information in "
                 "App2, subject to Rule 3.3.4.")  # fmt: skip
    ok = check(
        router, "It must contain the information in App2 (COB 3.3.2), subject to Rule 3.3.4.",
        (2, "COB 3.3.2"), blocks=[rule],
    )  # fmt: skip
    assert ok.ok, ok.violations
    assert set(ok.relayed) == {"* App2", "* 3.3.4"}

    # Resolved in the block's own module: GEN 3.3.4 is not what COB 3.3.2 refers to.
    wrong = check(router, "See GEN 3.3.4 (COB 3.3.2).", (2, "COB 3.3.2"), blocks=[rule])
    assert wrong.violations == ["GEN 3.3.4: mentioned, not in the context"]


def test_guidance_only_is_flagged(router):
    v = check(router, "It should be in writing (COB 3.3.2 Guidance).", (3, "COB 3.3.2 Guidance"))
    assert v.ok and v.guidance_only


def test_render_context_shows_number_citation_weight_and_path():
    text = render_context(BLOCKS[1:3])
    assert text.startswith("[2] COB 3.3.2 | Rule (binding)\nDFSA Rulebook > COB 3.3.2\n3.3.2 A")
    assert "\n\n[3] COB 3.3.2 Guidance | Guidance (non-binding)\n" in text
    assert cite_as(BLOCKS[2]) == "COB 3.3.2 Guidance"
