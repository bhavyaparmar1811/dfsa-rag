import pytest
import yaml
from pydantic import ValidationError

from src.config.settings import PathSettings
from src.eval.golden import GoldenSet, check_against_corpus, coverage, load_golden, provision


def item(id="cob-q", kind="single", turns=None, **kw) -> dict:
    turns = turns or [{"question": "Is a Client Agreement needed?", "expect": "answer",
                       "gold": ["COB 3.3.2"]}]  # fmt: skip
    return {"id": id, "kind": kind, "turns": turns, **kw}


def golden(*items) -> GoldenSet:
    return GoldenSet.model_validate({"version": 1, "items": list(items) or [item()]})


def test_provision_strips_part_suffix():
    assert provision("GEN 2.3.2#p1") == "GEN 2.3.2"
    assert provision("COB 3.3.2 Guidance") == "COB 3.3.2 Guidance"
    assert provision("Regulatory Law Art. 8#p12") == "Regulatory Law Art. 8"


def test_duplicate_ids_rejected():
    with pytest.raises(ValidationError, match="duplicate"):
        golden(item(), item())


@pytest.mark.parametrize(
    "turn, message",
    [
        ({"question": "What is X?", "expect": "answer"}, "needs gold"),
        ({"question": "VAT rate?", "expect": "refuse", "gold": ["COB 3.3.2"]}, "no gold"),
        ({"question": "What is X?", "expect": "answer", "gold": ["A"], "supporting": ["A"]},
         "gold and supporting"),
    ],
)  # fmt: skip
def test_turn_rules(turn, message):
    with pytest.raises(ValidationError, match=message):
        golden(item(turns=[turn]))


def test_kind_rules():
    two = [{"question": "Q one?", "expect": "answer", "gold": ["COB 3.3.2"]}] * 2
    with pytest.raises(ValidationError, match="only multi_turn"):
        golden(item(turns=two))
    with pytest.raises(ValidationError, match="at least 2 turns"):
        golden(item(kind="multi_turn"))
    answerable = [{"question": "Q one?", "expect": "answer", "gold": ["COB 3.3.2"]}]
    with pytest.raises(ValidationError, match="must expect a refusal"):
        golden(item(kind="unanswerable", turns=answerable))


def test_hash_ignores_notes_but_not_review_state():
    base = golden(item())
    noted = item()
    noted["turns"][0]["notes"] = "a comment"
    assert golden(noted).hash == base.hash
    assert golden(item(reviewed=True)).hash != base.hash


def test_select_reviewed_and_ids():
    g = golden(item("a", reviewed=True), item("b"))
    assert [i.id for i in g.select()] == ["a"]
    assert [i.id for i in g.select(include_unreviewed=True)] == ["a", "b"]
    assert [i.id for i in g.select(include_unreviewed=True, ids=["b"])] == ["b"]
    with pytest.raises(KeyError):
        g.select(ids=["nope"])


def test_check_against_corpus(loaded):
    engine, _ = loaded
    turns = [{"question": "Which rules?", "expect": "answer",
              "gold": ["COB 3.3.2", "COB 3.3.4", "COB 9.9.9", "COB 3.3"],
              "supporting": ["COB 3.3.4#p1", "GLO Retail Client"]}]  # fmt: skip
    issues = check_against_corpus(golden(item(turns=turns)), engine)
    assert issues == [
        "COB 3.3: a section parent is never retrieved; name its rules",
        "COB 3.3.4#p1: use the provision id 'COB 3.3.4', not a part",
        "COB 9.9.9: not in the corpus",
    ]  # the split rule COB 3.3.4 counts through its parts


def test_committed_golden_set_parses():
    g = load_golden()
    cov = coverage(g)
    assert cov["items"] >= 40
    assert cov["by_kind"]["unanswerable"] >= 8
    assert {"COB", "GEN", "AML", "MKT", "FPR", "GLO"} <= set(cov["by_module"])
    with open(PathSettings().golden_set, encoding="utf-8") as f:
        assert yaml.safe_load(f)["version"] == g.version
