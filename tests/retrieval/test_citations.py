import pytest

from src.retrieval.citations import CitationRouter
from src.retrieval.models import Filters

CODES = {"COB", "GEN", "AML", "MKT", "FPR", "GLO"}
LAWS = ["Regulatory Law", "Markets Law", "Law Regulating Islamic Financial Business"]


@pytest.fixture
def parser():
    return CitationRouter(None, None, CODES, LAWS)  # find() needs neither store


@pytest.mark.parametrize(
    "question, expected",
    [
        ("What does COB 3.3.2 require?", ["COB 3.3.2"]),
        ("explain cob rule 3.3.2(1)(a)", ["COB 3.3.2"]),
        ("Is Rule 3.3.2 of COB relevant?", ["COB 3.3.2"]),
        ("COB Rules 3.3.2 and 3.3.3", ["COB 3.3.2", "COB 3.3.3"]),
        ("what is in COB section 3.3", ["COB 3.3"]),
        ("GEN A2.1.1 and MKT App4", ["MKT App4", "GEN A2.1.1"]),
        ("Does Rule 2.3.2 apply?", ["* 2.3.2"]),
        ("Article 20 of the Regulatory Law", ["Regulatory Law Art. 20"]),
        ("regulatory law art. 76 and article 77",
         ["Regulatory Law Art. 76", "Regulatory Law Art. 77"]),
        ("Markets Law Article 2", ["Markets Law Art. 2"]),
        ("the Law Regulating Islamic Financial Business, Article 2",
         ["Law Regulating Islamic Financial Business Art. 2"]),
        ("Rule 1.2.1 of PIB", ["PIB 1.2.1"]),  # another module: kept, then unresolved
        # Prose, not provisions (a false "Rules 7" refused a correct answer, eval run #13):
        ("the relevant fee referred to in Chapter 7 of the Rules", []),
        ("section 3.3 of the Rules", []),
        ("Do we need a client agreement with a Market Counterparty?", []),
        ("Is a DFSA 2024 notice relevant?", []),
    ],
)  # fmt: skip
def test_find(parser, question, expected):
    assert parser.find(question) == expected


@pytest.fixture
def router(loaded):
    engine, store = loaded
    return CitationRouter(engine, store, CODES, LAWS, max_hits=8)


def test_a_rule_resolves_to_itself_and_its_guidance(router):
    lookup, hits = router.lookup("What does COB 3.3.2 require?")
    assert lookup.resolved == {"COB 3.3.2": ["COB 3.3.2", "COB 3.3.2 Guidance"]}
    assert [h.key for h in hits] == ["COB 3.3.2", "COB 3.3.2 Guidance"]
    assert all(h.source == "citation" for h in hits)


def test_a_split_rule_resolves_to_its_parts(router):
    _, hits = router.lookup("COB 3.3.4")
    assert [h.key for h in hits] == ["COB 3.3.4#p1", "COB 3.3.4#p2"]


def test_a_section_resolves_to_its_children(router):
    lookup, hits = router.lookup("summarise COB section 3.3")
    assert [h.key for h in hits] == [
        "COB 3.3.1", "COB 3.3.2", "COB 3.3.2 Guidance", "COB 3.3.3", "COB 3.3.4#p1", "COB 3.3.4#p2",
    ]  # fmt: skip


def test_a_bare_rule_number_is_looked_up_in_every_module(router):
    lookup, hits = router.lookup("what does Rule 3.4.1 say")
    assert lookup.resolved == {"* 3.4.1": ["COB 3.4.1"]}


def test_a_citation_not_in_the_corpus_is_reported(router):
    lookup, hits = router.lookup("Is COB 16.6.4 in force? And COB chapter 3?")
    assert lookup.unresolved == ["COB 16.6.4"] and lookup.too_broad == ["COB 3"]
    assert hits == []


def test_filters_apply_to_citation_hits(router):
    lookup, hits = router.lookup("COB 3.3.2", Filters(binding_only=True))
    assert [h.key for h in hits] == ["COB 3.3.2"]
    assert lookup.filtered_out == ["COB 3.3.2 Guidance"]
    lookup, hits = router.lookup("COB 3.3.2", Filters(modules=["AML"]))
    assert hits == [] and lookup.resolved == {} and len(lookup.filtered_out) == 2


def test_hits_are_capped(loaded):
    engine, store = loaded
    capped = CitationRouter(engine, store, CODES, LAWS, max_hits=2)
    _, hits = capped.lookup("COB section 3.3")
    assert len(hits) == 2
