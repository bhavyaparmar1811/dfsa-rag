"""Context assembly: from the reranked hits to what the LLM reads.

A hit is one rule-level chunk; answering from it alone can be wrong. COB 3.3.2
says what a Client Agreement needs, but COB 3.3.1 says the section doesn't
apply to Market Counterparties. So each hit brings, from SQLite:

    part -> whole       a split rule is shown whole (only its parts were searched)
    scope rules         "The Rules in this section do not apply to ...", always
    Guidance <-> rule   a rule's Guidance; the rule a Guidance block explains
    the section         the whole section if it fits parent_max_tokens, else a
                        window of siblings; the whole section anyway when
                        section_hit_min of its children were hits
    Defined Terms       their definitions (Glossary, or the Law's own Schedule);
                        terms in the question first, very common terms skipped
    references          optionally, one hop along cross-references

Every block is exactly one chunk, so each can be cited on its own. When the
token budget runs out, the least important blocks are dropped (and reported):
hits are always kept, then scope rules and Guidance, then definitions of terms
in the question, then section siblings, then other definitions, then references.
Blocks are shown grouped by section, the strongest hit's section first, each
section in document order, then the definitions, then the references.
"""

import re
from collections import Counter
from dataclasses import dataclass

from sqlalchemy import Engine

from src.config.settings import RetrievalSettings
from src.ingestion.terms import TermMatcher
from src.retrieval.models import Candidate, ContextBlock
from src.storage.db import session_scope
from src.storage.models import ChunkRow
from src.storage.repositories import ChunkRepository

GLOSSARY_DOC = "dfsa-rulebook-glo"
DEFINITIONS, REFERENCES = "definitions", "references"
# Lower = more important; tier 0 (the hits) is never dropped.
TIERS = {
    "hit": 0, "part": 0, "scope": 1, "guidance": 1, "rule": 1, "question_term": 2,
    "section": 3, "window": 3, "definition": 4, "ref": 5,
}  # fmt: skip
LABELS = {
    "rule": "Rule", "guidance": "Guidance", "definition": "Definition",
    "article": "Law article", "text": "Text",
}  # fmt: skip


@dataclass
class _Item:
    row: ChunkRow
    reason: str
    group: str
    rank: int  # rank of the hit that brought it in (0 = best)
    score: float | None


def label(row: ChunkRow) -> str:
    weight = "binding" if row.binding else "non-binding"
    return f"{LABELS.get(row.chunk_type, row.chunk_type)} ({weight})"


class ContextAssembler:
    def __init__(self, engine: Engine, cfg: RetrievalSettings):
        self.engine = engine
        self.cfg = cfg
        with session_scope(engine) as s:
            repo = ChunkRepository(s)
            self.shares = repo.term_shares()
            terms = repo.defined_terms()
        self.matcher = TermMatcher(terms) if terms else None
        # Typed questions are often lower case ("market counterparty"): multi-word
        # terms are matched case-insensitively there (one-word ones would match
        # ordinary words: "client", "person").
        multi = sorted((t for t in terms if " " in t), key=len, reverse=True)
        self._loose = (
            re.compile(rf"(?<!\w)({'|'.join(map(re.escape, multi))})(?:s|es)?(?!\w)", re.I)
            if multi
            else None
        )
        self._by_lower = {t.lower(): t for t in multi}

    def question_terms(self, question: str) -> list[str]:
        found = self.matcher.find(question) if self.matcher else []
        if self._loose:
            for m in self._loose.finditer(question):
                term = self._by_lower.get(m.group(1).lower())
                if term and term not in found:
                    found.append(term)
        return found

    def assemble(
        self, question: str, hits: list[Candidate]
    ) -> tuple[list[ContextBlock], list[str]]:
        """Blocks for the LLM, and the keys of blocks the budget left out."""
        cfg = self.cfg
        items: dict[str, _Item] = {}

        def add(row: ChunkRow, reason: str, group: str, rank: int, score: float | None):
            old = items.get(row.key)
            if old is None or TIERS[reason] < TIERS[old.reason]:
                items[row.key] = _Item(row, reason, group, rank, score)

        with session_scope(self.engine) as s:
            repo = ChunkRepository(s)
            primaries = self._primaries(repo, hits)
            hits_per_section = Counter(r.parent_key for r, *_ in primaries if r.parent_key)

            for row, rank, score, reason in primaries:
                group = row.parent_key or row.key
                add(row, reason, group, rank, score)
                if cfg.include_guidance:
                    for g in repo.by_keys(row.guidance_keys or []):
                        add(g, "guidance", group, rank, score)
                    if row.guidance_for:
                        rule = repo.by_key(row.guidance_for)
                        if rule is not None and rule.chunk_type != "section":
                            add(rule, "rule", group, rank, score)
                if row.parent_key and row.chunk_type != "definition":
                    self._section(repo, row, hits_per_section[row.parent_key], rank, score, add)

            if cfg.include_definitions:
                self._definitions(repo, question, [p[0] for p in primaries], items, add)
            if cfg.expand_refs:
                self._references(repo, [p[0] for p in primaries], items, add)

        kept, dropped = self._budget(list(items.values()))
        return self._blocks(kept), dropped

    # --- the steps -------------------------------------------------------------

    def _primaries(self, repo, hits):
        """(row, rank, score, reason) per hit; a part is replaced by its whole rule."""
        rows = {r.id: r for r in repo.by_ids([h.id for h in hits])}
        out, seen = [], set()
        for rank, hit in enumerate(hits):
            row = rows.get(hit.id)
            if row is None:
                continue
            score = hit.rerank_score if hit.rerank_score is not None else hit.fused_score
            reason = "hit"
            if row.part_of:
                whole = repo.by_key(row.part_of)
                if whole is not None:
                    row, reason = whole, "part"
            if row.key not in seen:
                seen.add(row.key)
                out.append((row, rank, score, reason))
        return out

    def _section(self, repo, row, hits_here, rank, score, add) -> None:
        cfg = self.cfg
        siblings = [r for r in repo.children_of(row.parent_key) if not r.deleted]
        if cfg.include_scope_rules:
            for sib in siblings:
                if sib.scope_rule:
                    add(sib, "scope", row.parent_key, rank, score)
        if cfg.parent_mode == "none":
            return
        size = sum(r.token_count for r in siblings)
        whole = hits_here >= cfg.section_hit_min or (
            cfg.parent_mode == "section" and size <= cfg.parent_max_tokens
        )
        if whole:
            for sib in siblings:
                add(sib, "section", row.parent_key, rank, score)
            return
        keys = [r.key for r in siblings]
        if row.key in keys:
            i, w = keys.index(row.key), cfg.parent_window
            for sib in siblings[max(0, i - w) : i + w + 1]:
                add(sib, "window", row.parent_key, rank, score)

    def _definitions(self, repo, question, primaries, items, add) -> None:
        cfg = self.cfg
        skip = set(cfg.skip_terms) | {r.term for r in primaries if r.term}  # already a hit
        in_question = [t for t in self.question_terms(question) if t not in skip]
        # Terms the hits use: those of the strongest hit first, rarer terms first.
        best_rank: dict[str, int] = {}
        for rank, row in enumerate(primaries):
            for t in repo.terms_of(row.id):
                best_rank.setdefault(t, rank)
        from_hits = [
            t for t in sorted(best_rank, key=lambda t: (best_rank[t], self.shares.get(t, 0)))
            if t not in skip and t not in in_question
            and self.shares.get(t, 0) <= cfg.term_max_share
        ][: cfg.max_hit_definitions]  # fmt: skip
        wanted = (in_question + from_hits)[: cfg.max_definitions]
        if not wanted:
            return
        docs_of_hits = [r.document_id for r in primaries]
        by_term: dict[str, list[ChunkRow]] = {}
        for d in repo.definitions_for(wanted):
            by_term.setdefault(d.term, []).append(d)
        for term in wanted:
            options = by_term.get(term)
            if not options:
                continue
            # The hit's own Law defines the term for that Law; otherwise the Glossary.
            best = min(
                options,
                key=lambda d: (
                    d.document_id not in docs_of_hits,
                    d.document_id != GLOSSARY_DOC,
                    d.document_id,
                ),
            )
            if best.key not in items:
                reason = "question_term" if term in in_question else "definition"
                add(best, reason, DEFINITIONS, len(items), None)

    def _references(self, repo, primaries, items, add) -> None:
        added = 0
        for row in primaries:
            for ref in repo.refs_of(row.id):
                targets = [
                    t for t in repo.by_citations([ref])
                    if t.part_of is None and t.chunk_type != "guidance" and not t.deleted
                ]  # fmt: skip
                for t in targets[:1]:
                    if t.key not in items and added < self.cfg.max_ref_blocks:
                        add(t, "ref", REFERENCES, len(items), None)
                        added += 1

    def _budget(self, items: list[_Item]) -> tuple[list[_Item], list[str]]:
        budget, used = self.cfg.context_budget_tokens, 0
        kept, dropped = [], []
        for item in sorted(items, key=lambda i: (TIERS[i.reason], i.rank, i.row.order_idx)):
            tokens = item.row.token_count
            if TIERS[item.reason] == 0 or used + tokens <= budget:
                kept.append(item)
                used += tokens
            else:
                dropped.append(item.row.key)
        return kept, dropped

    def _blocks(self, items: list[_Item]) -> list[ContextBlock]:
        tail = {DEFINITIONS: 1, REFERENCES: 2}
        group_rank: dict[str, int] = {}
        for item in items:
            group_rank[item.group] = min(group_rank.get(item.group, item.rank), item.rank)
        ordered = sorted(
            items,
            key=lambda i: (
                tail.get(i.group, 0),
                group_rank[i.group],
                i.group,
                i.row.document_id,
                i.row.order_idx,
            ),
        )
        return [
            ContextBlock(
                n=n,
                chunk_id=i.row.id,
                key=i.row.key,
                citation=i.row.citation,
                chunk_type=i.row.chunk_type,
                binding=i.row.binding,
                label=label(i.row),
                heading=i.row.header_text,
                text=i.row.text,
                tokens=i.row.token_count,
                reason=i.reason,
                group=i.group,
                score=i.score,
            )  # fmt: skip
            for n, i in enumerate(ordered, 1)
        ]
