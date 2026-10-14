"""The golden set: hand-checked questions with the provisions that answer them.

    uv run python -m src.eval.golden --check     # schema + every provision exists in SQLite

`data/eval/golden.yaml` holds conversations. A single question is a conversation
of one turn; a follow-up case has 2-3 turns, and each turn carries its own gold.

Gold labels are **provision ids**: a chunk key with any `#pN` part suffix
removed ("COB 3.3.2", "COB 3.3.2 Guidance", "GLO Retail Client",
"Regulatory Law Art. 20"). They survive re-ingestion, chunk ids don't. A rule
and its Guidance are different provisions, so a question answered by the
Guidance lists the Guidance.

Only items marked `reviewed: true` count for calibration and the release gate;
drafts are evaluated on request (`--include-unreviewed`) but never gate anything.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import Engine

from src.config.settings import PathSettings
from src.retrieval.models import Filters
from src.storage.db import make_engine, session_scope
from src.storage.repositories import ChunkRepository

Kind = Literal["single", "exemption", "laws_term", "multi_turn", "unanswerable"]
Expect = Literal["answer", "refuse"]

_PART = re.compile(r"#p\d+$")


def provision(key: str) -> str:
    """'GEN 2.3.2#p1' -> 'GEN 2.3.2'. A split rule's parts are one provision."""
    return _PART.sub("", key)


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3)
    expect: Expect
    gold: list[str] = []  # must be retrieved (relevance grade 2)
    supporting: list[str] = []  # helps, isn't required (grade 1)
    notes: str = ""

    @model_validator(mode="after")
    def _gold_matches_expect(self) -> "Turn":
        if self.expect == "answer" and not self.gold:
            raise ValueError(f"an answerable turn needs gold provisions: {self.question!r}")
        if self.expect == "refuse" and (self.gold or self.supporting):
            raise ValueError(f"a refusal turn has no gold provisions: {self.question!r}")
        if overlap := set(self.gold) & set(self.supporting):
            raise ValueError(f"listed as gold and supporting: {sorted(overlap)}")
        return self

    @property
    def relevance(self) -> dict[str, int]:
        """Graded relevance for nDCG: gold 2, supporting 1."""
        return {p: 1 for p in self.supporting} | {p: 2 for p in self.gold}


class Item(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    kind: Kind
    module: str | None = None  # the module the question is mainly about
    reviewed: bool = False
    filters: Filters = Filters()
    turns: list[Turn] = Field(min_length=1)

    @model_validator(mode="after")
    def _turns_match_kind(self) -> "Item":
        if self.kind == "multi_turn" and len(self.turns) < 2:
            raise ValueError(f"{self.id}: a multi_turn item needs at least 2 turns")
        if self.kind != "multi_turn" and len(self.turns) != 1:
            raise ValueError(f"{self.id}: only multi_turn items have more than one turn")
        if self.kind == "unanswerable" and self.turns[0].expect != "refuse":
            raise ValueError(f"{self.id}: an unanswerable item must expect a refusal")
        return self


class GoldenSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    items: list[Item]

    @model_validator(mode="after")
    def _unique_ids(self) -> "GoldenSet":
        dupes = [i for i, n in Counter(item.id for item in self.items).items() if n > 1]
        if dupes:
            raise ValueError(f"duplicate item ids: {dupes}")
        return self

    @property
    def hash(self) -> str:
        """sha256 of what is evaluated. Notes are left out, so commenting on an
        item doesn't invalidate earlier runs; flipping `reviewed` changes it."""
        data = self.model_dump(
            mode="json", exclude={"items": {"__all__": {"turns": {"__all__": {"notes"}}}}}
        )
        blob = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def select(
        self, *, include_unreviewed: bool = False, ids: list[str] | None = None
    ) -> list[Item]:
        items = [i for i in self.items if include_unreviewed or i.reviewed]
        if ids:
            unknown = set(ids) - {i.id for i in self.items}
            if unknown:
                raise KeyError(f"no such golden items: {sorted(unknown)}")
            items = [i for i in items if i.id in ids]
        return items

    def provisions(self) -> set[str]:
        return {p for item in self.items for t in item.turns for p in (*t.gold, *t.supporting)}


def load_golden(path: Path | None = None) -> GoldenSet:
    path = path or PathSettings().golden_set
    with open(path, encoding="utf-8") as f:
        return GoldenSet.model_validate(yaml.safe_load(f))


def check_against_corpus(golden: GoldenSet, engine: Engine) -> list[str]:
    """Every gold and supporting provision must be a live, retrievable chunk:
    present in SQLite, not deleted, not a section parent (never embedded)."""
    issues = []
    with session_scope(engine) as s:
        repo = ChunkRepository(s)
        for p in sorted(golden.provisions()):
            if p != provision(p):
                issues.append(f"{p}: use the provision id {provision(p)!r}, not a part")
                continue
            row = repo.by_key(p)
            if row is None:
                issues.append(f"{p}: not in the corpus")
            elif row.deleted:
                issues.append(f"{p}: a deleted provision")
            elif row.chunk_type == "section":
                issues.append(f"{p}: a section parent is never retrieved; name its rules")
            elif not row.is_embedded and not repo.parts_of(p):
                issues.append(f"{p}: not embedded (run ingest)")
    return issues


def coverage(golden: GoldenSet) -> dict:
    turns = [t for item in golden.items for t in item.turns]
    return {
        "items": len(golden.items),
        "reviewed": sum(i.reviewed for i in golden.items),
        "turns": len(turns),
        "expect_refuse": sum(t.expect == "refuse" for t in turns),
        "by_kind": dict(Counter(i.kind for i in golden.items)),
        "by_module": dict(Counter(i.module or "-" for i in golden.items)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate the golden set.")
    ap.add_argument("--path", type=Path, help="default: data/eval/golden.yaml")
    ap.add_argument("--check", action="store_true", help="also check provisions against SQLite")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    golden = load_golden(args.path)
    cov = coverage(golden)
    print(
        f"{cov['items']} items ({cov['reviewed']} reviewed), {cov['turns']} turns, "
        f"{cov['expect_refuse']} expect a refusal"
    )
    print("by kind:  ", ", ".join(f"{k} {n}" for k, n in sorted(cov["by_kind"].items())))
    print("by module:", ", ".join(f"{k} {n}" for k, n in sorted(cov["by_module"].items())))
    print(f"golden hash: {golden.hash[:16]}")
    if not args.check:
        return 0
    issues = check_against_corpus(golden, make_engine())
    for issue in issues:
        print(f"  ISSUE {issue}")
    print(
        f"{len(golden.provisions())} provisions checked: "
        f"{'OK' if not issues else f'{len(issues)} issue(s)'}"
    )
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
