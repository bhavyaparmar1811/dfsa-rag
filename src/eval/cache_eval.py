"""Calibrate and gate the semantic cache's similarity threshold (src/chat/cache.py).

    uv run python -m src.eval.cache_eval                       # reviewed pairs only
    uv run python -m src.eval.cache_eval --include-unreviewed  # the draft set
    uv run python -m src.eval.cache_eval --threshold 0.93      # evaluate a candidate value

Over data/eval/cache_pairs.yaml (a cached question and a new one, labelled
`positive` = same answer, or `negative` = a near miss with a different answer):

  1. embeds every question with the production EmbeddingClient (well under $0.001)
  2. applies the cache's guards (same provisions, Defined Terms and numbers),
     with the same parsers the cache uses
  3. sweeps the threshold, with and without the guards
  4. recommends the lowest threshold at which no guarded negative is served,
     plus a margin (and never below a floor), and reports the positive hit rate
  5. evaluates the threshold in force (CACHE_SIMILARITY_THRESHOLD, or --threshold)
     and stores the run as eval_runs kind="cache", gated by the `cache` block of
     eval/thresholds.yaml: 0 false hits is the release rule

The unguarded sweep is the baseline: what a similarity-only cache would serve at
the thresholds the popular tools use in their examples (LiteLLM's docs show 0.8
cosine similarity; RedisVL's default distance 0.1 is 0.9 similarity).
"""

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from src.chat.cache import normalise, question_features
from src.config.settings import CacheSettings, EmbeddingSettings, PathSettings, RetrievalSettings
from src.embeddings.client import EmbeddingClient
from src.eval import gate
from src.eval.run import git_commit
from src.ingestion.chunkers.law import law_name
from src.ingestion.manifest import load_manifest
from src.retrieval.citations import CitationRouter
from src.retrieval.context import ContextAssembler
from src.storage.db import assert_migrated, get_engine, session_scope
from src.storage.models import EvalRun
from src.storage.repositories import EvalRunRepository

# Thresholds a similarity-only cache would plausibly be configured with.
NAIVE_THRESHOLDS = (0.80, 0.85, 0.90, 0.95)
SWEEP = [round(0.80 + i * 0.005, 3) for i in range(40)]  # 0.800 .. 0.995


class Pair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    kind: Literal["positive", "negative"]
    cached: str
    asked: str
    why: str | None = None

    @model_validator(mode="after")
    def _negative_says_why(self) -> "Pair":
        if self.kind == "negative" and not self.why:
            raise ValueError(f"{self.id}: a negative pair must say why the answers differ")
        return self


class PairSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewed: bool
    pairs: list[Pair]

    @model_validator(mode="after")
    def _unique_ids(self) -> "PairSet":
        ids = [p.id for p in self.pairs]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate pair ids")
        return self


def load_pairs(path: Path | None = None) -> tuple[PairSet, str]:
    """The pairs and the sha256 of the file (the run records which set it used)."""
    path = path or PathSettings().cache_pairs
    raw = path.read_bytes()
    return PairSet.model_validate(yaml.safe_load(raw)), hashlib.sha256(raw).hexdigest()


@dataclass
class Scored:
    pair: Pair
    similarity: float
    exact: bool  # the same question once normalised: always a hit
    guards: list[str]  # the guards that differ; empty = the guards let it through

    def served(self, threshold: float | None, guarded: bool = True) -> bool:
        if self.exact:
            return True
        if threshold is None or self.similarity < threshold:
            return False
        return not (guarded and self.guards)


def score(pairs: list[Pair], embed, features) -> tuple[list[Scored], int, float]:
    """`embed(texts) -> EmbeddingResult`, `features(question) -> Features`."""
    texts = sorted({t for p in pairs for t in (p.cached, p.asked)})
    result = embed(texts)
    vectors = dict(zip(texts, result.vectors, strict=True))
    scored = [
        Scored(
            pair=p,
            similarity=_cosine(vectors[p.cached], vectors[p.asked]),
            exact=normalise(p.cached) == normalise(p.asked),
            guards=features(p.asked).differences(features(p.cached)),
        )
        for p in pairs
    ]
    return scored, result.prompt_tokens, result.cost_usd


def outcome(scored: list[Scored], threshold: float | None, guarded: bool = True) -> dict:
    pos = [s for s in scored if s.pair.kind == "positive"]
    neg = [s for s in scored if s.pair.kind == "negative"]
    false_hits = [s.pair.id for s in neg if s.served(threshold, guarded)]
    hits = [s.pair.id for s in pos if s.served(threshold, guarded)]
    return {
        "threshold": threshold,
        "false_hits": len(false_hits),
        "false_hit_ids": false_hits,
        "positive_hits": len(hits),
        "positive_hit_rate": round(len(hits) / len(pos), 4) if pos else None,
    }


def recommend(scored: list[Scored], margin: float, floor: float) -> float:
    """The lowest threshold that serves no guarded negative, plus `margin`, rounded
    up to 0.005, and never below `floor`."""
    passing = [s.similarity for s in scored if s.pair.kind == "negative" and not s.guards]
    lowest_safe = max(passing, default=0.0) + margin
    return min(0.995, max(floor, math.ceil(lowest_safe * 200) / 200))


def evaluate_pairs(scored: list[Scored], threshold: float | None, margin: float, floor: float):
    pos = [s for s in scored if s.pair.kind == "positive"]
    neg = [s for s in scored if s.pair.kind == "negative"]
    recommended = recommend(scored, margin, floor)
    guard_passing_negatives = sorted((s for s in neg if not s.guards), key=lambda s: -s.similarity)
    at = outcome(scored, threshold)
    return {
        "n_pairs": len(scored),
        "n_positive": len(pos),
        "n_negative": len(neg),
        # The threshold in force: what the gate checks.
        "threshold": threshold,
        "false_hits": at["false_hits"],
        "false_hit_ids": at["false_hit_ids"],
        "positive_hit_rate": at["positive_hit_rate"],
        "recommended_threshold": recommended,
        "at_recommended": outcome(scored, recommended),
        "negatives_stopped_by_guards": sum(bool(s.guards) for s in neg),
        "positives_stopped_by_guards": [s.pair.id for s in pos if s.guards],
        "max_guard_passing_negative": (
            {
                "id": guard_passing_negatives[0].pair.id,
                "similarity": round(guard_passing_negatives[0].similarity, 4),
            }
            if guard_passing_negatives
            else None
        ),  # fmt: skip
        "positive_similarity": _summary([s.similarity for s in pos]),
        "negative_similarity": _summary([s.similarity for s in neg]),
        "naive": [outcome(scored, t, guarded=False) for t in NAIVE_THRESHOLDS],
        "sweep": [
            {
                "threshold": t,
                "guarded_false_hits": outcome(scored, t)["false_hits"],
                "guarded_hit_rate": outcome(scored, t)["positive_hit_rate"],
                "naive_false_hits": outcome(scored, t, guarded=False)["false_hits"],
                "naive_hit_rate": outcome(scored, t, guarded=False)["positive_hit_rate"],
            }
            for t in SWEEP
        ],
        "pairs": [
            {
                "id": s.pair.id,
                "kind": s.pair.kind,
                "similarity": round(s.similarity, 4),
                "exact": s.exact,
                "guards": s.guards,
            }
            for s in scored
        ],  # fmt: skip
    }


def render(run_id: int, metrics: dict, reviewed: bool, cost: float, report: str) -> str:
    m = metrics
    lines = [
        f"# Semantic cache eval, run #{run_id}",
        "",
        f"{m['n_pairs']} pairs ({m['n_positive']} positive, {m['n_negative']} hard negatives)"
        f"{'' if reviewed else ', DRAFT (unreviewed)'}; embedding cost ${cost:.6f}.",
        "",
        "## A similarity-only cache (no guards)",
        "",
        "| Threshold | Wrong answers served (of negatives) | Paraphrases served |",
        "|---|---|---|",
        *(
            f"| {n['threshold']:.2f} | {n['false_hits']} / {m['n_negative']} "
            f"| {_pct(n['positive_hit_rate'])} |"
            for n in m["naive"]
        ),
        "",
    ]
    worst = [n for n in m["naive"] if n["false_hits"]]
    if worst:
        lines += [f"At {worst[0]['threshold']:.2f}: " + ", ".join(worst[0]["false_hit_ids"]), ""]
    lines += [
        "## With the guards (same provisions, Defined Terms and numbers)",
        "",
        f"- Negatives stopped by a guard: {m['negatives_stopped_by_guards']} / {m['n_negative']}",
        f"- Positives wrongly stopped by a guard: {len(m['positives_stopped_by_guards'])} "
        f"({', '.join(m['positives_stopped_by_guards']) or 'none'})",
    ]
    worst_neg = m["max_guard_passing_negative"]
    if worst_neg:
        lines.append(
            f"- Most similar negative the guards let through: {worst_neg['id']} "
            f"({worst_neg['similarity']:.4f})"
        )
    rec = m["at_recommended"]
    lines += [
        f"- **Recommended threshold: {m['recommended_threshold']:.3f}** "
        f"({rec['false_hits']} false hits, {_pct(rec['positive_hit_rate'])} of paraphrases served)",
        f"- Threshold in force: {m['threshold']} -> {m['false_hits']} false hits, "
        f"{_pct(m['positive_hit_rate'])} of paraphrases served",
        "",
        "| Threshold | Guarded: false hits | Guarded: hit rate | Naive: false hits | Naive: hit rate |",  # noqa: E501
        "|---|---|---|---|---|",
        *(
            f"| {r['threshold']:.3f} | {r['guarded_false_hits']} | {_pct(r['guarded_hit_rate'])} "
            f"| {r['naive_false_hits']} | {_pct(r['naive_hit_rate'])} |"
            for r in m["sweep"][::2]
        ),
        "",
        "## Pairs",
        "",
        "| Pair | Kind | Cosine | Guards that differ |",
        "|---|---|---|---|",
        *(
            f"| {p['id']} | {p['kind']} | {p['similarity']:.4f}{' (exact)' if p['exact'] else ''} "
            f"| {', '.join(p['guards']) or '-'} |"
            for p in sorted(m["pairs"], key=lambda p: (p["kind"], -p["similarity"]))
        ),
        "",
        report,
    ]
    return "\n".join(lines)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def _summary(xs: list[float]) -> dict | None:
    if not xs:
        return None
    xs = sorted(xs)
    return {"min": round(xs[0], 4), "median": round(xs[len(xs) // 2], 4), "max": round(xs[-1], 4)}


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.0%}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--include-unreviewed", action="store_true", help="run on the draft set")
    ap.add_argument("--threshold", type=float, help="evaluate this value, not the configured one")
    ap.add_argument("--margin", type=float, default=0.02, help="added above the worst negative")
    ap.add_argument("--floor", type=float, default=0.90, help="never recommend below this")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    pair_set, pairs_hash = load_pairs()
    if not pair_set.reviewed and not args.include_unreviewed:
        print("cache_pairs.yaml is not reviewed yet: review it, or pass --include-unreviewed.")
        return 1
    threshold = args.threshold
    if threshold is None:
        threshold = CacheSettings().similarity_threshold

    engine = get_engine()
    assert_migrated(engine)
    manifest = load_manifest()
    router = CitationRouter(
        engine, None,
        {e.module_code for e in manifest if e.doc_type == "rulebook"},
        [law_name(e) for e in manifest if e.doc_type == "law"],
    )  # fmt: skip
    terms = ContextAssembler(engine, RetrievalSettings())

    def features(q: str):
        return question_features(q, router.find, terms.question_terms)

    emb = EmbeddingSettings()
    scored, tokens, cost = score(pair_set.pairs, EmbeddingClient(emb).embed_documents, features)
    metrics = evaluate_pairs(scored, threshold, args.margin, args.floor)

    with session_scope(engine) as s:
        repo = EvalRunRepository(s)
        run = repo.start(
            EvalRun(
                kind="cache",
                config_name="semantic_cache",
                retrieval_config={
                    "embedding": emb.fingerprint,
                    "guards": ["citations", "terms", "numbers"],
                    "margin": args.margin,
                    "floor": args.floor,
                },
                golden_hash=pairs_hash,
                reviewed_only=pair_set.reviewed,
                n_items=len(scored),
                n_turns=len(scored),
                git_commit=git_commit(),
                notes=None if pair_set.reviewed else "draft (unreviewed) cache pairs",
            )
        )
        run_id = run.id
        repo.finish(
            run_id,
            metrics=metrics,
            cost_usd={"embed": round(cost, 6), "total": round(cost, 6)},
        )
    report = gate.apply(engine, run_id)

    text = render(run_id, metrics, pair_set.reviewed, cost, report.render())
    out_dir = PathSettings().eval_reports
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"cache-run{run_id}.md").write_text(text, encoding="utf-8")
    (out_dir / f"cache-run{run_id}.json").write_text(json.dumps(metrics, indent=2), "utf-8")
    print(text)
    print(f"\n{tokens} embedding tokens; report: {out_dir / f'cache-run{run_id}.md'}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
