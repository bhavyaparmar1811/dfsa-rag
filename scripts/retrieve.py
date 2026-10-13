"""Run retrieval for one question and show every stage.

    uv run python -m scripts.retrieve "Do we need a Client Agreement with a Market Counterparty?"
    uv run python -m scripts.retrieve "What does COB 3.3.2 require?" --no-rerank
    uv run python -m scripts.retrieve "customer due diligence" --module AML --binding-only
    uv run python -m scripts.retrieve "..." --mode dense --reranker BAAI/bge-reranker-base
    uv run python -m scripts.retrieve "..." --json            # the trace, as stored per request

Costs one query embedding (~$0.000001) unless --mode sparse.
"""

import argparse
import json
import logging
import sys
import textwrap

from src.config.settings import RetrievalSettings
from src.retrieval.models import Filters
from src.retrieval.pipeline import Retriever


def main() -> int:
    ap = argparse.ArgumentParser(description="Show retrieval for one question.")
    ap.add_argument("question")
    ap.add_argument("--mode", choices=["dense", "sparse", "hybrid"])
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--reranker", help="cross-encoder model name")
    ap.add_argument("--top-n", type=int)
    ap.add_argument("--parent-mode", choices=["none", "section", "window"])
    ap.add_argument("--expand-refs", action="store_true")
    ap.add_argument("--module", action="append", help="filter; repeatable (AML, REG_LAW, ...)")
    ap.add_argument("--binding-only", action="store_true")
    ap.add_argument("--show", type=int, default=12, help="candidates to list")
    ap.add_argument("--json", action="store_true", help="print the trace as JSON")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    overrides = {
        "mode": args.mode, "reranker_model": args.reranker, "top_n": args.top_n,
        "parent_mode": args.parent_mode,
    }  # fmt: skip
    cfg = RetrievalSettings(**{k: v for k, v in overrides.items() if v is not None})
    if args.no_rerank:
        cfg.rerank = False
    if args.expand_refs:
        cfg.expand_refs = True
    filters = Filters(modules=args.module, binding_only=args.binding_only)

    result = Retriever.build(cfg).retrieve(args.question, filters)
    if args.json:
        print(json.dumps(result.trace(), indent=2, ensure_ascii=False))
        return 0

    c = result.citations
    print(f"Question: {result.question}")
    print(f"Config:   {cfg.mode}, rerank={'off' if not cfg.rerank else cfg.reranker_model}, "
          f"top_n={cfg.top_n}, parent={cfg.parent_mode}")  # fmt: skip
    if result.config.get("reranker_device"):
        print(f"Reranker: {result.config['reranker_device']} ({result.config['reranker_dtype']})")
    if c.found:
        print(f"Citations found {c.found}; resolved {list(c.resolved)}; unresolved {c.unresolved};"
              f" too broad {c.too_broad}; filtered out {c.filtered_out}")  # fmt: skip

    print(f"\nCandidates ({len(result.candidates)}), best first:")
    print(f"  {'key':<46}{'src':>5}{'dense':>7}{'bm25':>6}{'rrf':>8}{'rerank':>8}")
    for cand in result.candidates[: args.show]:
        print(
            f"  {cand.key[:45]:<46}{cand.source[:4]:>5}{cand.dense_rank or '-':>7}"
            f"{cand.sparse_rank or '-':>6}{_f(cand.fused_score, 4):>8}"
            f"{_f(cand.rerank_score, 3):>8}{' (cut)' if cand.truncated else ''}"
        )

    conf = "n/a" if result.confidence is None else f"{result.confidence:.3f}"
    verdict = {True: "confident", False: "BELOW THRESHOLD", None: "threshold not calibrated"}
    print(f"\nHits: {[h.key for h in result.hits]}")
    print(f"Confidence: {conf} ({verdict[result.confident]})")

    print(f"\nContext: {len(result.context)} blocks, {result.context_tokens} tokens"
          + (f"; dropped {result.dropped}" if result.dropped else ""))  # fmt: skip
    group = None
    for b in result.context:
        if b.group != group:
            group = b.group
            print(f"  [{group}]")
        first = textwrap.shorten(b.text, 90)
        print(f"   {b.n:>2}. {b.key:<40} {b.label:<24} {b.reason:<13} {first}")

    timings = ", ".join(f"{k} {v:.0f}" for k, v in result.timings_ms.items())
    print(f"\nTimings (ms): {timings}")
    print(f"Embedding: {result.embed_tokens} tokens, ${result.embed_cost_usd:.6f}; "
          f"truncated by the reranker: {result.truncated}")  # fmt: skip
    return 0


def _f(value, digits):
    return "-" if value is None else f"{value:.{digits}f}"


if __name__ == "__main__":
    sys.exit(main())
