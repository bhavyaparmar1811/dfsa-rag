"""Ask a question in a conversation: rewrite, retrieve, answer (or refuse), store.

    uv run python -m scripts.ask "Do we need a Client Agreement with a Market Counterparty?"
    uv run python -m scripts.ask "what about retail clients?" --conversation <id>
    uv run python -m scripts.ask "What is a Retail Client?" --label staging
    uv run python -m scripts.ask "customer due diligence" --module AML --json
    uv run python -m scripts.ask "..." --chunk-type guidance   # Guidance only (labelled answer)

Goes through ChatService (the same path as the API): history comes from SQLite,
every turn is stored with its trace, and Langfuse gets one trace per turn,
grouped by conversation. A turn costs one query embedding and up to two GPT-4o
calls (rewrite on follow-ups, answer unless the gate refuses), about $0.01-0.03.
"""

import argparse
import json
import logging
import sys
import textwrap

from src.chat.service import ChatService
from src.retrieval.models import Filters
from src.storage.db import assert_migrated


def main() -> int:
    ap = argparse.ArgumentParser(description="Answer a question, with citations.")
    ap.add_argument("question")
    ap.add_argument("--conversation", help="continue this conversation (default: a new one)")
    ap.add_argument("--label", help="prompt label (default: production)")
    ap.add_argument("--module", action="append", help="filter; repeatable (AML, REG_LAW, ...)")
    ap.add_argument("--chunk-type", action="append", help="filter: rule, guidance, definition, ...")
    ap.add_argument("--binding-only", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the stored trace as JSON")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    service = ChatService.build()
    assert_migrated(service.engine)
    conversation = args.conversation or service.create_conversation().id
    filters = Filters(
        modules=args.module, chunk_types=args.chunk_type, binding_only=args.binding_only
    )
    turn = service.send(conversation, args.question, filters, label=args.label)
    answer, retrieval = turn.answer, turn.retrieval

    if args.json:
        trace = service.get_trace(turn.trace_id)
        row = {c.name: getattr(trace, c.name) for c in trace.__table__.columns}
        print(json.dumps(row, indent=2, ensure_ascii=False, default=str))
        return 0

    print(f"Conversation: {conversation}")
    print(f"Question: {args.question}")
    if turn.rewrite.changed:
        print(f"Rewritten: {turn.rewrite.question}")
    elif turn.rewrite.fallback_reason:
        print(f"Rewrite discarded: {turn.rewrite.fallback_reason}")
    if turn.cached:
        hit = turn.cache.hit
        similarity = "exact repeat" if hit.similarity is None else f"cosine {hit.similarity:.3f}"
        print(f"Cache: {hit.kind} hit ({similarity}) of \"{hit.cached_query}\"; "
              f"first answered in trace {hit.source_trace_id}")  # fmt: skip
    else:
        conf = "n/a" if retrieval.confidence is None else f"{retrieval.confidence:.3f}"
        print(f"Retrieval: confidence {conf}, gate {answer.gate}, {len(retrieval.context)} "
              f"blocks, {retrieval.context_tokens} tokens")  # fmt: skip
        if turn.cache is not None:
            stored = f"; stored as {turn.cache_entry_id}" if turn.cache_entry_id else ""
            print(f"Cache: {turn.cache.status} ({turn.cache.reason}){stored}")
    if answer.refused:
        print(f"\nREFUSED ({answer.refusal.code})")
    print()
    for para in answer.text.split("\n"):
        print(textwrap.fill(para, 100) if para else "")
    if answer.cited:
        print("\nSources:")
        for c in answer.cited:
            print(f"  [{c.context_id}] {c.citation:<32} {c.label}")
    if answer.violations:
        print(f"\nViolations: {answer.violations}")
    prompts = ([turn.rewrite.prompt] if turn.rewrite.prompt else []) + answer.prompts
    for p in prompts:
        version = f"v{p['version']}" if p["version"] else "git copy"
        print(f"Prompt: {p['name']} {p['label']} {version} ({p['source']}, {p['hash'][:12]})")
    stages = ", ".join(f"{k} {v:.0f} ms" for k, v in turn.latency_ms.items())
    print(f"Cost: ${turn.cost_usd:.6f}; latency: {stages}")
    print(f"Trace: {turn.trace_url or turn.trace_id}")
    print(f'\nFollow up: uv run python -m scripts.ask "..." --conversation {conversation}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
