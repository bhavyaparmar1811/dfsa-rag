"""Ask one question: retrieve, answer (or refuse), and show the sources.

    uv run python -m scripts.ask "Do we need a Client Agreement with a Market Counterparty?"
    uv run python -m scripts.ask "What is a Retail Client?" --label staging
    uv run python -m scripts.ask "customer due diligence" --module AML --json
    uv run python -m scripts.ask "..." --chunk-type guidance   # Guidance only (labelled answer)

One Langfuse trace per question: a retriever observation (with the query
embedding's usage and cost) and the answer step (generation + citation check).
Costs one query embedding and, unless the gate refuses, one GPT-4o call
(about $0.01-0.02).
"""

import argparse
import json
import logging
import sys
import textwrap

from src.config.settings import AzureOpenAISettings, EmbeddingSettings, RetrievalSettings
from src.generation.answer import Answerer
from src.generation.llm import LLMClient
from src.observability.langfuse import get_langfuse
from src.prompts.registry import PromptRegistry
from src.retrieval.models import Filters
from src.retrieval.pipeline import Retriever


def main() -> int:
    ap = argparse.ArgumentParser(description="Answer one question, with citations.")
    ap.add_argument("question")
    ap.add_argument("--label", help="prompt label (default: production)")
    ap.add_argument("--module", action="append", help="filter; repeatable (AML, REG_LAW, ...)")
    ap.add_argument("--binding-only", action="store_true")
    ap.add_argument("--chunk-type", action="append", help="filter: rule, guidance, definition, ...")
    ap.add_argument("--json", action="store_true", help="print the answer trace as JSON")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    retriever = Retriever.build(RetrievalSettings())
    answerer = Answerer(
        LLMClient(AzureOpenAISettings()), PromptRegistry.from_settings(), retriever.citations.find
    )
    filters = Filters(
        modules=args.module, chunk_types=args.chunk_type, binding_only=args.binding_only
    )
    langfuse, emb = get_langfuse(), EmbeddingSettings()

    with langfuse.start_as_current_observation(
        name="ask", as_type="span", input={"question": args.question}
    ) as root:
        with langfuse.start_as_current_observation(
            name="retrieve", as_type="retriever", input={"question": args.question}
        ) as obs:
            retrieval = retriever.retrieve(args.question, filters)
            with langfuse.start_as_current_observation(
                name="embed-query",
                as_type="embedding",
                model=emb.model,
                usage_details={"input": retrieval.embed_tokens},
                cost_details={"input": retrieval.embed_cost_usd},
            ):
                pass
            obs.update(
                output={
                    "hits": [h.key for h in retrieval.hits],
                    "context": [b.key for b in retrieval.context],
                },
                metadata={
                    "confidence": retrieval.confidence,
                    "confident": retrieval.confident,
                    "trace": retrieval.trace(),
                },
            )
        result = answerer.answer(retrieval, label=args.label)
        root.update(output={"status": result.status, "text": result.text})
        trace_url = langfuse.get_trace_url(trace_id=langfuse.get_current_trace_id())
    langfuse.flush()

    if args.json:
        print(json.dumps(result.trace(), indent=2, ensure_ascii=False))
        return 0

    conf = "n/a" if retrieval.confidence is None else f"{retrieval.confidence:.3f}"
    print(f"Question: {args.question}")
    print(
        f"Retrieval: confidence {conf}, gate {result.gate}, {len(retrieval.context)} blocks, "
        f"{retrieval.context_tokens} tokens"
    )
    if result.refused:
        print(f"\nREFUSED ({result.refusal.code})")
    print()
    for para in result.text.split("\n"):
        print(textwrap.fill(para, 100) if para else "")
    if result.cited:
        print("\nSources:")
        for c in result.cited:
            print(f"  [{c.context_id}] {c.citation:<32} {c.label}")
    if result.violations:
        print(f"\nViolations: {result.violations}")
    for p in result.prompts:
        version = f"v{p['version']}" if p["version"] else "git copy"
        print(f"\nPrompt: {p['name']} {p['label']} {version} ({p['source']}, {p['hash'][:12]})")
    cost = retrieval.embed_cost_usd + result.cost_usd
    llm = result.llm
    tokens = f"{llm.input_tokens} in / {llm.output_tokens} out" if llm else "no LLM call"
    print(
        f"Cost: ${cost:.6f} ({tokens}); latency: retrieve {retrieval.timings_ms['total']:.0f} ms"
        f", answer {result.timings_ms['total']:.0f} ms"
    )
    if trace_url:
        print(f"Trace: {trace_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
