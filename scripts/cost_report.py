"""Cost report over all spend: production requests, eval runs and ingestion.

    uv run python -m scripts.cost_report                    # everything recorded
    uv run python -m scripts.cost_report --days 30          # the last 30 UTC days, today included
    uv run python -m scripts.cost_report --since 2026-10-01 --until 2026-10-14
    uv run python -m scripts.cost_report --json

Production numbers come from the `traces` table (the same as GET /stats/cost),
including the semantic cache's hit rate and what its hits saved;
eval runs from `eval_runs`, ingestion from `ingestion_runs`. Spend is split by
provider: OpenAI for embeddings, Azure OpenAI for every GPT-4o call (rewrite,
answer, judges). The output is Markdown, so the tables can be pasted as they are.
"""

import argparse
import json
import sys
from datetime import UTC, date, datetime, timedelta

from sqlalchemy import Engine

from src.storage.db import get_engine, session_scope
from src.storage.repositories import (
    EvalRunRepository,
    IngestionRunRepository,
    TraceRepository,
)

# Eval-run stages billed by OpenAI; every other stage is a GPT-4o call on Azure.
OPENAI_STAGES = {"embed"}


def build_report(engine: Engine, since: date | None = None, until: date | None = None) -> dict:
    with session_scope(engine) as s:
        production = TraceRepository(s).cost_stats(since, until)
        evals = EvalRunRepository(s).cost_summary(since, until)
        ingestion = IngestionRunRepository(s).cost_summary(since, until)

    eval_openai = sum(
        v for k in evals["by_kind"].values()
        for stage, v in k["cost_usd"].items() if stage in OPENAI_STAGES
    )  # fmt: skip
    openai = production["cost_by_provider_usd"]["openai_embeddings"] + eval_openai
    openai += ingestion["cost_usd"]
    azure = production["cost_by_provider_usd"]["azure_generation"]
    azure += evals["cost_usd"] - eval_openai
    total = production["cost_usd"]["total"] + evals["cost_usd"] + ingestion["cost_usd"]
    return {
        "since": production["since"],
        "until": production["until"],
        "production": production,
        "evals": evals,
        "ingestion": ingestion,
        "total_usd": {
            "production": production["cost_usd"]["total"],
            "evals": evals["cost_usd"],
            "ingestion": ingestion["cost_usd"],
            "openai_embeddings": round(openai, 6),
            "azure_generation": round(azure, 6),
            "total": round(total, 6),
        },
    }


def render(report: dict) -> str:
    p, e, i, t = report["production"], report["evals"], report["ingestion"], report["total_usd"]
    window = f"{report['since'] or 'the first record'} to {report['until'] or 'now'}"
    lines = [f"# Cost report: {window} (UTC)", ""]

    lines += ["## Production requests", ""]
    if p["requests"]:
        gate = p["gate_refusals"]
        lines += [
            "| Requests | Refused (gate / after the LLM) | Errors | Refusal rate | Total | Per request | Latency p50 / p95 |",  # noqa: E501
            "|---|---|---|---|---|---|---|",
            f"| {p['requests']} | {p['refused']} ({gate} / {p['refused'] - gate}) | {p['errors']} "
            f"| {_pct(p['refusal_rate'])} | {_usd(p['cost_usd']['total'])} "
            f"| {_usd(p['cost_per_request_usd'])} | {_ms(p['latency_ms']['p50'])} / "
            f"{_ms(p['latency_ms']['p95'])} |",
            "",
            "| Stage | Provider | Tokens | Cost |",
            "|---|---|---|---|",
            f"| Query embedding | OpenAI | {p['tokens']['embedding']:,} "
            f"| {_usd(p['cost_usd']['embedding'])} |",
            f"| Query rewrite | Azure OpenAI | {p['tokens']['rewrite_input']:,} in / "
            f"{p['tokens']['rewrite_output']:,} out | {_usd(p['cost_usd']['rewrite'])} |",
            f"| Answer | Azure OpenAI | {p['tokens']['answer_input']:,} in / "
            f"{p['tokens']['answer_output']:,} out | {_usd(p['cost_usd']['answer'])} |",
            "",
        ]
        c = p["cache"]
        if c["lookups"]:
            lines += [
                f"Semantic cache: {c['hits']} hits of {c['lookups']} lookups "
                f"({_pct(c['hit_rate'])}; {c['exact_hits']} exact repeats), "
                f"saving {_usd(c['saved_usd'])}; a hit takes {_ms(c['hit_latency_p50_ms'])} (p50).",
                "",
            ]
        if p["refusals_by_code"]:
            codes = ", ".join(f"{c} {n}" for c, n in p["refusals_by_code"].items())
            lines += [f"Refusals by reason: {codes}.", ""]
        lines += [
            "| Day | Requests | Refusal rate | Cache hit rate | Cost | Per request |",
            "|---|---|---|---|---|---|",
            *(
                f"| {d['day']} | {d['requests']} | {_pct(d['refusal_rate'])} "
                f"| {_pct(d['cache']['hit_rate'])} | {_usd(d['cost_usd']['total'])} "
                f"| {_usd(d['cost_per_request_usd'])} |"
                for d in p["by_day"]
            ),
            "",
        ]
    else:
        lines += ["No production requests in this window.", ""]

    lines += ["## Eval runs", ""]
    if e["by_kind"]:
        lines += ["| Kind | Runs | Turns | Cost by stage | Total |", "|---|---|---|---|---|"]
        for kind, k in e["by_kind"].items():
            stages = ", ".join(
                f"{st} {_usd(v)}" for st, v in k["cost_usd"].items() if st != "total" and v
            )
            lines.append(
                f"| {kind} | {k['runs']} | {k['turns']} | {stages or '-'} "
                f"| {_usd(k['cost_usd']['total'])} |"
            )
        lines.append("")
    else:
        lines += ["No eval runs in this window.", ""]

    lines += ["## Ingestion", ""]
    lines += [
        "| Runs | Failed | Embedding tokens | Cost |",
        "|---|---|---|---|",
        f"| {i['runs']} | {i['failed']} | {i['embed_tokens']:,} | {_usd(i['cost_usd'])} |",
        "",
        "## Total",
        "",
        "| Production | Evals | Ingestion | OpenAI (embeddings) | Azure OpenAI (GPT-4o) | Total |",
        "|---|---|---|---|---|---|",
        f"| {_usd(t['production'])} | {_usd(t['evals'])} | {_usd(t['ingestion'])} "
        f"| {_usd(t['openai_embeddings'])} | {_usd(t['azure_generation'])} "
        f"| **{_usd(t['total'])}** |",
    ]
    return "\n".join(lines)


def _usd(value: float | None) -> str:
    if value is None:
        return "-"
    if 0 < value < 0.0001:  # a few query embeddings
        return f"${value:.6f}"
    return f"${value:.4f}" if value < 1 else f"${value:,.2f}"


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value / 1000:.1f} s"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--since", type=date.fromisoformat, help="first UTC day (YYYY-MM-DD)")
    parser.add_argument("--until", type=date.fromisoformat, help="last UTC day, inclusive")
    parser.add_argument("--days", type=int, help="the last N UTC days, today included")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    if args.days is not None and args.since is not None:
        parser.error("use --days or --since, not both")
    since = args.since
    if args.days is not None:
        if args.days < 1:
            parser.error("--days must be at least 1")
        since = datetime.now(UTC).date() - timedelta(days=args.days - 1)
    if since and args.until and since > args.until:
        parser.error("--since is after --until")

    report = build_report(get_engine(), since, args.until)
    print(json.dumps(report, indent=2) if args.json else render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
