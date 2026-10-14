"""Run the golden set through the pipeline and record the scores.

    # compare retrieval configs: no answers, no judges (~$0.0001 in query embeddings)
    uv run python -m src.eval.run --retrieval-only \
        --configs dense,sparse,hybrid,hybrid_rerank,hybrid_rerank_base,hybrid_rerank_minilm

    # end to end on production: rewrite -> retrieve -> answer -> judges -> gate (~$2)
    uv run python -m src.eval.run
    uv run python -m src.eval.run --label staging         # a prompt candidate
    uv run python -m src.eval.run --items aml-cdd-timing --include-unreviewed --no-judges
    uv run python -m src.eval.run --dry-run               # what it would run, and the cost

Each config gets one `eval_runs` row and one `eval_results` row per golden turn,
written as it goes, so an interrupted run keeps what it did. Eval turns never
touch `conversations`, `messages` or `traces`: `/stats/cost` stays production
only, and eval cost is recorded on the run.

A turn is what `ChatService.send` does (rewrite with the history, retrieve,
answer), but the history is kept in memory: the system's own earlier answers
in that golden conversation, so a follow-up tests real rewriting. In
retrieval-only mode there are no answers, so a follow-up is rewritten from the
earlier questions alone, once per invocation, and shared by every config.

An e2e run is gated (src/eval/gate.py) when it ends; the exit code is 1 if the
gate fails.
"""

import argparse
import gc
import logging
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace

from langfuse import propagate_attributes
from sqlalchemy import Engine

from src.chat.rewrite import QueryRewriter, Rewrite, render_history
from src.config.settings import (
    PROJECT_ROOT,
    AzureOpenAISettings,
    ChatSettings,
    EmbeddingSettings,
    LangfuseSettings,
    PathSettings,
    QdrantSettings,
    RetrievalSettings,
)
from src.embeddings.client import EmbeddingClient
from src.embeddings.sparse import SparseEncoder
from src.eval import gate
from src.eval import metrics as M
from src.eval.configs import CONFIGS, PRODUCTION_CONFIG, settings_for
from src.eval.dataset import RunLinker
from src.eval.golden import GoldenSet, Item, Turn, load_golden, provision
from src.eval.judges import FAITHFULNESS, RELEVANCE, Judge
from src.generation.answer import ANSWER_PROMPT, REFUSAL_PROMPT, Answerer
from src.generation.llm import LLMClient
from src.observability.langfuse import get_langfuse
from src.prompts.registry import PRODUCTION, PromptRegistry
from src.retrieval.pipeline import Retriever
from src.retrieval.rerank import CrossEncoderModel, ScoringModel
from src.storage.db import assert_migrated, get_engine, session_scope
from src.storage.models import EvalResult, EvalRun
from src.storage.qdrant_store import QdrantStore
from src.storage.repositories import EvalRunRepository

logger = logging.getLogger(__name__)

REWRITE_PROMPT = "query-rewrite"
RANKED_KEEP = 40  # provisions stored per turn, with their scores
# Rough per-turn cost, for --dry-run only: answer ~$0.017, judges ~$0.01, rewrite.
E2E_COST_PER_TURN = 0.03
EMBED_COST_PER_TURN = 0.000002
BY_KIND_KEYS = (
    "turns",
    "recall@6",
    "mrr",
    "ndcg@6",
    "context_recall",
    "faithfulness",
    "relevance",
    "false_refusal_rate",
    "refusal_recall",
)


@dataclass
class Components:
    """What every run shares: built once per invocation."""

    engine: Engine
    store: QdrantStore
    embedder: EmbeddingClient | None = None
    sparse: SparseEncoder | None = None
    llm: LLMClient | None = None  # None: retrieval only, follow-ups not rewritten
    registry: PromptRegistry | None = None
    scorer_factory: Callable[[RetrievalSettings], ScoringModel] = CrossEncoderModel
    link_dataset: bool = False  # link e2e traces to the Langfuse dataset (Langfuse on)


class EvalRunner:
    def __init__(
        self,
        parts: Components,
        golden: GoldenSet,
        items: list[Item],
        *,
        reviewed_only: bool = True,
        chat: ChatSettings | None = None,
    ):
        self.parts = parts
        self.golden = golden
        self.items = items
        self.reviewed_only = reviewed_only
        self.chat = chat or ChatSettings()
        self._scorer: ScoringModel | None = None
        self._scorer_name: str | None = None
        self._standalone: dict[tuple[str, int], Rewrite] = {}
        self._linker: RunLinker | None = None

    # --- one config -------------------------------------------------------------

    def run(
        self,
        name: str,
        cfg: RetrievalSettings,
        *,
        e2e: bool,
        label: str | None = None,
        judges: bool = True,
        judge_label: str = PRODUCTION,
        notes: str | None = None,
    ) -> EvalRun:
        p = self.parts
        if e2e and (p.llm is None or p.registry is None):
            raise ValueError("an e2e run needs the LLM client and the prompt registry")
        retriever = self._retriever(cfg)
        rewriter = QueryRewriter(p.llm, p.registry) if p.llm else None
        answerer = Answerer(p.llm, p.registry, retriever.citations.find) if e2e else None
        judge = Judge(p.llm, p.registry, judge_label) if e2e and judges else None

        prompts = []
        if e2e:
            prompts = [
                p.registry.get(n, label).trace()
                for n in (ANSWER_PROMPT, REFUSAL_PROMPT, REWRITE_PROMPT)
            ]
            if judge:
                prompts += [
                    p.registry.get(n, judge_label).trace() for n in (FAITHFULNESS, RELEVANCE)
                ]
        with session_scope(p.engine) as s:
            run = EvalRunRepository(s).start(
                EvalRun(
                    kind="e2e" if e2e else "retrieval",
                    config_name=name,
                    retrieval_config=cfg.model_dump(mode="json", exclude={"skip_terms"}),
                    prompts=prompts,
                    prompt_label=(label or PRODUCTION) if e2e else None,
                    golden_hash=self.golden.hash,
                    reviewed_only=self.reviewed_only,
                    n_items=len(self.items),
                    n_turns=sum(len(i.turns) for i in self.items),
                    git_commit=git_commit(),
                    notes=notes,
                )
            )
            run_id = run.id
            self._linker = None
            if e2e and p.link_dataset:
                run.langfuse_run = f"{name}-{label or PRODUCTION}-run{run_id}"
                self._linker = RunLinker(
                    get_langfuse(),
                    run.langfuse_run,
                    notes or "",
                    {"eval_run": run_id, "config": name, "golden_hash": self.golden.hash[:12]},
                )
        logger.info(
            "eval run #%d: %s, %s, %d items",
            run_id,
            name,
            "e2e" if e2e else "retrieval",
            len(self.items),
        )

        rows, costs, runtime = [], Counter(), {}
        try:
            for item in self.items:
                history: list[SimpleNamespace] = []
                for i, turn in enumerate(item.turns):
                    result, cost, config = self._turn(
                        run_id,
                        item,
                        i,
                        turn,
                        history,
                        retriever,
                        rewriter,
                        answerer,
                        judge,
                        e2e=e2e,
                        label=label,
                    )
                    runtime = runtime or config
                    costs.update(cost)
                    with session_scope(p.engine) as s:
                        EvalRunRepository(s).add_result(result)
                    rows.append(_row(result, item))
                    _log_turn(result)
        except BaseException as e:  # Ctrl-C too: the run is marked, results are kept
            with session_scope(p.engine) as s:
                EvalRunRepository(s).finish(
                    run_id,
                    metrics=M.aggregate(rows),
                    cost_usd=_costs(costs),
                    error=f"{type(e).__name__}: {e}"[:2000],
                )
            raise

        summary = M.aggregate(rows)
        summary["by_kind"] = _by_kind(rows)
        with session_scope(p.engine) as s:
            repo = EvalRunRepository(s)
            repo.finish(run_id, metrics=summary, cost_usd=_costs(costs))
            if runtime:  # reassigned: SQLAlchemy doesn't see in-place JSON changes
                run = repo.get(run_id)
                run.retrieval_config = {**run.retrieval_config, **runtime}
        if e2e:
            gate.apply(p.engine, run_id)
        with session_scope(p.engine) as s:
            return EvalRunRepository(s).get(run_id)

    # --- one turn ---------------------------------------------------------------

    def _turn(
        self,
        run_id,
        item: Item,
        i: int,
        turn: Turn,
        history,
        retriever,
        rewriter,
        answerer,
        judge,
        *,
        e2e: bool,
        label: str | None,
    ) -> tuple[EvalResult, Counter, dict]:
        start = time.perf_counter()
        result = EvalResult(
            run_id=run_id,
            item_id=item.id,
            turn=i,
            question=turn.question,
            expect=turn.expect,
            ranked=[],
            cited=[],
            metrics={},
            latency_ms={},
            cost_usd=0.0,
        )
        cost, latency, runtime = Counter(), {}, {}
        hist = render_history(history[-self.chat.history_turns :], self.chat.history_max_chars)
        langfuse = get_langfuse()
        with ExitStack() as stack:
            root = None
            if e2e:
                result.trace_id = langfuse.create_trace_id()
                root = stack.enter_context(
                    langfuse.start_as_current_observation(
                        name="eval-turn",
                        as_type="chain",
                        trace_context={"trace_id": result.trace_id},
                        input={"question": turn.question, "item": item.id, "turn": i},
                    )
                )
                stack.enter_context(
                    propagate_attributes(
                        session_id=f"eval-{run_id}-{item.id}", trace_name="eval-turn", tags=["eval"]
                    )
                )
            try:
                t = time.perf_counter()
                if e2e:
                    rewrite = rewriter.rewrite(turn.question, hist, label)
                else:
                    rewrite = self._standalone_query(item, i, rewriter, label)
                cost["rewrite"] += rewrite.cost_usd
                latency["rewrite"] = _ms(t)
                if rewrite.changed:
                    result.rewritten_query = rewrite.question

                retrieval = retriever.retrieve(rewrite.question, item.filters)
                runtime = {
                    k: retrieval.config.get(k)
                    for k in ("reranker_device", "reranker_dtype", "reranker_revision")
                }
                cost["embed"] += retrieval.embed_cost_usd
                latency["retrieve"] = retrieval.timings_ms.get("total")
                if "rerank" in retrieval.timings_ms:
                    latency["rerank"] = retrieval.timings_ms["rerank"]
                result.confidence = retrieval.confidence
                result.ranked = _ranked(retrieval)
                ranked = M.ranked_provisions(retrieval)
                metrics = M.retrieval_scores(ranked, M.context_provisions(retrieval), turn)
                metrics["truncated"] = retrieval.truncated

                if e2e:
                    t = time.perf_counter()
                    answer = answerer.answer(retrieval, history=hist, label=label)
                    latency["answer"] = _ms(t)
                    cost["answer"] += answer.cost_usd
                    result.status, result.answer = answer.status, answer.text
                    result.refusal_code = answer.refusal.code if answer.refusal else None
                    result.cited = M.dedupe(c.key for c in answer.cited)
                    metrics["outcome"] = M.refusal_outcome(turn.expect, answer.refused)
                    latency["turn"] = _ms(start)
                    if not answer.refused:
                        metrics |= M.citation_scores(result.cited, turn)
                        if judge:
                            t = time.perf_counter()
                            metrics |= self._judge(
                                judge, rewrite.question, retrieval, answer, result, cost
                            )
                            latency["judge"] = _ms(t)
                    history.extend(
                        [
                            SimpleNamespace(role="user", content=turn.question),
                            SimpleNamespace(role="assistant", content=answer.text),
                        ]
                    )
                    root.update(
                        output={"status": answer.status, "text": answer.text},
                        metadata={"run": run_id, "outcome": metrics["outcome"]},
                    )
                    if self._linker:
                        self._linker.link(item, i, result.trace_id, root.id, metrics)
                else:
                    history.append(SimpleNamespace(role="user", content=turn.question))
                    latency["turn"] = _ms(start)
                result.metrics = metrics
            except Exception as e:
                logger.exception("eval turn %s[%d] failed", item.id, i)
                result.status = "error"
                result.metrics = {"error": f"{type(e).__name__}: {e}"[:500]}
                latency["turn"] = _ms(start)
        result.latency_ms = latency
        result.cost_usd = round(sum(cost.values()), 6)
        return result, cost, runtime

    def _judge(self, judge: Judge, question, retrieval, answer, result, cost) -> dict:
        f = judge.faithfulness(question, retrieval.context, answer.text)
        r = judge.relevance(question, answer.text)
        cost["judge"] += f.cost_usd + r.cost_usd
        result.judges = {"faithfulness": f.trace(), "relevance": r.trace()}
        out = {}
        if f.score is not None:
            out["faithfulness"], out["contradicted"] = f.score, f.contradicted
        if r.score is not None:
            out["relevance"] = r.score
        if errors := [f"{j.name}: {j.error}" for j in (f, r) if j.error]:
            out["judge_error"] = "; ".join(errors)
        return out

    def _standalone_query(self, item: Item, i: int, rewriter, label) -> Rewrite:
        """Retrieval-only: a follow-up rewritten from the earlier questions, once."""
        turn = item.turns[i]
        if i == 0 or rewriter is None:
            if i and rewriter is None:
                logger.warning("%s[%d]: no LLM, follow-up not rewritten", item.id, i)
            return Rewrite(original=turn.question, question=turn.question)
        key = (item.id, i)
        if key in self._standalone:  # shared by every config: free after the first
            cached = self._standalone[key]
            return Rewrite(cached.original, cached.question, cached.changed)
        earlier = [SimpleNamespace(role="user", content=t.question) for t in item.turns[:i]]
        hist = render_history(earlier, self.chat.history_max_chars)
        self._standalone[key] = rewriter.rewrite(turn.question, hist, label)
        return self._standalone[key]

    # --- components ---------------------------------------------------------------

    def _retriever(self, cfg: RetrievalSettings) -> Retriever:
        p = self.parts
        return Retriever.build(
            cfg,
            engine=p.engine,
            store=p.store,
            embedder=p.embedder if cfg.mode in ("dense", "hybrid") else None,
            sparse=p.sparse if cfg.mode in ("sparse", "hybrid") else None,
            scorer=self._scorer_for(cfg) if cfg.rerank else None,
        )

    def _scorer_for(self, cfg: RetrievalSettings) -> ScoringModel:
        """One cross-encoder in memory at a time (the GPU has 4 GB)."""
        if self._scorer_name != cfg.reranker_model:
            self.release()
            self._scorer = self.parts.scorer_factory(cfg)
            self._scorer_name = cfg.reranker_model
        return self._scorer

    def release(self) -> None:
        if self._scorer is None:
            return
        self._scorer = self._scorer_name = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


# --- helpers ----------------------------------------------------------------------


def _ms(t: float) -> float:
    return round((time.perf_counter() - t) * 1000, 1)


def _ranked(retrieval) -> list[dict]:
    """The scored ranking, by provision: what calibration and audits read."""
    hit_keys = {c.key for c in retrieval.hits}
    out, seen = [], set()
    for c in [*retrieval.hits, *retrieval.candidates]:
        prov = provision(c.key)
        if prov in seen:
            continue
        seen.add(prov)
        out.append(
            {
                "provision": prov,
                "key": c.key,
                "source": c.source,
                "hit": c.key in hit_keys,
                "rerank": c.rerank_score,
                "fused": c.fused_score,
                "dense_rank": c.dense_rank,
                "sparse_rank": c.sparse_rank,
            }
        )
        if len(out) == RANKED_KEEP:
            break
    return out


def _row(result: EvalResult, item: Item) -> dict:
    return result.metrics | {
        "kind": item.kind,
        "refusal_code": result.refusal_code,
        "cost_usd": result.cost_usd,
        "latency_ms": result.latency_ms,
    }


def _by_kind(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[r["kind"]].append(r)
    return {
        kind: {k: v for k, v in M.aggregate(rs).items() if k in BY_KIND_KEYS}
        for kind, rs in sorted(groups.items())
    }


def _costs(costs: Counter) -> dict:
    out = {k: round(costs.get(k, 0.0), 6) for k in ("embed", "rewrite", "answer", "judge")}
    out["total"] = round(sum(out.values()), 6)
    return out


def _log_turn(r: EvalResult) -> None:
    m = r.metrics
    if "error" in m:
        logger.warning("  %s[%d] ERROR %s", r.item_id, r.turn, m["error"])
        return
    parts = [f"{r.item_id}[{r.turn}]"]
    if "mrr" in m:
        parts.append(f"mrr={m['mrr']:.2f} r@6={m['recall@6']:.2f}")
    if r.confidence is not None:
        parts.append(f"conf={r.confidence:.3f}")
    if "outcome" in m:
        parts.append(m["outcome"])
    if "faithfulness" in m:
        parts.append(f"faith={m['faithfulness']:.2f}")
    logger.info("  %s", " ".join(parts))


def git_commit() -> str | None:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return git("rev-parse", "--short", "HEAD") + (
            "-dirty" if git("status", "--porcelain") else ""
        )
    except (OSError, subprocess.CalledProcessError):
        return None


# --- report -----------------------------------------------------------------------


def _f(value, digits: int = 3) -> str:
    if value is None:
        return "–"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def report(runs: list[EvalRun]) -> str:
    """A Markdown comparison of runs: retrieval metrics, then answers if any."""
    lines = [
        "| Run | Config | r@1 | r@3 | r@6 | r@20 | r@40 | MRR | nDCG@6 | ctx recall "
        "| retrieve p50 / p95 ms | errors |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        m = r.metrics or {}
        lat = (m.get("latency_ms") or {}).get("retrieve", {})
        lines.append(
            f"| #{r.id} | {r.config_name} | "
            + " | ".join(
                _f(m.get(k))
                for k in (
                    "recall@1",
                    "recall@3",
                    "recall@6",
                    "recall@20",
                    "recall@40",
                    "mrr",
                    "ndcg@6",
                    "context_recall",
                )
            )
            + f" | {_f(lat.get('p50'), 0)} / {_f(lat.get('p95'), 0)} | {m.get('errors', 0)} |"
        )
    e2e = [r for r in runs if r.kind == "e2e"]
    if e2e:
        lines += [
            "",
            "| Run | Config | Label | Faithfulness | Relevance | Cit. precision "
            "| Cit. recall | False refusals | Refusal recall | Validator fails "
            "| Contradicted | Turn p50 / p95 ms | $ | Gate |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in e2e:
            m = r.metrics or {}
            lat = (m.get("latency_ms") or {}).get("turn", {})
            lines.append(
                f"| #{r.id} | {r.config_name} | {r.prompt_label} | {_f(m.get('faithfulness'))} "
                f"| {_f(m.get('relevance'), 2)} | {_f(m.get('citation_precision'))} "
                f"| {_f(m.get('citation_recall'))} | {_f(m.get('false_refusal_rate'))} "
                f"| {_f(m.get('refusal_recall'))} | {_f(m.get('validator_failure_rate'))} "
                f"| {m.get('contradicted_claims', '–')} "
                f"| {_f(lat.get('p50'), 0)} / {_f(lat.get('p95'), 0)} "
                f"| {_f((r.cost_usd or {}).get('total'), 4)} | {r.gate_status} |"
            )
    kinds = sorted({k for r in runs for k in (r.metrics or {}).get("by_kind", {})})
    if kinds:
        lines += [
            "",
            "By kind (r@6 / MRR / ctx recall; e2e: faithfulness, false refusals, refusal recall):",
            "",
            "| Run | " + " | ".join(kinds) + " |",
            "|---|" + "---|" * len(kinds),
        ]
        for r in runs:
            cells = []
            for k in kinds:
                b = (r.metrics or {}).get("by_kind", {}).get(k, {})
                cell = (
                    f"{_f(b.get('recall@6'), 2)} / {_f(b.get('mrr'), 2)} / "
                    f"{_f(b.get('context_recall'), 2)}"
                )
                if r.kind == "e2e":
                    cell += (
                        f"; {_f(b.get('faithfulness'), 2)}, {_f(b.get('false_refusal_rate'), 2)}"
                        f", {_f(b.get('refusal_recall'), 2)}"
                    )
                cells.append(cell)
            lines.append(f"| #{r.id} {r.config_name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --- CLI ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate the pipeline on the golden set.")
    ap.add_argument(
        "--configs", default=PRODUCTION_CONFIG, help=f"comma-separated, from: {', '.join(CONFIGS)}"
    )
    ap.add_argument("--retrieval-only", action="store_true", help="no answers, no judges")
    ap.add_argument("--label", help="prompt label for answer/refusal/rewrite (default production)")
    ap.add_argument("--items", help="comma-separated golden item ids")
    ap.add_argument("--limit", type=int, help="only the first N items")
    ap.add_argument("--no-judges", action="store_true")
    ap.add_argument(
        "--judge-label",
        default=PRODUCTION,
        help="prompt label of the judges (staging only to evaluate a judge edit)",
    )
    ap.add_argument(
        "--include-unreviewed",
        action="store_true",
        help="also draft items (such a run never gates a release)",
    )
    ap.add_argument("--notes")
    ap.add_argument("--dry-run", action="store_true", help="show what would run and its cost")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in (
        "httpx",
        "sentence_transformers",
        "openai",
        "langfuse",
        "src.retrieval",
        "src.embeddings",
        "src.generation",
        "src.chat",
        "src.prompts",
        "alembic",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Truncated rerank inputs are counted per turn (metrics "truncated") instead.
    logging.getLogger("src.retrieval.rerank").setLevel(logging.ERROR)
    # httpx's per-request INFO lines get through the level above (a library resets it).
    for handler in logging.getLogger().handlers:
        handler.addFilter(lambda r: r.levelno >= logging.WARNING or not r.name.startswith("httpx"))
    try:  # the reranker's "Loading weights" progress bar
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
    except ImportError:
        pass

    names = [n.strip() for n in args.configs.split(",") if n.strip()]
    base = RetrievalSettings()
    configs = {n: settings_for(n, base) for n in names}
    golden = load_golden()
    ids = [i.strip() for i in args.items.split(",")] if args.items else None
    items = golden.select(include_unreviewed=args.include_unreviewed, ids=ids)[: args.limit]
    if not items:
        print("No golden items selected: none are reviewed yet? Use --include-unreviewed.")
        return 1
    turns = sum(len(i.turns) for i in items)
    e2e = not args.retrieval_only
    per_turn = E2E_COST_PER_TURN if e2e else EMBED_COST_PER_TURN
    print(
        f"{len(items)} items, {turns} turns x {len(configs)} config(s) "
        f"({'e2e' if e2e else 'retrieval only'}), about ${turns * len(configs) * per_turn:.4f}"
        f"{'' if args.include_unreviewed else ', reviewed items only'}"
    )
    if args.dry_run:
        return 0

    engine = get_engine()
    assert_migrated(engine)
    emb, qdrant = EmbeddingSettings(), QdrantSettings()
    modes = {c.mode for c in configs.values()}
    followups = any(len(i.turns) > 1 for i in items)
    llm = registry = None
    if e2e or followups:
        llm, registry = LLMClient(AzureOpenAISettings()), PromptRegistry.from_settings()
    parts = Components(
        engine=engine,
        store=QdrantStore.from_settings(qdrant, emb),
        embedder=EmbeddingClient(emb) if modes & {"dense", "hybrid"} else None,
        sparse=SparseEncoder(qdrant) if modes & {"sparse", "hybrid"} else None,
        llm=llm,
        registry=registry,
        link_dataset=LangfuseSettings().active,
    )
    runner = EvalRunner(parts, golden, items, reviewed_only=not args.include_unreviewed)
    runs = []
    try:
        for name, cfg in configs.items():
            runs.append(
                runner.run(
                    name,
                    cfg,
                    e2e=e2e,
                    label=args.label,
                    judges=not args.no_judges,
                    judge_label=args.judge_label,
                    notes=args.notes,
                )
            )
    finally:
        runner.release()
        get_langfuse().flush()

    text = report(runs)
    print()
    print(text)
    out_dir = PathSettings().eval_reports
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{'-'.join(names)}.md"
    path.write_text(
        f"# Eval runs {', '.join(f'#{r.id}' for r in runs)}\n\n"
        f"golden {golden.hash[:12]}, {len(items)} items, {turns} turns\n\n{text}\n",
        encoding="utf-8",
    )
    print(f"\nReport: {path.relative_to(PROJECT_ROOT)}")
    for r in runs:
        if r.kind == "e2e":
            print()
            print(
                gate.GateReport(
                    r.id,
                    [gate.Check(**c) for c in r.gate_report["checks"]],
                    r.gate_report.get("baseline_run"),
                ).render()
            )
    return 1 if any(r.kind == "e2e" and r.gate_status != "passed" for r in runs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
