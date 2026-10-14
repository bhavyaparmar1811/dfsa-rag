"""The golden set as a Langfuse dataset, and e2e eval runs linked to it.

    uv run python -m scripts.sync_dataset [--dry-run]

Dataset `dfsa-golden` holds one item per golden **turn** (id
`golden-<item id>-<turn>`), so every turn's trace links to its own item:

    input            {"question", "earlier_questions"}  (the conversation so far)
    expected_output  {"expect", "gold", "supporting"}
    metadata         {"item", "turn", "kind", "module", "reviewed", "golden_hash", "notes"}

Sync upserts by id and skips unchanged items; items no longer in the golden set
are archived, not deleted, so old runs keep their links.

An e2e run links each turn's trace to its item under the run name
`<config>-<label>-run<id>` (what `run_experiment` does internally:
`api.dataset_run_items.create`) and posts the turn's scores on the trace.
SQLite stays the record of truth: the gate never reads Langfuse, and every
call here is best effort (logged, never raised).
"""

import logging

from langfuse import Langfuse

from src.eval.golden import GoldenSet, Item

logger = logging.getLogger(__name__)

DATASET = "dfsa-golden"
SCORES = ("faithfulness", "relevance", "recall@6", "mrr", "citation_precision")


def item_id(golden_item: str, turn: int) -> str:
    return f"golden-{golden_item}-{turn}"


def items_for(golden: GoldenSet) -> list[dict]:
    """The dataset items for the golden set, one per turn."""
    out = []
    for item in golden.items:
        for i, turn in enumerate(item.turns):
            out.append({
                "id": item_id(item.id, i),
                "input": {"question": turn.question,
                          "earlier_questions": [t.question for t in item.turns[:i]]},
                "expected_output": {"expect": turn.expect, "gold": turn.gold,
                                    "supporting": turn.supporting},
                "metadata": {"item": item.id, "turn": i, "kind": item.kind,
                             "module": item.module, "reviewed": item.reviewed,
                             "golden_hash": golden.hash, "notes": turn.notes},
            })  # fmt: skip
    return out


def _existing(client: Langfuse) -> dict[str, object] | None:
    """Current items by id, or None if the dataset doesn't exist yet."""
    try:
        dataset = client.get_dataset(DATASET)
    except Exception as e:  # the SDK raises its own NotFoundError (404)
        if "404" in str(e) or "not found" in str(e).lower():
            return None
        raise
    return {it.id: it for it in dataset.items}


def sync(client: Langfuse, golden: GoldenSet, *, dry_run: bool = False) -> dict[str, list[str]]:
    """Upsert the golden turns as dataset items; archive the ones that left."""
    existing = _existing(client)
    if existing is None and not dry_run:
        client.create_dataset(
            name=DATASET,
            description="Golden Q&A set (data/eval/golden.yaml): one item per turn.",
            metadata={"source": "data/eval/golden.yaml"},
        )
    existing = existing or {}
    wanted = items_for(golden)
    report: dict[str, list[str]] = {"created": [], "updated": [], "unchanged": [], "archived": []}
    for it in wanted:
        old = existing.get(it["id"])
        if old is None:
            status = "created"
        elif (old.input, old.expected_output, _meta(old.metadata)) == (
            it["input"], it["expected_output"], _meta(it["metadata"]),
        ) and str(getattr(old, "status", "ACTIVE")).upper().endswith("ACTIVE"):  # fmt: skip
            report["unchanged"].append(it["id"])
            continue
        else:
            status = "updated"
        report[status].append(it["id"])
        if not dry_run:
            client.create_dataset_item(dataset_name=DATASET, **it)
    for gone in sorted(set(existing) - {it["id"] for it in wanted}):
        old = existing[gone]
        if str(getattr(old, "status", "")).upper().endswith("ARCHIVED"):
            continue
        report["archived"].append(gone)
        if not dry_run:
            client.create_dataset_item(dataset_name=DATASET, id=gone, input=old.input,
                                       expected_output=old.expected_output,
                                       metadata=old.metadata, status="ARCHIVED")  # fmt: skip
    return report


def _meta(metadata) -> dict:
    """golden_hash changes whenever any item changes; compare the item itself."""
    return {k: v for k, v in (metadata or {}).items() if k != "golden_hash"}


class RunLinker:
    """Links the traces of one e2e eval run to the dataset items."""

    def __init__(self, client: Langfuse, run_name: str, description: str, metadata: dict):
        self.client = client
        self.run_name = run_name
        self.description = description
        self.metadata = metadata
        self.failures = 0

    def link(self, item: Item, turn: int, trace_id: str, observation_id: str | None,
             scores: dict) -> None:  # fmt: skip
        try:
            self.client.api.dataset_run_items.create(
                run_name=self.run_name,
                run_description=self.description,
                metadata=self.metadata,
                dataset_item_id=item_id(item.id, turn),
                trace_id=trace_id,
                observation_id=observation_id,
            )
            for name in SCORES:
                if isinstance(scores.get(name), int | float):
                    self.client.create_score(name=name, value=float(scores[name]),
                                             trace_id=trace_id)  # fmt: skip
            if outcome := scores.get("outcome"):
                self.client.create_score(
                    name="refusal_correct", trace_id=trace_id, data_type="BOOLEAN",
                    value=1.0 if outcome in ("correct_answer", "correct_refusal") else 0.0,
                    comment=outcome,
                )  # fmt: skip
        except Exception as e:  # tracing is optional; the eval is in SQLite
            self.failures += 1
            if self.failures <= 3:
                logger.warning("could not link %s[%d] to dataset run %s: %s",
                               item.id, turn, self.run_name, e)  # fmt: skip
