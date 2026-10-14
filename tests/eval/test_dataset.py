from types import SimpleNamespace

from src.eval.dataset import DATASET, RunLinker, items_for, sync
from src.eval.golden import GoldenSet

GOLDEN = GoldenSet.model_validate({"version": 1, "items": [
    {"id": "one", "kind": "single", "turns": [
        {"question": "What is a Retail Client?", "expect": "answer",
         "gold": ["GLO Retail Client"]}]},
    {"id": "conv", "kind": "multi_turn", "reviewed": True, "turns": [
        {"question": "Need a Client Agreement?", "expect": "answer", "gold": ["COB 3.3.2"]},
        {"question": "And with a Market Counterparty?", "expect": "answer",
         "gold": ["COB 3.3.1"]}]},
]})  # fmt: skip


class FakeDatasets:
    """Just enough of the Langfuse client for datasets, run items and scores."""

    def __init__(self):
        self.datasets: dict[str, dict] = {}
        self.writes: list[str] = []
        self.run_items: list[dict] = []
        self.scores: list[dict] = []
        self.api = SimpleNamespace(
            dataset_run_items=SimpleNamespace(create=lambda **kw: self.run_items.append(kw))
        )

    def get_dataset(self, name):
        if name not in self.datasets:
            raise RuntimeError("status_code: 404, body: Dataset not found")
        items = [SimpleNamespace(**it) for it in self.datasets[name].values()]
        return SimpleNamespace(items=items)

    def create_dataset(self, *, name, **_):
        self.datasets[name] = {}

    def create_dataset_item(self, *, dataset_name, id, status="ACTIVE", **fields):
        self.writes.append(id)
        self.datasets[dataset_name][id] = {"id": id, "status": status, **fields}

    def create_score(self, **kw):
        self.scores.append(kw)


def test_one_item_per_turn_with_the_conversation_so_far():
    items = {it["id"]: it for it in items_for(GOLDEN)}
    assert set(items) == {"golden-one-0", "golden-conv-0", "golden-conv-1"}
    follow_up = items["golden-conv-1"]
    assert follow_up["input"]["earlier_questions"] == ["Need a Client Agreement?"]
    assert follow_up["expected_output"]["gold"] == ["COB 3.3.1"]
    assert follow_up["metadata"]["reviewed"] is True


def test_sync_creates_then_skips_then_archives():
    fake = FakeDatasets()
    first = sync(fake, GOLDEN)
    assert len(first["created"]) == 3 and DATASET in fake.datasets

    fake.writes.clear()
    again = sync(fake, GOLDEN)
    assert len(again["unchanged"]) == 3 and fake.writes == []

    smaller = GoldenSet(version=1, items=GOLDEN.items[1:])  # also changes golden_hash
    third = sync(fake, smaller)
    assert third["archived"] == ["golden-one-0"] and len(third["unchanged"]) == 2
    assert fake.datasets[DATASET]["golden-one-0"]["status"] == "ARCHIVED"


def test_dry_run_writes_nothing():
    fake = FakeDatasets()
    report = sync(fake, GOLDEN, dry_run=True)
    assert len(report["created"]) == 3 and fake.datasets == {} and fake.writes == []


def test_linker_links_and_scores_and_survives_errors():
    fake = FakeDatasets()
    linker = RunLinker(fake, "hybrid_rerank-staging-run7", "", {"eval_run": 7})
    item = GOLDEN.items[1]
    scores = {"faithfulness": 0.8, "mrr": 1.0, "outcome": "correct_answer", "relevance": None}
    linker.link(item, 1, "trace-1", "obs-1", scores)
    assert fake.run_items[0]["dataset_item_id"] == "golden-conv-1"
    assert fake.run_items[0]["run_name"] == "hybrid_rerank-staging-run7"
    assert {s["name"]: s["value"] for s in fake.scores} == {
        "faithfulness": 0.8,
        "mrr": 1.0,
        "refusal_correct": 1.0,
    }

    def down(**kw):
        raise RuntimeError("down")

    fake.api.dataset_run_items.create = down
    linker.link(item, 0, "trace-2", None, {})
    assert linker.failures == 1
