"""Push the golden set (data/eval/golden.yaml) to the Langfuse dataset `dfsa-golden`.

One dataset item per golden turn; unchanged items are skipped, removed ones are
archived. e2e eval runs (`src.eval.run`) then link each turn's trace to its item.

    uv run python -m scripts.sync_dataset --dry-run
    uv run python -m scripts.sync_dataset
"""

import argparse
import sys

from src.config.settings import LangfuseSettings
from src.eval.dataset import DATASET, sync
from src.eval.golden import load_golden
from src.observability.langfuse import make_langfuse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="change nothing")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cfg = LangfuseSettings()
    if not cfg.active:
        print("FAIL: Langfuse is off (no keys, or LANGFUSE_ENABLED=false).")
        return 1
    golden = load_golden()
    client = make_langfuse(cfg)
    report = sync(client, golden, dry_run=args.dry_run)
    client.flush()
    prefix = "would be " if args.dry_run else ""
    counts = ", ".join(f"{len(v)} {prefix}{k}" for k, v in report.items())
    print(f"{DATASET} (golden {golden.hash[:12]}): {counts}")
    for kind in ("created", "updated", "archived"):
        for item in report[kind]:
            print(f"  {kind:<9} {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
