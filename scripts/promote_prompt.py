"""Move a prompt's `production` label in Langfuse to a tested version.

The version (default: the one labelled `staging`) must have exactly the content
of the committed git file, so production is always something in git. It also
has to pass the eval gate: the newest e2e eval run that used exactly this
content, on the current golden set, must have passed eval/thresholds.yaml
(src/eval/gate.py). --skip-gate REASON overrides it; the reason is printed so it
ends up in the terminal log.

    uv run python -m src.eval.run --label staging      # the e2e run the gate looks for
    uv run python -m scripts.promote_prompt answer
    uv run python -m scripts.promote_prompt answer --version 3
    uv run python -m scripts.promote_prompt answer --skip-gate "hotfix, eval to follow"

Judge prompts are graded with `--judge-label staging` on the eval run.
"""

import argparse
import subprocess
import sys

from src.config.settings import LangfuseSettings, PathSettings
from src.eval.gate import promotion_evidence
from src.eval.golden import load_golden
from src.observability.langfuse import make_langfuse
from src.prompts.files import load_prompts
from src.prompts.registry import PRODUCTION, fetch_version
from src.prompts.sync import PromotionError, check_promotable, promote
from src.storage.db import get_engine


def committed(paths: PathSettings, name: str) -> bool:
    """The prompt file is tracked by git and has no uncommitted changes."""
    path = paths.prompts_dir / f"{name}.yaml"
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            cwd=paths.root, capture_output=True,
        ).returncode == 0  # fmt: skip
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", str(path)],
            cwd=paths.root, capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return False
    return tracked and not status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name")
    parser.add_argument("--version", type=int, help="default: the version labelled staging")
    parser.add_argument(
        "--skip-gate",
        metavar="REASON",
        help="promote without a passing eval run (an explicit, logged override)",
    )
    args = parser.parse_args(argv)

    cfg, paths = LangfuseSettings(), PathSettings()
    if not cfg.active:
        print("FAIL: Langfuse is off (no keys, or LANGFUSE_ENABLED=false).")
        return 1
    prompts = load_prompts(paths.prompts_dir)
    if args.name not in prompts:
        print(f"FAIL: no prompt {args.name!r} in {paths.prompts_dir}")
        return 1
    if not committed(paths, args.name):
        print(f"FAIL: prompts/{args.name}.yaml has uncommitted changes; commit it first.")
        return 1

    client = make_langfuse(cfg)
    try:
        target = check_promotable(client, prompts[args.name], args.version)
    except PromotionError as e:
        print(f"FAIL: {e}")
        return 1

    if args.skip_gate:
        print(f"GATE SKIPPED: {args.skip_gate}")
    else:
        evidence = promotion_evidence(get_engine(), args.name, target.hash, load_golden().hash)
        if not evidence.ok:
            print(f"FAIL: eval gate: {evidence.reason}")
            return 1
        print(f"GATE PASSED: {evidence.reason}")

    current = fetch_version(client, args.name, PRODUCTION)
    if current and current.version == target.version:
        print(f"{args.name}: v{target.version} is already {PRODUCTION}.")
        return 0
    promote(client, args.name, target)
    before = f"v{current.version}" if current else "none"
    print(f"{args.name}: production {before} -> v{target.version} ({target.hash[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
