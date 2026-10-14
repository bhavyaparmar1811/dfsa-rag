"""Push the prompts in git (`prompts/*.yaml`) to Langfuse as the `staging` version.

A prompt whose newest Langfuse version already has the git content is left
alone (or just gets the `staging` label). `production` is never touched: that
is `promote_prompt`, after the eval gate. Each new version's commit message
records the content hash and the git commit it came from.

    uv run python -m scripts.sync_prompts             # all prompts
    uv run python -m scripts.sync_prompts answer      # just one
    uv run python -m scripts.sync_prompts --dry-run   # show what would change
"""

import argparse
import subprocess
import sys

from src.config.settings import LangfuseSettings, PathSettings
from src.observability.langfuse import make_langfuse
from src.prompts.files import load_prompts
from src.prompts.sync import sync_prompt


def git_commit(paths: PathSettings) -> str | None:
    """Short HEAD sha, with "+dirty" when prompts/ has uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=paths.root, capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", str(paths.prompts_dir)],
            cwd=paths.root, capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return None
    return sha + ("+dirty" if dirty else "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="prompt names (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="change nothing")
    args = parser.parse_args(argv)

    cfg, paths = LangfuseSettings(), PathSettings()
    if not cfg.active:
        print("FAIL: Langfuse is off (no keys, or LANGFUSE_ENABLED=false).")
        return 1
    prompts = load_prompts(paths.prompts_dir)
    if unknown := set(args.names) - set(prompts):
        print(f"FAIL: no such prompt(s) in {paths.prompts_dir}: {sorted(unknown)}")
        return 1

    client = make_langfuse(cfg)
    commit = git_commit(paths)
    print(f"Syncing to {cfg.host}" + (" (dry run)" if args.dry_run else "") + f", git {commit}")
    for name in args.names or prompts:
        r = sync_prompt(client, prompts[name], commit=commit, dry_run=args.dry_run)
        version = f"v{r.version}" if r.version else "new version"
        print(f"  - {name:20} {r.action:10} {version}  ({prompts[name].hash[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
