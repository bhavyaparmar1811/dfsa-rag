"""Fill in the self-hosted Langfuse secrets in `.env`, without printing them.

Adds only the LANGFUSE_* names that are missing or blank; an existing value is
never overwritten (changing LANGFUSE_ENCRYPTION_KEY or LANGFUSE_SALT after
Langfuse has stored data breaks it). Prints only the names it added or kept,
so no secret ends up in a terminal log.

    uv run python -m scripts.init_langfuse_env --email you@example.com
"""

import argparse
import re
import secrets
import sys
import uuid
from collections.abc import Callable
from pathlib import Path

from src.config.settings import ENV_FILE

# name -> how to generate it. Hex only: these go into a Postgres URL and a shell command line.
GENERATED: dict[str, Callable[[], str]] = {
    "LANGFUSE_POSTGRES_PASSWORD": lambda: secrets.token_hex(24),
    "LANGFUSE_CLICKHOUSE_PASSWORD": lambda: secrets.token_hex(24),
    "LANGFUSE_REDIS_AUTH": lambda: secrets.token_hex(24),
    "LANGFUSE_MINIO_PASSWORD": lambda: secrets.token_hex(24),
    "LANGFUSE_NEXTAUTH_SECRET": lambda: secrets.token_hex(32),
    "LANGFUSE_SALT": lambda: secrets.token_hex(32),
    "LANGFUSE_ENCRYPTION_KEY": lambda: secrets.token_hex(32),  # must be exactly 64 hex chars
    "LANGFUSE_PUBLIC_KEY": lambda: f"pk-lf-{uuid.uuid4()}",
    "LANGFUSE_SECRET_KEY": lambda: f"sk-lf-{uuid.uuid4()}",
    "LANGFUSE_INIT_USER_PASSWORD": lambda: secrets.token_urlsafe(18),
}
HOST = ("LANGFUSE_HOST", "http://127.0.0.1:3000")
EMAIL = "LANGFUSE_INIT_USER_EMAIL"
HEADER = "# --- Langfuse (self-hosted), written by scripts/init_langfuse_env.py ---"

_LINE = re.compile(r"^\s*([A-Z0-9_]+)\s*=(.*)$")


def _existing(env_text: str) -> dict[str, str]:
    values = {}
    for line in env_text.splitlines():
        if m := _LINE.match(line):
            values[m.group(1)] = m.group(2).strip().strip("'\"")
    return values


def fill_missing(env_text: str, email: str | None) -> tuple[str, list[str], list[str]]:
    """Return (new .env text, names added, names kept). Never changes a set value.

    A name present but blank (`NAME=`) is filled in place; a missing name is
    appended under a Langfuse header.
    """
    existing = _existing(env_text)
    wanted: dict[str, Callable[[], str] | str | None] = {
        **GENERATED,
        HOST[0]: HOST[1],
        EMAIL: email,
    }
    added, kept, values = [], [], {}
    for name, make in wanted.items():
        if existing.get(name):
            kept.append(name)
        elif make is not None:
            values[name] = make() if callable(make) else make
            added.append(name)

    lines = env_text.splitlines()
    for i, line in enumerate(lines):
        if (m := _LINE.match(line)) and m.group(1) in values:
            lines[i] = f"{m.group(1)}={values.pop(m.group(1))}"
    if values:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(HEADER)
        lines += [f"{name}={value}" for name, value in values.items()]
    return "\n".join(lines) + "\n", added, kept


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--email", help="login email for the Langfuse admin user")
    parser.add_argument("--env-file", type=Path, default=ENV_FILE)
    args = parser.parse_args()

    if not args.env_file.exists():
        print(f"FAIL: {args.env_file} not found. Copy .env.example to .env first.")
        return 1
    text = args.env_file.read_text(encoding="utf-8")
    new_text, added, kept = fill_missing(text, args.email)

    if EMAIL not in added and EMAIL not in kept:
        print(f"FAIL: {EMAIL} is not set. Pass --email you@example.com.")
        return 1
    if added:
        args.env_file.write_text(new_text, encoding="utf-8")
    for name in added:
        print(f"added  {name}")
    for name in kept:
        print(f"kept   {name} (already set, not changed)")
    print(f"\n{len(added)} added, {len(kept)} kept in {args.env_file}. Values are not printed;")
    print("the admin login password is LANGFUSE_INIT_USER_PASSWORD in .env.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
