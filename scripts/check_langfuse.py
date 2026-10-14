"""Smoke test: is the self-hosted Langfuse in `.env` up, do the keys work, and
does a trace make it all the way in?

1. health endpoint (server version), 2. the project keys, 3. a test trace sent
through the SDK and read back from the API (this exercises web, the worker,
Redis, MinIO and ClickHouse), 4. the prompt drift check: is `production` in
Langfuse what's in git? Exits 0 when 1-3 pass (drift is reported, not fatal).

    uv run python -m scripts.check_langfuse
    uv run python -m scripts.check_langfuse --no-trace   # skip step 3
"""

import argparse
import sys
import time

import httpx
from pydantic import ValidationError

from src.config.settings import LangfuseSettings
from src.observability.langfuse import make_langfuse
from src.prompts.registry import PromptRegistry

TRACE_WAIT_S = 30


def _trace_round_trip(client) -> bool:
    with client.start_as_current_observation(
        name="check_langfuse", as_type="span", input={"check": "round trip"}
    ) as span:
        span.update(output={"ok": True})
        trace_id = client.get_current_trace_id()
    client.flush()

    # Langfuse v4 stores observations; the legacy trace endpoint (api.trace.get)
    # answers 404 for traces ingested by v4, so read the observations back.
    start = time.perf_counter()
    while time.perf_counter() - start < TRACE_WAIT_S:
        found = client.api.observations.get_many(trace_id=trace_id, fields="core,basic,io").data
        if found and found[0].output:
            elapsed = time.perf_counter() - start
            print(f"Trace:        {found[0].name} read back after {elapsed:.1f} s")
            print(f"              {client.get_trace_url(trace_id=trace_id)}")
            return True
        time.sleep(1)
    print(f"FAIL: trace {trace_id} not readable after {TRACE_WAIT_S} s. Is langfuse-worker up?")
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-trace", action="store_true", help="skip the trace round trip")
    args = parser.parse_args(argv)

    try:
        cfg = LangfuseSettings()
    except ValidationError as e:
        print(f"FAIL: Langfuse settings invalid in .env:\n{e}")
        return 1

    print(f"Host:         {cfg.host}")
    print(f"Enabled:      {cfg.enabled}")
    print(f"Keys:         {'set' if cfg.public_key and cfg.secret_key else 'NOT SET'}")
    if not cfg.active:
        print("FAIL: Langfuse is off. Set its keys in .env (scripts.init_langfuse_env).")
        return 1

    try:
        health = httpx.get(f"{cfg.host}/api/public/health", timeout=5).json()
        print(f"Server:       Langfuse {health.get('version', '?')} ({health.get('status', '?')})")
    except (httpx.HTTPError, ValueError) as e:
        print(f"FAIL: cannot reach {cfg.host}: {e}. Are the containers running?")
        return 1

    client = make_langfuse(cfg)
    try:
        client.auth_check()
        print("Keys:         accepted")
    except Exception as e:
        print(f"FAIL: keys rejected ({type(e).__name__}). Do they match .env on the server?")
        return 1

    if not args.no_trace and not _trace_round_trip(client):
        return 1

    print("Prompts (git vs Langfuse):")
    for d in PromptRegistry.from_settings().drift():
        versions = f"production v{d.production or '-'}, staging v{d.staging or '-'}"
        print(f"  - {d.name:20} {d.state:19} {versions}  {d.detail}")

    print("OK: Langfuse is reachable, the keys work" + ("" if args.no_trace else ", traces arrive"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
