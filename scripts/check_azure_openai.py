"""Smoke test: is the Azure OpenAI GPT-4o deployment in `.env` reachable?

Sends one chat completion and reports latency, model, and token usage.
Exits 0 on success, 1 on failure. Asks DEFAULT_QUESTION unless another
question is passed on the command line.

    uv run python -m scripts.check_azure_openai
    uv run python -m scripts.check_azure_openai "What is the DFSA?"
"""

import sys
import time

from openai import APIConnectionError, APIStatusError, AzureOpenAI
from pydantic import ValidationError

from src.config.settings import AzureOpenAISettings

DEFAULT_QUESTION = "Who wrote Harry Potter?"


def main(question: str = DEFAULT_QUESTION) -> int:
    # Load only the Azure section, so a missing OPENAI_API_KEY (embeddings)
    # doesn't block this check.
    try:
        cfg = AzureOpenAISettings()
    except ValidationError as e:
        print(f"FAIL: Azure OpenAI settings missing or invalid in .env:\n{e}")
        return 1

    print(f"Endpoint:    {cfg.endpoint}")
    print(f"Deployment:  {cfg.deployment}")
    print(f"API version: {cfg.api_version}")
    print(f"Question:    {question}")

    client = AzureOpenAI(
        azure_endpoint=cfg.endpoint,
        api_key=cfg.api_key.get_secret_value(),
        api_version=cfg.api_version,
        timeout=30,
        max_retries=0,
    )

    start = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=cfg.deployment,
            messages=[{"role": "user", "content": question}],
            max_tokens=300,
            temperature=0,
        )
    except APIStatusError as e:
        print(f"FAIL: HTTP {e.status_code}: {e.message}")
        return 1
    except APIConnectionError as e:
        print(f"FAIL: could not connect to endpoint: {e}")
        return 1
    latency_ms = (time.perf_counter() - start) * 1000

    usage = resp.usage
    cost = (
        usage.prompt_tokens * cfg.input_cost_per_1m
        + usage.completion_tokens * cfg.output_cost_per_1m
    ) / 1_000_000
    print(f"Model:       {resp.model}")
    print(f"Reply:       {resp.choices[0].message.content!r}")
    print(f"Latency:     {latency_ms:.0f} ms")
    print(f"Tokens:      {usage.prompt_tokens} in / {usage.completion_tokens} out (${cost:.6f})")
    print("OK: Azure OpenAI endpoint is working.")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:]) or DEFAULT_QUESTION))
