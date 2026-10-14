"""The one client for generation: GPT-4o on Azure OpenAI.

    complete(prompt, variables, name=...) -> LLMResult

The prompt comes from the PromptRegistry (Langfuse, or its git copy); its
`config` holds the request parameters, mapped through an allow-list so a typo
in a prompt file fails loudly instead of being ignored. Every call is one
Langfuse generation, linked to the prompt version it used, with token usage and
cost passed explicitly: Langfuse's price table doesn't know Azure deployment
names.

API errors (after the SDK's own retries) are raised, never turned into a
refusal: "no confident answer" would misdescribe an outage.
"""

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

from langfuse import Langfuse
from openai import AzureOpenAI

from src.config.settings import AzureOpenAISettings
from src.observability.langfuse import get_langfuse
from src.prompts.registry import ResolvedPrompt

logger = logging.getLogger(__name__)

# Prompt config key -> how it becomes a chat.completions.create argument.
PARAMS = {
    "temperature": lambda v: ("temperature", v),
    "top_p": lambda v: ("top_p", v),
    "max_tokens": lambda v: ("max_tokens", v),
    "seed": lambda v: ("seed", v),
    "response_format": lambda v: ("response_format", {"type": v}),
}


def request_params(config: dict[str, Any]) -> dict[str, Any]:
    if unknown := set(config) - set(PARAMS):
        raise ValueError(f"unknown prompt config keys: {sorted(unknown)} (known: {sorted(PARAMS)})")
    return dict(PARAMS[k](v) for k, v in config.items())


def messages_for(prompt: ResolvedPrompt, variables: dict[str, Any]) -> list[dict[str, str]]:
    compiled = prompt.compile(**variables)
    return [{"role": "user", "content": compiled}] if isinstance(compiled, str) else compiled


@dataclass
class LLMResult:
    text: str
    finish_reason: str | None
    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    cost_usd: float
    latency_ms: float

    def trace(self) -> dict:
        row = asdict(self)
        row.pop("text")
        return row


class LLMClient:
    def __init__(
        self,
        cfg: AzureOpenAISettings,
        client: AzureOpenAI | None = None,
        langfuse: Langfuse | None = None,
    ):
        self.cfg = cfg
        self._client = client or AzureOpenAI(
            azure_endpoint=cfg.endpoint,
            api_key=cfg.api_key.get_secret_value(),
            api_version=cfg.api_version,
            timeout=cfg.timeout_s,
            max_retries=cfg.max_retries,
        )
        self._langfuse = langfuse

    @property
    def langfuse(self) -> Langfuse:
        return self._langfuse or get_langfuse()

    def cost(self, input_tokens: int, output_tokens: int) -> tuple[float, float]:
        return (
            input_tokens * self.cfg.input_cost_per_1m / 1_000_000,
            output_tokens * self.cfg.output_cost_per_1m / 1_000_000,
        )

    def complete(
        self, prompt: ResolvedPrompt, variables: dict[str, Any], *, name: str
    ) -> LLMResult:
        messages = messages_for(prompt, variables)
        params = request_params(prompt.config)
        with self.langfuse.start_as_current_observation(
            name=name,
            as_type="generation",
            model=self.cfg.deployment,
            model_parameters=params,
            input=messages,
            prompt=prompt.client,
            metadata={"prompt": prompt.trace()},
        ) as gen:
            start = time.perf_counter()
            try:
                resp = self._client.chat.completions.create(
                    model=self.cfg.deployment, messages=messages, **params
                )
            except Exception as e:
                gen.update(level="ERROR", status_message=f"{type(e).__name__}: {e}"[:500])
                raise
            latency_ms = (time.perf_counter() - start) * 1000

            choice, usage = resp.choices[0], resp.usage
            details = getattr(usage, "prompt_tokens_details", None)
            cached = (getattr(details, "cached_tokens", None) or 0) if details else 0
            cost_in, cost_out = self.cost(usage.prompt_tokens, usage.completion_tokens)
            result = LLMResult(
                text=choice.message.content or "",
                finish_reason=choice.finish_reason,
                model=resp.model,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                cached_tokens=cached,
                cost_usd=cost_in + cost_out,
                latency_ms=latency_ms,
            )
            gen.update(
                output=result.text,
                usage_details={"input": result.input_tokens, "output": result.output_tokens},
                cost_details={"input": cost_in, "output": cost_out},
                metadata={
                    "prompt": prompt.trace(),
                    "finish_reason": result.finish_reason,
                    "model": result.model,
                    "cached_tokens": cached,
                },
            )
            if result.finish_reason != "stop":  # cut off (length) or filtered
                gen.update(level="WARNING", status_message=f"finish_reason {result.finish_reason}")
        logger.info(
            "%s: %s %s v%s, %d in / %d out tokens, $%.6f, %.0f ms, finish %s",
            name,
            prompt.name,
            prompt.source,
            prompt.version,
            result.input_tokens,
            result.output_tokens,
            result.cost_usd,
            result.latency_ms,
            result.finish_reason,
        )
        return result
