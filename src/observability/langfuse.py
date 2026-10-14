"""The one Langfuse client: request traces (Phases F-G) and the prompt store (E3-E4).

`get_langfuse()` always returns a client. When Langfuse is disabled or has no
keys, it is a no-op client: observations can still be opened and updated, but
nothing is sent, so tests and offline runs need no server. Code that would
*fetch* from Langfuse (prompts) checks `LangfuseSettings.active` instead, see
`src.prompts.registry`.

The SDK reads LANGFUSE_* from the process environment, not from `.env`, so
everything is passed explicitly from `LangfuseSettings`.
"""

import logging
from functools import lru_cache

from langfuse import Langfuse

from src.config.settings import LangfuseSettings

logger = logging.getLogger(__name__)


def make_langfuse(cfg: LangfuseSettings) -> Langfuse:
    if not cfg.active:
        if cfg.enabled:
            logger.warning("Langfuse keys are not set: tracing is off, prompts come from git")
        return Langfuse(
            public_key="disabled",
            secret_key="disabled",
            base_url=cfg.host,
            tracing_enabled=False,
        )
    return Langfuse(
        public_key=cfg.public_key,
        secret_key=cfg.secret_key.get_secret_value(),
        base_url=cfg.host,
        timeout=cfg.timeout_s,
        environment=cfg.environment,
        sample_rate=cfg.sample_rate,
    )


@lru_cache
def get_langfuse() -> Langfuse:
    """The process-wide client (the SDK batches and flushes in the background)."""
    return make_langfuse(LangfuseSettings())
