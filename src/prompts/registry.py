"""PromptRegistry: the prompt the app uses, from Langfuse, or from git when it can't be.

    get(name)            the `production` version (LANGFUSE_PROMPT_LABEL)
    get(name, "staging") the candidate an eval run tests

Where the prompt came from is part of the result and goes into every trace:

    langfuse        the labelled version from Langfuse (the SDK caches it for
                    prompt_cache_ttl_s, and keeps serving a stale copy if a refresh fails)
    local-fallback  Langfuse was on but failed (unreachable, no version with that
                    label, or its variables don't match the code): the git copy
    local           Langfuse is off (no keys, or LANGFUSE_ENABLED=false): the git copy

Every result carries the content hash, so a trace pins the exact text used even
when it came from git. Note: the git copy can be ahead of `production` (edited,
synced to staging, not yet promoted), so a fallback can serve a prompt that
hasn't passed the eval gate. `drift()` reports that state, and the app checks it
at startup.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Literal

from langfuse import Langfuse
from langfuse.api import NotFoundError
from langfuse.model import ChatPromptClient, TextPromptClient

from src.config.settings import LangfuseSettings, PathSettings
from src.observability.langfuse import get_langfuse
from src.prompts.files import (
    Content,
    PromptFile,
    PromptType,
    compile_prompt,
    content_hash,
    load_prompts,
    normalise_messages,
    variables_in,
)

logger = logging.getLogger(__name__)

Source = Literal["langfuse", "local-fallback", "local"]
STAGING, PRODUCTION, LATEST = "staging", "production", "latest"


def describe(e: Exception) -> str:
    """A short reason. The API's errors print every response header."""
    if (status := getattr(e, "status_code", None)) is not None:
        body = getattr(e, "body", None)
        message = body.get("message", "") if isinstance(body, dict) else body or ""
        return f"{type(e).__name__}: HTTP {status} {message}".strip()[:200]
    return f"{type(e).__name__}: {e}"[:200]


@dataclass(frozen=True)
class ResolvedPrompt:
    name: str
    type: PromptType
    content: Content
    config: dict[str, Any]
    source: Source
    label: str
    version: int | None  # the Langfuse version; None for a git copy
    hash: str
    fallback_reason: str | None = None
    # The SDK object, to link a generation to its prompt version (Langfuse only).
    client: TextPromptClient | ChatPromptClient | None = None

    def compile(self, **variables: Any) -> Content:
        return compile_prompt(self.content, variables)

    def trace(self) -> dict:
        """What a trace records about the prompt."""
        return {
            "name": self.name,
            "label": self.label,
            "version": self.version,
            "source": self.source,
            "hash": self.hash,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class RemoteVersion:
    """One version of a prompt in Langfuse."""

    version: int
    labels: list[str]
    hash: str
    variables: set[str]


def remote_version(prompt: Any) -> RemoteVersion:
    """From an API prompt (`client.api.prompts.get`)."""
    content = prompt.prompt if prompt.type == "text" else normalise_messages(prompt.prompt)
    return RemoteVersion(
        version=prompt.version,
        labels=list(prompt.labels),
        hash=content_hash(prompt.type, content, prompt.config),
        variables=variables_in(content),
    )


def fetch_version(client: Langfuse, name: str, label: str) -> RemoteVersion | None:
    """The version holding `label` ("latest" = the newest), or None. Uncached."""
    try:
        return remote_version(client.api.prompts.get(name, label=label))
    except NotFoundError:
        return None


@dataclass(frozen=True)
class Drift:
    name: str
    state: Literal["in-sync", "awaiting-promotion", "not-synced", "unreachable"]
    local_hash: str
    production: int | None  # version numbers
    staging: int | None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.state == "in-sync"


class PromptRegistry:
    def __init__(
        self,
        prompts: dict[str, PromptFile],
        client: Langfuse | None,
        cfg: LangfuseSettings,
    ):
        """`client` None means Langfuse is off: every prompt comes from git."""
        self.prompts = prompts
        self.client = client
        self.cfg = cfg
        # After a connection failure, serve git without retrying for a while:
        # a refused connection costs seconds (~2 s per attempt on Windows), and
        # without a cached copy every request would pay it again.
        self._down_until = 0.0
        self._down_reason = ""

    @classmethod
    def from_settings(cls) -> "PromptRegistry":
        cfg = LangfuseSettings()
        client = get_langfuse() if cfg.active else None
        return cls(load_prompts(PathSettings().prompts_dir), client, cfg)

    def local(self, name: str) -> PromptFile:
        if name not in self.prompts:
            raise KeyError(f"no prompt {name!r} in git (prompts/{name}.yaml)")
        return self.prompts[name]

    def get(self, name: str, label: str | None = None) -> ResolvedPrompt:
        local = self.local(name)
        label = label or self.cfg.prompt_label
        if self.client is None:
            return self._from_git(local, label, "local")
        if time.monotonic() < self._down_until:
            return self._from_git(local, label, "local-fallback", self._down_reason)
        try:
            remote = self.client.get_prompt(
                name,
                label=label,
                type=local.type,
                cache_ttl_seconds=self.cfg.prompt_cache_ttl_s,
                max_retries=self.cfg.prompt_max_retries,
                fetch_timeout_seconds=self.cfg.prompt_fetch_timeout_s,
            )
        except Exception as e:  # unreachable, auth, not found: all fall back to git
            reason = describe(e)
            if getattr(e, "status_code", None) is None:  # no answer from the server
                self._down_until = time.monotonic() + self.cfg.prompt_cache_ttl_s
                self._down_reason = f"{reason} (not retried for {self.cfg.prompt_cache_ttl_s} s)"
            return self._fallback(local, label, reason)

        content = remote.prompt if local.type == "text" else normalise_messages(remote.prompt)
        # The code fills the variables the git copy declares; a version that
        # expects others would fail (or leave gaps) at compile time.
        if (used := variables_in(content)) != set(local.variables):
            return self._fallback(
                local,
                label,
                f"version {remote.version} uses variables {sorted(used)}, "
                f"the code fills {sorted(local.variables)}",
            )
        return ResolvedPrompt(
            name=name,
            type=local.type,
            content=content,
            config=dict(remote.config or {}),
            source="langfuse",
            label=label,
            version=remote.version,
            hash=content_hash(local.type, content, remote.config),
            client=remote,
        )

    def _fallback(self, local: PromptFile, label: str, reason: str) -> ResolvedPrompt:
        logger.warning("prompt %r (%s): using the git copy: %s", local.name, label, reason)
        return self._from_git(local, label, "local-fallback", reason)

    @staticmethod
    def _from_git(
        local: PromptFile, label: str, source: Source, reason: str | None = None
    ) -> ResolvedPrompt:
        return ResolvedPrompt(
            name=local.name,
            type=local.type,
            content=local.content,
            config=dict(local.config),
            source=source,
            label=label,
            version=None,
            hash=local.hash,
            fallback_reason=reason,
        )

    def drift(self) -> list[Drift]:
        """Git vs Langfuse, per prompt: is `production` what's in git?
        Stops asking after the first prompt Langfuse doesn't answer for."""
        results: list[Drift] = []
        for p in self.prompts.values():
            if results and results[-1].state == "unreachable":
                results.append(Drift(p.name, "unreachable", p.hash, None, None, results[-1].detail))
            else:
                results.append(self._drift(p))
        return results

    def _drift(self, local: PromptFile) -> Drift:
        if self.client is None:
            return Drift(local.name, "unreachable", local.hash, None, None, "Langfuse is off")
        try:
            prod = fetch_version(self.client, local.name, self.cfg.prompt_label)
            staging = fetch_version(self.client, local.name, STAGING)
        except Exception as e:
            return Drift(local.name, "unreachable", local.hash, None, None, describe(e))

        versions = (prod and prod.version, staging and staging.version)
        if prod and prod.hash == local.hash:
            return Drift(local.name, "in-sync", local.hash, *versions)
        detail = f"production is v{prod.version}" if prod else "no production version yet"
        if staging and staging.hash == local.hash:
            return Drift(local.name, "awaiting-promotion", local.hash, *versions, detail)
        return Drift(local.name, "not-synced", local.hash, *versions, f"{detail}; run sync_prompts")


def report_drift(registry: PromptRegistry) -> bool:
    """Log the drift check (at app startup). True when every prompt is in sync."""
    results = registry.drift()
    for d in results:
        log = logger.info if d.ok else logger.warning
        log("prompt %s: %s %s", d.name, d.state, d.detail)
    return all(d.ok for d in results)
