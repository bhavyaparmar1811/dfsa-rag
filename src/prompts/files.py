"""Prompts as files in git: `prompts/<name>.yaml`.

Git is the source of truth for a prompt's content; Langfuse holds its versions
and labels (`staging`, `production`). Each file is either a chat prompt:

    name: answer
    type: chat
    description: what it is for (not sent to Langfuse, not hashed)
    variables: [question, context]
    config: {temperature: 0, max_tokens: 1200}
    messages:
      - role: system
        content: ...

or a text prompt (`type: text`, `prompt: ...` instead of `messages`).
Variables are written `{{name}}` (Langfuse's syntax), and the declared list must
match what the messages use: it is the contract with the code that fills them.

`content_hash` identifies the content the LLM sees plus its config, computed the
same way for a git file and for a version fetched from Langfuse, so the two can
be compared (sync, drift check) and every trace records exactly what was used.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PromptType = Literal["chat", "text"]
Content = str | list[dict[str, str]]

NAME = re.compile(r"^[a-z][a-z0-9-]*$")
VARIABLE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
ROLES = {"system", "user", "assistant"}


class PromptFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    type: PromptType
    description: str = ""
    variables: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    messages: list[dict[str, str]] | None = None  # chat
    prompt: str | None = None  # text

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not NAME.match(v):
            raise ValueError(f"prompt name {v!r} must be lowercase letters, digits and '-'")
        return v

    @field_validator("messages")
    @classmethod
    def _messages(cls, v: list[dict[str, str]] | None) -> list[dict[str, str]] | None:
        for m in v or []:
            if set(m) != {"role", "content"} or m["role"] not in ROLES:
                raise ValueError(f"each message needs exactly a role in {ROLES} and content: {m}")
        return v

    @model_validator(mode="after")
    def _shape(self) -> "PromptFile":
        if self.type == "chat" and (not self.messages or self.prompt is not None):
            raise ValueError(f"{self.name}: a chat prompt has `messages` and no `prompt`")
        if self.type == "text" and (not self.prompt or self.messages is not None):
            raise ValueError(f"{self.name}: a text prompt has `prompt` and no `messages`")
        used = variables_in(self.content)
        if used != set(self.variables):
            raise ValueError(
                f"{self.name}: declared variables {sorted(self.variables)} "
                f"!= used in the prompt {sorted(used)}"
            )
        return self

    @property
    def content(self) -> Content:
        return self.messages if self.type == "chat" else self.prompt

    @property
    def hash(self) -> str:
        return content_hash(self.type, self.content, self.config)


def variables_in(content: Content) -> set[str]:
    texts = [content] if isinstance(content, str) else [m["content"] for m in content]
    return {v for t in texts for v in VARIABLE.findall(t)}


def _canonical(value: Any) -> Any:
    # Langfuse stores config as JSON and hands 0.0 back as 0; whole floats become ints.
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: _canonical(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_canonical(v) for v in value]
    return value


def normalise_messages(messages: list[Any]) -> list[dict[str, str]]:
    """Chat messages as plain {role, content} dicts, from the SDK's dicts or the API's models."""
    out = []
    for m in messages:
        get = m.get if isinstance(m, dict) else lambda k, m=m: getattr(m, k, None)
        if get("type") == "placeholder":
            raise ValueError("chat placeholders are not used in this project")
        out.append({"role": get("role"), "content": get("content")})
    return out


def content_hash(type_: str, content: Content, config: dict | None) -> str:
    if type_ == "chat":
        content = normalise_messages(content)
    blob = json.dumps(
        {"type": type_, "prompt": content, "config": _canonical(config or {})},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_prompt(path: Path) -> PromptFile:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    prompt = PromptFile.model_validate(data)
    if prompt.name != path.stem:
        raise ValueError(f"{path.name}: name {prompt.name!r} must match the file name")
    return prompt


def load_prompts(directory: Path) -> dict[str, PromptFile]:
    prompts = {p.name: p for p in map(load_prompt, sorted(directory.glob("*.yaml")))}
    if not prompts:
        raise FileNotFoundError(f"no prompts in {directory}")
    return prompts


def compile_prompt(content: Content, variables: dict[str, Any]) -> Content:
    """Fill `{{name}}` placeholders. Strict: every placeholder needs a value and
    every value a placeholder (Langfuse's own compile silently leaves gaps).
    One pass, so text inside a value (e.g. a rule quoting "{{x}}") is never
    substituted again."""
    used = variables_in(content)
    if missing := used - set(variables):
        raise KeyError(f"missing prompt variables: {sorted(missing)}")
    if extra := set(variables) - used:
        raise KeyError(f"unknown prompt variables: {sorted(extra)}")

    def fill(text: str) -> str:
        return VARIABLE.sub(lambda m: str(variables[m.group(1)]), text)

    if isinstance(content, str):
        return fill(content)
    return [{"role": m["role"], "content": fill(m["content"])} for m in content]
