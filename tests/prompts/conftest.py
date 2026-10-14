"""An in-memory stand-in for the Langfuse prompt store, built from the SDK's own
prompt models, so registry and sync tests run without a server."""

from types import SimpleNamespace

import pytest
from langfuse.api import NotFoundError, Prompt_Chat, Prompt_Text
from langfuse.model import ChatPromptClient, TextPromptClient

from src.config.settings import LangfuseSettings
from src.prompts.files import PromptFile


class FakeLangfuse:
    def __init__(self):
        self.store: dict[str, list] = {}  # name -> versions, oldest first
        self.down = False
        self.api = SimpleNamespace(prompts=SimpleNamespace(get=self._api_get))

    def _find(self, name: str, label: str | None = None, version: int | None = None):
        if self.down:
            raise ConnectionError("Langfuse is down")
        versions = self.store.get(name, [])
        if version is not None:
            match = [p for p in versions if p.version == version]
        elif label == "latest":
            match = versions[-1:]
        else:
            match = [p for p in versions if (label or "production") in p.labels]
        if not match:
            raise NotFoundError(body={"message": "Prompt not found"})
        return match[0]

    def _api_get(self, name, *, version=None, label=None):
        return self._find(name, label, version)

    def get_prompt(self, name, *, label=None, type="text", **_):
        prompt = self._find(name, label)
        return ChatPromptClient(prompt) if prompt.type == "chat" else TextPromptClient(prompt)

    def create_prompt(self, *, name, prompt, type, config, labels, commit_message=None):
        versions = self.store.setdefault(name, [])
        model = Prompt_Chat if type == "chat" else Prompt_Text
        created = model(
            type=type, prompt=prompt, name=name, version=len(versions) + 1,
            config=config, labels=[], tags=[], commit_message=commit_message,
        )  # fmt: skip
        versions.append(created)
        self.update_prompt(name=name, version=created.version, new_labels=labels)
        return self.store[name][-1]

    def update_prompt(self, *, name, version, new_labels):
        # As the server does: the labels are added to the version (not replacing
        # its others), and, being unique across versions, removed from the rest.
        self.store[name] = [
            p.model_copy(
                update={
                    "labels": sorted({*p.labels, *new_labels})
                    if p.version == version
                    else [label for label in p.labels if label not in new_labels]
                }
            )
            for p in self.store[name]
        ]

    def labels(self, name: str) -> dict[int, list[str]]:
        return {p.version: sorted(p.labels) for p in self.store[name]}


def chat_prompt(system: str = "You answer from {{context}}.", **config) -> PromptFile:
    return PromptFile(
        name="answer",
        type="chat",
        variables=["context", "question"],
        config={"temperature": 0, **config},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": "Q: {{question}}"},
        ],
    )


@pytest.fixture
def fake():
    return FakeLangfuse()


@pytest.fixture
def lf_cfg():
    return LangfuseSettings(enabled=True, public_key="pk", secret_key="sk")
