from src.prompts.registry import PromptRegistry
from tests.prompts.conftest import chat_prompt


def push(fake, prompt, labels):
    fake.create_prompt(
        name=prompt.name, prompt=prompt.content, type=prompt.type,
        config=prompt.config, labels=labels,
    )  # fmt: skip


def registry(fake, cfg, local=None):
    local = local or chat_prompt()
    return PromptRegistry({local.name: local}, fake, cfg)


def test_langfuse_off_serves_git(lf_cfg):
    local = chat_prompt()
    got = PromptRegistry({"answer": local}, None, lf_cfg).get("answer")

    assert (got.source, got.version, got.hash) == ("local", None, local.hash)
    assert got.compile(context="c", question="q")[1]["content"] == "Q: q"


def test_serves_the_production_version(fake, lf_cfg):
    push(fake, chat_prompt(system="Old {{context}}."), ["production"])
    push(fake, chat_prompt(), ["staging"])
    reg = registry(fake, lf_cfg)

    prod = reg.get("answer")
    assert (prod.source, prod.version, prod.label) == ("langfuse", 1, "production")
    assert prod.content[0]["content"] == "Old {{context}}."
    assert prod.client is not None  # links generations to the version

    staging = reg.get("answer", "staging")
    assert (staging.version, staging.hash) == (2, chat_prompt().hash)


def test_unreachable_falls_back_to_git_with_the_reason(fake, lf_cfg):
    fake.down = True
    got = registry(fake, lf_cfg).get("answer")

    assert got.source == "local-fallback"
    assert got.hash == chat_prompt().hash
    assert "ConnectionError" in got.fallback_reason
    assert got.trace()["source"] == "local-fallback"


def test_a_down_server_is_not_retried_on_every_request(fake, lf_cfg):
    reg = registry(fake, lf_cfg)
    fake.down = True
    reg.get("answer")
    fake.down = False  # back, but the registry waits out prompt_cache_ttl_s
    push(fake, chat_prompt(), ["production"])

    got = reg.get("answer")
    assert got.source == "local-fallback" and "not retried" in got.fallback_reason
    reg._down_until = 0.0
    assert reg.get("answer").source == "langfuse"


def test_no_production_version_falls_back(fake, lf_cfg):
    push(fake, chat_prompt(), ["staging"])
    got = registry(fake, lf_cfg).get("answer")
    assert got.source == "local-fallback"
    assert got.fallback_reason == "NotFoundError: HTTP 404 Prompt not found"


def test_a_version_with_other_variables_falls_back(fake, lf_cfg):
    # The code fills {{context}} and {{question}}; this version wants {{rules}}.
    old = chat_prompt().model_copy(
        update={"messages": [{"role": "user", "content": "{{rules}} {{question}}"}]}
    )
    push(fake, old, ["production"])
    got = registry(fake, lf_cfg).get("answer")
    assert got.source == "local-fallback" and "rules" in got.fallback_reason


def test_drift_states(fake, lf_cfg):
    reg = registry(fake, lf_cfg)
    assert reg.drift()[0].state == "not-synced"  # nothing in Langfuse

    push(fake, chat_prompt(), ["staging"])
    d = reg.drift()[0]
    assert (d.state, d.production, d.staging) == ("awaiting-promotion", None, 1)

    fake.update_prompt(name="answer", version=1, new_labels=["staging", "production"])
    assert reg.drift()[0].ok

    edited = chat_prompt(system="Edited {{context}}.")
    assert registry(fake, lf_cfg, edited).drift()[0].state == "not-synced"

    fake.down = True
    assert reg.drift()[0].state == "unreachable"
    two = PromptRegistry({"a": chat_prompt(), "b": chat_prompt()}, fake, lf_cfg).drift()
    assert [d.state for d in two] == ["unreachable", "unreachable"]
    assert PromptRegistry(reg.prompts, None, lf_cfg).drift()[0].state == "unreachable"
