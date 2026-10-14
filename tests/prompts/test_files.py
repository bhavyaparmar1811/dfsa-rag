import pytest
from pydantic import ValidationError

from src.config.settings import PathSettings
from src.prompts.files import PromptFile, compile_prompt, load_prompt, load_prompts
from src.prompts.registry import remote_version
from tests.prompts.conftest import chat_prompt

EXPECTED = {"answer", "query-rewrite", "refusal", "judge-faithfulness", "judge-relevance"}


def test_the_prompts_in_git_load():
    prompts = load_prompts(PathSettings().prompts_dir)

    assert set(prompts) == EXPECTED
    assert prompts["refusal"].type == "text"
    assert set(prompts["answer"].variables) == {"history", "context", "question"}
    # JSON mode on Azure OpenAI needs the word "JSON" in the messages.
    for p in prompts.values():
        if p.config.get("response_format") == "json_object":
            assert "JSON" in str(p.content), p.name


def test_declared_variables_must_match_the_prompt():
    with pytest.raises(ValidationError, match="declared variables"):
        PromptFile(name="x", type="text", variables=["a"], prompt="{{a}} and {{b}}")


@pytest.mark.parametrize(
    "fields",
    [
        {"type": "chat", "prompt": "hi"},
        {"type": "text", "messages": [{"role": "user", "content": "hi"}]},
        {"type": "chat", "messages": [{"role": "robot", "content": "hi"}]},
        {"type": "text", "prompt": "hi", "extra": 1},
    ],
)
def test_malformed_files_are_rejected(fields):
    with pytest.raises(ValidationError):
        PromptFile(name="x", **fields)


def test_name_must_match_the_file(tmp_path):
    path = tmp_path / "other.yaml"
    path.write_text("name: answer\ntype: text\nprompt: hi\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must match the file name"):
        load_prompt(path)


def test_hash_ignores_description_but_not_content_or_config():
    base = chat_prompt()
    assert base.model_copy(update={"description": "new words"}).hash == base.hash
    assert chat_prompt(system="You answer only from {{context}}.").hash != base.hash
    assert chat_prompt(max_tokens=100).hash != base.hash


def test_hash_is_the_same_after_a_round_trip_through_langfuse(fake):
    # 0.0 comes back from Langfuse's JSON as 0, and messages as API models.
    local = chat_prompt(temperature=0.0, top_p=1.0)
    fake.create_prompt(
        name=local.name, prompt=local.content, type="chat", config=local.config, labels=[]
    )
    assert remote_version(fake.store["answer"][0]).hash == local.hash


def test_compile_is_strict_and_single_pass():
    content = [{"role": "user", "content": "{{context}} | {{question}}"}]
    out = compile_prompt(content, {"context": "COB says {{question}}", "question": "Q?"})
    assert out == [{"role": "user", "content": "COB says {{question}} | Q?"}]

    with pytest.raises(KeyError, match="missing"):
        compile_prompt(content, {"context": "c"})
    with pytest.raises(KeyError, match="unknown"):
        compile_prompt(content, {"context": "c", "question": "q", "history": "h"})
