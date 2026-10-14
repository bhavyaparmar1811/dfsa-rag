import pytest
from openai import APIConnectionError

from src.generation.llm import LLMClient, request_params
from src.prompts.registry import ResolvedPrompt
from tests.generation.conftest import FakeAzure, azure_settings


def text_prompt(config=None) -> ResolvedPrompt:
    return ResolvedPrompt(
        name="refusal", type="text", content="Say {{reason}}", config=config or {},
        source="local", label="production", version=None, hash="h",
    )  # fmt: skip


def test_prompt_config_becomes_request_parameters():
    params = request_params({"temperature": 0, "max_tokens": 50, "response_format": "json_object"})
    assert params == {
        "temperature": 0, "max_tokens": 50, "response_format": {"type": "json_object"}
    }  # fmt: skip
    with pytest.raises(ValueError, match="max_token"):
        request_params({"max_token": 50})  # a typo fails loudly


def test_a_call_sends_the_compiled_prompt_and_reports_usage_and_cost():
    fake = FakeAzure("ok")
    llm = LLMClient(azure_settings(), client=fake)
    result = llm.complete(text_prompt({"temperature": 0}), {"reason": "hi"}, name="t")

    call = fake.calls[0]
    assert call["model"] == "gpt-4o" and call["temperature"] == 0
    assert call["messages"] == [{"role": "user", "content": "Say hi"}]  # text -> one user message
    assert (result.text, result.finish_reason) == ("ok", "stop")
    assert (result.input_tokens, result.output_tokens) == (1000, 200)
    # $2.50 / 1M in, $10 / 1M out
    assert result.cost_usd == pytest.approx(1000 * 2.5e-6 + 200 * 10e-6)


def test_api_errors_are_raised_not_swallowed():
    fake = FakeAzure(APIConnectionError(request=None))
    with pytest.raises(APIConnectionError):
        LLMClient(azure_settings(), client=fake).complete(text_prompt(), {"reason": "x"}, name="t")
