from src.config.settings import LangfuseSettings
from src.observability.langfuse import make_langfuse


def test_active_needs_both_keys_and_enabled():
    assert LangfuseSettings(public_key="pk", secret_key="sk", enabled=True).active
    assert not LangfuseSettings(public_key="pk", secret_key="sk", enabled=False).active
    assert not LangfuseSettings(public_key=None, secret_key="sk", enabled=True).active


def test_disabled_client_is_a_no_op():
    # Port 1: nothing listens there, so any network call would fail.
    client = make_langfuse(LangfuseSettings(enabled=False, host="http://127.0.0.1:1"))
    with client.start_as_current_observation(name="request", as_type="span") as span:
        span.update(input="question", output="answer")
        with client.start_as_current_observation(name="llm", as_type="generation") as gen:
            gen.update(usage_details={"input": 10}, cost_details={"input": 0.001})
    client.flush()
