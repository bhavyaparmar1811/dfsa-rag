import re

from scripts.init_langfuse_env import EMAIL, GENERATED, HOST, _existing, fill_missing


def test_fills_every_name_on_an_empty_env():
    text, added, kept = fill_missing("OPENAI_API_KEY=abc\n", "me@example.com")
    values = _existing(text)

    assert kept == []
    assert set(added) == {*GENERATED, HOST[0], EMAIL}
    assert values["OPENAI_API_KEY"] == "abc"  # unrelated lines untouched
    assert values[EMAIL] == "me@example.com"
    assert values[HOST[0]] == "http://127.0.0.1:3000"
    assert re.fullmatch(r"[0-9a-f]{64}", values["LANGFUSE_ENCRYPTION_KEY"])
    assert values["LANGFUSE_PUBLIC_KEY"].startswith("pk-lf-")
    assert values["LANGFUSE_SECRET_KEY"].startswith("sk-lf-")


def test_never_overwrites_a_set_value():
    first, _, _ = fill_missing("", "me@example.com")
    second, added, kept = fill_missing(first, "other@example.com")

    assert second == first
    assert added == []
    assert _existing(second)[EMAIL] == "me@example.com"
    assert len(kept) == len(GENERATED) + 2


def test_blank_placeholders_are_filled_in_place():
    env = "LANGFUSE_SALT=\nLANGFUSE_ENCRYPTION_KEY=keep-me\n"
    text, added, kept = fill_missing(env, None)
    lines = text.splitlines()

    assert lines[0].startswith("LANGFUSE_SALT=") and lines[0] != "LANGFUSE_SALT="
    assert lines[1] == "LANGFUSE_ENCRYPTION_KEY=keep-me"
    assert "LANGFUSE_SALT" in added and "LANGFUSE_ENCRYPTION_KEY" in kept
    assert text.count("LANGFUSE_SALT=") == 1
    assert EMAIL not in added  # no email given, so it stays unset
