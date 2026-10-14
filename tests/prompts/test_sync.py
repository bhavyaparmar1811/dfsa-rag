import pytest

from src.prompts.sync import PromotionError, check_promotable, promote, sync_prompt
from tests.prompts.conftest import chat_prompt

V1 = chat_prompt()
V2 = chat_prompt(system="Answer only from {{context}}.")


def test_sync_creates_staging_once(fake):
    first = sync_prompt(fake, V1, commit="abc123")
    again = sync_prompt(fake, V1)

    assert (first.action, first.version) == ("created", 1)
    assert (again.action, again.version) == ("unchanged", 1)
    assert fake.labels("answer") == {1: ["staging"]}
    assert fake.store["answer"][0].commit_message == f"git {V1.hash[:12]} @ abc123"


def test_an_edit_moves_staging_and_leaves_production(fake):
    sync_prompt(fake, V1)
    promote(fake, "answer", check_promotable(fake, V1, None))
    assert sync_prompt(fake, V2).version == 2

    assert fake.labels("answer") == {1: ["production"], 2: ["staging"]}


def test_a_revert_is_a_new_version(fake):
    sync_prompt(fake, V1)
    sync_prompt(fake, V2)
    assert sync_prompt(fake, V1).version == 3
    assert fake.labels("answer") == {1: [], 2: [], 3: ["staging"]}


def test_newest_version_without_staging_is_relabelled(fake):
    sync_prompt(fake, V1)
    sync_prompt(fake, V2)
    fake.update_prompt(name="answer", version=1, new_labels=["staging"])  # moved back by hand
    r = sync_prompt(fake, V2)
    assert (r.action, fake.labels("answer")) == ("labelled", {1: [], 2: ["staging"]})


def test_dry_run_changes_nothing(fake):
    assert sync_prompt(fake, V1, dry_run=True).action == "created"
    assert "answer" not in fake.store


def test_promote_only_what_is_in_git(fake):
    with pytest.raises(PromotionError, match="run sync"):
        check_promotable(fake, V1, None)

    sync_prompt(fake, V1)
    with pytest.raises(PromotionError, match="not the git copy"):
        check_promotable(fake, V2, None)  # git has moved on since the sync
    with pytest.raises(PromotionError, match="no version 7"):
        check_promotable(fake, V1, 7)

    promote(fake, "answer", check_promotable(fake, V1, 1))
    assert fake.labels("answer") == {1: ["production", "staging"]}
