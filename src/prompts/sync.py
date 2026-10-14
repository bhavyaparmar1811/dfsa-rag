"""Moving prompts from git into Langfuse, and promoting them.

    edit prompts/<name>.yaml -> sync (new version, label `staging`)
      -> eval run against `staging` -> promote (`production` moves to it)

Sync never touches `production`. Promote moves it only to a version whose
content is the git copy, so production is always something in git: a rollback
is a revert in git, a sync and a promote (Langfuse just gets a new version
equal to the old one).
"""

from dataclasses import dataclass
from typing import Literal

from langfuse import Langfuse
from langfuse.api import NotFoundError

from src.prompts.files import PromptFile
from src.prompts.registry import (
    LATEST,
    PRODUCTION,
    STAGING,
    RemoteVersion,
    fetch_version,
    remote_version,
)


@dataclass(frozen=True)
class SyncResult:
    name: str
    action: Literal["created", "labelled", "unchanged"]
    version: int | None  # None on a dry run that would create one


def _with_label(labels: list[str], label: str) -> list[str]:
    # "latest" is managed by Langfuse and can't be set.
    return sorted({*labels, label} - {LATEST})


def sync_prompt(
    client: Langfuse, local: PromptFile, *, commit: str | None = None, dry_run: bool = False
) -> SyncResult:
    """Make sure a version with the git content exists and carries `staging`.

    Compared with the newest version only: if git matches an older version
    (a revert), a new version is created, so the history reads in order.
    """
    latest = fetch_version(client, local.name, LATEST)
    if latest and latest.hash == local.hash:
        if STAGING in latest.labels:
            return SyncResult(local.name, "unchanged", latest.version)
        if not dry_run:
            client.update_prompt(
                name=local.name,
                version=latest.version,
                new_labels=_with_label(latest.labels, STAGING),
            )
        return SyncResult(local.name, "labelled", latest.version)

    if dry_run:
        return SyncResult(local.name, "created", None)
    message = f"git {local.hash[:12]}" + (f" @ {commit}" if commit else "")
    created = client.create_prompt(
        name=local.name,
        type=local.type,
        prompt=local.content,
        config=local.config,
        labels=[STAGING],
        commit_message=message,
    )
    return SyncResult(local.name, "created", created.version)


class PromotionError(Exception):
    pass


def check_promotable(client: Langfuse, local: PromptFile, version: int | None) -> RemoteVersion:
    """The version to promote (default: the one labelled `staging`), after
    checking it is the git copy. Raises PromotionError otherwise."""
    if version is None:
        target = fetch_version(client, local.name, STAGING)
        if target is None:
            raise PromotionError(f"{local.name}: no version is labelled {STAGING}; run sync first")
    else:
        try:
            target = remote_version(client.api.prompts.get(local.name, version=version))
        except NotFoundError:
            raise PromotionError(f"{local.name}: there is no version {version}") from None
    if target.hash != local.hash:
        raise PromotionError(
            f"{local.name} v{target.version} is not the git copy "
            f"({target.hash[:12]} != {local.hash[:12]}); production must be what's in git"
        )
    return target


def promote(client: Langfuse, name: str, target: RemoteVersion) -> None:
    client.update_prompt(
        name=name, version=target.version, new_labels=_with_label(target.labels, PRODUCTION)
    )
