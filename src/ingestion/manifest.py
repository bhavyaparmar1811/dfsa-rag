"""Loads `data/manifest.yaml`, the source of truth for the corpus.

Every ingestion step starts here and only ever reads local files listed in
the manifest. A file that is missing, or whose sha256 doesn't match the
manifest, fails loudly: silently ingesting a different version of a rulebook
than the one recorded would break every citation's provenance.
"""

import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

import yaml
from pydantic import BaseModel, field_validator

from src.config.settings import PathSettings

DocType = Literal["rulebook", "law"]


class ManifestError(Exception):
    """The manifest is malformed or disagrees with the files on disk."""


class ManifestEntry(BaseModel):
    id: str
    regulator: str
    rulebook: str
    module_code: str
    module_name: str
    version: str
    effective_date: str
    source_url: str
    file_path: str
    downloaded_on: str
    file_hash: str
    notes: str = ""

    @field_validator("file_path")
    @classmethod
    def _relative_path(cls, v: str) -> str:
        if PurePosixPath(v).is_absolute() or PureWindowsPath(v).is_absolute() or ":" in v:
            raise ValueError(f"file_path must be relative to the repo root, got {v!r}")
        return v

    @field_validator("file_hash")
    @classmethod
    def _sha256_hex(cls, v: str) -> str:
        v = v.strip().lower()
        if len(v) != 64 or any(c not in "0123456789abcdef" for c in v):
            raise ValueError(f"file_hash must be a 64-char sha256 hex digest, got {v!r}")
        return v

    @property
    def doc_type(self) -> DocType:
        """Laws and Rulebook modules number their provisions differently, so
        they get different chunkers."""
        return "law" if "law" in self.rulebook.lower() else "rulebook"

    def path(self, root: Path) -> Path:
        return root / self.file_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path | None = None) -> list[ManifestEntry]:
    path = path or PathSettings().manifest
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    docs = raw.get("documents")
    if not docs:
        raise ManifestError(f"No documents listed in {path}.")

    try:
        entries = [ManifestEntry(**d) for d in docs]
    except ValueError as e:
        raise ManifestError(f"Invalid manifest entry in {path}:\n{e}") from e

    seen: set[str] = set()
    for entry in entries:
        if entry.id in seen:
            raise ManifestError(f"Duplicate manifest id {entry.id!r}.")
        seen.add(entry.id)
    return entries


def verify_file(entry: ManifestEntry, root: Path | None = None) -> Path:
    """Return the entry's local path, after checking it exists and matches its hash."""
    path = entry.path(root or PathSettings().root)
    if not path.is_file():
        raise ManifestError(f"{entry.id}: file not found at {path}.")
    actual = sha256_file(path)
    if actual != entry.file_hash:
        raise ManifestError(
            f"{entry.id}: sha256 of {path.name} is {actual}, manifest says {entry.file_hash}. "
            "Re-download the document or update file_hash in the manifest."
        )
    return path
