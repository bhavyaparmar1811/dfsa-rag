import hashlib

import pytest
import yaml

from src.ingestion.manifest import ManifestError, load_manifest, verify_file

ENTRY = {
    "id": "dfsa-rulebook-cob",
    "regulator": "DFSA",
    "rulebook": "DFSA Rulebook",
    "module_code": "COB",
    "module_name": "Conduct of Business Module",
    "version": "51",
    "effective_date": "2026-07",
    "source_url": "https://example.com/cob.pdf",
    "file_path": "data/raw/cob.pdf",
    "downloaded_on": "2026-09-26",
    "file_hash": "0" * 64,
}


def write_manifest(tmp_path, *entries):
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump({"documents": list(entries)}), encoding="utf-8")
    return path


def test_loads_entries_and_doc_types(tmp_path):
    law = {**ENTRY, "id": "dfsa-laws-markets", "rulebook": "DIFC Laws", "module_code": "MKT_LAW"}
    entries = load_manifest(write_manifest(tmp_path, ENTRY, law))

    assert [e.doc_type for e in entries] == ["rulebook", "law"]


def test_hash_is_normalised_to_lowercase(tmp_path):
    entry = load_manifest(write_manifest(tmp_path, {**ENTRY, "file_hash": "AB" * 32}))[0]
    assert entry.file_hash == "ab" * 32


@pytest.mark.parametrize(
    "bad_path", [r"D:\Projects\DFSA RAG\data\raw\cob.pdf", "/data/raw/cob.pdf"]
)
def test_rejects_absolute_paths(tmp_path, bad_path):
    with pytest.raises(ManifestError, match="relative"):
        load_manifest(write_manifest(tmp_path, {**ENTRY, "file_path": bad_path}))


def test_rejects_duplicate_ids(tmp_path):
    with pytest.raises(ManifestError, match="Duplicate"):
        load_manifest(write_manifest(tmp_path, ENTRY, ENTRY))


def test_verify_file_checks_hash(tmp_path):
    pdf = tmp_path / "data" / "raw" / "cob.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.7 fake")
    good_hash = hashlib.sha256(b"%PDF-1.7 fake").hexdigest()

    entry = load_manifest(write_manifest(tmp_path, {**ENTRY, "file_hash": good_hash}))[0]
    assert verify_file(entry, root=tmp_path) == pdf

    entry = load_manifest(write_manifest(tmp_path, ENTRY))[0]
    with pytest.raises(ManifestError, match="sha256"):
        verify_file(entry, root=tmp_path)


def test_verify_file_reports_missing_file(tmp_path):
    entry = load_manifest(write_manifest(tmp_path, ENTRY))[0]
    with pytest.raises(ManifestError, match="not found"):
        verify_file(entry, root=tmp_path)


def test_real_manifest_is_valid():
    """Every entry in the committed manifest parses (no file access)."""
    entries = load_manifest()
    assert {e.module_code for e in entries} >= {"GEN", "COB", "AML", "MKT", "FPR", "GLO"}
