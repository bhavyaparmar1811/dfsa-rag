from src.ingestion import parse


def test_cache_key_changes_with_pipeline_options(monkeypatch):
    before = parse.cache_key("ab" * 32)
    monkeypatch.setitem(parse.PIPELINE_OPTIONS, "do_ocr", True)
    assert parse.cache_key("ab" * 32) != before


def test_cache_key_is_per_document():
    assert parse.cache_key("ab" * 32) != parse.cache_key("cd" * 32)
