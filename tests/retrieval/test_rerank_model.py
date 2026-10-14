import pytest

from src.config.settings import RetrievalSettings
from src.retrieval.rerank import CrossEncoderModel


def test_the_default_rerankers_are_pinned():
    cfg = RetrievalSettings(_env_file=None)
    assert cfg.reranker_local_only
    assert cfg.reranker_revision == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    base = RetrievalSettings(_env_file=None, reranker_model="BAAI/bge-reranker-base")
    assert base.reranker_revision == "2cfc18c9415c912f9d8155881c133215df768a70"


def test_an_uncached_model_fails_with_a_hint_and_no_download():
    cfg = RetrievalSettings(_env_file=None, reranker_model="example/not-a-cached-reranker")
    with pytest.raises(RuntimeError, match="RETRIEVAL_RERANKER_LOCAL_ONLY=false"):
        CrossEncoderModel(cfg)
