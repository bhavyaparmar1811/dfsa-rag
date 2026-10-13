"""The real fastembed BM25 encoder (local; skipped if the model can't be fetched)."""

import pytest

from src.embeddings.sparse import SparseEncoder
from tests.factories import qdrant_settings


@pytest.fixture(scope="module")
def encoder():
    try:
        return SparseEncoder(qdrant_settings())
    except Exception as e:  # first use downloads the model
        pytest.skip(f"fastembed Qdrant/bm25 unavailable: {e}")


def test_settings_reach_the_model(encoder):
    assert encoder.fingerprint == "Qdrant/bm25-k1.2-b0.75-avg124"
    assert encoder._model.model.avg_len == 124.0


def test_documents_are_tf_weighted_and_queries_are_binary(encoder):
    doc = encoder.encode_documents(
        ["A Retail Client is a Client who is not a Professional Client."]
    )[0]
    query = encoder.encode_query("retail client")
    assert len(query) == 2 and set(query.values) == {1.0}
    assert set(query.indices) <= set(doc.indices)  # same stemming and hashing on both sides
    vocab = encoder.vocabulary("A Retail Client is a Client")
    weights = {vocab[i]: v for i, v in zip(doc.indices, doc.values, strict=True) if i in vocab}
    assert weights["client"] > weights["retail"]  # "client" occurs three times


def test_stopwords_are_dropped(encoder):
    assert len(encoder.encode_query("the of and")) == 0
