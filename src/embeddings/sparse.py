"""The one sparse (BM25) encoder, used by ingestion AND the query path.

fastembed's `Qdrant/bm25` runs locally (no API, no cost). It turns text into
{token id: weight}, where token ids are hashes of stemmed words:

    documents   BM25 term-frequency weights, normalised by length (k, b, avg_len)
    queries     weight 1 per distinct term

Qdrant applies the IDF part itself (the collection's sparse vector has
`Modifier.IDF`), so IDF always reflects what is in the collection right now.

Documents and queries must go through the same encoder with the same settings,
just like dense embeddings: `QdrantSettings.sparse_fingerprint` identifies them
and is checked against the collection.
"""

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass

from fastembed import SparseTextEmbedding

from src.config.settings import QdrantSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]

    def __len__(self) -> int:
        return len(self.indices)


class SparseEncoder:
    def __init__(self, cfg: QdrantSettings, model: SparseTextEmbedding | None = None):
        self.cfg = cfg
        self.fingerprint = cfg.sparse_fingerprint
        self._model = model or SparseTextEmbedding(
            cfg.sparse_model, k=cfg.sparse_k, b=cfg.sparse_b, avg_len=cfg.sparse_avg_len
        )

    def encode_documents(self, texts: list[str]) -> list[SparseVector]:
        start = time.perf_counter()
        vectors = [_to_vector(e) for e in self._model.embed(texts, batch_size=256)]
        logger.info(
            "BM25-encoded %d documents with %s in %.0f ms",
            len(texts), self.fingerprint, (time.perf_counter() - start) * 1000,
        )  # fmt: skip
        return vectors

    def encode_query(self, text: str) -> SparseVector:
        return _to_vector(next(iter(self._model.query_embed(text))))

    def token_count(self, texts: Iterable[str]) -> int:
        """BM25 tokens (after stopwords and stemming): what avg_len is measured in."""
        return self._model.token_count(list(texts))

    def vocabulary(self, text: str) -> dict[int, str]:
        """{token id: stemmed word} for `text`, to make a sparse vector readable.

        BM25 ids are one-way hashes, so this re-tokenises the text. It uses
        fastembed internals and is for inspection only, never for retrieval.
        """
        inner = self._model.model
        try:
            words = inner._stem(inner.tokenizer.tokenize(text))
        except AttributeError:  # fastembed changed its internals
            return {}
        return {inner.compute_token_id(w): w for w in words}


def _to_vector(embedding) -> SparseVector:
    return SparseVector(embedding.indices.tolist(), embedding.values.tolist())
