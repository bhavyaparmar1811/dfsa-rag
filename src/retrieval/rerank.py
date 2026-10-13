"""Cross-encoder reranking (precision), and the confidence signal.

A cross-encoder reads (question, chunk) together and scores how well the chunk
answers the question: slower than vector search but far more precise, so it
runs only on the ~40 candidates. Scores go through a sigmoid (0..1), so the top
score can be thresholded for refusal (calibrated on the eval set, Phase H).

The reranker sees the chunk's contextual header + text, exactly what was
embedded. Most cross-encoders read ~512 tokens: a longer (question, chunk) pair
is cut off, which for legal text can hide the operative clause, so every
truncation is counted with the reranker's own tokenizer and logged.

Load the model once (at app start), not per request: it takes seconds.
"""

import logging
import time
from typing import Protocol

from src.config.settings import RetrievalSettings
from src.retrieval.models import Candidate

logger = logging.getLogger(__name__)


class ScoringModel(Protocol):
    """What the Reranker needs from a model; the tests use a fake."""

    def score(self, question: str, texts: list[str]) -> list[float]: ...

    def count_tokens(self, question: str, text: str) -> int: ...


class CrossEncoderModel:
    """sentence-transformers CrossEncoder, scores in 0..1."""

    def __init__(self, cfg: RetrievalSettings):
        import torch
        from sentence_transformers import CrossEncoder

        start = time.perf_counter()
        self.name = cfg.reranker_model
        self.batch_size = cfg.reranker_batch_size
        self.device = cfg.reranker_device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Half precision on a GPU: half the memory (v2-m3 is ~1.1 GB instead of ~2.3 GB)
        # and tensor cores. CPUs gain nothing from it, so they stay in float32.
        half = cfg.reranker_fp16 and self.device.startswith("cuda")
        self.model = CrossEncoder(
            cfg.reranker_model,
            max_length=cfg.reranker_max_length,
            device=self.device,
            activation_fn=torch.nn.Sigmoid(),
            model_kwargs={"dtype": torch.float16} if half else None,
        )
        self.tokenizer = self.model.tokenizer
        self.dtype = str(next(self.model.parameters()).dtype).removeprefix("torch.")
        logger.info(
            "Loaded reranker %s on %s (%s) in %.1f s",
            self.name, self.device, self.dtype, time.perf_counter() - start,
        )  # fmt: skip

    def score(self, question: str, texts: list[str]) -> list[float]:
        pairs = [(question, t) for t in texts]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]

    def count_tokens(self, question: str, text: str) -> int:
        return len(self.tokenizer(question, text, truncation=False)["input_ids"])


class Reranker:
    def __init__(self, model: ScoringModel, max_length: int):
        self.model = model
        self.max_length = max_length

    def rerank(self, question: str, candidates: list[Candidate]) -> list[Candidate]:
        """Score every candidate (in place) and return them best first."""
        if not candidates:
            return []
        texts = [c.text_for_rerank for c in candidates]
        for c, text in zip(candidates, texts, strict=True):
            n = self.model.count_tokens(question, text)
            c.truncated = n > self.max_length
            if c.truncated:
                logger.warning(
                    "rerank: %s is %d tokens with the question, over %d; the reranker "
                    "sees only its start", c.key, n, self.max_length,
                )  # fmt: skip
        for c, score in zip(candidates, self.model.score(question, texts), strict=True):
            c.rerank_score = score
        return sorted(candidates, key=lambda c: -c.rerank_score)
