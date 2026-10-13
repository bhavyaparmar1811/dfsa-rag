"""Application settings, loaded from environment variables and `.env`.

Each external provider has its own section with its own env prefix, so the
Azure OpenAI (generation) and OpenAI (embeddings) credentials can never be
mixed up.

`EmbeddingSettings` is the single source of truth for the embedding model and
dimension. Both ingestion and the query path must read it from here; never
hardcode the model name anywhere else.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


def _section(prefix: str) -> SettingsConfigDict:
    return SettingsConfigDict(
        env_prefix=prefix,
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )


class AzureOpenAISettings(BaseSettings):
    """Generation: GPT-4o on Azure OpenAI."""

    model_config = _section("AZURE_OPENAI_")

    endpoint: str
    api_key: SecretStr
    api_version: str = "2024-10-21"
    deployment: str = Field(
        "gpt-4o",
        validation_alias=AliasChoices("AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_DEPLOYMENT_NAME"),
    )

    # USD per 1M tokens, used for cost logging. Check against your Azure pricing.
    input_cost_per_1m: float = 2.50
    output_cost_per_1m: float = 10.00


class EmbeddingSettings(BaseSettings):
    """Embeddings: OpenAI API (not Azure). Used by ingestion AND queries."""

    model_config = _section("EMBEDDING_")

    # Read from OPENAI_API_KEY, not EMBEDDING_API_KEY. Optional here so that
    # chunking, loading SQLite and dry runs (which only need the model name and
    # dimension) work without it; EmbeddingClient refuses to start without it.
    api_key: SecretStr | None = Field(None, validation_alias="OPENAI_API_KEY")
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 100
    # Per-input token limit of the text-embedding-3 models. Inputs over this
    # raise instead of being silently truncated.
    max_input_tokens: int = 8191
    # Retries (with exponential backoff) for rate limits, timeouts and 5xx
    # errors; done by the OpenAI SDK itself.
    max_retries: int = 5
    timeout_s: float = 60.0

    # USD per 1M tokens, used for cost logging.
    cost_per_1m: float = 0.02

    @property
    def fingerprint(self) -> str:
        """Identifies the vector space, e.g. 'text-embedding-3-small-1536'.

        Stored with every chunk and used in the Qdrant collection name, so
        vectors from different models or dimensions never share a collection.
        """
        return f"{self.model}-{self.dimensions}"


class QdrantSettings(BaseSettings):
    model_config = _section("QDRANT_")

    # 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 (::1) first, and
    # the container only listens on 127.0.0.1, so every new connection waited ~2 s.
    url: str = "http://127.0.0.1:6333"
    api_key: SecretStr | None = None
    collection_prefix: str = "dfsa_rules"
    # The one sparse (BM25) encoder, for documents and queries alike. Like the
    # embedding fingerprint, it is recorded with the collection and with every
    # ingestion run; changing it means re-encoding the corpus.
    sparse_model: str = "Qdrant/bm25"
    # BM25 parameters. avg_len is the corpus's mean chunk length in BM25 tokens
    # (after stopwords and stemming): 419,845 tokens over 3,397 embedded chunks
    # in corpus v1. fastembed's default of 256 would under-penalise long chunks.
    # Document weights depend on all three, so they are part of the fingerprint.
    sparse_k: float = 1.2
    sparse_b: float = 0.75
    sparse_avg_len: float = 124.0
    timeout_s: int = 30
    upsert_batch_size: int = 100

    @property
    def sparse_fingerprint(self) -> str:
        """e.g. 'Qdrant/bm25-k1.2-b0.75-avg124': stored with the collection and every run."""
        params = f"k{self.sparse_k:g}-b{self.sparse_b:g}-avg{self.sparse_avg_len:g}"
        return f"{self.sparse_model}-{params}"


def collection_name(qdrant: QdrantSettings, embedding: EmbeddingSettings) -> str:
    """Collection name tied to the embedding model, e.g.
    'dfsa_rules__text-embedding-3-small-1536'. Changing the embedding model
    means a new collection and a full re-embed."""
    return f"{qdrant.collection_prefix}__{embedding.fingerprint}"


class ChunkingSettings(BaseSettings):
    model_config = _section("CHUNK_")

    # Token counts for sizing decisions. cl100k_base is the tokenizer of the
    # text-embedding-3 models; the embedding client still enforces the exact
    # per-input limit at embed time. Kept separate from EmbeddingSettings so
    # chunking runs without an OpenAI key.
    tokenizer: str = "cl100k_base"
    # Chunks longer than this are split (at sub-rules, or Guidance paragraphs).
    # 512 matches the reranker's input window, so no chunk is silently cut
    # off at rerank time; on COB this splits 18 of 698 chunks (2.6%).
    max_chunk_tokens: int = 512
    # Cross-encoders often read ~512 tokens; longer chunks are reported because
    # the reranker would only see their beginning. Approximate: the reranker's
    # own tokenizer differs, and the reranker logs real truncation itself.
    rerank_warn_tokens: int = 512


class ChatSettings(BaseSettings):
    model_config = _section("CHAT_")

    # Number of recent turns used for query rewriting and answer context.
    history_turns: int = 6


class PathSettings(BaseModel):
    root: Path = PROJECT_ROOT
    manifest: Path = PROJECT_ROOT / "data" / "manifest.yaml"
    raw_data: Path = PROJECT_ROOT / "data" / "raw"
    # Derived, regenerable artifacts (Docling output, chunks, reports). Gitignored.
    parsed_dir: Path = PROJECT_ROOT / "data" / "parsed"
    sqlite_db: Path = PROJECT_ROOT / "data" / "app.db"

    @property
    def sqlite_url(self) -> str:
        return f"sqlite:///{self.sqlite_db.as_posix()}"


class Settings(BaseModel):
    azure_openai: AzureOpenAISettings
    embedding: EmbeddingSettings
    qdrant: QdrantSettings
    chat: ChatSettings
    paths: PathSettings

    @property
    def qdrant_collection(self) -> str:
        return collection_name(self.qdrant, self.embedding)


@lru_cache
def get_settings() -> Settings:
    return Settings(
        azure_openai=AzureOpenAISettings(),
        embedding=EmbeddingSettings(),
        qdrant=QdrantSettings(),
        chat=ChatSettings(),
        paths=PathSettings(),
    )
