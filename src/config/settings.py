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

    # Read from OPENAI_API_KEY, not EMBEDDING_API_KEY.
    api_key: SecretStr = Field(validation_alias="OPENAI_API_KEY")
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 100

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

    url: str = "http://localhost:6333"
    api_key: SecretStr | None = None
    collection_prefix: str = "dfsa_rules"


class ChatSettings(BaseSettings):
    model_config = _section("CHAT_")

    # Number of recent turns used for query rewriting and answer context.
    history_turns: int = 6


class PathSettings(BaseModel):
    root: Path = PROJECT_ROOT
    manifest: Path = PROJECT_ROOT / "data" / "manifest.yaml"
    raw_data: Path = PROJECT_ROOT / "data" / "raw"
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
        """Collection name tied to the embedding model, e.g.
        'dfsa_rules__text-embedding-3-small-1536'. Changing the embedding model
        means a new collection and a full re-embed."""
        return f"{self.qdrant.collection_prefix}__{self.embedding.fingerprint}"


@lru_cache
def get_settings() -> Settings:
    return Settings(
        azure_openai=AzureOpenAISettings(),
        embedding=EmbeddingSettings(),
        qdrant=QdrantSettings(),
        chat=ChatSettings(),
        paths=PathSettings(),
    )
