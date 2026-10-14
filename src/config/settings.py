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
from typing import Literal

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

    # The OpenAI SDK retries rate limits, timeouts and 5xx errors itself, with backoff.
    max_retries: int = 3
    timeout_s: float = 60.0

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
    # fastembed's default cache is the system temp directory, which gets cleaned.
    sparse_cache_dir: Path = PROJECT_ROOT / "data" / "cache" / "fastembed"
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


class RetrievalSettings(BaseSettings):
    """Question -> rule-level chunks -> LLM context. Every stage is switchable, so
    evals can compare dense / hybrid / hybrid + rerank and the parent policies."""

    model_config = _section("RETRIEVAL_")

    # --- candidates (recall) ---
    # dense: embeddings only; sparse: BM25 only; hybrid: both, fused by RRF.
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    candidate_k: int = 40  # per retriever, and after fusion
    rrf_k: int = 60  # reciprocal rank fusion: score = sum(1 / (rrf_k + rank))
    # Exact citations in the question ("COB 3.3.2") looked up directly; kept first.
    citation_lookup: bool = True
    max_citation_hits: int = 8

    # --- rerank (precision) ---
    rerank: bool = True
    # How many of the fused candidates the cross-encoder scores (plus any exact
    # citation hits); time is about linear in it. On a GPU all 40 take ~1.5 s with
    # bge-reranker-v2-m3; on CPU the same takes ~90 s, so lower it (or use
    # bge-reranker-base) on a machine without one.
    rerank_k: int = 40
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # Pinned Hugging Face commits of the rerankers we use, so a model can't change
    # under the eval results. A model not listed here loads its latest commit (logged).
    reranker_revisions: dict[str, str] = Field(
        default_factory=lambda: {
            "BAAI/bge-reranker-v2-m3": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
            "BAAI/bge-reranker-base": "2cfc18c9415c912f9d8155881c133215df768a70",
        }
    )
    # Load the reranker from the local Hugging Face cache only: no request to the Hub
    # at startup, and it works offline. Set false once to download a model (or a
    # newly pinned commit) that isn't cached yet.
    reranker_local_only: bool = True
    reranker_max_length: int = 512  # tokens of (question, chunk); longer pairs are cut
    reranker_batch_size: int = 16
    reranker_device: str | None = None  # None: "cuda" if a GPU is available, else "cpu"
    reranker_fp16: bool = True  # half precision, on a GPU only
    top_n: int = 6

    @property
    def reranker_revision(self) -> str | None:
        return self.reranker_revisions.get(self.reranker_model)

    # The top reranker score is the confidence signal. None = not calibrated yet:
    # Phase H calibrates it on the golden set. Never guess it.
    refusal_threshold: float | None = None

    # --- context assembly (what the LLM sees) ---
    # none: the hits only; section: the whole section when it fits in
    # parent_max_tokens, else a window; window: always a window of siblings.
    parent_mode: Literal["none", "section", "window"] = "section"
    parent_max_tokens: int = 1500
    parent_window: int = 1  # siblings on each side of a hit
    section_hit_min: int = 3  # this many hits in one section -> the whole section
    include_scope_rules: bool = True  # "does not apply to ..." rules change answers
    include_guidance: bool = True  # a hit rule's Guidance, and a hit Guidance's rule
    expand_refs: bool = False  # one hop along cross-references
    max_ref_blocks: int = 4
    include_definitions: bool = True
    max_definitions: int = 6  # in all
    max_hit_definitions: int = 3  # of terms the hits use but the question doesn't
    # Terms used in more than this share of chunks are too common to be worth
    # defining (unless the question itself uses them).
    term_max_share: float = 0.05
    skip_terms: list[str] = Field(
        default_factory=lambda: [
            "Rule", "Rules", "DFSA", "DIFC", "Person", "person", "Law", "Module",
            "GEN", "COB", "AML", "MKT", "FPR", "GLO",
        ]
    )  # fmt: skip
    context_budget_tokens: int = 6000


class LangfuseSettings(BaseSettings):
    """Tracing and the prompt store: self-hosted Langfuse (infra/docker-compose.yml).

    Optional: without keys, or with LANGFUSE_ENABLED=false, tracing is a no-op and
    prompts come from the git copies in prompts/, so tests and offline runs work.
    """

    model_config = _section("LANGFUSE_")

    enabled: bool = True
    # 127.0.0.1, not localhost (see QdrantSettings.url).
    host: str = "http://127.0.0.1:3000"
    public_key: str | None = None
    secret_key: SecretStr | None = None
    # Separates traces by where they ran; lowercase, no "langfuse" prefix.
    environment: str = "development"
    timeout_s: int = 10
    sample_rate: float = 1.0

    # --- prompts ---
    # The label the app serves. Evals run a candidate with label "staging".
    prompt_label: str = "production"
    # The SDK caches each prompt this long, and keeps serving the stale copy if
    # a refresh fails.
    prompt_cache_ttl_s: int = 60
    # A first fetch that fails falls back to the git copy; keep it short so an
    # unreachable Langfuse doesn't stall a request. No retries: once a prompt is
    # cached, a failed refresh serves the stale copy anyway.
    prompt_fetch_timeout_s: int = 3
    prompt_max_retries: int = 0

    @property
    def active(self) -> bool:
        """Enabled, and both keys set."""
        return self.enabled and bool(self.public_key and self.secret_key)


class ChatSettings(BaseSettings):
    model_config = _section("CHAT_")

    # Recent messages (one turn = one message) used for query rewriting and answer
    # context. Loaded from SQLite, never taken from the client.
    history_turns: int = 6
    # Each history message is cut to this many characters, to bound token cost.
    history_max_chars: int = 1500
    # A longer user message is rejected (422).
    max_message_chars: int = 2000


class ApiSettings(BaseSettings):
    """FastAPI. Every endpoint except /health needs the X-API-Key header."""

    model_config = _section("API_")

    key: SecretStr | None = None  # API_KEY
    # Without a key the app refuses to start, unless this is set (local experiments).
    # Bind it to 127.0.0.1: uvicorn src.api.app:app --host 127.0.0.1 --port 8000
    auth_disabled: bool = False


class PathSettings(BaseModel):
    root: Path = PROJECT_ROOT
    manifest: Path = PROJECT_ROOT / "data" / "manifest.yaml"
    raw_data: Path = PROJECT_ROOT / "data" / "raw"
    # Derived, regenerable artifacts (Docling output, chunks, reports). Gitignored.
    parsed_dir: Path = PROJECT_ROOT / "data" / "parsed"
    sqlite_db: Path = PROJECT_ROOT / "data" / "app.db"
    # Prompts live in git; Langfuse holds the versions and labels (Phase E3).
    prompts_dir: Path = PROJECT_ROOT / "prompts"

    @property
    def sqlite_url(self) -> str:
        return f"sqlite:///{self.sqlite_db.as_posix()}"


class Settings(BaseModel):
    azure_openai: AzureOpenAISettings
    embedding: EmbeddingSettings
    qdrant: QdrantSettings
    langfuse: LangfuseSettings
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
        langfuse=LangfuseSettings(),
        chat=ChatSettings(),
        paths=PathSettings(),
    )
