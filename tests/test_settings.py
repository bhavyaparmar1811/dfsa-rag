import pytest

from src.config.settings import get_settings


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_provider_credentials_stay_separate(env):
    settings = get_settings()
    assert settings.azure_openai.api_key.get_secret_value() == "azure-key"
    assert settings.embedding.api_key.get_secret_value() == "openai-key"


def test_collection_name_tracks_embedding_model(env, monkeypatch):
    assert get_settings().qdrant_collection == "dfsa_rules__text-embedding-3-small-1536"

    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "512")
    get_settings.cache_clear()
    assert get_settings().qdrant_collection == "dfsa_rules__text-embedding-3-small-512"


def test_manifest_path_exists(env):
    assert get_settings().paths.manifest.is_file()
