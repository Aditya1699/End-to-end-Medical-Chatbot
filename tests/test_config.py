from pathlib import Path

import pytest

from src.config import AppConfig, ConfigError


def test_runtime_validation_requires_pinecone_key(tmp_path):
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    config = AppConfig(base_dir=tmp_path, pinecone_api_key=None, model_path=model)

    with pytest.raises(ConfigError, match="PINECONE_API_KEY"):
        config.validate_for_runtime()


def test_runtime_validation_requires_existing_model(tmp_path):
    config = AppConfig(base_dir=tmp_path, pinecone_api_key="test-key", model_path=tmp_path / "missing.bin")

    with pytest.raises(ConfigError, match="Model file"):
        config.validate_for_runtime()


def test_from_env_resolves_relative_model_path(monkeypatch, tmp_path):
    monkeypatch.setenv("PINECONE_API_KEY", "test-key")
    monkeypatch.setenv("MODEL_PATH", "model/test.bin")

    config = AppConfig.from_env(tmp_path)

    assert config.model_path == Path(tmp_path / "model/test.bin")
