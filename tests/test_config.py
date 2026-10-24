from __future__ import annotations

import pytest
from pydantic import ValidationError

from config import DEFAULT_DATABASE_URL, Settings


def clear_settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "GROQ_API_KEY",
        "GROQ_MODEL_FAST",
        "GROQ_MODEL_REASONING",
        "RETRIEVAL_CONFIDENCE_THRESHOLD",
        "DATABASE_URL",
        "PREDICT_MODEL",
        "PREDICT_FALLBACK_MODEL",
        "PREDICT_LLM_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_settings_load_valid_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_settings_env(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_placeholder")
    monkeypatch.setenv("RETRIEVAL_CONFIDENCE_THRESHOLD", "0.42")
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@host/db")

    settings = Settings(_env_file=None)

    assert settings.groq_api_key == "gsk_test_placeholder"
    assert settings.retrieval_confidence_threshold == 0.42
    assert settings.database_url == "postgresql+asyncpg://u:p@host/db"


def test_settings_database_url_defaults_to_local_sqlite(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_settings_env(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_placeholder")

    settings = Settings(_env_file=None)

    assert settings.database_url == DEFAULT_DATABASE_URL


def test_settings_missing_groq_key_raises_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_settings_env(monkeypatch)

    with pytest.raises(ValidationError, match="GROQ_API_KEY"):
        Settings(_env_file=None)


def test_settings_malformed_threshold_raises_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_settings_env(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_placeholder")
    monkeypatch.setenv("RETRIEVAL_CONFIDENCE_THRESHOLD", "not-a-float")

    with pytest.raises(ValidationError, match="RETRIEVAL_CONFIDENCE_THRESHOLD"):
        Settings(_env_file=None)


def test_predict_chain_defaults_to_gemma4_then_groq(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_settings_env(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_placeholder")
    monkeypatch.setenv("GROQ_MODEL_FAST", "qwen/qwen3.8-27b")

    settings = Settings(_env_file=None)

    assert settings.predict_model_chain() == ("gemma4", "groq/qwen/qwen3.8-27b")
    assert settings.predict_llm_timeout_seconds == 30.0


@pytest.mark.parametrize("disabled", ["none", "OFF", ""])
def test_predict_fallback_can_be_disabled(monkeypatch: pytest.MonkeyPatch, disabled: str) -> None:
    clear_settings_env(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_placeholder")
    monkeypatch.setenv("PREDICT_FALLBACK_MODEL", disabled)

    assert Settings(_env_file=None).predict_model_chain() == ("gemma4",)


def test_predict_chain_honours_explicit_models(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_settings_env(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_placeholder")
    monkeypatch.setenv("PREDICT_MODEL", "ollama_chat/gemma2:2b")
    monkeypatch.setenv("PREDICT_FALLBACK_MODEL", "groq/openai/gpt-oss-20b")

    assert Settings(_env_file=None).predict_model_chain() == (
        "ollama_chat/gemma2:2b",
        "groq/openai/gpt-oss-20b",
    )
