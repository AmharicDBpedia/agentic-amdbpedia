"""Runtime configuration for the mapping assistant."""

from __future__ import annotations

from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_RETRIEVAL_CONFIDENCE_THRESHOLD = 0.35
# A local SQLite file by default -- zero-config for local dev; the real
# docker-compose Postgres service is opt-in via DATABASE_URL (refs 14.1).
DEFAULT_DATABASE_URL = "sqlite+aiosqlite:///./data/review_queue.db"
DEFAULT_MEDIAWIKI_BASE_URL = "https://mappings.dbpedia.org"
# Pipeline LLM (rerank + propose-from-scratch). The primary model is the
# local Gemma 2 9B via Ollama (rag/predict.py's registry alias); when Ollama
# is unreachable or too slow, the same call is retried on the fallback model
# -- by default Groq's GROQ_MODEL_FAST, so a configured GROQ_API_KEY alone is
# enough for the LLM layer to work. Set PREDICT_FALLBACK_MODEL=none to
# disable the fallback entirely.
DEFAULT_PREDICT_MODEL = "gemma2"
# Gemma 2 9B on CPU (no usable GPU) takes several seconds for the first
# token, and much longer on the first call while Ollama loads the weights.
DEFAULT_PREDICT_LLM_TIMEOUT_SECONDS = 60.0
PREDICT_FALLBACK_DISABLED = {"", "none", "off", "false", "0"}


class Settings(BaseSettings):
    """Environment-backed application settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def __init__(self, **values: Any) -> None:
        super().__init__(**values)

    groq_api_key: str = Field(alias="GROQ_API_KEY")
    # llama-3.1-8b-instant/llama-3.3-70b-versatile (the original defaults) were
    # removed from Groq's model catalog at some point after this project was
    # last run live -- confirmed directly: both now 400/401 on a real key.
    # qwen/qwen3.8-27b is confirmed live (classify() JSON mode + full ReAct
    # tool-calling) as of 2026-09-02; re-check Groq's current catalog
    # (GET https://api.groq.com/openai/v1/models) if this goes stale again.
    groq_model_fast: str = Field(default="qwen/qwen3.8-27b", alias="GROQ_MODEL_FAST")
    groq_model_reasoning: str = Field(default="qwen/qwen3.8-27b", alias="GROQ_MODEL_REASONING")
    retrieval_confidence_threshold: float = Field(
        default=DEFAULT_RETRIEVAL_CONFIDENCE_THRESHOLD,
        ge=0.0,
        le=1.0,
        alias="RETRIEVAL_CONFIDENCE_THRESHOLD",
    )
    predict_model: str = Field(default=DEFAULT_PREDICT_MODEL, alias="PREDICT_MODEL")
    # None -> "groq/<GROQ_MODEL_FAST>" (see predict_model_chain()).
    predict_fallback_model: str | None = Field(default=None, alias="PREDICT_FALLBACK_MODEL")
    predict_llm_timeout_seconds: float = Field(
        default=DEFAULT_PREDICT_LLM_TIMEOUT_SECONDS, gt=0.0, alias="PREDICT_LLM_TIMEOUT_SECONDS"
    )
    database_url: str = Field(default=DEFAULT_DATABASE_URL, alias="DATABASE_URL")
    # Bot Password only (Special:BotPasswords) -- never a real MediaWiki
    # account password. Both unset by default; publish_mapping() fails
    # clearly (MediaWikiCredentialsError) rather than attempting a request
    # with empty credentials (refs 14.3).
    mediawiki_base_url: str = Field(default=DEFAULT_MEDIAWIKI_BASE_URL, alias="MEDIAWIKI_BASE_URL")
    mediawiki_bot_username: str | None = Field(default=None, alias="MEDIAWIKI_BOT_USERNAME")
    mediawiki_bot_password: str | None = Field(default=None, alias="MEDIAWIKI_BOT_PASSWORD")

    def predict_model_chain(self) -> tuple[str, ...]:
        """Models the pipeline LLM tries in order: primary, then fallback."""

        fallback = self.predict_fallback_model
        if fallback is None:
            fallback = f"groq/{self.groq_model_fast}"
        chain = [self.predict_model]
        if fallback.strip().casefold() not in PREDICT_FALLBACK_DISABLED and fallback not in chain:
            chain.append(fallback.strip())
        return tuple(chain)
