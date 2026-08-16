from __future__ import annotations

import os
from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    FASTAPI_HOST: str = "localhost"
    FASTAPI_PORT: int = 8100
    DATABASE_URL: str = "postgresql+asyncpg://postgres:password@localhost:5442/simplemem"
    embedding_dim: int = 1024
    db_pool_size: int = 10
    db_max_overflow: int = 20
    ingestion_mode: str = "llm" # "pattern" or "llm"
    retrieval_mode: str = "hybrid" # "keyword", "embedding", "hybrid", or "auto"
    embedder_mode: str = "semantic" # "hashing" or "semantic"
    default_top_k: int = 10
    cors_allowed_origins: str = "*"
    OPENAI_API_KEY: str = ""

    # Service-to-service auth
    APP_ENV: str = "local"       # set to "production" (or anything non-"local") in AWS
    INTERNAL_API_KEY: str = ""   # generate with: python -c "import secrets; print(secrets.token_hex(32))"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # OpenTelemetry → self-hosted Langfuse (OTLP/HTTP). Tracing is off unless
    # OTEL_ENABLED is true AND langfuse host + keys are set. When on, spans include
    # raw dialogue / memory text.
    OTEL_ENABLED: bool = False
    LANGFUSE_HOST: str = ""            # e.g. http://localhost:3000
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @model_validator(mode="after")
    def _strip_internal_api_key_from_env(self) -> Settings:
        os.environ.pop("INTERNAL_API_KEY", None)
        return self

@lru_cache
def get_settings() -> Settings:
    return Settings()
