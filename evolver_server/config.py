from __future__ import annotations

from functools import lru_cache

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
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
