from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8100
    db_path: str = "~/.simplemem/evolver_server.db"
    retrieval_mode: str = "hybrid"
    default_top_k: int = 10
    cors_allowed_origins: str = "*"

    model_config = SettingsConfigDict(env_prefix="EVOLVER_", env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
