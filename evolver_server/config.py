"""Settings for the Evolver API server.

Env-backed dataclass with an ``lru_cache``d accessor, mirroring the pattern in
``MCP/config/settings.py``. Every value resolves from an environment variable
with a built-in default, so the server runs with zero configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass
class Settings:
    # HTTP server
    host: str = field(default_factory=lambda: os.getenv("EVOLVER_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("EVOLVER_PORT", "8100")))

    # Evolver store — single shared SQLite DB; tenant isolation is via scope_id.
    db_path: str = field(
        default_factory=lambda: os.getenv("EVOLVER_DB_PATH", "~/.simplemem/evolver_server.db")
    )

    # Retrieval behaviour
    retrieval_mode: str = field(default_factory=lambda: os.getenv("EVOLVER_RETRIEVAL_MODE", "hybrid"))
    default_top_k: int = field(default_factory=lambda: int(os.getenv("EVOLVER_DEFAULT_TOP_K", "10")))

    # CORS — comma-separated origins, or "*" for any.
    cors_allowed_origins: str = field(
        default_factory=lambda: os.getenv("EVOLVER_CORS_ALLOWED_ORIGINS", "*")
    )


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
