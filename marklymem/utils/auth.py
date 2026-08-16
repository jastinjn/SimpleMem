from __future__ import annotations

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from marklymem.config import get_settings

_api_key_header = APIKeyHeader(name="API-Key", auto_error=False)


async def verify_internal_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    settings = get_settings()
    if settings.APP_ENV == "local":
        return
    if api_key != settings.INTERNAL_API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")
