# pyright: reportMissingImports=false
"""E2E tests for the /health endpoint."""

from __future__ import annotations


async def test_health(authed_client):
    r = await authed_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


async def test_health_exempt_from_auth(unauthed_client):
    assert (await unauthed_client.get("/health")).status_code == 200
