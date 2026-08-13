# pyright: reportMissingImports=false
"""E2E tests for the /health endpoint."""

from __future__ import annotations


async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
