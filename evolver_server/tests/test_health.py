# pyright: reportMissingImports=false
"""E2E tests for the /health endpoint."""

from __future__ import annotations


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
