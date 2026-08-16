"""Launcher for the Evolver API server.

    uv run python run.py

Uses the uvicorn import-string form so ``--reload`` works.
"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from marklymem.config import get_settings


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="SimpleMem Evolver API server")
    parser.add_argument("--host", default=settings.FASTAPI_HOST)
    parser.add_argument("--port", type=int, default=settings.FASTAPI_PORT)
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    print("=" * 60)
    print("  SimpleMem Evolver API")
    print(f"  http://{args.host}:{args.port}")
    print("  Routes: POST /memory/{add_dialogue,retrieve,clear,stats,clone_namespace}")
    print("          GET  /health   |   docs: /docs")
    print("=" * 60)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    uvicorn.run(
        "marklymem.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
