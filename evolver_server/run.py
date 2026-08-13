"""Launcher for the Evolver API server.

    uv run python evolver_server/run.py --port 8100

Uses the uvicorn import-string form so ``--reload`` works.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the repo root is importable whether launched as a script
# (`python evolver_server/run.py`) or a module (`python -m evolver_server.run`).
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import uvicorn  # noqa: E402

from evolver_server.config import get_settings  # noqa: E402


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="SimpleMem Evolver API server")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    print("=" * 60)
    print("  SimpleMem Evolver API")
    print(f"  http://{args.host}:{args.port}")
    print("  Routes: POST /memory/{add,add_batch,retrieve,clear,stats}")
    print("          GET  /health   |   docs: /docs")
    print("=" * 60)

    uvicorn.run(
        "evolver_server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        app_dir=_REPO_ROOT,
    )


if __name__ == "__main__":
    main()
