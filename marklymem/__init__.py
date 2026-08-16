"""MarklyMem Evolver API — standalone FastAPI REST server over the evolver.

Exposes the evolver's hybrid memory engine (simplemem/evolver) as a small set of
REST endpoints. No authentication; every endpoint requires a caller-supplied
``namespace`` that isolates where memories are stored and retrieved.
"""
