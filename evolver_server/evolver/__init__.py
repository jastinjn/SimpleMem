"""
EvolveMem — Self-Evolving Memory Architecture for LLM Agents.

A typed memory system with hybrid retrieval and automatic consolidation,
backed by PostgreSQL + pgvector.
"""

from .consolidator import MemoryConsolidator
from .manager import MemoryManager
from .models import MemoryQuery, MemoryStatus, MemoryType, MemoryUnit
from .store import MemoryStore

__all__ = [
    "MemoryManager",
    "MemoryStore",
    "MemoryConsolidator",
    "MemoryQuery",
    "MemoryStatus",
    "MemoryType",
    "MemoryUnit",
]
