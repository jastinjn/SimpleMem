# pyright: reportMissingImports=false
"""Shared test-corpus helpers for both the evolver unit tests and the API integration tests.

Corpus layout (returned by create_test_units):
  USER_ID  + SCOPE       — 6 units (unit-001..006): diverse types / topics
  USER_ID  + OTHER_NAMESPACE — 2 units (unit-s01..s02): Terraform / CloudFormation
  OTHER_USER + SCOPE     — 1 unit  (unit-u01):       Ansible

All updated_at values are distinct so any (score, updated_at) sort is total.
"""

from __future__ import annotations

from marklymem.evolver.models import MemoryType, MemoryUnit

# --- Identity constants ---
USER_ID = "user-test"
OTHER_USER = "user-bob"
SCOPE = "test"
OTHER_NAMESPACE = "test-b"

# --- Corpus size constants ---
CORPUS_SIZE = 6       # units for USER_ID + SCOPE
OTHER_NAMESPACE_SIZE = 2  # units for USER_ID + OTHER_NAMESPACE
OTHER_USER_SIZE = 1   # units for OTHER_USER + SCOPE


def create_test_units() -> list[MemoryUnit]:
    """Return the fixed multi-user corpus with explicit timestamps."""
    def ts(n: int) -> str:
        return f"2025-01-15T14:00:{n:02d}+00:00"

    return [
        # --- primary user, primary namespace ---
        MemoryUnit(
            memory_id="unit-001",
            user_id=USER_ID,
            namespace=SCOPE,
            memory_type=MemoryType.SEMANTIC,
            content="The project uses PostgreSQL as the primary database",
            entities=["PostgreSQL", "database"],
            topics=["infrastructure", "database"],
            tags=["db", "infra"],
            importance=0.8,
            confidence=0.9,
            created_at=ts(1),
            updated_at=ts(1),
        ),
        MemoryUnit(
            memory_id="unit-002",
            user_id=USER_ID,
            namespace=SCOPE,
            memory_type=MemoryType.EPISODIC,
            content="Alice and Bob discussed the authentication strategy for the API",
            entities=["Alice", "Bob", "API"],
            topics=["authentication", "api"],
            tags=["auth", "api"],
            importance=0.6,
            confidence=0.8,
            created_at=ts(2),
            updated_at=ts(2),
        ),
        MemoryUnit(
            memory_id="unit-003",
            user_id=USER_ID,
            namespace=SCOPE,
            memory_type=MemoryType.PREFERENCE,
            content="User prefers TypeScript over JavaScript for frontend development",
            entities=["TypeScript", "JavaScript"],
            topics=["frontend", "typescript"],
            tags=["frontend", "ts"],
            importance=0.7,
            confidence=0.85,
            created_at=ts(3),
            updated_at=ts(3),
        ),
        MemoryUnit(
            memory_id="unit-004",
            user_id=USER_ID,
            namespace=SCOPE,
            memory_type=MemoryType.PROJECT_STATE,
            content="The deployment pipeline uses Kubernetes for container orchestration",
            entities=["Kubernetes", "deployment"],
            topics=["deployment", "kubernetes"],
            tags=["k8s", "deploy"],
            importance=0.75,
            confidence=0.9,
            created_at=ts(4),
            updated_at=ts(4),
        ),
        MemoryUnit(
            memory_id="unit-005",
            user_id=USER_ID,
            namespace=SCOPE,
            memory_type=MemoryType.PROCEDURAL_OBSERVATION,
            content="Running tests requires the pytest framework with coverage enabled",
            entities=["pytest"],
            topics=["testing", "ci"],
            tags=["test", "pytest"],
            importance=0.5,
            confidence=0.7,
            created_at=ts(5),
            updated_at=ts(5),
        ),
        MemoryUnit(
            memory_id="unit-006",
            user_id=USER_ID,
            namespace=SCOPE,
            memory_type=MemoryType.SEMANTIC,
            content="Redis is used for caching session tokens and rate limiting",
            entities=["Redis", "session"],
            topics=["caching", "authentication"],
            tags=["cache", "auth"],
            importance=0.65,
            confidence=0.8,
            created_at=ts(6),
            updated_at=ts(6),
        ),
        # --- primary user, secondary namespace ---
        MemoryUnit(
            memory_id="unit-s01",
            user_id=USER_ID,
            namespace=OTHER_NAMESPACE,
            memory_type=MemoryType.SEMANTIC,
            content="The team uses Terraform for infrastructure provisioning",
            entities=["Terraform"],
            topics=["infrastructure", "iac"],
            tags=["terraform", "iac"],
            importance=0.7,
            confidence=0.85,
            created_at=ts(7),
            updated_at=ts(7),
        ),
        MemoryUnit(
            memory_id="unit-s02",
            user_id=USER_ID,
            namespace=OTHER_NAMESPACE,
            memory_type=MemoryType.SEMANTIC,
            content="We deploy services to AWS using CloudFormation",
            entities=["AWS", "CloudFormation"],
            topics=["deployment", "aws"],
            tags=["aws", "cloudformation"],
            importance=0.65,
            confidence=0.8,
            created_at=ts(8),
            updated_at=ts(8),
        ),
        # --- secondary user, primary namespace ---
        MemoryUnit(
            memory_id="unit-u01",
            user_id=OTHER_USER,
            namespace=SCOPE,
            memory_type=MemoryType.SEMANTIC,
            content="We use Ansible for configuration management",
            entities=["Ansible"],
            topics=["configuration", "automation"],
            tags=["ansible", "config"],
            importance=0.6,
            confidence=0.8,
            created_at=ts(9),
            updated_at=ts(9),
        ),
    ]


def make_unit(
    memory_id: str,
    *,
    user_id: str,
    namespace: str,
    content: str = "Test memory content",
    memory_type: MemoryType = MemoryType.SEMANTIC,
) -> MemoryUnit:
    """Build a MemoryUnit for cases not covered by the seeded corpus."""
    ts = "2025-06-01T00:00:00+00:00"
    return MemoryUnit(
        memory_id=memory_id,
        user_id=user_id,
        namespace=namespace,
        memory_type=memory_type,
        content=content,
        entities=[],
        topics=[],
        tags=[],
        importance=0.5,
        confidence=0.7,
        created_at=ts,
        updated_at=ts,
    )
