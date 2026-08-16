"""Pydantic request/response models for the Evolver API.

``user_id`` is required (min_length=1) on every request model — omitting it is a
422.  ``namespace`` is optional; omitting it queries or writes across all namespaces
for that user (no namespace filter applied).
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from marklymem.utils.sanitize import normalise_namespace


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
class NamespacedRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="Owner of the memory")
    namespace: str | None = Field(None, description="Optional sub-context within the user")

    @field_validator("namespace", mode="before")
    @classmethod
    def _check_namespace(cls, v: object) -> str | None:
        return normalise_namespace(v)


class TurnIn(BaseModel):
    prompt_text: str = ""
    response_text: str = ""

    @model_validator(mode="after")
    def _at_least_one_side(self) -> "TurnIn":
        if not self.prompt_text.strip() and not self.response_text.strip():
            raise ValueError("at least one of prompt_text or response_text must be non-empty")
        return self


class AddDialogueRequest(NamespacedRequest):
    session_id: str | None = Field(None)
    turns: list[TurnIn] = Field(..., min_length=1, max_length=50)


class RetrieveRequest(NamespacedRequest):
    session_id: str | None = None
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=100)


class CloneNamespaceRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    source_namespace: str = Field(..., min_length=1)
    target_namespace: str = Field(..., min_length=1)

    @field_validator("source_namespace", "target_namespace", mode="before")
    @classmethod
    def _check_namespaces(cls, v: object) -> str | None:
        return normalise_namespace(v)



# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #
class AddResponse(BaseModel):
    user_id: str
    namespace: str | None
    session_id: str | None
    units_added: int
    units_consolidated: int


class MemoryHit(BaseModel):
    memory_id: str
    content: str
    memory_type: str
    importance: float
    score: float
    matched_terms: list[str]
    entities: list[str]
    topics: list[str]
    updated_at: str


class RetrieveResponse(BaseModel):
    user_id: str
    namespace: str | None
    query: str
    results: list[MemoryHit]
    total: int


class ClearResponse(BaseModel):
    user_id: str
    namespace: str | None
    archived: int
    pinned_kept: int
    total_before: int


class StatsResponse(BaseModel):
    user_id: str
    namespace: str | None
    entry_count: int
    total: int
    superseded: int
    active_by_type: dict[str, int]
    type_count: int
    dominant_type: str


class CloneNamespaceResponse(BaseModel):
    user_id: str
    source_namespace: str
    target_namespace: str
    cloned: int
