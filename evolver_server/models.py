"""Pydantic request/response models for the Evolver API.

``scope_id`` is required (min_length=1) on every request model — omitting it is a
422, never a silent fall-through to the ``"default"`` scope. This is the primary
tenant-isolation guardrail.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
class AddRequest(BaseModel):
    scope_id: str = Field(..., min_length=1, description="Caller-supplied tenant/user id")
    session_id: str = Field("api", description="Groups turns from the same conversation")
    prompt_text: str = Field("", description="User side of the turn")
    response_text: str = Field("", description="Assistant side of the turn")

    @model_validator(mode="after")
    def _at_least_one_side(self) -> "AddRequest":
        if not self.prompt_text.strip() and not self.response_text.strip():
            raise ValueError("at least one of prompt_text or response_text must be non-empty")
        return self


class TurnIn(BaseModel):
    prompt_text: str = ""
    response_text: str = ""


class AddBatchRequest(BaseModel):
    scope_id: str = Field(..., min_length=1)
    session_id: str = Field("api")
    turns: list[TurnIn] = Field(..., min_length=1)


class RetrieveRequest(BaseModel):
    scope_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=100)


class ClearRequest(BaseModel):
    scope_id: str = Field(..., min_length=1)


class StatsRequest(BaseModel):
    scope_id: str = Field(..., min_length=1)


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #
class AddResponse(BaseModel):
    scope_id: str
    session_id: str
    units_added: int


class MemoryHit(BaseModel):
    memory_id: str
    content: str
    summary: str
    memory_type: str
    importance: float
    score: float
    matched_terms: list[str]
    entities: list[str]
    topics: list[str]
    updated_at: str


class RetrieveResponse(BaseModel):
    scope_id: str
    query: str
    results: list[MemoryHit]
    total: int


class ClearResponse(BaseModel):
    scope_id: str
    archived: int
    pinned_kept: int
    total_before: int


class StatsResponse(BaseModel):
    scope_id: str
    entry_count: int
    total: int
    superseded: int
    active_by_type: dict[str, int]
    type_count: int
    dominant_type: str
