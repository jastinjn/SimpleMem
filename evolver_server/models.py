"""Pydantic request/response models for the Evolver API.

``user_id`` is required (min_length=1) on every request model — omitting it is a
422.  ``scope_id`` is optional; omitting it queries or writes across all scopes
for that user (no scope filter applied).
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
class AddRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="Owner of the memory")
    scope_id: str | None = Field(None, description="Optional sub-context within the user")
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
    user_id: str = Field(..., min_length=1)
    scope_id: str | None = None
    session_id: str = Field("api")
    turns: list[TurnIn] = Field(..., min_length=1, max_length=50)


class RetrieveRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    scope_id: str | None = None
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=100)


class ClearRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    scope_id: str | None = None


class StatsRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    scope_id: str | None = None


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #
class AddResponse(BaseModel):
    user_id: str
    scope_id: str | None
    session_id: str
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
    scope_id: str | None
    query: str
    results: list[MemoryHit]
    total: int


class ClearResponse(BaseModel):
    user_id: str
    scope_id: str | None
    archived: int
    pinned_kept: int
    total_before: int


class StatsResponse(BaseModel):
    user_id: str
    scope_id: str | None
    entry_count: int
    total: int
    superseded: int
    active_by_type: dict[str, int]
    type_count: int
    dominant_type: str
