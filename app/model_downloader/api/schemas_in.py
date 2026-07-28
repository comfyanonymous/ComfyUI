"""Request schemas for the download manager API.

Pydantic enforces shape at the boundary; handlers operate only on validated
values past that point.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class EnqueueRequest(BaseModel):
    url: str
    model_id: str
    priority: int = 0
    expected_sha256: Optional[str] = None
    allow_any_extension: bool = False

    @field_validator("url")
    @classmethod
    def _strip_url(cls, v: str) -> str:
        return v.strip()


class PriorityRequest(BaseModel):
    priority: int


class AvailabilityRequest(BaseModel):
    """``{model_id: url}`` — the URLs declared in the workflow JSON."""

    models: dict[str, str] = Field(default_factory=dict)

    @field_validator("models")
    @classmethod
    def _strip_urls(cls, v: dict[str, str]) -> dict[str, str]:
        return {k: url.strip() for k, url in v.items()}


__all__ = [
    "EnqueueRequest",
    "PriorityRequest",
    "AvailabilityRequest",
]
