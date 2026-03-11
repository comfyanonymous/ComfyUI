"""Pydantic models for fal.ai queue API envelope.

These are shared by all fal.ai-routed nodes. Individual model results
are returned as plain dicts (model-specific output shapes).
"""

from pydantic import BaseModel, Field


class FalQueueSubmitResponse(BaseModel):
    """Response from POST queue.fal.run/{model_id}."""

    request_id: str = Field(...)
    response_url: str = Field(...)
    status_url: str = Field(...)
    cancel_url: str = Field(default="")


class FalQueueStatusResponse(BaseModel):
    """Response from GET queue.fal.run/{model_id}/requests/{id}/status."""

    status: str = Field(...)  # IN_QUEUE, IN_PROGRESS, COMPLETED
    queue_position: int | None = Field(default=None)
    response_url: str = Field(default="")


class FalErrorDetail(BaseModel):
    """Single error detail from fal.ai."""

    loc: list[str] = Field(default_factory=list)
    msg: str = Field(default="")
    type: str = Field(default="")


class FalErrorResponse(BaseModel):
    """Error response from fal.ai endpoints."""

    detail: list[FalErrorDetail] = Field(default_factory=list)
