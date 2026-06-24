"""Request schemas for the model-downloader API.

Each endpoint accepts a small JSON body. Pydantic enforces the shape at
the boundary; route handlers operate only on validated values past that.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AvailabilityStatusRequest(BaseModel):
    """``POST /api/models-availability-status``.

    Sent by the frontend on each poll. Each entry is ``{model_id: url}``;
    the URL is the one declared in ``properties.models[i].url`` in the
    workflow JSON and lets the server compute per-id metadata
    (``file_size`` + ``is_hf_downloadable``) on the same request.
    """
    models: dict[str, str] = Field(default_factory=dict)


class DownloadModelsRequest(BaseModel):
    """``POST /api/download-models``.

    Same shape as the metadata request — the URL for each model_id.
    Returns immediately after validation and scheduling.
    """
    models: dict[str, str] = Field(default_factory=dict)


class CancelDownloadSessionRequest(BaseModel):
    """``POST /api/cancel-model-download-session``."""
    model_id: str


__all__ = [
    "AvailabilityStatusRequest",
    "DownloadModelsRequest",
    "CancelDownloadSessionRequest",
]
