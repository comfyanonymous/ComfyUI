"""Response schemas for the model-downloader API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


ModelState = Literal["available", "missing", "downloading"]


class DownloadProgress(BaseModel):
    """Embedded in a model entry when its state is ``downloading``."""
    bytes_downloaded: int
    total_bytes: Optional[int] = None
    progress: Optional[float] = None  # fraction in [0,1]; null until total known


class ModelStatusEntry(BaseModel):
    """Everything the UI needs to render one row, in one shot.

    ``state`` reflects what the server has on disk + in-flight; ``file_size``
    and ``is_hf_downloadable`` come from probes (intrinsic; cached).
    The HF fields are populated for every poll (cached on the server),
    so license-acceptance flips show up within one poll interval without
    any frontend cache invalidation.
    """
    state: ModelState
    progress: Optional[DownloadProgress] = None
    file_size: Optional[int] = None
    # HF-only: True iff the server can fetch this URL with current auth
    # state. False iff gated and lacking access. None for non-HF URLs.
    is_hf_downloadable: Optional[bool] = None


class HfAuthStatus(BaseModel):
    """Snapshot of HF login state, embedded in availability response."""
    token_available: bool
    eligible: bool


class AvailabilityStatusResponse(BaseModel):
    models: dict[str, ModelStatusEntry]
    hf_auth: HfAuthStatus


class DownloadModelsResponse(BaseModel):
    accepted: bool
    scheduled: list[str]


class CancelDownloadSessionResponse(BaseModel):
    cancelled: bool


class HfAuthTokenStatusResponse(BaseModel):
    token_available: bool
    username: Optional[str] = None


class HfAuthLoginStartResponse(BaseModel):
    authorize_url: str


class HfAuthLogoutResponse(BaseModel):
    logged_out: bool


__all__ = [
    "ModelState",
    "DownloadProgress",
    "ModelStatusEntry",
    "HfAuthStatus",
    "AvailabilityStatusResponse",
    "DownloadModelsResponse",
    "CancelDownloadSessionResponse",
    "HfAuthTokenStatusResponse",
    "HfAuthLoginStartResponse",
    "HfAuthLogoutResponse",
]
