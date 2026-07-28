"""Response helpers for the download manager API.

The download/status read models are plain dicts produced by the manager. This
module serializes the per-provider auth status (never a token) for the API.
"""

from __future__ import annotations

from app.model_downloader.auth.providers import PROVIDERS
from app.model_downloader.auth.store import AUTH_STORE


def auth_status() -> list[dict]:
    """Per-provider auth status — never includes a token."""
    return [AUTH_STORE.status(p) for p in PROVIDERS.values()]
