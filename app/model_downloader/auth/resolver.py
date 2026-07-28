"""Per-hop auth resolution (https only).

Recomputed from scratch on every redirect hop: a hop only gets a bearer token
when *its own host* matches a configured provider, so a token bound to
``huggingface.co`` is silently dropped when the request is redirected to a
presigned CDN host — which is exactly what these hubs expect.

For a matching hop: env API key first, then the provider's OAuth access token
(refreshed if expired), else no auth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.model_downloader.auth.providers import provider_for_host
from app.model_downloader.auth.store import AUTH_STORE


@dataclass
class RequestAuth:
    """How to modify a single request to carry a bearer token."""

    headers: dict[str, str] = field(default_factory=dict)


async def resolve_auth_for_hop(host: str, scheme: str) -> RequestAuth | None:
    """Resolve the bearer token (if any) to attach for one request hop."""
    if scheme.lower() != "https":
        return None
    provider = provider_for_host(host)
    if provider is None:
        return None

    token = provider.env_token()
    if token:
        return RequestAuth(headers={"Authorization": f"Bearer {token}"})

    access = await AUTH_STORE.get_valid_token(provider)
    if access:
        return RequestAuth(headers={"Authorization": f"Bearer {access}"})
    return None
