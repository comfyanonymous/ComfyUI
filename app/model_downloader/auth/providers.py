"""Provider registry for download authentication.

A :class:`Provider` describes a hub that can authenticate downloads either from
an environment API key or from an OAuth 2.0 access token. Both HuggingFace and
Civitai are public PKCE clients, so no client secret is ever stored; the public
``client_id`` is a placeholder overridable via env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlsplit


def normalize_host(host: str) -> str:
    """Lowercase, strip port, IDNA-encode."""
    if not host:
        return ""
    host = host.strip()
    if "://" in host:  # a full URL was pasted — extract just the host
        host = urlsplit(host).hostname or ""
    host = host.lower()
    if host.startswith("[") and "]" in host:  # bracketed IPv6 literal
        host = host[1 : host.index("]")]
    elif host.count(":") == 1:  # host:port (not IPv6)
        host = host.split(":", 1)[0]
    try:
        host = host.encode("idna").decode("ascii")
    except (UnicodeError, ValueError):
        pass
    return host


@dataclass(frozen=True)
class Provider:
    name: str
    host: str
    authorize_url: str
    token_url: str
    scope: str
    # Env vars to try, in order, for a plain API key.
    env_keys: tuple[str, ...]
    # Env var overriding the public OAuth client id.
    client_id_env: str
    # Public PKCE client id. Empty means "not configured" until the env sets it.
    default_client_id: str = ""

    @property
    def client_id(self) -> str:
        return os.environ.get(self.client_id_env, self.default_client_id) or ""

    def env_token(self) -> str | None:
        for var in self.env_keys:
            token = os.environ.get(var)
            if token:
                return token
        return None


PROVIDERS: dict[str, Provider] = {
    "huggingface": Provider(
        name="huggingface",
        host="huggingface.co",
        authorize_url="https://huggingface.co/oauth/authorize",
        token_url="https://huggingface.co/oauth/token",
        scope="openid read-repos gated-repos",
        env_keys=("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
        client_id_env="COMFY_HF_OAUTH_CLIENT_ID",
    ),
    "civitai": Provider(
        name="civitai",
        host="civitai.com",
        authorize_url="https://auth.civitai.com/api/auth/oauth/authorize",
        token_url="https://auth.civitai.com/api/auth/oauth/token",
        scope="4",  # ModelsRead; UserRead is auto-granted
        env_keys=("CIVITAI_API_TOKEN", "CIVITAI_API_KEY"),
        client_id_env="COMFY_CIVITAI_OAUTH_CLIENT_ID",
    ),
}

_HOST_TO_PROVIDER = {p.host: p for p in PROVIDERS.values()}


def provider_for_host(host: str) -> Provider | None:
    """Return the provider whose host exactly matches ``host`` (normalized)."""
    return _HOST_TO_PROVIDER.get(normalize_host(host))
