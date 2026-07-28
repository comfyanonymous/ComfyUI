"""In-memory OAuth token cache over the on-disk token store.

:data:`AUTH_STORE` is the process singleton the resolver and API talk to. It
lazily loads each provider's token from disk, refreshes an expired access token
via its refresh token, and orchestrates the login flow (delegating the loopback
callback server to :mod:`oauth`).
"""

from __future__ import annotations

from app.model_downloader.auth import oauth, token_store
from app.model_downloader.auth.providers import Provider
from app.model_downloader.auth.token_store import Token


class AuthStore:
    def __init__(self) -> None:
        # provider name -> Token, or None when known to be absent. A missing key
        # means "not yet loaded from disk".
        self._cache: dict[str, Token | None] = {}

    def _load(self, name: str) -> Token | None:
        if name not in self._cache:
            self._cache[name] = token_store.load(name)
        return self._cache[name]

    def set_token(self, name: str, token: Token) -> None:
        self._cache[name] = token
        token_store.save(name, token)

    def clear(self, name: str) -> None:
        self._cache[name] = None
        token_store.delete(name)

    async def get_valid_token(self, provider: Provider) -> str | None:
        """Return a valid access token string for ``provider``, or ``None``.

        Refreshes an expired token when a refresh token is available.
        """
        token = self._load(provider.name)
        if token is None or not token.access_token:
            return None
        if token.is_expired():
            if not token.refresh_token:
                return None
            token = await oauth.refresh_access_token(provider, token)
            self.set_token(provider.name, token)
        return token.access_token

    async def begin_login(self, provider: Provider) -> str:
        """Start a login flow; returns the authorize URL to open in a browser."""
        return await oauth.start_login_flow(provider, self.set_token)

    def status(self, provider: Provider) -> dict:
        token = self._load(provider.name)
        return {
            "provider": provider.name,
            "logged_in": token is not None and bool(token.access_token),
            "login_in_progress": oauth.login_in_progress(provider.name),
            "env_key_present": provider.env_token() is not None,
        }


AUTH_STORE = AuthStore()
