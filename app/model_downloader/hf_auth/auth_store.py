"""In-memory token cache with lazy disk persistence + refresh.

Public surface is the ``HF_AUTH_STORE`` singleton. Callers ask
``get_valid_token()``; the store transparently refreshes from disk
on first use, refreshes via the OAuth refresh_token if the cached
access_token is expired, and returns ``None`` if neither path works.

The refresh path imports ``oauth.refresh_access_token`` lazily to
avoid an import cycle (oauth needs the store to save tokens it
acquires).
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from app.model_downloader.hf_auth.token_store import (
    Token,
    delete_token,
    load_token,
    save_token,
)


class HfAuthStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._token: Optional[Token] = None
        self._loaded_from_disk = False

    def _ensure_loaded(self) -> None:
        """Read the disk token into memory on first access."""
        if self._loaded_from_disk:
            return
        with self._lock:
            if self._loaded_from_disk:
                return
            self._token = load_token()
            self._loaded_from_disk = True

    def has_token(self) -> bool:
        """Cheap check: is there any token in memory?

        Does not attempt refresh; an expired-but-refreshable token still
        counts as "logged in" from the user's perspective.
        """
        self._ensure_loaded()
        return self._token is not None

    def _store_token_locked(self, token: Token) -> None:
        """Set the in-memory token and persist it to disk.

        Caller must already hold ``self._lock``. Keeping the disk write inside
        the lock means memory and disk flip together — a concurrent ``clear()``
        or refresh can't interleave between them.
        """
        self._token = token
        self._loaded_from_disk = True
        save_token(token)

    def set_token(self, token: Token) -> None:
        """Replace the in-memory token and persist to disk (atomically)."""
        with self._lock:
            self._store_token_locked(token)

    def clear(self) -> None:
        """Forget the token in memory and on disk (logout)."""
        with self._lock:
            self._token = None
            self._loaded_from_disk = True
            delete_token()

    def get_token_sync(self) -> Optional[Token]:
        """Return the cached token without refreshing.

        Sync callers (e.g. constructing an Authorization header in a
        non-async path) use this. They accept an expired token over
        ``None``; HF will simply return 401 and the caller can decide
        what to do.
        """
        self._ensure_loaded()
        return self._token

    async def get_valid_token(self) -> Optional[Token]:
        """Return a fresh token, refreshing via OAuth if necessary.

        Returns ``None`` if there's no cached token at all, or if the
        cached token is expired and refresh failed. Callers should
        treat that as "user is not logged in".
        """
        self._ensure_loaded()
        with self._lock:
            tok = self._token
        if tok is None:
            return None
        if tok.is_valid():
            return tok
        if not tok.refresh_token:
            return None

        # Lazy import to avoid the oauth ↔ store import cycle.
        from app.model_downloader.hf_auth.oauth import refresh_access_token

        try:
            refreshed = await refresh_access_token(tok.refresh_token)
        except Exception as e:
            logging.warning("[hf_auth] token refresh failed: %s", e)
            return None

        with self._lock:
            # If a logout (clear) or another update replaced the token while we
            # were awaiting the refresh, don't resurrect the old session.
            if self._token is not tok:
                return None
            self._store_token_locked(refreshed)
        return refreshed


HF_AUTH_STORE = HfAuthStore()
