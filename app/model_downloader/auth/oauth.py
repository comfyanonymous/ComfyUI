"""Generic OAuth 2.0 PKCE engine + transient loopback callback server.

The flow, per provider:

1. :func:`start_login_flow` builds a PKCE challenge, binds a loopback callback
   server on ``127.0.0.1:<CALLBACK_PORT>`` at ``/callback/<provider>``, and
   returns the provider's authorize URL for the user to open.
2. The provider redirects the browser back to the loopback URL with a ``code``
   and the ``state`` we generated. The server validates ``state``, exchanges the
   code for a :class:`Token`, hands it to the ``deliver`` sink, and tears down.
3. If no callback arrives within :data:`_LOGIN_TIMEOUT`, the server tears down.

The callback runs on its own bare server, not ComfyUI's main server: the main
server rejects cross-site navigations (``Sec-Fetch-Site: cross-site`` → 403),
and an OAuth redirect from the provider is exactly such a navigation. The port
is fixed because HuggingFace and Civitai require an exact registered
``redirect_uri`` (port included); only one login runs at a time so the port
never contends with itself.

Only public PKCE clients are supported (no client secret). All outbound calls
go to the provider's own authorize/token endpoints, strictly user-initiated.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
import secrets
import time
from typing import Callable
from urllib.parse import urlencode

from aiohttp import web

from app.model_downloader.auth.providers import Provider
from app.model_downloader.auth.token_store import Token
from app.model_downloader.net.session import get_session, ssl_context

CALLBACK_HOST = "127.0.0.1"
# Fixed loopback port for the OAuth redirect. Must match the redirect URI
# registered on the provider's OAuth app; override in lockstep if you change it.
CALLBACK_PORT = int(os.environ.get("COMFY_OAUTH_CALLBACK_PORT", "41954"))
_LOGIN_TIMEOUT = 300.0  # seconds to wait for the browser callback

# The auth tab is opened by the frontend via window.open, so window.close() is
# allowed here; the visible text is the fallback when the browser blocks it.
_SUCCESS_HTML = (
    "<!doctype html><meta charset=utf-8><title>ComfyUI</title>"
    "<p>Login successful. You can close this window and return to ComfyUI.</p>"
    "<script>window.close()</script>"
)

# Token sink: called with the provider name and the exchanged Token.
TokenSink = Callable[[str, Token], None]


class OAuthError(Exception):
    """A user-facing OAuth failure."""


class OAuthNotConfigured(OAuthError):
    """The provider has no public client id configured."""


class LoginInProgress(OAuthError):
    """A login flow for this provider is already running."""


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _make_pkce() -> tuple[str, str]:
    """Return ``(verifier, challenge)`` for the S256 PKCE method."""
    verifier = _b64url(secrets.token_bytes(32))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def build_authorize_url(
    provider: Provider, challenge: str, state: str, redirect_uri: str
) -> str:
    params = {
        "response_type": "code",
        "client_id": provider.client_id,
        "redirect_uri": redirect_uri,
        "scope": provider.scope,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    return f"{provider.authorize_url}?{urlencode(params)}"


def _token_from_payload(payload: dict) -> Token:
    expires_in = payload.get("expires_in")
    expires_at = int(time.time()) + int(expires_in) if expires_in else 0
    return Token(
        access_token=payload["access_token"],
        refresh_token=payload.get("refresh_token"),
        expires_at=expires_at,
        token_type=payload.get("token_type", "Bearer"),
        scope=payload.get("scope"),
    )


async def _post_token(provider: Provider, data: dict) -> Token:
    session = await get_session()
    resp = await session.post(
        provider.token_url,
        data=data,
        headers={"Accept": "application/json"},
        ssl=ssl_context(),
    )
    try:
        if resp.status != 200:
            body = await resp.text()
            raise OAuthError(
                f"{provider.name} token endpoint returned HTTP {resp.status}: {body[:200]}"
            )
        payload = await resp.json()
    finally:
        await resp.release()
    if "access_token" not in payload:
        raise OAuthError(f"{provider.name} token response missing access_token")
    return _token_from_payload(payload)


async def exchange_code(
    provider: Provider, code: str, verifier: str, redirect_uri: str
) -> Token:
    return await _post_token(
        provider,
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": provider.client_id,
            "code_verifier": verifier,
        },
    )


async def refresh_access_token(provider: Provider, token: Token) -> Token:
    if not token.refresh_token:
        raise OAuthError(f"{provider.name} token is not refreshable")
    refreshed = await _post_token(
        provider,
        {
            "grant_type": "refresh_token",
            "refresh_token": token.refresh_token,
            "client_id": provider.client_id,
        },
    )
    # Some providers omit a new refresh token on refresh; keep the old one.
    if refreshed.refresh_token is None:
        refreshed.refresh_token = token.refresh_token
    return refreshed


class _LoginFlow:
    """A single in-flight login: owns the loopback server and PKCE state."""

    def __init__(self, provider: Provider, deliver: TokenSink) -> None:
        self.provider = provider
        self.deliver = deliver
        self.verifier, self.challenge = _make_pkce()
        self.state = secrets.token_urlsafe(24)
        self.redirect_uri = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}/callback/{provider.name}"
        self._runner: web.AppRunner | None = None
        self._timeout_handle: asyncio.TimerHandle | None = None

    async def start(self) -> str:
        app = web.Application()
        app.router.add_get("/callback/{provider}", self._handle_callback)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, CALLBACK_HOST, CALLBACK_PORT, reuse_address=True)
        try:
            await site.start()
        except OSError as e:
            await self._runner.cleanup()
            self._runner = None
            raise OAuthError(f"could not bind callback port {CALLBACK_PORT}: {e}")
        loop = asyncio.get_running_loop()
        self._timeout_handle = loop.call_later(
            _LOGIN_TIMEOUT, lambda: asyncio.ensure_future(self._teardown())
        )
        return build_authorize_url(
            self.provider, self.challenge, self.state, self.redirect_uri
        )

    async def _handle_callback(self, request: web.Request) -> web.Response:
        if request.match_info.get("provider") != self.provider.name:
            return web.Response(text="Unknown login.", content_type="text/plain", status=404)
        error = request.query.get("error")
        if error:
            asyncio.ensure_future(self._teardown())
            return web.Response(
                text=f"Login failed: {error}", content_type="text/plain", status=400
            )
        if request.query.get("state") != self.state:
            return web.Response(
                text="Login failed: state mismatch.",
                content_type="text/plain",
                status=400,
            )
        code = request.query.get("code")
        if not code:
            return web.Response(
                text="Login failed: no authorization code.",
                content_type="text/plain",
                status=400,
            )
        try:
            token = await exchange_code(
                self.provider, code, self.verifier, self.redirect_uri
            )
        except OAuthError as e:
            logging.warning("[model_downloader] %s login failed: %s", self.provider.name, e)
            asyncio.ensure_future(self._teardown())
            return web.Response(
                text=f"Login failed: {e}", content_type="text/plain", status=502
            )
        self.deliver(self.provider.name, token)
        asyncio.ensure_future(self._teardown())
        return web.Response(text=_SUCCESS_HTML, content_type="text/html")

    async def _teardown(self) -> None:
        _ACTIVE.pop(self.provider.name, None)
        if self._timeout_handle is not None:
            self._timeout_handle.cancel()
            self._timeout_handle = None
        if self._runner is not None:
            try:
                await self._runner.cleanup()
            except Exception:
                logging.debug("[model_downloader] callback server cleanup error", exc_info=True)
            self._runner = None


_ACTIVE: dict[str, _LoginFlow] = {}


def login_in_progress(provider_name: str) -> bool:
    return provider_name in _ACTIVE


async def start_login_flow(provider: Provider, deliver: TokenSink) -> str:
    """Begin a login flow and return the authorize URL to open in a browser.

    Binds the fixed-port loopback callback server; only one login may run at a
    time since that port is shared.
    """
    if not provider.client_id:
        raise OAuthNotConfigured(
            f"OAuth app not configured for {provider.name}; set "
            f"{provider.client_id_env} or use an env API key."
        )
    if _ACTIVE:
        active = next(iter(_ACTIVE))
        raise LoginInProgress(f"A login for {active} is already in progress.")
    flow = _LoginFlow(provider, deliver)
    authorize_url = await flow.start()
    _ACTIVE[provider.name] = flow
    return authorize_url
