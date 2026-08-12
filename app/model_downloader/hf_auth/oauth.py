"""OAuth 2.0 PKCE flow against HuggingFace's authorization server.

Wired so that ``POST /api/hf-auth-login-start`` can:
  1. Generate state + PKCE verifier/challenge in this process.
  2. Spin up a short-lived loopback HTTP server at port 41954 to
     receive the redirect callback from HF.
  3. Return the ``authorize_url`` for the frontend to open in a new tab.

After the user grants consent on huggingface.co, HF redirects to the
local callback URL with ``code`` and ``state``. The callback server
validates ``state`` (CSRF), exchanges the code for tokens via PKCE,
hands the resulting Token to ``HF_AUTH_STORE.set_token``, and shuts
itself down.

Before this can be exercised end-to-end a maintainer must register a
HuggingFace OAuth app and substitute the ``HF_CLIENT_ID`` placeholder
below. See the comment above the constant for the exact steps.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import secrets
import threading
import time

import aiohttp
from aiohttp import web

from app.model_downloader.hf_auth.auth_store import HF_AUTH_STORE
from app.model_downloader.hf_auth.token_store import Token
from app.model_downloader.http_client import get_session


# --- HF OAuth app registration -------------------------------------------- #
# NOTE: The OAuth client_id below is a placeholder. Before this feature can be
# exercised end-to-end, a maintainer must register a HuggingFace OAuth app
# under a Comfy-Org-controlled HF account and substitute its client_id here.
# Detailed walkthrough is in docs/server-side-model-downloads-handover.html
# ("HuggingFace OAuth app setup" section). Short version:
#   1. huggingface.co → Settings → Connected Apps → "Create app"
#   2. Default Scopes: check ``openid`` + ``profile`` (User Info) and
#      ``gated-repos`` (Repository Access). Leave everything else off.
#   3. Redirect URLs: exactly ``http://127.0.0.1:41954/api/auth/huggingface/callback``
#      — must match ``REDIRECT_URI`` below; change both in lockstep if you
#      change ``CALLBACK_PORT``.
#   4. Save → copy the resulting Client ID into ``HF_CLIENT_ID`` below.
# The client_id is not a secret (it travels through the user's browser in
# plaintext); HF's "Public app" type means there's no client secret to
# manage — PKCE replaces it.
HF_CLIENT_ID = "REPLACE_ME_WITH_COMFY_ORG_HF_OAUTH_CLIENT_ID"

CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 41954
CALLBACK_PATH = "/api/auth/huggingface/callback"
REDIRECT_URI = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"

AUTHORIZE_URL = "https://huggingface.co/oauth/authorize"
TOKEN_URL = "https://huggingface.co/oauth/token"
_TOKEN_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=30)
# Minimal scope set for the feature:
#   - openid       : required by HF when the app uses OIDC at all
#   - profile      : lets ``HfApi.whoami(token=...)`` return a username for the
#                    settings UI; cosmetic but expected
#   - gated-repos  : grants the token enough to call ``auth_check`` and
#                    download files from public gated repos the user has
#                    accepted the license for. The wider ``read-repos`` scope
#                    would also work (it includes ``gated-repos``) but it
#                    additionally grants private-repo read access, which we
#                    don't need and which makes the consent screen scarier
#                    for the user.
SCOPE = "openid profile gated-repos"

# Maximum time the callback server stays up waiting for the user to
# complete consent on huggingface.co. Past this, the port closes and
# the user has to click "Log in" again.
CALLBACK_TIMEOUT_SECS = 300


# Process-wide lock so two simultaneous /api/hf-auth-login-start
# requests don't fight over port CALLBACK_PORT.
_OAUTH_LOCK = threading.Lock()


class OAuthInProgressError(Exception):
    """Another OAuth attempt is already running."""


class OAuthCallbackError(Exception):
    """The OAuth callback returned an error (HF denied, port stolen, etc.)."""


# --- PKCE primitives ------------------------------------------------------ #


def _make_pkce() -> tuple[str, str, str]:
    """Return ``(verifier, challenge, state)``.

    Verifier never leaves this process. Challenge and state travel
    through the user's browser. State is checked on the callback to
    prevent a malicious cross-origin redirect from injecting a token.
    """
    verifier = secrets.token_urlsafe(64)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    state = secrets.token_urlsafe(32)
    return verifier, challenge, state


def _build_authorize_url(challenge: str, state: str) -> str:
    from urllib.parse import urlencode

    params = {
        "client_id": HF_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": SCOPE,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


# --- Token exchange ------------------------------------------------------- #


async def _exchange_code(code: str, verifier: str) -> Token:
    """Trade the authorization code for an access+refresh token pair."""
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "client_id": HF_CLIENT_ID,
        "code_verifier": verifier,
    }
    session = await get_session()
    async with session.post(TOKEN_URL, data=data, timeout=_TOKEN_REQUEST_TIMEOUT) as resp:
        resp.raise_for_status()
        body = await resp.json()
    return Token(
        access_token=body["access_token"],
        refresh_token=body.get("refresh_token"),
        expires_at=time.time() + float(body.get("expires_in", 3600)),
        scope=body.get("scope", SCOPE),
    )


async def refresh_access_token(refresh_token: str) -> Token:
    """Trade a refresh_token for a new access (+ possibly refresh) token."""
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": HF_CLIENT_ID,
    }
    session = await get_session()
    async with session.post(TOKEN_URL, data=data, timeout=_TOKEN_REQUEST_TIMEOUT) as resp:
        resp.raise_for_status()
        body = await resp.json()
    return Token(
        access_token=body["access_token"],
        # If HF doesn't rotate refresh tokens, keep using the existing one.
        refresh_token=body.get("refresh_token", refresh_token),
        expires_at=time.time() + float(body.get("expires_in", 3600)),
        scope=body.get("scope", SCOPE),
    )


# --- Callback server ------------------------------------------------------ #


async def start_login_flow() -> str:
    """Begin one OAuth attempt: spawn the callback server, return the URL.

    Returns the URL the frontend should open in a new tab. Raises
    ``OAuthInProgressError`` if another attempt is already running.
    The callback server runs in the background until the user
    completes consent or until ``CALLBACK_TIMEOUT_SECS`` elapses;
    either way the lock + port are released afterward.
    """
    if not _OAUTH_LOCK.acquire(blocking=False):
        raise OAuthInProgressError()

    try:
        verifier, challenge, state = _make_pkce()
        authorize_url = _build_authorize_url(challenge, state)
        ready: asyncio.Future[None] = asyncio.get_event_loop().create_future()
    except BaseException:
        # Failed before handing the lock to the callback-server task: release it
        # here. (Once the task is spawned, it owns releasing the lock.)
        _OAUTH_LOCK.release()
        raise

    asyncio.create_task(_run_callback_server(verifier, state, ready))
    # Don't return the URL until the callback server is actually bound and
    # listening — otherwise HF could redirect to a port nothing is serving and
    # the login would silently dead-end. ``ready`` raises if the bind failed.
    await ready
    return authorize_url


async def _run_callback_server(
    verifier: str, expected_state: str, ready: "asyncio.Future[None]"
) -> None:
    """Listen for HF's redirect once, capture the token, then shut down.

    Signals ``ready`` once the port is bound (or with an exception if the bind
    fails), so ``start_login_flow`` only hands back a URL on a live server.
    """
    received: asyncio.Future[Token] = asyncio.get_event_loop().create_future()

    async def handler(request: web.Request) -> web.Response:
        try:
            if request.query.get("state") != expected_state:
                return web.Response(status=400, text="state mismatch")
            err = request.query.get("error")
            if err:
                received.set_exception(OAuthCallbackError(f"HF returned: {err}"))
                return web.Response(status=400, text=f"OAuth error: {err}")
            code = request.query.get("code")
            if not code:
                return web.Response(status=400, text="missing code")
            tok = await _exchange_code(code, verifier)
            if not received.done():
                received.set_result(tok)
            return web.Response(
                content_type="text/html",
                text=(
                    "<html><body style='font-family:sans-serif;padding:40px'>"
                    "<h2>HuggingFace login successful</h2>"
                    "<p>You can close this tab and return to ComfyUI.</p>"
                    "</body></html>"
                ),
            )
        except Exception as exc:
            if not received.done():
                received.set_exception(exc)
            return web.Response(status=500, text=str(exc))

    app = web.Application()
    app.router.add_get(CALLBACK_PATH, handler)
    runner = web.AppRunner(app)
    try:
        await runner.setup()
        site = web.TCPSite(runner, CALLBACK_HOST, CALLBACK_PORT, reuse_address=True)
        await site.start()
    except Exception as e:
        # Couldn't bind the callback port (commonly already in use). Tell the
        # waiting start_login_flow via ``ready`` so it surfaces a clear error
        # instead of returning a dead URL, and release the lock for next time.
        logging.warning("[hf_auth] could not start callback server: %s", e)
        if not ready.done():
            ready.set_exception(
                OAuthCallbackError(f"could not bind callback port {CALLBACK_PORT}: {e}")
            )
        _OAUTH_LOCK.release()
        return

    # Bound and listening — now it's safe for start_login_flow to return the URL.
    if not ready.done():
        ready.set_result(None)

    try:
        token = await asyncio.wait_for(received, timeout=CALLBACK_TIMEOUT_SECS)
    except asyncio.TimeoutError:
        logging.info("[hf_auth] OAuth login timed out after %ds", CALLBACK_TIMEOUT_SECS)
        return
    except OAuthCallbackError as e:
        logging.warning("[hf_auth] OAuth callback error: %s", e)
        return
    except Exception as e:
        logging.warning("[hf_auth] unexpected OAuth failure: %s", e)
        return
    else:
        HF_AUTH_STORE.set_token(token)
        logging.info("[hf_auth] OAuth login complete")
    finally:
        await runner.cleanup()
        if _OAUTH_LOCK.locked():
            _OAUTH_LOCK.release()


def is_login_in_progress() -> bool:
    """True iff a callback server is currently bound + waiting."""
    return _OAUTH_LOCK.locked()


# Re-export for callers that only want the URL builder (e.g. tests).
__all__ = [
    "start_login_flow",
    "refresh_access_token",
    "is_login_in_progress",
    "OAuthInProgressError",
    "CALLBACK_TIMEOUT_SECS",
]
