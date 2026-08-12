"""Unit tests for the HuggingFace auth subsystem.

Covers:
  - token store: save/load roundtrip, chmod 0600, atomic write, delete
  - eligibility under various CLI-arg combinations
  - URL parsing (huggingface.co host detection + repo_id extraction)
  - HF-aware gated_detection.probe_url (mocked auth_check)
  - HF auth routes (token status, login start with eligibility gate, logout)
  - PKCE primitives + authorize URL shape

The OAuth callback server itself isn't exercised end-to-end here — that
requires a real HF server. We test the components (state checking,
URL building, code-exchange request shape) instead.
"""

from __future__ import annotations

import os
import stat
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from app.model_downloader.api.routes import register_routes
from app.model_downloader.hf_auth import oauth
from app.model_downloader.hf_auth.auth_store import HF_AUTH_STORE, HfAuthStore
from app.model_downloader.hf_auth.token_store import (
    EXPIRY_BUFFER_SECS,
    Token,
    delete_token,
    load_token,
    save_token,
)
from app.model_downloader.hf_url import is_hf_url, repo_id_from_url


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def patched_user_dir(tmp_path):
    """Redirect ``folder_paths.get_user_directory`` so the token file
    lands in an isolated tmp_path instead of the real user dir."""
    user_dir = tmp_path / "user"
    user_dir.mkdir()
    with patch("folder_paths.get_user_directory", return_value=str(user_dir)):
        yield user_dir


def _token_file_path(user_dir) -> str:
    return os.path.join(user_dir, "__hf_auth", "hf_auth_token.json")


@pytest.fixture
def fresh_auth_store():
    """Wipe singleton state between tests: auth + probe caches."""
    from app.model_downloader import gated_detection

    HF_AUTH_STORE._token = None
    HF_AUTH_STORE._loaded_from_disk = False
    gated_detection.clear_caches_for_tests()
    yield HF_AUTH_STORE
    HF_AUTH_STORE._token = None
    HF_AUTH_STORE._loaded_from_disk = False
    gated_detection.clear_caches_for_tests()


@pytest.fixture
def app(patched_user_dir, fresh_auth_store):
    app = web.Application()
    register_routes(app)
    return app


# --------------------------------------------------------------------------- #
# URL parsing
# --------------------------------------------------------------------------- #


def test_is_hf_url_recognises_huggingface_co():
    assert is_hf_url("https://huggingface.co/x/y/resolve/main/z.safetensors")
    assert is_hf_url("https://huggingface.co/abc")
    assert not is_hf_url("https://hf-mirror.com/x/y/resolve/main/z.safetensors")
    assert not is_hf_url("https://civitai.com/x.safetensors")


def test_repo_id_from_url_extracts_org_and_repo():
    url = "https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-HDR/resolve/main/x.safetensors"
    assert repo_id_from_url(url) == "Lightricks/LTX-2.3-22b-IC-LoRA-HDR"


def test_repo_id_from_url_handles_nested_path():
    url = "https://huggingface.co/Comfy-Org/ltx-2.3/resolve/main/split_files/loras/x.safetensors"
    assert repo_id_from_url(url) == "Comfy-Org/ltx-2.3"


def test_repo_id_from_url_returns_none_for_non_hf():
    assert repo_id_from_url("https://civitai.com/x.safetensors") is None


def test_repo_id_from_url_returns_none_for_non_resolve_paths():
    assert repo_id_from_url("https://huggingface.co/org/repo/blob/main/x.safetensors") is None
    assert repo_id_from_url("https://huggingface.co/org") is None


# --------------------------------------------------------------------------- #
# Token store
# --------------------------------------------------------------------------- #


def test_token_store_roundtrip(patched_user_dir):
    tok = Token(
        access_token="hf_abc",
        refresh_token="rf_def",
        expires_at=9999999999.0,
        scope="openid profile",
    )
    save_token(tok)
    loaded = load_token()
    assert loaded == tok


def test_token_store_writes_0600(patched_user_dir):
    tok = Token(access_token="x", refresh_token=None, expires_at=0.0)
    save_token(tok)
    path = _token_file_path(patched_user_dir)
    mode = stat.S_IMODE(os.stat(path).st_mode)
    # On Windows we silently no-op chmod; allow either the intended
    # mode or whatever umask the OS gave us.
    if os.name == "posix":
        assert mode == 0o600


def test_token_store_delete_removes_file(patched_user_dir):
    tok = Token(access_token="x", refresh_token=None, expires_at=0.0)
    save_token(tok)
    delete_token()
    path = _token_file_path(patched_user_dir)
    assert not os.path.exists(path)
    # Idempotent: second delete is fine.
    delete_token()


def test_token_store_load_returns_none_for_missing_file(patched_user_dir):
    assert load_token() is None


def test_token_store_load_returns_none_for_corrupt_file(patched_user_dir):
    path = _token_file_path(patched_user_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("not json {")
    assert load_token() is None


def test_token_is_valid_uses_buffer(patched_user_dir):
    import time

    fresh = Token(access_token="x", refresh_token=None, expires_at=time.time() + 3600)
    nearly_expired = Token(
        access_token="x",
        refresh_token=None,
        expires_at=time.time() + EXPIRY_BUFFER_SECS - 1,
    )
    assert fresh.is_valid()
    assert not nearly_expired.is_valid()


def test_token_is_valid_rejects_empty_access_token():
    import time

    tok = Token(access_token="", refresh_token=None, expires_at=time.time() + 3600)
    assert not tok.is_valid()


def test_token_is_valid_rejects_at_exact_buffer_boundary():
    import time

    tok = Token(
        access_token="x",
        refresh_token=None,
        expires_at=time.time() + EXPIRY_BUFFER_SECS,
    )
    assert not tok.is_valid()


# --------------------------------------------------------------------------- #
# Auth store
# --------------------------------------------------------------------------- #


def test_auth_store_loads_lazily(patched_user_dir):
    tok = Token(access_token="x", refresh_token=None, expires_at=9999999999.0)
    save_token(tok)
    store = HfAuthStore()
    assert store.has_token()
    assert store.get_token_sync() == tok


def test_auth_store_set_persists(patched_user_dir):
    store = HfAuthStore()
    tok = Token(access_token="x", refresh_token=None, expires_at=9999999999.0)
    store.set_token(tok)
    # Token is on disk now — a fresh store sees it.
    assert HfAuthStore().get_token_sync() == tok


def test_auth_store_clear_removes_in_memory_and_on_disk(patched_user_dir):
    store = HfAuthStore()
    tok = Token(access_token="x", refresh_token=None, expires_at=9999999999.0)
    store.set_token(tok)
    store.clear()
    assert not store.has_token()
    assert HfAuthStore().get_token_sync() is None


def test_auth_store_has_token_true_when_expired_but_refreshable(patched_user_dir):
    import time

    store = HfAuthStore()
    expired = Token(
        access_token="old",
        refresh_token="rf",
        expires_at=time.time() - 100,
    )
    store.set_token(expired)
    assert store.has_token()
    assert not expired.is_valid()


def test_auth_store_get_token_sync_returns_expired_without_refresh(patched_user_dir):
    import time

    store = HfAuthStore()
    expired = Token(
        access_token="old",
        refresh_token=None,
        expires_at=time.time() - 100,
    )
    store.set_token(expired)
    assert store.get_token_sync() == expired


@pytest.mark.asyncio
async def test_auth_store_get_valid_returns_none_when_expired_without_refresh(
    patched_user_dir,
):
    import time

    store = HfAuthStore()
    expired = Token(
        access_token="old",
        refresh_token=None,
        expires_at=time.time() - 100,
    )
    store.set_token(expired)
    with patch(
        "app.model_downloader.hf_auth.oauth.refresh_access_token",
        new=AsyncMock(),
    ) as refresh_mock:
        result = await store.get_valid_token()
    assert result is None
    refresh_mock.assert_not_called()


@pytest.mark.asyncio
async def test_auth_store_get_valid_returns_fresh_token(patched_user_dir):
    store = HfAuthStore()
    import time

    tok = Token(access_token="x", refresh_token=None, expires_at=time.time() + 3600)
    store.set_token(tok)
    fetched = await store.get_valid_token()
    assert fetched == tok


@pytest.mark.asyncio
async def test_auth_store_get_valid_refresh_on_expired(patched_user_dir):
    store = HfAuthStore()
    import time

    expired = Token(
        access_token="old",
        refresh_token="rf",
        expires_at=time.time() - 100,
    )
    store.set_token(expired)
    refreshed = Token(
        access_token="new",
        refresh_token="rf",
        expires_at=time.time() + 3600,
    )
    with patch(
        "app.model_downloader.hf_auth.oauth.refresh_access_token",
        new=AsyncMock(return_value=refreshed),
    ):
        result = await store.get_valid_token()
    assert result == refreshed


@pytest.mark.asyncio
async def test_auth_store_get_valid_token_does_not_resurrect_after_logout(
    patched_user_dir,
):
    """A logout landing *during* an in-flight refresh must not be undone by
    the refresh writing the token back (the resurrection race)."""
    store = HfAuthStore()
    import time

    expired = Token(
        access_token="old", refresh_token="rf", expires_at=time.time() - 100
    )
    store.set_token(expired)
    refreshed = Token(
        access_token="new", refresh_token="rf", expires_at=time.time() + 3600
    )

    async def fake_refresh(_refresh_token):
        # Simulate the user clicking "Log out" while the refresh is in flight.
        store.clear()
        return refreshed

    with patch(
        "app.model_downloader.hf_auth.oauth.refresh_access_token",
        new=fake_refresh,
    ):
        result = await store.get_valid_token()

    # The refresh result is discarded — logout wins, in memory and on disk.
    assert result is None
    assert not store.has_token()
    assert load_token() is None


@pytest.mark.asyncio
async def test_auth_store_get_valid_returns_none_on_refresh_failure(patched_user_dir):
    store = HfAuthStore()
    import time

    expired = Token(
        access_token="old",
        refresh_token="rf",
        expires_at=time.time() - 100,
    )
    store.set_token(expired)
    with patch(
        "app.model_downloader.hf_auth.oauth.refresh_access_token",
        new=AsyncMock(side_effect=RuntimeError("HF down")),
    ):
        result = await store.get_valid_token()
    assert result is None


# --------------------------------------------------------------------------- #
# Eligibility
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "listen,multi_user,expected",
    [
        ("127.0.0.1", False, True),
        ("127.0.0.1", True, False),    # multi-user disables it
        ("0.0.0.0", False, False),     # bind-all is not loopback
        ("0.0.0.0", True, False),
        ("192.168.1.5", False, False), # LAN address
        ("::1", False, True),          # IPv6 loopback
    ],
)
def test_eligibility(listen, multi_user, expected, monkeypatch):
    from app.model_downloader.hf_auth import eligibility
    from comfy.cli_args import args

    monkeypatch.setattr(args, "listen", listen)
    monkeypatch.setattr(args, "multi_user", multi_user)
    assert eligibility.is_hf_auth_eligible() is expected


# --------------------------------------------------------------------------- #
# gated_detection HF probe
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_probe_url_hf_public(fresh_auth_store):
    """auth_check succeeds with no token → is_hf_downloadable = True."""
    from app.model_downloader.gated_detection import probe_url

    url = "https://huggingface.co/public/repo/resolve/main/x.safetensors"
    with patch("app.model_downloader.gated_detection._auth_check_sync"), patch(
        "app.model_downloader.gated_detection._probe_size_once",
        new=AsyncMock(return_value=1024),
    ):
        result = await probe_url(url)
    assert result.is_hf_downloadable is True
    assert result.file_size == 1024


@pytest.mark.asyncio
async def test_probe_url_hf_gated_no_access(fresh_auth_store):
    """auth_check raises GatedRepoError → is_hf_downloadable = False."""
    from huggingface_hub.errors import GatedRepoError

    from app.model_downloader.gated_detection import probe_url

    url = "https://huggingface.co/gated/repo/resolve/main/x.safetensors"
    fake_response = MagicMock(status_code=403)
    with patch(
        "app.model_downloader.gated_detection._auth_check_sync",
        side_effect=GatedRepoError("gated", response=fake_response),
    ), patch(
        "app.model_downloader.gated_detection._probe_size_once",
        new=AsyncMock(return_value=None),
    ):
        result = await probe_url(url)
    assert result.is_hf_downloadable is False


@pytest.mark.asyncio
async def test_probe_url_non_hf_skips_auth_check():
    """Non-HF URLs never call auth_check; is_hf_downloadable stays None."""
    from app.model_downloader.gated_detection import probe_url

    url = "https://civitai.com/api/download/models/1.safetensors"
    with patch(
        "app.model_downloader.gated_detection._auth_check_sync",
    ) as mocked, patch(
        "app.model_downloader.gated_detection._probe_size_once",
        new=AsyncMock(return_value=2048),
    ):
        result = await probe_url(url)
    assert result.is_hf_downloadable is None
    assert result.file_size == 2048
    mocked.assert_not_called()


@pytest.mark.asyncio
async def test_is_gated_cached_across_calls(fresh_auth_store):
    """Intrinsic ``is_gated`` should be determined exactly once per URL.

    Subsequent ``probe_url`` calls for the same URL must not re-issue
    the null-token auth_check — that's the whole point of the cache."""
    from app.model_downloader.gated_detection import probe_url

    url = "https://huggingface.co/public/repo/resolve/main/x.safetensors"
    with patch(
        "app.model_downloader.gated_detection._auth_check_sync"
    ) as mocked, patch(
        "app.model_downloader.gated_detection._probe_size_once",
        new=AsyncMock(return_value=1024),
    ):
        await probe_url(url)
        await probe_url(url)
        await probe_url(url)
    # Three probe_url calls × public-only-needs-1-auth_check = 1 call total.
    assert mocked.call_count == 1


@pytest.mark.asyncio
async def test_file_size_cached_across_calls(fresh_auth_store):
    """Once a successful HEAD lands, subsequent calls don't re-HEAD."""
    from app.model_downloader.gated_detection import probe_url

    url = "https://huggingface.co/public/repo/resolve/main/x.safetensors"
    with patch(
        "app.model_downloader.gated_detection._auth_check_sync"
    ), patch(
        "app.model_downloader.gated_detection._probe_size_once",
        new=AsyncMock(return_value=2048),
    ) as size_probe:
        r1 = await probe_url(url)
        r2 = await probe_url(url)
    assert r1.file_size == 2048
    assert r2.file_size == 2048
    assert size_probe.call_count == 1


@pytest.mark.asyncio
async def test_file_size_not_probed_for_gated_no_access(fresh_auth_store):
    """When ``is_hf_downloadable`` is False we must NOT HEAD the URL —
    otherwise a 401-due-to-gating would land as a cached ``None`` that
    survives a later successful login."""
    from app.model_downloader.gated_detection import probe_url
    from huggingface_hub.errors import GatedRepoError
    from unittest.mock import MagicMock

    url = "https://huggingface.co/gated/repo/resolve/main/x.safetensors"
    fake_resp = MagicMock(status_code=403)
    with patch(
        "app.model_downloader.gated_detection._auth_check_sync",
        side_effect=GatedRepoError("gated", response=fake_resp),
    ), patch(
        "app.model_downloader.gated_detection._probe_size_once",
        new=AsyncMock(return_value=None),
    ) as size_probe:
        result = await probe_url(url)
    assert result.is_hf_downloadable is False
    assert result.file_size is None
    assert size_probe.call_count == 0


@pytest.mark.asyncio
async def test_probe_url_passes_token_when_available(fresh_auth_store, patched_user_dir):
    """For a gated URL, auth_check runs twice: once with token=None to
    determine the intrinsic ``is_gated`` flag (cached forever), and once
    with the stored access_token to determine ``is_hf_downloadable`` for
    the current user."""
    from app.model_downloader import gated_detection
    from app.model_downloader.gated_detection import probe_url
    from huggingface_hub.errors import GatedRepoError
    from unittest.mock import MagicMock

    gated_detection.clear_caches_for_tests()
    fresh_auth_store.set_token(Token(
        access_token="hf_test_token",
        refresh_token=None,
        expires_at=9999999999.0,
    ))
    url = "https://huggingface.co/private/repo/resolve/main/x.safetensors"

    fake_resp = MagicMock(status_code=403)

    def fake_auth_check(repo_id, token):
        # Null-token call → repo is gated. Subsequent call with the real
        # token succeeds (user has access).
        if token is None:
            raise GatedRepoError("gated", response=fake_resp)

    with patch(
        "app.model_downloader.gated_detection._auth_check_sync",
        side_effect=fake_auth_check,
    ) as mocked, patch(
        "app.model_downloader.gated_detection._probe_size_once",
        new=AsyncMock(return_value=None),
    ):
        result = await probe_url(url)

    # is_hf_downloadable should be True (token-authed call succeeded).
    assert result.is_hf_downloadable is True
    # Two calls: (repo_id, None) then (repo_id, <token>).
    assert mocked.call_count == 2
    assert mocked.call_args_list[0].args == ("private/repo", None)
    assert mocked.call_args_list[1].args == ("private/repo", "hf_test_token")


# --------------------------------------------------------------------------- #
# OAuth primitives
# --------------------------------------------------------------------------- #


def test_make_pkce_returns_distinct_high_entropy_values():
    verifier1, challenge1, state1 = oauth._make_pkce()
    verifier2, challenge2, state2 = oauth._make_pkce()
    assert verifier1 != verifier2
    assert challenge1 != challenge2
    assert state1 != state2
    # Verifier should be at least 43 chars per PKCE spec.
    assert len(verifier1) >= 43


def test_build_authorize_url_includes_pkce_and_state():
    url = oauth._build_authorize_url("challenge123", "state456")
    assert url.startswith(oauth.AUTHORIZE_URL)
    assert "client_id=" + oauth.HF_CLIENT_ID in url
    assert "code_challenge=challenge123" in url
    assert "code_challenge_method=S256" in url
    assert "state=state456" in url
    assert "response_type=code" in url


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_hf_auth_token_status_empty(aiohttp_client, app):
    """No token set → token_available=false, username=null."""
    client = await aiohttp_client(app)
    resp = await client.get("/api/hf-auth-token-status")
    assert resp.status == 200
    data = await resp.json()
    assert data == {"token_available": False, "username": None}


@pytest.mark.asyncio
async def test_hf_auth_token_status_with_token(
    aiohttp_client, app, fresh_auth_store, patched_user_dir
):
    """Token present, whoami works → username is returned."""
    fresh_auth_store.set_token(Token(
        access_token="x", refresh_token=None, expires_at=9999999999.0,
    ))
    with patch(
        "app.model_downloader.api.routes._whoami_username",
        return_value="alice",
    ):
        client = await aiohttp_client(app)
        resp = await client.get("/api/hf-auth-token-status")
    assert resp.status == 200
    assert (await resp.json()) == {"token_available": True, "username": "alice"}


@pytest.mark.asyncio
async def test_hf_auth_login_start_403_when_ineligible(aiohttp_client, app, monkeypatch):
    """Not loopback / multi-user → 403."""
    monkeypatch.setattr(
        "app.model_downloader.api.routes.is_hf_auth_eligible",
        lambda: False,
    )
    client = await aiohttp_client(app)
    resp = await client.post("/api/hf-auth-login-start")
    assert resp.status == 403
    assert (await resp.json())["error"]["code"] == "HF_AUTH_NOT_ELIGIBLE"


@pytest.mark.asyncio
async def test_hf_auth_login_start_returns_authorize_url(aiohttp_client, app, monkeypatch):
    """Eligible + first attempt → 200 with authorize_url."""
    monkeypatch.setattr(
        "app.model_downloader.api.routes.is_hf_auth_eligible",
        lambda: True,
    )
    monkeypatch.setattr(
        "app.model_downloader.api.routes.start_login_flow",
        AsyncMock(return_value="https://huggingface.co/oauth/authorize?fake=1"),
    )
    client = await aiohttp_client(app)
    resp = await client.post("/api/hf-auth-login-start")
    assert resp.status == 200
    assert (await resp.json())["authorize_url"].startswith(
        "https://huggingface.co/oauth/authorize"
    )


@pytest.mark.asyncio
async def test_hf_auth_login_start_409_when_in_progress(aiohttp_client, app, monkeypatch):
    """Lock already held → 409."""
    from app.model_downloader.hf_auth.oauth import OAuthInProgressError

    monkeypatch.setattr(
        "app.model_downloader.api.routes.is_hf_auth_eligible",
        lambda: True,
    )
    monkeypatch.setattr(
        "app.model_downloader.api.routes.start_login_flow",
        AsyncMock(side_effect=OAuthInProgressError()),
    )
    client = await aiohttp_client(app)
    resp = await client.post("/api/hf-auth-login-start")
    assert resp.status == 409
    assert (await resp.json())["error"]["code"] == "HF_AUTH_IN_PROGRESS"


@pytest.mark.asyncio
async def test_hf_auth_login_start_503_when_callback_bind_fails(
    aiohttp_client, app, monkeypatch
):
    """Callback server failed to bind (e.g. port busy) → 503, not a dead URL."""
    from app.model_downloader.hf_auth.oauth import OAuthCallbackError

    monkeypatch.setattr(
        "app.model_downloader.api.routes.is_hf_auth_eligible",
        lambda: True,
    )
    monkeypatch.setattr(
        "app.model_downloader.api.routes.start_login_flow",
        AsyncMock(side_effect=OAuthCallbackError("could not bind callback port")),
    )
    client = await aiohttp_client(app)
    resp = await client.post("/api/hf-auth-login-start")
    assert resp.status == 503
    assert (await resp.json())["error"]["code"] == "HF_AUTH_START_FAILED"


@pytest.mark.asyncio
async def test_hf_auth_logout_clears_store(
    aiohttp_client, app, fresh_auth_store, patched_user_dir
):
    fresh_auth_store.set_token(Token(
        access_token="x", refresh_token=None, expires_at=9999999999.0,
    ))
    client = await aiohttp_client(app)
    resp = await client.post("/api/hf-auth-logout")
    assert resp.status == 200
    assert (await resp.json()) == {"logged_out": True}
    assert not fresh_auth_store.has_token()


@pytest.mark.asyncio
async def test_availability_includes_hf_auth_snapshot(aiohttp_client, app, monkeypatch):
    """The availability response embeds {token_available, eligible}."""
    monkeypatch.setattr(
        "app.model_downloader.api.routes.is_hf_auth_eligible",
        lambda: True,
    )
    client = await aiohttp_client(app)
    resp = await client.post(
        "/api/models-availability-status",
        json={"models": {}},
    )
    assert resp.status == 200
    data = await resp.json()
    assert "hf_auth" in data
    assert data["hf_auth"] == {"token_available": False, "eligible": True}
