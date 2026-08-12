"""Per-URL probes for the unified availability endpoint.

Three cached/derived facts per URL:

  - ``is_gated``         intrinsic to the model; cached forever once known.
                         Determined by ``auth_check(repo_id, token=None)``:
                         ``GatedRepoError`` → True, success → False.

  - ``is_hf_downloadable`` depends on the *current* token; recomputed every
                         call. For non-gated URLs this is trivially True
                         (no HF call needed). For gated URLs we run
                         ``auth_check`` with the stored token each call.

  - ``file_size``        intrinsic to the file. Cached forever once
                         determined (including ``None`` on transient
                         failure — we don't retry). We only attempt the
                         HEAD when we already know the URL is downloadable
                         to us; that way a failed-because-gated probe
                         never lands as a cached ``None``.

Caches are per-process, in-memory; small, no eviction needed for the
workflow-scale (~tens of URLs). Concurrent calls for the same URL
deduplicate via per-URL ``asyncio.Lock``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import aiohttp
from huggingface_hub import HfApi
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    RepositoryNotFoundError,
)

from app.model_downloader.allowlist import is_url_allowed
from app.model_downloader.hf_auth.auth_store import HF_AUTH_STORE
from app.model_downloader.hf_url import is_hf_url, repo_id_from_url
from app.model_downloader.http_client import get_session, parse_content_length


_HEAD_TIMEOUT = aiohttp.ClientTimeout(total=15)


@dataclass
class ProbeResult:
    file_size: Optional[int]
    is_hf_downloadable: Optional[bool]


# --- caches -------------------------------------------------------------- #


# url → bool. Whether this URL's HF repo gates access. Intrinsic to the
# model — never changes for a given URL.
_is_gated_cache: dict[str, bool] = {}

# url → Optional[int]. The file's size in bytes, ``None`` if a probe
# was attempted and produced no answer. **Only populated when we knew
# the URL was downloadable to us at probe time** — so gated-without-
# access never lands a ``None`` here that we'd be stuck with after login.
_file_size_cache: dict[str, Optional[int]] = {}

# Per-URL locks for single-flight probes — when multiple polls arrive
# in the same tick for the same URL, exactly one of them runs the HF
# call and the others wait on the result.
_locks: dict[str, asyncio.Lock] = {}


def _lock_for(url: str) -> asyncio.Lock:
    lock = _locks.get(url)
    if lock is None:
        lock = asyncio.Lock()
        _locks[url] = lock
    return lock


def clear_caches_for_tests() -> None:
    """Test-only: drop everything."""
    _is_gated_cache.clear()
    _file_size_cache.clear()
    _locks.clear()


# --- public entrypoint --------------------------------------------------- #


async def probe_url(url: str) -> ProbeResult:
    """Return downloadability + size for one URL, using caches where safe."""
    if not is_url_allowed(url):
        return ProbeResult(file_size=None, is_hf_downloadable=None)
    if not is_hf_url(url):
        # Non-HF: ``is_hf_downloadable`` is "not applicable" (None).
        # Size we still cache so we don't HEAD on every poll.
        size = await _get_or_probe_size(url, token=None)
        return ProbeResult(file_size=size, is_hf_downloadable=None)

    repo_id = repo_id_from_url(url)
    if repo_id is None:
        return ProbeResult(file_size=None, is_hf_downloadable=None)

    # Determine intrinsic gating once.
    gated = await _resolve_is_gated(url, repo_id)
    if gated is None:
        return ProbeResult(file_size=None, is_hf_downloadable=None)

    # Compute current-token downloadability per call.
    tok = await HF_AUTH_STORE.get_valid_token()
    token_str: Optional[str] = tok.access_token if tok else None
    if not gated:
        is_hf_downloadable: Optional[bool] = True
    else:
        is_hf_downloadable = await _auth_check_with_token(repo_id, token_str)

    if is_hf_downloadable is True:
        size = await _get_or_probe_size(url, token=token_str)
    else:
        # Skip the HEAD entirely — would 401 and we'd be stuck with
        # cached None that survives a later login.
        size = None

    return ProbeResult(file_size=size, is_hf_downloadable=is_hf_downloadable)


# --- gated/auth probes --------------------------------------------------- #


async def _resolve_is_gated(url: str, repo_id: str) -> Optional[bool]:
    """Decide once whether ``repo_id`` is a gated repo."""
    cached = _is_gated_cache.get(url)
    if cached is not None:
        return cached

    async with _lock_for(url):
        cached = _is_gated_cache.get(url)
        if cached is not None:
            return cached
        # Probe anonymously (token=None) on purpose: an unauthenticated
        # auth_check is what makes HF raise GatedRepoError for gated repos.
        # With a token, a gated-but-accepted repo would succeed and look
        # ungated.
        try:
            await asyncio.to_thread(_auth_check_sync, repo_id, None)
            _is_gated_cache[url] = False
            return False
        except GatedRepoError:
            _is_gated_cache[url] = True
            return True
        except RepositoryNotFoundError:
            # Repo doesn't exist publicly. Treat as gated — we can't
            # serve it without auth, and an authenticated check might
            # still succeed if it's a private repo the user can see.
            _is_gated_cache[url] = True
            return True
        except (HfHubHTTPError, Exception) as e:
            logging.debug(
                "[hf_auth] is_gated probe failed for %s (will retry): %s",
                repo_id, e,
            )
            return None  # don't cache; retry next call


async def _auth_check_with_token(
    repo_id: str, token: Optional[str]
) -> Optional[bool]:
    """Auth-check with the supplied token. True/False/None per outcome."""
    try:
        await asyncio.to_thread(_auth_check_sync, repo_id, token)
        return True
    except GatedRepoError:
        return False
    except RepositoryNotFoundError:
        return False
    except HfHubHTTPError as e:
        # 401/403 covers org-SSO-required, revoked tokens, and similar —
        # all of which mean "can't fetch right now" from the user's POV.
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status in (401, 403):
            return False
        logging.debug(
            "[hf_auth] auth_check transient failure for %s: %s", repo_id, e,
        )
        return None
    except Exception as e:
        logging.warning("[hf_auth] unexpected auth_check error for %s: %s", repo_id, e)
        return None


def _auth_check_sync(repo_id: str, token: Optional[str]) -> None:
    """Thin sync wrapper around ``HfApi.auth_check`` for ``asyncio.to_thread``."""
    HfApi().auth_check(repo_id, token=token)


# --- size probe ---------------------------------------------------------- #


async def _get_or_probe_size(url: str, token: Optional[str]) -> Optional[int]:
    """Return the cached size or HEAD the URL once and cache the result."""
    if url in _file_size_cache:
        return _file_size_cache[url]

    async with _lock_for(url):
        if url in _file_size_cache:
            return _file_size_cache[url]
        size = await _probe_size_once(url, token=token)
        _file_size_cache[url] = size
        return size


async def _probe_size_once(url: str, token: Optional[str]) -> Optional[int]:
    """HEAD the URL and return the file size in bytes, or None on failure.

    HuggingFace serves LFS-tracked files via a 302 to a signed CDN URL.
    The real file size lives in the non-standard ``X-Linked-Size`` header
    on that 302 response (``Content-Length`` is the redirect-body length).
    Disabling redirect-follow lets us read either header on the same
    response:

      - LFS files: 302 + ``X-Linked-Size``
      - Small/non-LFS files: 200 + ``Content-Length``
    """
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        session = await get_session()
        async with session.head(
            url, allow_redirects=False, timeout=_HEAD_TIMEOUT, headers=headers,
        ) as resp:
            linked = parse_content_length(resp.headers.get("X-Linked-Size"))
            if linked is not None:
                return linked
            if resp.status == 200:
                return parse_content_length(resp.headers.get("Content-Length"))
            return None
    except (aiohttp.ClientError, TimeoutError, OSError):
        return None


# Backward-compat shim so consumers that still import the old name keep
# building during the refactor; can be removed once routes are updated.
MetadataProbeResult = ProbeResult
