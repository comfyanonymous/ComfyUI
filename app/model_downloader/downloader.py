"""Streaming download worker with progress reporting and cancellation.

Each download writes to ``<final_path>.tmp`` and atomically renames into
place on success. Between chunks the worker checks the registry for
cancellation (via ``DownloadServer.is_active``) and rolls back its
``.tmp`` on cancel or on any error.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import aiohttp

from app.model_downloader.download_server import (
    DOWNLOAD_SERVER,
    DownloadCancelled,
    DownloadSession,
)
from app.model_downloader.hf_auth.auth_store import HF_AUTH_STORE
from app.model_downloader.hf_url import is_hf_url
from app.model_downloader.http_client import get_session, parse_content_length
from app.model_downloader.paths import resolve_destination


CHUNK_SIZE = 64 * 1024  # 64 KiB — same scale as other ComfyUI download paths.
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=120)


async def stream_to_disk(session: DownloadSession) -> str:
    """Run a single download to completion or cancellation.

    Returns the final on-disk path on success. Removes the ``.tmp`` and
    raises on cancellation or failure. The session is finished
    (removed from the registry) exactly once, here — callers do not
    need to call ``DOWNLOAD_SERVER.finish`` themselves.
    """
    final_path, tmp_path = resolve_destination(session.model_id, session.epoch)
    os.makedirs(os.path.dirname(final_path), exist_ok=True)

    # Wipe any stale .tmp from a previous failed attempt before we start —
    # otherwise a partial body could masquerade as our completed download
    # when the rename finally happens.
    _remove_if_exists(tmp_path)

    bytes_seen = 0
    try:
        http = await get_session()
        headers = _auth_headers_for(session.url)
        logging.info(
            "[model_downloader] starting GET %s (auth=%s)",
            session.url, "yes" if "Authorization" in headers else "no",
        )
        async with http.get(
            session.url,
            allow_redirects=True,
            timeout=REQUEST_TIMEOUT,
            headers=headers,
        ) as resp:
            if resp.status != 200:
                # Capture a snippet of the response body so 4xx/5xx aren't
                # opaque in the logs — HF returns JSON or HTML with a
                # human-readable reason on failures.
                body_snippet = await _read_short(resp)
                logging.warning(
                    "[model_downloader] GET %s failed: status=%d final_url=%s body=%s",
                    session.url, resp.status, str(resp.url), body_snippet,
                )
                raise DownloadError(
                    f"unexpected HTTP {resp.status} fetching {session.url}: {body_snippet}",
                    status=resp.status,
                )

            total = parse_content_length(resp.headers.get("Content-Length"))
            DOWNLOAD_SERVER.update_progress(session, 0, total)

            with open(tmp_path, "wb") as f:
                async for chunk in resp.content.iter_chunked(CHUNK_SIZE):
                    # Cancellation check between chunks. Cheap and means
                    # cancellation latency is bounded by one chunk plus
                    # one ``write()`` — typically well under a second
                    # even on slow disks.
                    if not DOWNLOAD_SERVER.is_active(session):
                        raise DownloadCancelled()
                    f.write(chunk)
                    bytes_seen += len(chunk)
                    DOWNLOAD_SERVER.update_progress(session, bytes_seen, total)

        # Final cancellation check before we promote the .tmp to the real
        # filename — avoids the awkward case where cancel arrives during
        # the very last chunk and we'd otherwise commit anyway.
        if not DOWNLOAD_SERVER.is_active(session):
            raise DownloadCancelled()

        # Size verification before commit. aiohttp already raises
        # ClientPayloadError on a truncated Content-Length/chunked body,
        # but this also catches the HTTP/1.0-style case (no Content-Length
        # + Connection: close) where a short read can masquerade as a
        # complete download.
        if total is not None and bytes_seen != total:
            raise DownloadError(
                f"size mismatch for {session.model_id}: "
                f"got {bytes_seen} of {total} bytes from {session.url}"
            )

        # Atomic rename. os.replace is atomic within the same filesystem,
        # which is guaranteed here because tmp lives alongside final_path.
        os.replace(tmp_path, final_path)
        logging.info(
            "[model_downloader] downloaded %s (%d bytes) from %s",
            session.model_id, bytes_seen, session.url,
        )
        return final_path

    except DownloadCancelled:
        logging.info("[model_downloader] cancelled: %s", session.model_id)
        _remove_if_exists(tmp_path)
        raise
    except Exception as e:
        logging.warning(
            "[model_downloader] failed: %s from %s: %s: %s",
            session.model_id, session.url, type(e).__name__, e,
            exc_info=True,
        )
        _remove_if_exists(tmp_path)
        raise
    finally:
        # In all terminal states (success / cancel / error) drop the
        # session from the registry. Idempotent — only removes if we're
        # still the live epoch for this model_id.
        DOWNLOAD_SERVER.finish(session)


class DownloadError(Exception):
    """Network / protocol error during a download."""

    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


async def _read_short(resp: aiohttp.ClientResponse, limit: int = 512) -> str:
    """Read up to ``limit`` bytes of a response body for logging.

    Used to surface the JSON/HTML reason from an HF non-2xx response in
    server logs instead of just the status code. Best-effort: any
    error here is swallowed.
    """
    try:
        raw = await resp.content.read(limit)
        return raw.decode("utf-8", errors="replace").strip()
    except Exception:
        return "<unreadable>"


def _auth_headers_for(url: str) -> dict[str, str]:
    """Return any auth headers we should add to the GET for ``url``.

    For HuggingFace URLs we inject the user's OAuth access token as a
    Bearer header — this is HF's documented way to access gated repos
    (see ``huggingface_hub.hf_hub_download``'s wire format). For every
    other host we send no extra headers; allowlisted public files
    don't need them and we don't want to leak tokens to other hosts.
    """
    if not is_hf_url(url):
        return {}
    tok = HF_AUTH_STORE.get_token_sync()
    if tok is None or not tok.access_token:
        return {}
    return {"Authorization": f"Bearer {tok.access_token}"}


def _remove_if_exists(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as e:
        logging.warning("[model_downloader] could not remove %s: %s", path, e)


async def run_batch_sequential(sessions: list[DownloadSession]) -> None:
    """Run a list of sessions one after the other.

    Each session is independent: a failure or cancellation of one does
    not abort the rest. Cancellations are observable via the registry
    *before* a given download starts, so a session that's been
    pre-cancelled (cancel before the worker reached it) just gets skipped.
    """
    for session in sessions:
        # If the session got cancelled before its turn, skip without
        # touching disk. This is what makes the per-request "sequential
        # but cancellable" semantic work.
        if not DOWNLOAD_SERVER.is_active(session):
            DOWNLOAD_SERVER.finish(session)
            continue
        try:
            await stream_to_disk(session)
        except DownloadCancelled:
            # Already logged + tmp removed inside stream_to_disk.
            continue
        except Exception:
            # stream_to_disk already logged. Continue with the rest of the batch.
            continue


def schedule_batch(sessions: list[DownloadSession]) -> asyncio.Task:
    """Kick off ``run_batch_sequential`` on the running event loop.

    Returned task is fire-and-forget; the API handler returns immediately
    after scheduling and clients observe progress via the polling endpoints.
    """
    return asyncio.create_task(run_batch_sequential(sessions))
