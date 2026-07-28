"""Public facade for the download manager.

This is the only object the server imports. It validates requests, owns the
:class:`Scheduler`, and exposes a small async API plus read models for status.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Callable, Optional

from app.model_downloader.constants import DownloadStatus
from app.model_downloader.database import queries
from app.model_downloader.net.probe import gated_error_message, probe
from app.model_downloader.scheduler import SCHEDULER
from app.model_downloader.security import paths
from app.model_downloader.net.http import redact_url
from app.model_downloader.security.allowlist import (
    ALLOWED_MODEL_EXTENSIONS,
    filename_extension,
    is_host_allowed_url,
    is_url_downloadable,
    url_path_extension,
)
from app.model_downloader.security.paths import InvalidModelId

# Non-terminal statuses: an existing row in one of these blocks a re-enqueue.
_LIVE_STATUSES = (
    DownloadStatus.QUEUED,
    DownloadStatus.ACTIVE,
    DownloadStatus.PAUSED,
    DownloadStatus.VERIFYING,
)


class DownloadError(Exception):
    """A user-facing error with a stable machine-readable code."""

    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = status


class DownloadManager:
    def __init__(self) -> None:
        self._scheduler = SCHEDULER
        self._notify_cb: Optional[Callable[[str], None]] = None
        # Serializes the "check for a live download, then write" critical section
        # per model_id. ``downloads`` has no uniqueness constraint on model_id
        # (history rows are kept), so without this two concurrent enqueue/resume
        # calls could both pass the live check and admit two jobs sharing one
        # temp/dest path. The manager is a process singleton over a local SQLite
        # DB, so an in-process lock is sufficient (and avoids a migration).
        self._model_locks: dict[str, asyncio.Lock] = {}

    def set_notify(self, cb: Optional[Callable[[str], None]]) -> None:
        self._notify_cb = cb
        self._scheduler.set_notify(cb)

    async def start(self) -> None:
        await self._scheduler.start()

    # ----- enqueue -----

    async def enqueue(
        self,
        url: str,
        model_id: str,
        *,
        priority: int = 0,
        expected_sha256: Optional[str] = None,
        allow_any_extension: bool = False,
    ) -> str:
        # Coarse gate first: host/scheme must be allowlisted, and any extension
        # present in the URL path must be a known model type. A URL whose path
        # carries NO extension (e.g. Civitai's ``/api/download/models/<id>``) is
        # admitted here and its real extension is resolved from the network
        # below before the download is finally accepted.
        if allow_any_extension:
            if not is_host_allowed_url(url):
                raise DownloadError(
                    "URL_NOT_ALLOWED",
                    "URL is not on the download allowlist (host/scheme).",
                )
        elif not is_url_downloadable(url):
            raise DownloadError(
                "URL_NOT_ALLOWED",
                "URL is not on the download allowlist (host/scheme/extension).",
            )

        # When the URL path has no extension, follow it to where it resolves and
        # adopt the real extension from the response, forcing the stored
        # filename to match. Skipped when the caller opted into any extension.
        if not allow_any_extension and url_path_extension(url) == "":
            resolved_ext = await self._resolve_extension(url)
            model_id = paths.apply_extension(model_id, resolved_ext)

        try:
            paths.parse_model_id(model_id, allow_any_extension)
            dest_path, temp_path = paths.resolve_destination(model_id, allow_any_extension)
        except InvalidModelId as e:
            raise DownloadError("INVALID_MODEL_ID", str(e))

        if await asyncio.to_thread(
            paths.resolve_existing, model_id, allow_any_extension
        ):
            raise DownloadError(
                "ALREADY_AVAILABLE",
                f"Model already exists on disk: {model_id}",
                status=409,
            )
        download_id = str(uuid.uuid4())
        # Hold the per-model lock across the live check and the insert so a
        # concurrent enqueue/resume for the same model_id cannot interleave
        # between them and create a second job against the same temp/dest path.
        async with self._model_lock(model_id):
            if await self._has_live_download(model_id):
                raise DownloadError(
                    "ALREADY_DOWNLOADING",
                    f"A download for {model_id} is already in progress.",
                    status=409,
                )
            await asyncio.to_thread(
                queries.insert_download,
                {
                    "id": download_id,
                    "url": url,
                    "model_id": model_id,
                    "dest_path": dest_path,
                    "temp_path": temp_path,
                    "status": DownloadStatus.QUEUED,
                    "priority": priority,
                    "expected_sha256": expected_sha256,
                    "allow_any_extension": allow_any_extension,
                },
            )
        logging.info("[model_downloader] enqueued %s -> %s", redact_url(url), model_id)
        await self._scheduler.pump()
        return download_id

    async def _resolve_extension(self, url: str) -> str:
        """Follow ``url`` to its final response and return the real extension.

        Used for allowlisted URLs whose path has no extension (e.g. Civitai
        download endpoints): the filename lives in the ``Content-Disposition``
        header or the post-redirect URL. Raises :class:`DownloadError` when the
        URL can't be resolved, needs authentication, or resolves to something
        that is not a known model file — so we never persist a bogus destination.
        """
        pr = await probe(url)
        if not pr.ok:
            if pr.gated:
                raise DownloadError(
                    "GATED_REPO" if pr.is_gated_repo else "CREDENTIALS_REQUIRED",
                    gated_error_message(url, pr),
                    status=401,
                )
            raise DownloadError(
                "URL_RESOLVE_FAILED",
                f"Could not resolve {redact_url(url)}: {pr.error or 'unknown error'}",
                status=502,
            )
        ext = filename_extension(pr.filename) if pr.filename else ""
        if ext not in ALLOWED_MODEL_EXTENSIONS:
            raise DownloadError(
                "URL_NOT_ALLOWED",
                f"URL resolves to {pr.filename or '<unknown>'!r}, which is not a "
                f"known model file type {ALLOWED_MODEL_EXTENSIONS}.",
            )
        return ext

    def _model_lock(self, model_id: str) -> asyncio.Lock:
        # Lazily create one lock per model_id. There is no ``await`` between the
        # lookup and the insert, so under the single asyncio thread this is
        # atomic and cannot hand out two different locks for the same model_id.
        lock = self._model_locks.get(model_id)
        if lock is None:
            lock = asyncio.Lock()
            self._model_locks[model_id] = lock
        return lock

    async def _has_live_download(
        self, model_id: str, *, exclude_id: Optional[str] = None
    ) -> bool:
        return await asyncio.to_thread(
            queries.has_live_download_for_model, model_id, _LIVE_STATUSES, exclude_id
        )

    # ----- control -----

    async def pause(self, download_id: str) -> None:
        job = self._scheduler.get_job(download_id)
        if job is not None:
            job.request_pause()
            return
        row = await asyncio.to_thread(queries.get_download, download_id)
        if row is None:
            raise DownloadError("NOT_FOUND", "No such download.", status=404)
        if row.status == DownloadStatus.QUEUED:
            await asyncio.to_thread(
                queries.update_download, download_id, status=DownloadStatus.PAUSED
            )

    async def resume(self, download_id: str) -> None:
        row = await asyncio.to_thread(queries.get_download, download_id)
        if row is None:
            raise DownloadError("NOT_FOUND", "No such download.", status=404)
        if row.status not in (DownloadStatus.PAUSED, DownloadStatus.FAILED):
            return
        # Re-queueing a paused/failed row must respect the single-live-per-model
        # invariant: another download (e.g. a newer enqueue) may already be live
        # for this model_id and would share this row's temp/dest path. Hold the
        # per-model lock across the check and the status flip, and exclude this
        # row itself (a paused row is already a "live" status).
        async with self._model_lock(row.model_id):
            if await self._has_live_download(row.model_id, exclude_id=download_id):
                raise DownloadError(
                    "ALREADY_DOWNLOADING",
                    f"A download for {row.model_id} is already in progress.",
                    status=409,
                )
            await asyncio.to_thread(
                queries.update_download,
                download_id,
                status=DownloadStatus.QUEUED,
                error=None,
            )
        await self._scheduler.pump()

    async def cancel(self, download_id: str) -> None:
        job = self._scheduler.get_job(download_id)
        if job is not None:
            job.request_cancel()
            return
        row = await asyncio.to_thread(queries.get_download, download_id)
        if row is None:
            raise DownloadError("NOT_FOUND", "No such download.", status=404)
        if row.status in _LIVE_STATUSES:
            try:
                os.remove(row.temp_path)
            except OSError:
                pass
            await asyncio.to_thread(
                queries.update_download, download_id, status=DownloadStatus.CANCELLED
            )

    async def set_priority(self, download_id: str, priority: int) -> None:
        row = await asyncio.to_thread(queries.get_download, download_id)
        if row is None:
            raise DownloadError("NOT_FOUND", "No such download.", status=404)
        await asyncio.to_thread(
            queries.update_download, download_id, priority=priority
        )
        # Admission-order only; a higher priority is
        # picked up the next time a slot frees. Pump in case a slot is free now.
        await self._scheduler.pump()

    async def delete(self, download_id: str) -> None:
        """Delete a terminal download so it stays gone from history.

        Refuses to delete a live download so a record is never removed out from
        under a running worker; cancel it first. Any leftover ``.part`` temp
        file (e.g. from a failed transfer) is removed, but the finished model
        file on disk is never touched.
        """
        if self._scheduler.get_job(download_id) is not None:
            raise DownloadError(
                "DOWNLOAD_ACTIVE",
                "Cannot delete a download that is still in progress.",
                status=409,
            )
        row = await asyncio.to_thread(queries.get_download, download_id)
        if row is None:
            raise DownloadError("NOT_FOUND", "No such download.", status=404)
        if row.status in _LIVE_STATUSES:
            raise DownloadError(
                "DOWNLOAD_ACTIVE",
                "Cannot delete a download that is still in progress.",
                status=409,
            )

        try:
            os.remove(row.temp_path)
        except OSError:
            pass
        await asyncio.to_thread(queries.delete_download, download_id)

    async def clear(self) -> int:
        """Delete all terminal downloads from history in one transaction.

        Skips anything still live (queued/active/paused/verifying, or a running
        job) so an in-flight download is never removed out from under a worker.
        Finished model files on disk are never touched; only leftover ``.part``
        temp files from failed/cancelled transfers are removed. Returns the
        number of history rows deleted.
        """

        rows = await asyncio.to_thread(queries.list_downloads)
        deletable = [
            r
            for r in rows
            if r.status not in _LIVE_STATUSES
            and self._scheduler.get_job(r.id) is None
        ]
        if not deletable:
            return 0
        for r in deletable:
            try:
                os.remove(r.temp_path)
            except OSError:
                pass
        return await asyncio.to_thread(
            queries.delete_downloads, [r.id for r in deletable]
        )

    # ----- read models -----

    def _view(self, row) -> dict:
        """Combine the persisted row with live in-memory progress, if running."""
        job = self._scheduler.get_job(row.id)
        bytes_done = row.bytes_done
        total = row.total_bytes
        speed = None
        eta = None
        segments = None
        if job is not None:
            st = job.state
            bytes_done = st.bytes_done
            total = st.total_bytes if st.total_bytes is not None else total
            speed = st.speed_bps
            eta = st.eta_seconds
            segments = [
                {"idx": s.idx, "bytes_done": s.bytes_done, "length": s.length}
                for s in st.segments
                if s.end >= s.start
            ]
        progress = (bytes_done / total) if total else None
        return {
            "download_id": row.id,
            "model_id": row.model_id,
            "url": redact_url(row.url),
            "status": row.status,
            "priority": row.priority,
            "total_bytes": total,
            "bytes_done": bytes_done,
            "progress": progress,
            "speed_bps": speed,
            "eta_seconds": eta,
            "segments": segments,
            "error": row.error,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }

    def _view_from_state(self, job) -> dict:
        """Build a view purely from the live in-memory job state (no DB)."""
        st = job.state
        return {
            "download_id": st.download_id,
            "model_id": st.model_id,
            "url": redact_url(st.url),
            "status": st.status,
            "priority": st.priority,
            "total_bytes": st.total_bytes,
            "bytes_done": st.bytes_done,
            "progress": st.progress,
            "speed_bps": st.speed_bps,
            "eta_seconds": st.eta_seconds,
            "segments": [
                {"idx": s.idx, "bytes_done": s.bytes_done, "length": s.length}
                for s in st.segments
                if s.end >= s.start
            ],
            "error": st.error,
        }

    def status_sync(self, download_id: str) -> Optional[dict]:
        """Synchronous status read for the websocket notify path.

        Uses live in-memory state when the job is running (no DB round-trip on
        the hot path); falls back to a quick DB read otherwise.
        """
        job = self._scheduler.get_job(download_id)
        if job is not None:
            return self._view_from_state(job)
        row = queries.get_download(download_id)
        return self._view(row) if row is not None else None

    async def status(self, download_id: str) -> Optional[dict]:
        row = await asyncio.to_thread(queries.get_download, download_id)
        return self._view(row) if row is not None else None

    async def list(self) -> list[dict]:
        rows = await asyncio.to_thread(queries.list_downloads)
        return [self._view(r) for r in rows]

    async def availability(self, models: dict[str, str]) -> dict[str, dict]:
        """Bulk per-id ``{state, progress, ...}`` for the frontend poll.

        ``state`` is ``available`` (on disk), ``downloading`` (live row), or
        ``missing``. Cheap: a path lookup plus an in-memory/DB status check.
        """
        rows = await asyncio.to_thread(queries.list_downloads)
        by_model: dict[str, object] = {}
        for r in rows:
            if r.status in _LIVE_STATUSES or r.model_id not in by_model:
                by_model[r.model_id] = r

        # ``url_allowed`` mirrors the coarse enqueue gate (host/scheme + a
        # non-disallowed extension); URLs whose extension is only known after a
        # network resolve — e.g. Civitai download endpoints — report allowed.
        out: dict[str, dict] = {}
        for model_id, url in models.items():
            try:
                exists = await asyncio.to_thread(paths.resolve_existing, model_id)
            except InvalidModelId:
                out[model_id] = {"state": "missing", "url_allowed": is_url_downloadable(url)}
                continue
            if exists:
                out[model_id] = {"state": "available", "url_allowed": is_url_downloadable(url)}
                continue
            row = by_model.get(model_id)
            if row is not None and row.status in _LIVE_STATUSES:
                view = self._view(row)
                out[model_id] = {
                    "state": "downloading",
                    "url_allowed": is_url_downloadable(url),
                    "download_id": view["download_id"],
                    "progress": view["progress"],
                    "bytes_done": view["bytes_done"],
                    "total_bytes": view["total_bytes"],
                    "speed_bps": view["speed_bps"],
                }
            else:
                out[model_id] = {"state": "missing", "url_allowed": is_url_downloadable(url)}
        return out


DOWNLOAD_MANAGER = DownloadManager()
