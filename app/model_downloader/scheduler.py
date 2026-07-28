"""Priority scheduler + lifecycle.

Owns the set of running jobs and admits queued downloads up to a global
concurrency limit (K), highest priority first, FIFO within a priority. Runs
entirely on the existing ComfyUI asyncio loop; blocking work (disk, hashing,
DB) is offloaded by the job/writer layers.

On startup it reconciles DB vs. disk: ``active``/``verifying`` rows left by a
previous run are reset to ``queued`` and resumed from persisted offsets, and
orphaned ``.part`` files with no live download row are swept.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Callable, Optional

from comfy.cli_args import args
from app.model_downloader.constants import DownloadStatus
from app.model_downloader.database import queries
from app.model_downloader.engine.job import DownloadJob, JobSpec
from app.model_downloader.security import paths

# Backoff for retryable failures
_BACKOFF_BASE = 2.0
_BACKOFF_CAP = 300.0
_MAX_ATTEMPTS = 6


class Scheduler:
    def __init__(self) -> None:
        self._jobs: dict[str, DownloadJob] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._backoff_until: dict[str, float] = {}
        self._pump_lock = asyncio.Lock()
        self._notify_cb: Optional[Callable[[str], None]] = None
        self._started = False

    @property
    def max_active(self) -> int:
        return max(1, getattr(args, "download_max_active", 3))

    def set_notify(self, cb: Optional[Callable[[str], None]]) -> None:
        self._notify_cb = cb

    def get_job(self, download_id: str) -> Optional[DownloadJob]:
        return self._jobs.get(download_id)

    def is_active(self, download_id: str) -> bool:
        return download_id in self._tasks

    # ----- startup -----

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        try:
            await asyncio.to_thread(queries.reconcile_live_downloads)
            await asyncio.to_thread(self._sweep_orphan_temp_files)
        except Exception as e:
            logging.warning("[model_downloader] startup reconcile failed: %s", e)
        await self.pump()

    @staticmethod
    def _sweep_orphan_temp_files() -> None:
        """Remove ``.part`` files not referenced by a resumable download row.

        Resumable partials are preserved; only truly orphaned temp files from
        crashed runs are deleted. ``FAILED`` is included because
        :meth:`DownloadManager.resume` explicitly permits resuming a
        retry-exhausted failed row: deleting its partial here while the
        per-segment offsets survive in the DB would make the next resume
        preallocate a fresh sparse file, skip every "complete" segment, and
        leave zero-filled holes that pass the size-only verification gate.
        """
        live = {
            row.temp_path
            for row in queries.list_downloads()
            if row.status
            in (
                DownloadStatus.QUEUED,
                DownloadStatus.PAUSED,
                DownloadStatus.FAILED,
            )
        }
        for path in paths.iter_all_tmp_paths():
            if path in live:
                continue
            try:
                os.remove(path)
                logging.info("[model_downloader] removed orphan temp file: %s", path)
            except OSError as e:
                logging.warning("[model_downloader] could not remove %s: %s", path, e)

    # ----- admission -----

    async def pump(self) -> None:
        async with self._pump_lock:
            slots = self.max_active - len(self._tasks)
            if slots <= 0:
                return
            now = time.monotonic()
            candidates = await asyncio.to_thread(queries.list_queued_downloads)
            for row in candidates:
                if slots <= 0:
                    break
                if row.id in self._tasks:
                    continue
                if self._backoff_until.get(row.id, 0.0) > now:
                    continue
                self._admit(row)
                slots -= 1

    def _admit(self, row) -> None:
        spec = JobSpec(
            download_id=row.id,
            url=row.url,
            model_id=row.model_id,
            dest_path=row.dest_path,
            temp_path=row.temp_path,
            priority=row.priority,
            expected_sha256=row.expected_sha256,
            allow_any_extension=row.allow_any_extension,
            etag=row.etag,
            attempts=row.attempts,
        )
        job = DownloadJob(spec, notify_cb=self._notify_cb)
        self._jobs[row.id] = job
        self._tasks[row.id] = asyncio.ensure_future(self._run_job(job))

    async def _run_job(self, job: DownloadJob) -> None:
        download_id = job.spec.download_id
        status = DownloadStatus.FAILED
        try:
            status = await job.run()
        except Exception as e:  # run() is defensive, but never let a task die silently
            logging.error("[model_downloader] job %s crashed: %s", download_id, e)
            queries.update_download(
                download_id,
                status=DownloadStatus.FAILED,
                error=f"internal error: {e}",
            )
            if self._notify_cb:
                self._notify_cb(download_id)
        finally:
            self._tasks.pop(download_id, None)
            self._jobs.pop(download_id, None)

        if status == DownloadStatus.QUEUED:
            if job.spec.attempts >= _MAX_ATTEMPTS:
                queries.update_download(
                    download_id,
                    status=DownloadStatus.FAILED,
                    error=f"giving up after {job.spec.attempts} attempts",
                )
                if self._notify_cb:
                    self._notify_cb(download_id)
            else:
                delay = min(
                    _BACKOFF_CAP, _BACKOFF_BASE ** job.spec.attempts
                ) + random.uniform(0, 1.0)
                self._backoff_until[download_id] = time.monotonic() + delay
                asyncio.ensure_future(self._delayed_pump(delay))
        await self.pump()

    async def _delayed_pump(self, delay: float) -> None:
        await asyncio.sleep(delay)
        await self.pump()


SCHEDULER = Scheduler()
