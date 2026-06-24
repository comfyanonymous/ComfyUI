"""Process-wide registry of in-flight model downloads.

A single instance, ``DOWNLOAD_SERVER``, tracks every currently-running
server-side model fetch. Designed to be safe with multiple concurrent
clients hitting the API: each model_id has at most one active session,
and the API rejects requests that conflict with in-flight downloads.

Cancellation is cooperative — the download loop checks ``is_active`` on
its own session between chunks and raises ``DownloadCancelled`` when the
session has been removed from the registry. This avoids the complications
of ``Task.cancel()`` from outside the loop while still giving deterministic
rollback semantics (the worker is responsible for deleting its own
``.tmp`` on the cancel path).
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Optional

from app.model_downloader.paths import iter_all_tmp_paths


class DownloadCancelled(Exception):
    """Raised by the streaming loop when its session has been removed
    from the registry (cancellation request) and the worker should roll
    back its ``.tmp`` file."""


@dataclass
class DownloadSession:
    """One in-flight download.

    ``progress`` is a fraction in ``[0.0, 1.0]``; ``None`` until the first
    byte arrives and we know whether the response carries a
    ``Content-Length``. ``total_bytes`` mirrors that header when present.
    """
    model_id: str
    url: str
    progress: Optional[float] = None
    bytes_downloaded: int = 0
    total_bytes: Optional[int] = None
    # Sequence number used solely as identity for the cancellation check —
    # so that "cancel + restart" doesn't get confused by stale workers.
    epoch: int = field(default_factory=lambda: 0)


class DownloadServer:
    """Singleton registry of active downloads.

    All mutation goes through this object so concurrent route handlers
    see a consistent view. The ``_lock`` is a plain threading lock
    because the registry is consulted from both the asyncio event-loop
    thread (route handlers) and from any worker coroutines spawned to
    perform downloads.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, DownloadSession] = {}
        self._epoch_counter = 0
        self._orphan_sweep_done = False

    # ----- lifecycle -----

    def sweep_orphan_tmp_files(self) -> None:
        """Idempotently sweep ``*.tmp`` files left by crashed downloads.

        Deferred off the import path so module load doesn't block on
        filesystem I/O against potentially-slow mounts. Each route handler
        that might create a new ``.tmp`` runs this exactly once.
        """
        with self._lock:
            if self._orphan_sweep_done:
                return
            self._orphan_sweep_done = True
        for path in iter_all_tmp_paths():
            try:
                os.remove(path)
                logging.info("[model_downloader] removed orphan tmp file: %s", path)
            except OSError as e:
                logging.warning("[model_downloader] could not remove %s: %s", path, e)

    # ----- queries -----

    def is_downloading(self, model_id: str) -> bool:
        with self._lock:
            return model_id in self._sessions

    def get(self, model_id: str) -> Optional[DownloadSession]:
        with self._lock:
            return self._sessions.get(model_id)

    def snapshot(self) -> dict[str, DownloadSession]:
        """Return a shallow copy of the current sessions map."""
        with self._lock:
            return dict(self._sessions)

    # ----- mutations -----

    def try_register(self, model_id: str, url: str) -> Optional[DownloadSession]:
        """Atomically register a new session iff none exists for ``model_id``.

        Returns the new session on success, ``None`` if a session is already
        in flight. Callers must check the return value — the caller is the
        sole owner of the session it gets back.
        """
        with self._lock:
            if model_id in self._sessions:
                return None
            self._epoch_counter += 1
            session = DownloadSession(
                model_id=model_id,
                url=url,
                epoch=self._epoch_counter,
            )
            self._sessions[model_id] = session
            return session

    def update_progress(
        self,
        session: DownloadSession,
        bytes_downloaded: int,
        total_bytes: Optional[int],
    ) -> None:
        """Update progress on a session. No-op if the session has been
        removed (cancelled) — caller should check ``is_active`` separately."""
        with self._lock:
            current = self._sessions.get(session.model_id)
            if current is None or current.epoch != session.epoch:
                return
            current.bytes_downloaded = bytes_downloaded
            current.total_bytes = total_bytes
            if total_bytes and total_bytes > 0:
                current.progress = min(1.0, bytes_downloaded / total_bytes)

    def is_active(self, session: DownloadSession) -> bool:
        """True iff this exact session is still the registered one for
        its model_id. False after cancellation, after completion, or if
        another session has replaced it."""
        with self._lock:
            current = self._sessions.get(session.model_id)
            return current is not None and current.epoch == session.epoch

    def finish(self, session: DownloadSession) -> None:
        """Remove a completed (or cancelled) session from the registry.

        Safe to call multiple times. Only removes if the epoch matches —
        we never accidentally evict a *newer* session for the same model_id.
        """
        with self._lock:
            current = self._sessions.get(session.model_id)
            if current is not None and current.epoch == session.epoch:
                del self._sessions[session.model_id]

    def reset_for_tests(self) -> None:
        """Clear all sessions and reset the epoch counter. Test-only."""
        with self._lock:
            self._sessions.clear()
            self._epoch_counter = 0

    def cancel(self, model_id: str) -> bool:
        """Remove the session registered for ``model_id``.

        Returns True if there was an active session to cancel. The worker
        will discover the cancellation on its next ``is_active`` check
        and roll back its ``.tmp`` file.
        """
        with self._lock:
            if model_id in self._sessions:
                del self._sessions[model_id]
                return True
            return False


DOWNLOAD_SERVER = DownloadServer()
