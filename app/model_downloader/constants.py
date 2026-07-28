"""Shared constants for the download manager.

Status values are persisted as TEXT in the ``downloads`` table; keep them
stable. The lifecycle is:

    queued -> active -> verifying -> completed
       |        |-> paused -> (resume) -> active
       |        |-> failed (network, retryable) -> queued (backoff)
       |-> cancelled
"""

from __future__ import annotations


class DownloadStatus:
    QUEUED = "queued"
    ACTIVE = "active"
    PAUSED = "paused"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    #: States from which a worker is doing (or about to do) network I/O.
    LIVE = (QUEUED, ACTIVE, VERIFYING)
    #: Terminal states — the job will not transition again on its own.
    TERMINAL = (COMPLETED, FAILED, CANCELLED)


# Default temp-file suffix. Distinctive so the startup orphan sweep only
# removes files THIS subsystem created, never unrelated *.tmp files.
TMP_SUFFIX = ".comfy-download.part"
