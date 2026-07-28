"""Synchronous DB access for the download manager.

All functions open their own short-lived session via ``create_session`` and
commit before returning, mirroring ``app/assets`` usage. They are blocking
(SQLite) and should be called from async code through ``asyncio.to_thread``.
"""

from __future__ import annotations

import time
from typing import Optional

from sqlalchemy import delete, select

from app.database.db import create_session
from app.model_downloader.constants import DownloadStatus
from app.model_downloader.database.models import Download, DownloadSegment


# ----- downloads -----


def insert_download(values: dict) -> None:
    with create_session() as session:
        session.add(Download(**values))
        session.commit()


def get_download(download_id: str) -> Optional[Download]:
    with create_session() as session:
        row = session.get(Download, download_id)
        if row is not None:
            session.expunge_all()
        return row


def list_downloads() -> list[Download]:
    with create_session() as session:
        rows = list(
            session.execute(
                select(Download).order_by(Download.created_at.desc())
            ).scalars()
        )
        session.expunge_all()
        return rows


def has_live_download_for_model(
    model_id: str, live_statuses: tuple[str, ...], exclude_id: Optional[str] = None
) -> bool:
    with create_session() as session:
        stmt = select(Download.id).where(
            Download.model_id == model_id,
            Download.status.in_(live_statuses),
        ).limit(1)
        if exclude_id is not None:
            stmt = stmt.where(Download.id != exclude_id)
        return session.execute(stmt).first() is not None


def list_segments(download_id: str) -> list[DownloadSegment]:
    with create_session() as session:
        rows = list(
            session.execute(
                select(DownloadSegment)
                .where(DownloadSegment.download_id == download_id)
                .order_by(DownloadSegment.idx)
            ).scalars()
        )
        session.expunge_all()
        return rows


def update_download(download_id: str, **fields) -> None:
    if not fields:
        return
    fields.setdefault("updated_at", int(time.time()))
    with create_session() as session:
        row = session.get(Download, download_id)
        if row is None:
            return
        for key, value in fields.items():
            setattr(row, key, value)
        session.commit()


def delete_download(download_id: str) -> None:
    with create_session() as session:
        row = session.get(Download, download_id)
        if row is not None:
            session.delete(row)
            session.commit()


def delete_downloads(download_ids: list[str]) -> int:
    """Delete many downloads in one transaction; returns the number removed.

    Uses a bulk ``DELETE ... WHERE id IN (...)``. Segment rows are removed by
    the ``ON DELETE CASCADE`` foreign key (SQLite ``PRAGMA foreign_keys=ON`` is
    set in ``app/database/db.py``), so this stays consistent without loading the
    ORM relationship.
    """
    if not download_ids:
        return 0
    with create_session() as session:
        result = session.execute(
            delete(Download).where(Download.id.in_(download_ids))
        )
        session.commit()
        return result.rowcount or 0


def replace_segments(download_id: str, segments: list[dict]) -> None:
    """Atomically replace the segment plan for a download."""
    with create_session() as session:
        session.query(DownloadSegment).filter(
            DownloadSegment.download_id == download_id
        ).delete()
        for seg in segments:
            session.add(DownloadSegment(download_id=download_id, **seg))
        session.commit()


def update_segment_progress(download_id: str, idx: int, bytes_done: int) -> None:
    with create_session() as session:
        row = session.get(DownloadSegment, {"download_id": download_id, "idx": idx})
        if row is None:
            return
        row.bytes_done = bytes_done
        session.commit()


def list_queued_downloads() -> list[Download]:
    """Queued rows ordered for admission (priority desc, then FIFO)."""
    with create_session() as session:
        rows = list(
            session.execute(
                select(Download)
                .where(Download.status == DownloadStatus.QUEUED)
                .order_by(Download.priority.desc(), Download.created_at.asc())
            ).scalars()
        )
        session.expunge_all()
        return rows


def reconcile_live_downloads() -> list[Download]:
    """Reset any ``active``/``verifying`` rows left by a previous run.

    On a clean restart there can be no live worker, so anything still marked
    live is stale. Move it back to ``queued`` (offsets are preserved on the
    segment rows) so the scheduler re-admits it. Returns the rows that should
    be re-queued by the scheduler (queued + paused).
    """
    with create_session() as session:
        stale = list(
            session.execute(
                select(Download).where(
                    Download.status.in_([DownloadStatus.ACTIVE, DownloadStatus.VERIFYING])
                )
            ).scalars()
        )
        now = int(time.time())
        for row in stale:
            row.status = DownloadStatus.QUEUED
            row.updated_at = now
        session.commit()

        resumable = list(
            session.execute(
                select(Download)
                .where(Download.status == DownloadStatus.QUEUED)
                .order_by(Download.priority.desc(), Download.created_at.asc())
            ).scalars()
        )
        session.expunge_all()
        return resumable
