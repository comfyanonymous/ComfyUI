from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import folder_paths
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.assets import mode
from app.assets.database.models import AssetContent


def is_temp_path(path: str) -> bool:
    try:
        temp_root = Path(os.path.abspath(folder_paths.get_temp_directory()))
        candidate = Path(os.path.abspath(path))
        return candidate.is_relative_to(temp_root)
    except OSError:
        return False


def _stat_consistent(content: AssetContent) -> bool:
    try:
        stat = os.stat(content.path)
    except FileNotFoundError:
        return False
    if content.mtime_ns is not None and stat.st_mtime_ns != content.mtime_ns:
        return False
    if content.mtime_ns is not None and stat.st_size != content.size_bytes:
        return False
    return True


def _qualifies(content: AssetContent) -> bool:
    return (
        not content.is_missing
        and os.path.isfile(content.path)
        and _stat_consistent(content)
        and not is_temp_path(content.path)
    )


def qualified_content_iterator(session: Session, hash: str) -> Iterator[AssetContent]:
    rows = session.scalars(
        select(AssetContent)
        .where(AssetContent.hash == hash, AssetContent.is_missing.is_(False))
        .order_by(AssetContent.created_at, AssetContent.id)
    )
    for row in rows:
        if _qualifies(row):
            yield row


def claim_qualified_content(session: Session, content_id: str, hash: str) -> bool:
    """Atomically re-affirm that a content row is still live and still holds ``hash``.

    A writer that selected a row through ``qualified_content_iterator`` and is
    about to attach a record to it calls this, inside the same transaction,
    immediately before creating the record. It is a conditional UPDATE, not a
    SELECT, and that difference is the whole point: SQLite is single-writer,
    and pysqlite only opens an implicit transaction (taking the write lock)
    before a DML statement, never before a SELECT. A read-then-write
    revalidation leaves a real window - between the read and this session's
    own commit - where another connection can commit a conflicting change
    (retire the row, or correct its hash, e.g. ``detect_content_change``) and
    this session would never see it. Making the revalidation itself the first
    write closes that window: once it executes, this connection holds SQLite's
    write lock continuously through to its own commit.

    That lock is database-file-wide, not scoped to this row: this app's default
    rollback-journal SQLite locking has no row-level granularity, so from the
    moment this statement runs until this session commits or rolls back, no
    OTHER connection can commit ANY write anywhere in the database - a
    different content row, a different table, an unrelated upload entirely -
    not just a conflicting write to this row. That breadth is exactly what
    makes the guarantee hold, not an accident to narrow down: it briefly
    serializes every SQLite writer in the app for the length of this critical
    section (claim, the filesystem re-check, record creation, commit), which is
    an accepted tradeoff because that section is short. Do not read this as
    "safe to rely on row-scoped locking elsewhere" - there is no such thing
    here.

    The WHERE clause is the entire correctness contract: ``hash`` catches a row
    whose recorded content changed identity since the caller's lookup, and
    ``is_missing`` catches a row retired since then. ``rowcount`` reports
    whether both were still true at the instant of the write. The value
    written back is a no-op (``is_missing`` set to the value it must already
    have to match) - the claim exists to take and hold the lock and to prove
    the row's state, not to change anything.
    """
    # `session.connection()` is the ORM session's own connection for its current
    # transaction (autobegin-ing it if needed, and shared with every ORM call
    # this function's caller makes afterward, through to that same session's
    # own commit) - issuing the claim as a plain Core execute on it, rather than
    # through `Session.execute()`, sidesteps the ORM-enabled-statement overloads
    # that type this as the base `Result` (no `rowcount`); a Core connection
    # always returns the actual runtime type, `CursorResult`, so no cast is
    # needed to reach the attribute SQLAlchemy's own docs use for exactly this
    # check.
    result = session.connection().execute(
        update(AssetContent)
        .where(
            AssetContent.id == content_id,
            AssetContent.hash == hash,
            AssetContent.is_missing.is_(False),
        )
        .values(is_missing=False)
    )
    return result.rowcount == 1


def refresh_qualified_content(session: Session, content_id: str) -> AssetContent | None:
    """Re-read a claimed content row and apply the filesystem-level checks.

    ``claim_qualified_content`` proves the DB-level facts (liveness, hash)
    atomically; a database transaction cannot make the filesystem atomic
    against an external process touching the file, so this reruns the same
    ``_qualifies`` predicate ``qualified_content_iterator`` uses, against a
    forced re-read (``populate_existing``) so a stale identity-map copy from
    the caller's earlier lookup is never what gets checked.
    """
    content = session.get(AssetContent, content_id, populate_existing=True)
    if content is None or not _qualifies(content):
        return None
    return content


def lookup_for_from_hash(session: Session, hash: str) -> AssetContent | None:
    if not mode.hashing_enabled():
        return None
    return next(qualified_content_iterator(session, hash), None)


def lookup_for_view(session: Session, hash: str) -> AssetContent | None:
    return next(qualified_content_iterator(session, hash), None)
