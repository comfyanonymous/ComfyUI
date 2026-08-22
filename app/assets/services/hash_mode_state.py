"""Hash-mode persistence and OFF→ON transition logic."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.assets import mode as _mode
from app.assets.database.models import AssetContent, AssetSystemState
from app.assets.database.queries.records import mark_content_missing
from app.assets.services.snapshot_hash import snapshot_hash

_KEY = "hash_mode"
_PENDING_QUEUE: list[str] = []


def clear_transition_queue() -> None:
    _PENDING_QUEUE.clear()


def pending_transition_count() -> int:
    return len(_PENDING_QUEUE)


def read_stored_mode(session: Session) -> str | None:
    row = session.get(AssetSystemState, _KEY)
    return row.value if row else None


def write_stored_mode(session: Session, value: str) -> None:
    row = session.get(AssetSystemState, _KEY)
    if row is None:
        session.add(AssetSystemState(key=_KEY, value=value))
    else:
        row.value = value
    session.flush()


def record_transition_intent(session: Session) -> str | None:
    stored = read_stored_mode(session)
    runtime = "on" if _mode.hashing_enabled() else "off"
    if stored is None:
        write_stored_mode(session, runtime)
        return None
    if stored == "off" and runtime == "on":
        return "off_to_on"
    if stored == "on" and runtime == "off":
        write_stored_mode(session, "off")
        return "on_to_off"
    return None


def enqueue_transition_work(session: Session, transition: str | None) -> None:
    if transition != "off_to_on":
        return
    rows = session.scalars(
        select(AssetContent).where(AssetContent.is_missing.is_(False))
    )
    for row in rows:
        if row.path not in _PENDING_QUEUE:
            _PENDING_QUEUE.append(row.path)


def drain_transition_queue(session: Session) -> None:
    pending_count = len(_PENDING_QUEUE)
    for _ in range(pending_count):
        path = _PENDING_QUEUE.pop(0)
        digest = snapshot_hash(path)
        if digest is None:
            _PENDING_QUEUE.append(path)
            continue
        content = session.scalars(
            select(AssetContent).where(
                AssetContent.path == path, AssetContent.is_missing.is_(False)
            )
        ).first()
        if content is None:
            continue
        current_hash = digest
        if content.hash is None:
            content.hash = current_hash
        elif content.hash != current_hash:
            mark_content_missing(session, content.id)
            replacement = AssetContent(
                path=path,
                hash=current_hash,
                size_bytes=content.size_bytes,
                mtime_ns=content.mtime_ns,
            )
            session.add(replacement)
    if not _PENDING_QUEUE:
        write_stored_mode(session, "on")
