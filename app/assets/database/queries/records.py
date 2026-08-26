from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetContent, AssetTag, Tag
from app.assets.helpers import get_utc_now


def create_content(session: Session, path: str, hash: str | None = None, size_bytes: int = 0, mtime_ns: int | None = None) -> AssetContent:
    content = AssetContent(path=path, hash=hash, size_bytes=size_bytes, mtime_ns=mtime_ns)
    try:
        with session.begin_nested():
            session.add(content)
            session.flush()
            return content
    except IntegrityError:
        winner = session.execute(sa.select(AssetContent).where(AssetContent.path == path, AssetContent.is_missing.is_(False))).scalar_one()
        return winner


def create_record(session: Session, content_id: str, name: str, mime_type: str | None = None, job_id: str | None = None, loader_path: str | None = None, tags: Sequence[str] | None = None) -> Asset:
    record = Asset(content_id=content_id, name=name, mime_type=mime_type, job_id=job_id, loader_path=loader_path)
    session.add(record)
    session.flush()
    for tag_name in tags or ():
        if session.get(Tag, tag_name) is None:
            session.add(Tag(name=tag_name))
            session.flush()
        session.add(AssetTag(asset_id=record.id, tag_name=tag_name))
    session.flush()
    return record


def get_record_by_id(session: Session, id: str) -> Asset | None:
    return session.get(Asset, id)


def get_preview_file_paths_by_ids(
    session: Session,
    preview_ids: Sequence[str],
) -> dict[str, str]:
    """Map live preview asset ids to their content paths in one statement."""
    if not preview_ids:
        return {}

    rows = session.execute(
        sa.select(Asset.id, AssetContent.path)
        .join(AssetContent, Asset.content_id == AssetContent.id)
        .where(
            Asset.id.in_(preview_ids),
            AssetContent.is_missing.is_(False),
        )
    )
    return {preview_id: path for preview_id, path in rows}


def get_record_by_path_or_none(session: Session, path: str) -> Asset | None:
    return session.scalar(
        sa.select(Asset)
        .join(AssetContent, Asset.content_id == AssetContent.id)
        .where(AssetContent.path == path, AssetContent.is_missing.is_(False))
        .order_by(Asset.created_at.desc(), Asset.id.desc())
        .limit(1)
    )


def fetch_record_tags(session: Session, record_id: str) -> list[str]:
    return list(
        session.scalars(
            sa.select(AssetTag.tag_name)
            .where(AssetTag.asset_id == record_id)
            .order_by(AssetTag.tag_name)
        )
    )


def update_record_access_time(
    session: Session,
    record_id: str,
    ts: datetime | None = None,
    only_if_newer: bool = True,
) -> None:
    ts = ts or get_utc_now()
    stmt = sa.update(Asset).where(Asset.id == record_id)
    if only_if_newer:
        stmt = stmt.where(
            sa.or_(
                Asset.last_access_time.is_(None),
                Asset.last_access_time < ts,
            )
        )
    session.execute(stmt.values(last_access_time=ts))


def list_records_page(session: Session, cursor: str | None = None, limit: int = 50, include_tags: Sequence[str] | None = None, exclude_tags: Sequence[str] | None = None) -> tuple[list[Asset], str | None]:
    statement = sa.select(Asset).order_by(Asset.created_at.asc(), Asset.id.asc()).limit(limit)
    if cursor is not None:
        statement = statement.where(Asset.id > cursor)
    for tag_name in include_tags or ():
        statement = statement.where(sa.exists(sa.select(AssetTag.asset_id).where(AssetTag.asset_id == Asset.id, AssetTag.tag_name == tag_name)))
    for tag_name in exclude_tags or ():
        statement = statement.where(~sa.exists(sa.select(AssetTag.asset_id).where(AssetTag.asset_id == Asset.id, AssetTag.tag_name == tag_name)))
    records = list(session.execute(statement).scalars())
    return records, records[-1].id if len(records) == limit else None


def rename_record(session: Session, id: str, name: str) -> Asset:
    record = session.get(Asset, id)
    if record is None:
        raise LookupError(id)
    record.name = name
    record.updated_at = get_utc_now()
    session.flush()
    return record


def delete_record(session: Session, id: str) -> None:
    record = session.get(Asset, id)
    if record is None:
        return
    preview_id = record.preview_id
    session.delete(record)
    session.flush()
    if preview_id is not None and session.scalar(sa.select(sa.func.count()).select_from(Asset).where(Asset.preview_id == preview_id)) == 0:
        preview = session.get(Asset, preview_id)
        if preview is not None:
            session.delete(preview)
    session.flush()


def mark_content_missing(session: Session, content_id: str) -> None:
    content = session.get(AssetContent, content_id)
    if content is None:
        raise LookupError(content_id)
    content.is_missing = True
    if session.get(Tag, "missing") is None:
        session.add(Tag(name="missing"))
        session.flush()
    for record_id in session.scalars(sa.select(Asset.id).where(Asset.content_id == content_id)):
        if session.get(AssetTag, {"asset_id": record_id, "tag_name": "missing"}) is None:
            session.add(AssetTag(asset_id=record_id, tag_name="missing", origin="automatic"))
    session.flush()


def unset_content_missing(session: Session, content_id: str) -> None:
    content = session.get(AssetContent, content_id)
    if content is None:
        raise LookupError(content_id)
    content.is_missing = False
    session.execute(sa.delete(AssetTag).where(AssetTag.tag_name == "missing", AssetTag.asset_id.in_(sa.select(Asset.id).where(Asset.content_id == content_id))))
    session.flush()
