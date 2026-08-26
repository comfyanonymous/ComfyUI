from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import folder_paths
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.assets import mode
from app.assets.database.models import Asset, AssetContent


def is_temp_path(path: str) -> bool:
    try:
        temp_root = Path(os.path.abspath(folder_paths.get_temp_directory()))
        candidate = Path(os.path.abspath(path))
        return candidate.is_relative_to(temp_root)
    except OSError:
        return False


def _stat_consistent(content: AssetContent) -> bool:
    """Check file exists and stored stat values are consistent.

    mtime_ns=None means "not yet measured" — always consistent.
    size_bytes=0 with mtime_ns=None means "stub row" — always consistent.
    """
    try:
        stat = os.stat(content.path)
    except FileNotFoundError:
        return False
    if content.mtime_ns is not None and stat.st_mtime_ns != content.mtime_ns:
        return False
    # Only check size when mtime is also stored (fully enriched row)
    if content.mtime_ns is not None and stat.st_size != content.size_bytes:
        return False
    return True


def qualified_content_iterator(session: Session, hash: str) -> Iterator[AssetContent]:
    rows = session.scalars(
        select(AssetContent)
        .where(AssetContent.hash == hash, AssetContent.is_missing.is_(False))
        .order_by(AssetContent.created_at, AssetContent.id)
    )
    for row in rows:
        if os.path.isfile(row.path) and _stat_consistent(row) and not is_temp_path(row.path):
            yield row


def lookup_for_from_hash(session: Session, hash: str) -> AssetContent | None:
    if not mode.hashing_enabled():
        return None
    return next(qualified_content_iterator(session, hash), None)


def lookup_for_upload_dedup(
    session: Session, hash: str, name: str
) -> Asset | AssetContent | None:
    first_content = None
    for content in qualified_content_iterator(session, hash):
        if first_content is None:
            first_content = content
        match = session.scalars(
            select(Asset).where(Asset.content_id == content.id, Asset.name == name)
        ).first()
        if match is not None:
            return match
    return first_content


def lookup_for_view(session: Session, hash: str) -> AssetContent | None:
    return next(qualified_content_iterator(session, hash), None)
