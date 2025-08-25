import contextlib
import os
import logging
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from typing import Any, Sequence, Optional, Iterable

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, exists, func
from sqlalchemy.orm import contains_eager
from sqlalchemy.exc import IntegrityError

from .models import Asset, AssetInfo, AssetInfoTag, AssetLocatorState, Tag, AssetInfoMeta
from .timeutil import utcnow
from .._assets_helpers import normalize_tags


async def asset_exists_by_hash(session: AsyncSession, *, asset_hash: str) -> bool:
    row = (
        await session.execute(
            select(sa.literal(True)).select_from(Asset).where(Asset.hash == asset_hash).limit(1)
        )
    ).first()
    return row is not None


async def get_asset_by_hash(session: AsyncSession, *, asset_hash: str) -> Optional[Asset]:
    return await session.get(Asset, asset_hash)


async def check_fs_asset_exists_quick(
    session,
    *,
    file_path: str,
    size_bytes: Optional[int] = None,
    mtime_ns: Optional[int] = None,
) -> bool:
    """
    Returns 'True' if there is already an Asset present whose canonical locator matches this absolute path,
    AND (if provided) mtime_ns matches stored locator-state,
    AND (if provided) size_bytes matches verified size when known.
    """
    locator = os.path.abspath(file_path)

    stmt = select(sa.literal(True)).select_from(Asset)

    conditions = [
        Asset.storage_backend == "fs",
        Asset.storage_locator == locator,
    ]

    # If size_bytes provided require equality when the asset has a verified (non-zero) size.
    # If verified size is 0 (unknown), we don't force equality.
    if size_bytes is not None:
        conditions.append(sa.or_(Asset.size_bytes == 0, Asset.size_bytes == int(size_bytes)))

    # If mtime_ns provided require the locator-state to exist and match.
    if mtime_ns is not None:
        stmt = stmt.join(AssetLocatorState, AssetLocatorState.asset_hash == Asset.hash)
        conditions.append(AssetLocatorState.mtime_ns == int(mtime_ns))

    stmt = stmt.where(*conditions).limit(1)

    row = (await session.execute(stmt)).first()
    return row is not None


async def ingest_fs_asset(
    session: AsyncSession,
    *,
    asset_hash: str,
    abs_path: str,
    size_bytes: int,
    mtime_ns: int,
    mime_type: Optional[str] = None,
    info_name: Optional[str] = None,
    owner_id: str = "",
    preview_hash: Optional[str] = None,
    user_metadata: Optional[dict] = None,
    tags: Sequence[str] = (),
    tag_origin: str = "manual",
    added_by: Optional[str] = None,
    require_existing_tags: bool = False,
) -> dict:
    """
    Creates or updates Asset record for a local (fs) asset.

    Always:
      - Insert Asset if missing; else update size_bytes (and updated_at) if different.
      - Insert AssetLocatorState if missing; else update mtime_ns if different.

    Optionally (when info_name is provided):
      - Create an AssetInfo (no refcount changes).
      - Link provided tags to that AssetInfo.
        * If the require_existing_tags=True, raises ValueError if any tag does not exist in `tags` table.
        * If False (default), create unknown tags.

    Returns flags and ids:
      {
        "asset_created": bool,
        "asset_updated": bool,
        "state_created": bool,
        "state_updated": bool,
        "asset_info_id": int | None,
      }
    """
    locator = os.path.abspath(abs_path)
    datetime_now = utcnow()

    out = {
        "asset_created": False,
        "asset_updated": False,
        "state_created": False,
        "state_updated": False,
        "asset_info_id": None,
    }

    # ---- Step 1: INSERT Asset or UPDATE size_bytes/updated_at if exists ----
    with contextlib.suppress(IntegrityError):
        async with session.begin_nested():
            session.add(
                Asset(
                    hash=asset_hash,
                    size_bytes=int(size_bytes),
                    mime_type=mime_type,
                    refcount=0,
                    storage_backend="fs",
                    storage_locator=locator,
                    created_at=datetime_now,
                    updated_at=datetime_now,
                )
            )
            await session.flush()
            out["asset_created"] = True

    if not out["asset_created"]:
        existing = await session.get(Asset, asset_hash)
        if existing is not None:
            changed = False
            if existing.size_bytes != size_bytes:
                existing.size_bytes = size_bytes
                changed = True
            if mime_type and existing.mime_type != mime_type:
                existing.mime_type = mime_type
                changed = True
            if existing.storage_locator != locator:
                existing.storage_locator = locator
                changed = True
            if changed:
                existing.updated_at = datetime_now
                out["asset_updated"] = True
        else:
            logging.error("Asset %s not found after PK conflict; skipping update.", asset_hash)

    # ---- Step 2: INSERT/UPDATE AssetLocatorState (mtime_ns) ----
    with contextlib.suppress(IntegrityError):
        async with session.begin_nested():
            session.add(
                AssetLocatorState(
                    asset_hash=asset_hash,
                    mtime_ns=int(mtime_ns),
                )
            )
            await session.flush()
            out["state_created"] = True

    if not out["state_created"]:
        state = await session.get(AssetLocatorState, asset_hash)
        if state is not None:
            desired_mtime = int(mtime_ns)
            if state.mtime_ns != desired_mtime:
                state.mtime_ns = desired_mtime
                out["state_updated"] = True
        else:
            logging.error("Locator state missing for %s after conflict; skipping update.", asset_hash)

    # ---- Optional: AssetInfo + tag links ----
    if info_name:
        # 2a) Upsert AssetInfo idempotently on (asset_hash, owner_id, name)
        with contextlib.suppress(IntegrityError):
            async with session.begin_nested():
                info = AssetInfo(
                    owner_id=owner_id,
                    name=info_name,
                    asset_hash=asset_hash,
                    preview_hash=preview_hash,
                    created_at=datetime_now,
                    updated_at=datetime_now,
                    last_access_time=datetime_now,
                )
                session.add(info)
                await session.flush()  # get info.id
                out["asset_info_id"] = info.id

        existing_info = (
            await session.execute(
                select(AssetInfo)
                .where(
                    AssetInfo.asset_hash == asset_hash,
                    AssetInfo.name == info_name,
                    (AssetInfo.owner_id == owner_id),
                )
                .limit(1)
            )
        ).unique().scalar_one_or_none()
        if not existing_info:
            raise RuntimeError("Failed to update or insert AssetInfo.")

        if preview_hash is not None and existing_info.preview_hash != preview_hash:
            existing_info.preview_hash = preview_hash
        existing_info.updated_at = datetime_now
        if existing_info.last_access_time < datetime_now:
            existing_info.last_access_time = datetime_now
        await session.flush()
        out["asset_info_id"] = existing_info.id

        # 2b) Link tags (if any). We DO NOT create new Tag rows here by default.
        norm = [t.strip().lower() for t in (tags or []) if (t or "").strip()]
        if norm and out["asset_info_id"] is not None:
            if not require_existing_tags:
                await _ensure_tags_exist(session, norm, tag_type="user")

            # Which tags exist?
            existing_tag_names = set(
                name for (name,) in (await session.execute(select(Tag.name).where(Tag.name.in_(norm)))).all()
            )
            missing = [t for t in norm if t not in existing_tag_names]
            if missing and require_existing_tags:
                raise ValueError(f"Unknown tags: {missing}")

            # Which links already exist?
            existing_links = set(
                tag_name
                for (tag_name,) in (
                    await session.execute(
                        select(AssetInfoTag.tag_name).where(AssetInfoTag.asset_info_id == out["asset_info_id"])
                    )
                ).all()
            )
            to_add = [t for t in norm if t in existing_tag_names and t not in existing_links]
            if to_add:
                session.add_all(
                    [
                        AssetInfoTag(
                            asset_info_id=out["asset_info_id"],
                            tag_name=t,
                            origin=tag_origin,
                            added_by=added_by,
                            added_at=datetime_now,
                        )
                        for t in to_add
                    ]
                )
                await session.flush()

        # 2c) Rebuild metadata projection if provided
        if user_metadata is not None and out["asset_info_id"] is not None:
            await replace_asset_info_metadata_projection(
                session,
                asset_info_id=out["asset_info_id"],
                user_metadata=user_metadata,
            )
    return out


async def touch_asset_infos_by_fs_path(
    session: AsyncSession,
    *,
    abs_path: str,
    ts: Optional[datetime] = None,
    only_if_newer: bool = True,
) -> int:
    locator = os.path.abspath(abs_path)
    ts = ts or utcnow()

    stmt = sa.update(AssetInfo).where(
        sa.exists(
            sa.select(sa.literal(1))
            .select_from(Asset)
            .where(
                Asset.hash == AssetInfo.asset_hash,
                Asset.storage_backend == "fs",
                Asset.storage_locator == locator,
            )
        )
    )

    if only_if_newer:
        stmt = stmt.where(
            sa.or_(
                AssetInfo.last_access_time.is_(None),
                AssetInfo.last_access_time < ts,
            )
        )

    stmt = stmt.values(last_access_time=ts)

    res = await session.execute(stmt)
    return int(res.rowcount or 0)


async def touch_asset_info_by_id(
    session: AsyncSession,
    *,
    asset_info_id: int,
    ts: Optional[datetime] = None,
    only_if_newer: bool = True,
) -> int:
    ts = ts or utcnow()
    stmt = sa.update(AssetInfo).where(AssetInfo.id == asset_info_id)
    if only_if_newer:
        stmt = stmt.where(
            sa.or_(AssetInfo.last_access_time.is_(None), AssetInfo.last_access_time < ts)
        )
    stmt = stmt.values(last_access_time=ts)
    res = await session.execute(stmt)
    return int(res.rowcount or 0)


async def list_asset_infos_page(
    session: AsyncSession,
    *,
    include_tags: Optional[Sequence[str]] = None,
    exclude_tags: Optional[Sequence[str]] = None,
    name_contains: Optional[str] = None,
    metadata_filter: Optional[dict] = None,
    limit: int = 20,
    offset: int = 0,
    sort: str = "created_at",
    order: str = "desc",
) -> tuple[list[AssetInfo], dict[int, list[str]], int]:
    """
    Returns a page of AssetInfo rows with their Asset eagerly loaded (no N+1),
    plus a map of asset_info_id -> [tags], and the total count.

    We purposely collect tags in a separate (single) query to avoid row explosion.
    """
    # Clamp
    if limit <= 0:
        limit = 1
    if limit > 100:
        limit = 100
    if offset < 0:
        offset = 0

    # Build base query
    base = (
        select(AssetInfo)
        .join(Asset, Asset.hash == AssetInfo.asset_hash)
        .options(contains_eager(AssetInfo.asset))
    )

    # Filters
    if name_contains:
        base = base.where(AssetInfo.name.ilike(f"%{name_contains}%"))

    base = _apply_tag_filters(base, include_tags, exclude_tags)

    base = _apply_metadata_filter(base, metadata_filter)

    # Sort
    sort = (sort or "created_at").lower()
    order = (order or "desc").lower()
    sort_map = {
        "name": AssetInfo.name,
        "created_at": AssetInfo.created_at,
        "updated_at": AssetInfo.updated_at,
        "last_access_time": AssetInfo.last_access_time,
        "size": Asset.size_bytes,
    }
    sort_col = sort_map.get(sort, AssetInfo.created_at)
    sort_exp = sort_col.desc() if order == "desc" else sort_col.asc()

    base = base.order_by(sort_exp).limit(limit).offset(offset)

    # Total count (same filters, no ordering/limit/offset)
    count_stmt = (
        select(func.count())
        .select_from(AssetInfo)
        .join(Asset, Asset.hash == AssetInfo.asset_hash)
    )
    if name_contains:
        count_stmt = count_stmt.where(AssetInfo.name.ilike(f"%{name_contains}%"))
    count_stmt = _apply_tag_filters(count_stmt, include_tags, exclude_tags)
    count_stmt = _apply_metadata_filter(count_stmt, metadata_filter)

    total = (await session.execute(count_stmt)).scalar_one()

    # Fetch rows
    infos = (await session.execute(base)).scalars().unique().all()

    # Collect tags in bulk (single query)
    id_list = [i.id for i in infos]
    tag_map: dict[int, list[str]] = defaultdict(list)
    if id_list:
        rows = await session.execute(
            select(AssetInfoTag.asset_info_id, Tag.name)
            .join(Tag, Tag.name == AssetInfoTag.tag_name)
            .where(AssetInfoTag.asset_info_id.in_(id_list))
        )
        for aid, tag_name in rows.all():
            tag_map[aid].append(tag_name)

    return infos, tag_map, total


async def fetch_asset_info_and_asset(session: AsyncSession, *, asset_info_id: int) -> Optional[tuple[AssetInfo, Asset]]:
    row = await session.execute(
        select(AssetInfo, Asset)
        .join(Asset, Asset.hash == AssetInfo.asset_hash)
        .where(AssetInfo.id == asset_info_id)
        .limit(1)
    )
    pair = row.first()
    if not pair:
        return None
    return pair[0], pair[1]


async def create_asset_info_for_existing_asset(
    session: AsyncSession,
    *,
    asset_hash: str,
    name: str,
    user_metadata: Optional[dict] = None,
    tags: Optional[Sequence[str]] = None,
    tag_origin: str = "manual",
    added_by: Optional[str] = None,
) -> AssetInfo:
    """Create a new AssetInfo referencing an existing Asset (no content write)."""
    now = utcnow()
    info = AssetInfo(
        owner_id="",
        name=name,
        asset_hash=asset_hash,
        preview_hash=None,
        created_at=now,
        updated_at=now,
        last_access_time=now,
    )
    session.add(info)
    await session.flush()  # get info.id

    if user_metadata is not None:
        await replace_asset_info_metadata_projection(
            session, asset_info_id=info.id, user_metadata=user_metadata
        )

    if tags is not None:
        await set_asset_info_tags(
            session,
            asset_info_id=info.id,
            tags=tags,
            origin=tag_origin,
            added_by=added_by,
        )
    return info


async def set_asset_info_tags(
    session: AsyncSession,
    *,
    asset_info_id: int,
    tags: Sequence[str],
    origin: str = "manual",
    added_by: Optional[str] = None,
) -> dict:
    """
    Replace the tag set on an AssetInfo with `tags`. Idempotent.
    Creates missing tag names as 'user'.
    """
    desired = normalize_tags(tags)

    # current links
    current = set(
        tag_name for (tag_name,) in (
            await session.execute(select(AssetInfoTag.tag_name).where(AssetInfoTag.asset_info_id == asset_info_id))
        ).all()
    )

    to_add = [t for t in desired if t not in current]
    to_remove = [t for t in current if t not in desired]

    if to_add:
        await _ensure_tags_exist(session, to_add, tag_type="user")
        session.add_all([
            AssetInfoTag(asset_info_id=asset_info_id, tag_name=t, origin=origin, added_by=added_by, added_at=utcnow())
            for t in to_add
        ])
        await session.flush()

    if to_remove:
        await session.execute(
            delete(AssetInfoTag)
            .where(AssetInfoTag.asset_info_id == asset_info_id, AssetInfoTag.tag_name.in_(to_remove))
        )
        await session.flush()

    return {"added": to_add, "removed": to_remove, "total": desired}


async def update_asset_info_full(
    session: AsyncSession,
    *,
    asset_info_id: int,
    name: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    user_metadata: Optional[dict] = None,
    tag_origin: str = "manual",
    added_by: Optional[str] = None,
) -> AssetInfo:
    """
    Update AssetInfo fields:
      - name (if provided)
      - user_metadata blob + rebuild projection (if provided)
      - replace tags with provided set (if provided)
    Returns the updated AssetInfo.
    """
    info = await session.get(AssetInfo, asset_info_id)
    if not info:
        raise ValueError(f"AssetInfo {asset_info_id} not found")

    touched = False
    if name is not None and name != info.name:
        info.name = name
        touched = True

    if user_metadata is not None:
        await replace_asset_info_metadata_projection(
            session, asset_info_id=asset_info_id, user_metadata=user_metadata
        )
        touched = True

    if tags is not None:
        await set_asset_info_tags(
            session,
            asset_info_id=asset_info_id,
            tags=tags,
            origin=tag_origin,
            added_by=added_by,
        )
        touched = True

    if touched and user_metadata is None:
        info.updated_at = utcnow()
        await session.flush()

    return info


async def delete_asset_info_by_id(session: AsyncSession, *, asset_info_id: int) -> bool:
    """Delete the user-visible AssetInfo row. Cascades clear tags and metadata."""
    res = await session.execute(delete(AssetInfo).where(AssetInfo.id == asset_info_id))
    return bool(res.rowcount)


async def replace_asset_info_metadata_projection(
    session: AsyncSession,
    *,
    asset_info_id: int,
    user_metadata: Optional[dict],
) -> None:
    """Replaces the `assets_info.user_metadata` AND rebuild the projection rows in `asset_info_meta`."""
    info = await session.get(AssetInfo, asset_info_id)
    if not info:
        raise ValueError(f"AssetInfo {asset_info_id} not found")

    info.user_metadata = user_metadata or {}
    info.updated_at = utcnow()
    await session.flush()

    await session.execute(delete(AssetInfoMeta).where(AssetInfoMeta.asset_info_id == asset_info_id))
    await session.flush()

    if not user_metadata:
        return

    rows: list[AssetInfoMeta] = []
    for k, v in user_metadata.items():
        for r in _project_kv(k, v):
            rows.append(
                AssetInfoMeta(
                    asset_info_id=asset_info_id,
                    key=r["key"],
                    ordinal=int(r["ordinal"]),
                    val_str=r.get("val_str"),
                    val_num=r.get("val_num"),
                    val_bool=r.get("val_bool"),
                    val_json=r.get("val_json"),
                )
            )
    if rows:
        session.add_all(rows)
        await session.flush()


async def get_asset_tags(session: AsyncSession, *, asset_info_id: int) -> list[str]:
    return [
            tag_name
            for (tag_name,) in (
                await session.execute(
                    sa.select(AssetInfoTag.tag_name).where(AssetInfoTag.asset_info_id == asset_info_id)
                )
            ).all()
        ]


async def list_tags_with_usage(
    session,
    *,
    prefix: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    include_zero: bool = True,
    order: str = "count_desc",        # "count_desc" | "name_asc"
) -> tuple[list[tuple[str, str, int]], int]:
    """
    Returns:
      rows: list of (name, tag_type, count)
      total: number of tags matching filter (independent of pagination)
    """
    # Subquery with counts by tag_name
    counts_sq = (
        select(
            AssetInfoTag.tag_name.label("tag_name"),
            func.count(AssetInfoTag.asset_info_id).label("cnt"),
        )
        .group_by(AssetInfoTag.tag_name)
        .subquery()
    )

    # Base select with LEFT JOIN so we can include zero-usage tags
    q = (
        select(
            Tag.name,
            Tag.tag_type,
            func.coalesce(counts_sq.c.cnt, 0).label("count"),
        )
        .select_from(Tag)
        .join(counts_sq, counts_sq.c.tag_name == Tag.name, isouter=True)
    )

    # Prefix filter (tags are lowercase by check constraint)
    if prefix:
        q = q.where(Tag.name.like(prefix.strip().lower() + "%"))

    # Include_zero toggles: if False, drop zero-usage tags
    if not include_zero:
        q = q.where(func.coalesce(counts_sq.c.cnt, 0) > 0)

    # Ordering
    if order == "name_asc":
        q = q.order_by(Tag.name.asc())
    else:  # default "count_desc"
        q = q.order_by(func.coalesce(counts_sq.c.cnt, 0).desc(), Tag.name.asc())

    # Total (without limit/offset, same filters)
    total_q = select(func.count()).select_from(Tag)
    if prefix:
        total_q = total_q.where(Tag.name.like(prefix.strip().lower() + "%"))
    if not include_zero:
        # count only names that appear in counts subquery
        total_q = total_q.where(
            Tag.name.in_(select(AssetInfoTag.tag_name).group_by(AssetInfoTag.tag_name))
        )

    rows = (await session.execute(q.limit(limit).offset(offset))).all()
    total = (await session.execute(total_q)).scalar_one()

    # Normalize counts to int for SQLite/Postgres consistency
    rows_norm = [(name, ttype, int(count or 0)) for (name, ttype, count) in rows]
    return rows_norm, int(total or 0)


async def add_tags_to_asset_info(
    session: AsyncSession,
    *,
    asset_info_id: int,
    tags: Sequence[str],
    origin: str = "manual",
    added_by: Optional[str] = None,
    create_if_missing: bool = True,
) -> dict:
    """Adds tags to an AssetInfo.
    If create_if_missing=True, missing tag rows are created as 'user'.
    Returns: {"added": [...], "already_present": [...], "total_tags": [...]}
    """
    info = await session.get(AssetInfo, asset_info_id)
    if not info:
        raise ValueError(f"AssetInfo {asset_info_id} not found")

    norm = normalize_tags(tags)
    if not norm:
        total = await get_asset_tags(session, asset_info_id=asset_info_id)
        return {"added": [], "already_present": [], "total_tags": total}

    # Ensure tag rows exist if requested.
    if create_if_missing:
        await _ensure_tags_exist(session, norm, tag_type="user")

    # Snapshot current links
    current = {
        tag_name
        for (tag_name,) in (
            await session.execute(
                sa.select(AssetInfoTag.tag_name).where(AssetInfoTag.asset_info_id == asset_info_id)
            )
        ).all()
    }

    want = set(norm)
    to_add = sorted(want - current)

    if to_add:
        async with session.begin_nested() as nested:
            try:
                session.add_all(
                    [
                        AssetInfoTag(
                            asset_info_id=asset_info_id,
                            tag_name=t,
                            origin=origin,
                            added_by=added_by,
                            added_at=utcnow(),
                        )
                        for t in to_add
                    ]
                )
                await session.flush()
            except IntegrityError:
                await nested.rollback()

    after = set(await get_asset_tags(session, asset_info_id=asset_info_id))
    return {
        "added": sorted(((after - current) & want)),
        "already_present": sorted(want & current),
        "total_tags": sorted(after),
    }


async def remove_tags_from_asset_info(
    session: AsyncSession,
    *,
    asset_info_id: int,
    tags: Sequence[str],
) -> dict:
    """Removes tags from an AssetInfo.
    Returns: {"removed": [...], "not_present": [...], "total_tags": [...]}
    """
    info = await session.get(AssetInfo, asset_info_id)
    if not info:
        raise ValueError(f"AssetInfo {asset_info_id} not found")

    norm = normalize_tags(tags)
    if not norm:
        total = await get_asset_tags(session, asset_info_id=asset_info_id)
        return {"removed": [], "not_present": [], "total_tags": total}

    existing = {
        tag_name
        for (tag_name,) in (
            await session.execute(
                sa.select(AssetInfoTag.tag_name).where(AssetInfoTag.asset_info_id == asset_info_id)
            )
        ).all()
    }

    to_remove = sorted(set(t for t in norm if t in existing))
    not_present = sorted(set(t for t in norm if t not in existing))

    if to_remove:
        await session.execute(
            delete(AssetInfoTag)
            .where(
                AssetInfoTag.asset_info_id == asset_info_id,
                AssetInfoTag.tag_name.in_(to_remove),
            )
        )
        await session.flush()

    total = await get_asset_tags(session, asset_info_id=asset_info_id)
    return {"removed": to_remove, "not_present": not_present, "total_tags": total}


async def _ensure_tags_exist(session: AsyncSession, names: Iterable[str], tag_type: str = "user") -> list[Tag]:
    wanted = normalize_tags(list(names))
    if not wanted:
        return []
    existing = (await session.execute(select(Tag).where(Tag.name.in_(wanted)))).scalars().all()
    by_name = {t.name: t for t in existing}
    to_create = [Tag(name=n, tag_type=tag_type) for n in wanted if n not in by_name]
    if to_create:
        session.add_all(to_create)
        await session.flush()
        by_name.update({t.name: t for t in to_create})
    return [by_name[n] for n in wanted]


def _apply_tag_filters(
    stmt: sa.sql.Select,
    include_tags: Optional[Sequence[str]],
    exclude_tags: Optional[Sequence[str]],
) -> sa.sql.Select:
    """include_tags: every tag must be present; exclude_tags: none may be present."""
    include_tags = normalize_tags(include_tags)
    exclude_tags = normalize_tags(exclude_tags)

    if include_tags:
        for tag_name in include_tags:
            stmt = stmt.where(
                exists().where(
                    (AssetInfoTag.asset_info_id == AssetInfo.id)
                    & (AssetInfoTag.tag_name == tag_name)
                )
            )

    if exclude_tags:
        stmt = stmt.where(
            ~exists().where(
                (AssetInfoTag.asset_info_id == AssetInfo.id)
                & (AssetInfoTag.tag_name.in_(exclude_tags))
            )
        )
    return stmt

def _apply_metadata_filter(
    stmt: sa.sql.Select,
    metadata_filter: Optional[dict],
) -> sa.sql.Select:
    """Apply metadata filters using the projection table asset_info_meta.

    Semantics:
      - For scalar values: require EXISTS(asset_info_meta) with matching key + typed value.
      - For None: key is missing OR key has explicit null (val_json IS NULL).
      - For list values: ANY-of the list elements matches (EXISTS for any).
        (Change to ALL-of by 'for each element: stmt = stmt.where(_meta_exists_clause(key, elem))')
    """
    if not metadata_filter:
        return stmt

    def _exists_for_pred(key: str, *preds) -> sa.sql.ClauseElement:
        subquery = (
            select(sa.literal(1))
            .select_from(AssetInfoMeta)
            .where(
                AssetInfoMeta.asset_info_id == AssetInfo.id,
                AssetInfoMeta.key == key,
                *preds,
            )
            .limit(1)
        )
        return sa.exists(subquery)

    def _exists_clause_for_value(key: str, value) -> sa.sql.ClauseElement:
        # Missing OR null:
        if value is None:
            # either: no row for key OR a row for key with explicit null
            no_row_for_key = ~sa.exists(
                select(sa.literal(1))
                .select_from(AssetInfoMeta)
                .where(
                    AssetInfoMeta.asset_info_id == AssetInfo.id,
                    AssetInfoMeta.key == key,
                )
                .limit(1)
            )
            null_row = _exists_for_pred(key, AssetInfoMeta.val_json.is_(None))
            return sa.or_(no_row_for_key, null_row)

        # Typed scalar matches:
        if isinstance(value, bool):
            return _exists_for_pred(key, AssetInfoMeta.val_bool == bool(value))
        if isinstance(value, (int, float, Decimal)):
            # store as Decimal for equality against NUMERIC(38,10)
            num = value if isinstance(value, Decimal) else Decimal(str(value))
            return _exists_for_pred(key, AssetInfoMeta.val_num == num)
        if isinstance(value, str):
            return _exists_for_pred(key, AssetInfoMeta.val_str == value)

        # Complex: compare JSON (no index, but supported)
        return _exists_for_pred(key, AssetInfoMeta.val_json == value)

    for k, v in metadata_filter.items():
        if isinstance(v, list):
            # ANY-of (exists for any element)
            ors = [ _exists_clause_for_value(k, elem) for elem in v ]
            if ors:
                stmt = stmt.where(sa.or_(*ors))
        else:
            stmt = stmt.where(_exists_clause_for_value(k, v))
    return stmt


def _is_scalar(v: Any) -> bool:
    if v is None:  # treat None as a value (explicit null) so it can be indexed for "is null" queries
        return True
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float, Decimal, str)):
        return True
    return False


def _project_kv(key: str, value: Any) -> list[dict]:
    """
    Turn a metadata key/value into one or more projection rows:
    - scalar -> one row (ordinal=0) in the proper typed column
    - list of scalars -> one row per element with ordinal=i
    - dict or list with non-scalars -> single row with val_json (or one per element w/ val_json if list)
    - None -> single row with val_json = None
    Each row: {"key": key, "ordinal": i, "val_str"/"val_num"/"val_bool"/"val_json": ...}
    """
    rows: list[dict] = []

    # None
    if value is None:
        rows.append({"key": key, "ordinal": 0, "val_json": None})
        return rows

    # Scalars
    if _is_scalar(value):
        if isinstance(value, bool):
            rows.append({"key": key, "ordinal": 0, "val_bool": bool(value)})
        elif isinstance(value, (int, float, Decimal)):
            # store numeric; SQLAlchemy will coerce to Numeric
            rows.append({"key": key, "ordinal": 0, "val_num": value})
        elif isinstance(value, str):
            rows.append({"key": key, "ordinal": 0, "val_str": value})
        else:
            # Fallback to json
            rows.append({"key": key, "ordinal": 0, "val_json": value})
        return rows

    # Lists
    if isinstance(value, list):
        # list of scalars?
        if all(_is_scalar(x) for x in value):
            for i, x in enumerate(value):
                if x is None:
                    rows.append({"key": key, "ordinal": i, "val_json": None})
                elif isinstance(x, bool):
                    rows.append({"key": key, "ordinal": i, "val_bool": bool(x)})
                elif isinstance(x, (int, float, Decimal)):
                    rows.append({"key": key, "ordinal": i, "val_num": x})
                elif isinstance(x, str):
                    rows.append({"key": key, "ordinal": i, "val_str": x})
                else:
                    rows.append({"key": key, "ordinal": i, "val_json": x})
            return rows
        # list contains objects -> one val_json per element
        for i, x in enumerate(value):
            rows.append({"key": key, "ordinal": i, "val_json": x})
        return rows

    # Dict or any other structure -> single json row
    rows.append({"key": key, "ordinal": 0, "val_json": value})
    return rows
