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
from sqlalchemy.orm import contains_eager, noload
from sqlalchemy.exc import IntegrityError

from .models import Asset, AssetInfo, AssetInfoTag, AssetCacheState, Tag, AssetInfoMeta, AssetLocation
from .timeutil import utcnow
from .._assets_helpers import normalize_tags, visible_owner_clause, compute_model_relative_filename


async def asset_exists_by_hash(session: AsyncSession, *, asset_hash: str) -> bool:
    row = (
        await session.execute(
            select(sa.literal(True)).select_from(Asset).where(Asset.hash == asset_hash).limit(1)
        )
    ).first()
    return row is not None


async def get_asset_by_hash(session: AsyncSession, *, asset_hash: str) -> Optional[Asset]:
    return await session.get(Asset, asset_hash)


async def get_asset_info_by_id(session: AsyncSession, *, asset_info_id: str) -> Optional[AssetInfo]:
    return await session.get(AssetInfo, asset_info_id)


async def check_fs_asset_exists_quick(
    session,
    *,
    file_path: str,
    size_bytes: Optional[int] = None,
    mtime_ns: Optional[int] = None,
) -> bool:
    """
    Returns 'True' if there is already AssetCacheState record that matches this absolute path,
    AND (if provided) mtime_ns matches stored locator-state,
    AND (if provided) size_bytes matches verified size when known.
    """
    locator = os.path.abspath(file_path)

    stmt = select(sa.literal(True)).select_from(AssetCacheState).join(
        Asset, Asset.hash == AssetCacheState.asset_hash
    ).where(AssetCacheState.file_path == locator).limit(1)

    conds = []
    if mtime_ns is not None:
        conds.append(AssetCacheState.mtime_ns == int(mtime_ns))
    if size_bytes is not None:
        conds.append(sa.or_(Asset.size_bytes == 0, Asset.size_bytes == int(size_bytes)))

    if conds:
        stmt = stmt.where(*conds)

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
    require_existing_tags: bool = False,
) -> dict:
    """
    Upsert Asset identity row + cache state pointing at local file.

    Always:
      - Insert Asset if missing;
      - Insert AssetCacheState if missing; else update mtime_ns if different.

    Optionally (when info_name is provided):
      - Create an AssetInfo.
      - Link provided tags to that AssetInfo.
        * If the require_existing_tags=True, raises ValueError if any tag does not exist in `tags` table.
        * If False (default), create unknown tags.

    Returns flags and ids:
      {
        "asset_created": bool,
        "asset_updated": bool,
        "state_created": bool,
        "state_updated": bool,
        "asset_info_id": str | None,
      }
    """
    locator = os.path.abspath(abs_path)
    datetime_now = utcnow()

    out: dict[str, Any] = {
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
                    created_at=datetime_now,
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
            if changed:
                out["asset_updated"] = True
        else:
            logging.error("Asset %s not found after PK conflict; skipping update.", asset_hash)

    # ---- Step 2: INSERT/UPDATE AssetCacheState (mtime_ns, file_path) ----
    with contextlib.suppress(IntegrityError):
        async with session.begin_nested():
            session.add(
                AssetCacheState(
                    asset_hash=asset_hash,
                    file_path=locator,
                    mtime_ns=int(mtime_ns),
                )
            )
            await session.flush()
            out["state_created"] = True

    if not out["state_created"]:
        state = await session.get(AssetCacheState, asset_hash)
        if state is not None:
            changed = False
            if state.file_path != locator:
                state.file_path = locator
                changed = True
            if state.mtime_ns != int(mtime_ns):
                state.mtime_ns = int(mtime_ns)
                changed = True
            if changed:
                await session.flush()
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
                await session.flush()  # get info.id (UUID)
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
                            added_at=datetime_now,
                        )
                        for t in to_add
                    ]
                )
                await session.flush()

        # 2c) Rebuild metadata projection if provided
        # Uncomment next code, and remove code after it, once the hack with "metadata[filename" is not needed anymore
        # if user_metadata is not None and out["asset_info_id"] is not None:
        #     await replace_asset_info_metadata_projection(
        #         session,
        #         asset_info_id=out["asset_info_id"],
        #         user_metadata=user_metadata,
        #     )
        # start of adding metadata["filename"]
        if out["asset_info_id"] is not None:
            computed_filename = compute_model_relative_filename(abs_path)

            # Start from current metadata on this AssetInfo, if any
            current_meta = existing_info.user_metadata or {}
            new_meta = dict(current_meta)

            # Merge caller-provided metadata, if any (caller keys override current)
            if user_metadata is not None:
                for k, v in user_metadata.items():
                    new_meta[k] = v

            # Enforce correct model-relative filename when known
            if computed_filename:
                new_meta["filename"] = computed_filename

            # Only write when there is a change
            if new_meta != current_meta:
                await replace_asset_info_metadata_projection(
                    session,
                    asset_info_id=out["asset_info_id"],
                    user_metadata=new_meta,
                )
        # end of adding metadata["filename"]
    return out


async def touch_asset_infos_by_fs_path(
    session: AsyncSession,
    *,
    file_path: str,
    ts: Optional[datetime] = None,
    only_if_newer: bool = True,
) -> int:
    locator = os.path.abspath(file_path)
    ts = ts or utcnow()

    stmt = sa.update(AssetInfo).where(
        sa.exists(
            sa.select(sa.literal(1))
            .select_from(AssetCacheState)
            .where(
                AssetCacheState.asset_hash == AssetInfo.asset_hash,
                AssetCacheState.file_path == locator,
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
    asset_info_id: str,
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
    owner_id: str = "",
    include_tags: Optional[Sequence[str]] = None,
    exclude_tags: Optional[Sequence[str]] = None,
    name_contains: Optional[str] = None,
    metadata_filter: Optional[dict] = None,
    limit: int = 20,
    offset: int = 0,
    sort: str = "created_at",
    order: str = "desc",
) -> tuple[list[AssetInfo], dict[str, list[str]], int]:
    """Return page of AssetInfo rows in the viewers visibility."""
    base = (
        select(AssetInfo)
        .join(Asset, Asset.hash == AssetInfo.asset_hash)
        .options(contains_eager(AssetInfo.asset))
        .where(visible_owner_clause(owner_id))
    )

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
        .where(visible_owner_clause(owner_id))
    )
    if name_contains:
        count_stmt = count_stmt.where(AssetInfo.name.ilike(f"%{name_contains}%"))
    count_stmt = _apply_tag_filters(count_stmt, include_tags, exclude_tags)
    count_stmt = _apply_metadata_filter(count_stmt, metadata_filter)

    total = int((await session.execute(count_stmt)).scalar_one() or 0)

    # Fetch rows
    infos = (await session.execute(base)).scalars().unique().all()

    # Collect tags in bulk (single query)
    id_list: list[str] = [i.id for i in infos]
    tag_map: dict[str, list[str]] = defaultdict(list)
    if id_list:
        rows = await session.execute(
            select(AssetInfoTag.asset_info_id, Tag.name)
            .join(Tag, Tag.name == AssetInfoTag.tag_name)
            .where(AssetInfoTag.asset_info_id.in_(id_list))
        )
        for aid, tag_name in rows.all():
            tag_map[aid].append(tag_name)

    return infos, tag_map, total


async def fetch_asset_info_and_asset(
    session: AsyncSession,
    *,
    asset_info_id: str,
    owner_id: str = "",
) -> Optional[tuple[AssetInfo, Asset]]:
    stmt = (
        select(AssetInfo, Asset)
        .join(Asset, Asset.hash == AssetInfo.asset_hash)
        .where(
            AssetInfo.id == asset_info_id,
            visible_owner_clause(owner_id),
        )
        .limit(1)
    )
    row = await session.execute(stmt)
    pair = row.first()
    if not pair:
        return None
    return pair[0], pair[1]


async def fetch_asset_info_asset_and_tags(
    session: AsyncSession,
    *,
    asset_info_id: str,
    owner_id: str = "",
) -> Optional[tuple[AssetInfo, Asset, list[str]]]:
    stmt = (
        select(AssetInfo, Asset, Tag.name)
        .join(Asset, Asset.hash == AssetInfo.asset_hash)
        .join(AssetInfoTag, AssetInfoTag.asset_info_id == AssetInfo.id, isouter=True)
        .join(Tag, Tag.name == AssetInfoTag.tag_name, isouter=True)
        .where(
            AssetInfo.id == asset_info_id,
            visible_owner_clause(owner_id),
        )
        .options(noload(AssetInfo.tags))
        .order_by(Tag.name.asc())
    )

    rows = (await session.execute(stmt)).all()
    if not rows:
        return None

    # First row contains the mapped entities; tags may repeat across rows
    first_info, first_asset, _ = rows[0]
    tags: list[str] = []
    seen: set[str] = set()
    for _info, _asset, tag_name in rows:
        if tag_name and tag_name not in seen:
            seen.add(tag_name)
            tags.append(tag_name)
    return first_info, first_asset, tags


async def get_cache_state_by_asset_hash(session: AsyncSession, *, asset_hash: str) -> Optional[AssetCacheState]:
    return await session.get(AssetCacheState, asset_hash)


async def list_asset_locations(
        session: AsyncSession, *, asset_hash: str, provider: Optional[str] = None
) -> list[AssetLocation] | Sequence[AssetLocation]:
    stmt = select(AssetLocation).where(AssetLocation.asset_hash == asset_hash)
    if provider:
        stmt = stmt.where(AssetLocation.provider == provider)
    return (await session.execute(stmt)).scalars().all()


async def upsert_asset_location(
    session: AsyncSession,
    *,
    asset_hash: str,
    provider: str,
    locator: str,
    expected_size_bytes: Optional[int] = None,
    etag: Optional[str] = None,
    last_modified: Optional[str] = None,
) -> AssetLocation:
    loc = (
        await session.execute(
            select(AssetLocation).where(
                AssetLocation.asset_hash == asset_hash,
                AssetLocation.provider == provider,
                AssetLocation.locator == locator,
            ).limit(1)
        )
    ).scalars().first()
    if loc:
        changed = False
        if expected_size_bytes is not None and loc.expected_size_bytes != expected_size_bytes:
            loc.expected_size_bytes = expected_size_bytes
            changed = True
        if etag is not None and loc.etag != etag:
            loc.etag = etag
            changed = True
        if last_modified is not None and loc.last_modified != last_modified:
            loc.last_modified = last_modified
            changed = True
        if changed:
            await session.flush()
        return loc

    loc = AssetLocation(
        asset_hash=asset_hash,
        provider=provider,
        locator=locator,
        expected_size_bytes=expected_size_bytes,
        etag=etag,
        last_modified=last_modified,
    )
    session.add(loc)
    await session.flush()
    return loc


async def create_asset_info_for_existing_asset(
    session: AsyncSession,
    *,
    asset_hash: str,
    name: str,
    user_metadata: Optional[dict] = None,
    tags: Optional[Sequence[str]] = None,
    tag_origin: str = "manual",
    owner_id: str = "",
) -> AssetInfo:
    """Create a new AssetInfo referencing an existing Asset (no content write)."""
    now = utcnow()
    info = AssetInfo(
        owner_id=owner_id,
        name=name,
        asset_hash=asset_hash,
        preview_hash=None,
        created_at=now,
        updated_at=now,
        last_access_time=now,
    )
    session.add(info)
    await session.flush()  # get info.id

    # Uncomment next code, and remove code after it, once the hack with "metadata[filename" is not needed anymore
    # if user_metadata is not None:
    #     await replace_asset_info_metadata_projection(
    #         session, asset_info_id=info.id, user_metadata=user_metadata
    #     )

    # start of adding metadata["filename"]
    new_meta = dict(user_metadata or {})

    computed_filename = None
    try:
        state = await get_cache_state_by_asset_hash(session, asset_hash=asset_hash)
        if state and state.file_path:
            computed_filename = compute_model_relative_filename(state.file_path)
    except Exception:
        computed_filename = None

    if computed_filename:
        new_meta["filename"] = computed_filename

    if new_meta:
        await replace_asset_info_metadata_projection(
            session,
            asset_info_id=info.id,
            user_metadata=new_meta,
        )
    # end of adding metadata["filename"]

    if tags is not None:
        await set_asset_info_tags(
            session,
            asset_info_id=info.id,
            tags=tags,
            origin=tag_origin,
        )
    return info


async def set_asset_info_tags(
    session: AsyncSession,
    *,
    asset_info_id: str,
    tags: Sequence[str],
    origin: str = "manual",
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
            AssetInfoTag(asset_info_id=asset_info_id, tag_name=t, origin=origin, added_at=utcnow())
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
    asset_info_id: str,
    name: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    user_metadata: Optional[dict] = None,
    tag_origin: str = "manual",
    asset_info_row: Any = None,
) -> AssetInfo:
    """
    Update AssetInfo fields:
      - name (if provided)
      - user_metadata blob + rebuild projection (if provided)
      - replace tags with provided set (if provided)
    Returns the updated AssetInfo.
    """
    if not asset_info_row:
        info = await session.get(AssetInfo, asset_info_id)
        if not info:
            raise ValueError(f"AssetInfo {asset_info_id} not found")
    else:
        info = asset_info_row

    touched = False
    if name is not None and name != info.name:
        info.name = name
        touched = True

    # Uncomment next code, and remove code after it, once the hack with "metadata[filename" is not needed anymore
    # if user_metadata is not None:
    #     await replace_asset_info_metadata_projection(
    #         session, asset_info_id=asset_info_id, user_metadata=user_metadata
    #     )
    #     touched = True

    # start of adding metadata["filename"]
    computed_filename = None
    try:
        state = await get_cache_state_by_asset_hash(session, asset_hash=info.asset_hash)
        if state and state.file_path:
            computed_filename = compute_model_relative_filename(state.file_path)
    except Exception:
        computed_filename = None

    if user_metadata is not None:
        new_meta = dict(user_metadata)
        if computed_filename:
            new_meta["filename"] = computed_filename
        await replace_asset_info_metadata_projection(
            session, asset_info_id=asset_info_id, user_metadata=new_meta
        )
        touched = True
    else:
        if computed_filename:
            current_meta = info.user_metadata or {}
            if current_meta.get("filename") != computed_filename:
                new_meta = dict(current_meta)
                new_meta["filename"] = computed_filename
                await replace_asset_info_metadata_projection(
                    session, asset_info_id=asset_info_id, user_metadata=new_meta
                )
                touched = True
    # end of adding metadata["filename"]

    if tags is not None:
        await set_asset_info_tags(
            session,
            asset_info_id=asset_info_id,
            tags=tags,
            origin=tag_origin,
        )
        touched = True

    if touched and user_metadata is None:
        info.updated_at = utcnow()
        await session.flush()

    return info


async def delete_asset_info_by_id(session: AsyncSession, *, asset_info_id: str, owner_id: str) -> bool:
    """Delete the user-visible AssetInfo row. Cascades clear tags and metadata."""
    res = await session.execute(delete(AssetInfo).where(
        AssetInfo.id == asset_info_id,
        visible_owner_clause(owner_id),
    ))
    return bool(res.rowcount)


async def replace_asset_info_metadata_projection(
    session: AsyncSession,
    *,
    asset_info_id: str,
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


async def get_asset_tags(session: AsyncSession, *, asset_info_id: str) -> list[str]:
    return [
            tag_name
            for (tag_name,) in (
                await session.execute(
                    sa.select(AssetInfoTag.tag_name).where(AssetInfoTag.asset_info_id == asset_info_id)
                )
            ).all()
        ]


async def list_tags_with_usage(
    session: AsyncSession,
    *,
    prefix: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    include_zero: bool = True,
    order: str = "count_desc",  # "count_desc" | "name_asc"
    owner_id: str = "",
) -> tuple[list[tuple[str, str, int]], int]:
    # Subquery with counts by tag_name and owner_id
    counts_sq = (
        select(
            AssetInfoTag.tag_name.label("tag_name"),
            func.count(AssetInfoTag.asset_info_id).label("cnt"),
        )
        .select_from(AssetInfoTag)
        .join(AssetInfo, AssetInfo.id == AssetInfoTag.asset_info_id)
        .where(visible_owner_clause(owner_id))
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
    asset_info_id: str,
    tags: Sequence[str],
    origin: str = "manual",
    create_if_missing: bool = True,
    asset_info_row: Any = None,
) -> dict:
    """Adds tags to an AssetInfo.
    If create_if_missing=True, missing tag rows are created as 'user'.
    Returns: {"added": [...], "already_present": [...], "total_tags": [...]}
    """
    if not asset_info_row:
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
    asset_info_id: str,
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
            num = value if isinstance(value, Decimal) else Decimal(str(value))
            rows.append({"key": key, "ordinal": 0, "val_num": num})
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
                    num = x if isinstance(x, Decimal) else Decimal(str(x))
                    rows.append({"key": key, "ordinal": i, "val_num": num})
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
