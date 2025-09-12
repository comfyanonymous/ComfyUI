from collections import defaultdict
from datetime import datetime
from typing import Any, Optional, Sequence

import sqlalchemy as sa
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import contains_eager, noload

from ..._assets_helpers import compute_model_relative_filename, normalize_tags
from ..helpers import (
    apply_metadata_filter,
    apply_tag_filters,
    ensure_tags_exist,
    project_kv,
    visible_owner_clause,
)
from ..models import Asset, AssetInfo, AssetInfoMeta, AssetInfoTag, Tag
from ..timeutil import utcnow
from .queries import get_asset_by_hash, get_cache_state_by_asset_id


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
    base = (
        select(AssetInfo)
        .join(Asset, Asset.id == AssetInfo.asset_id)
        .options(contains_eager(AssetInfo.asset), noload(AssetInfo.tags))
        .where(visible_owner_clause(owner_id))
    )

    if name_contains:
        base = base.where(AssetInfo.name.ilike(f"%{name_contains}%"))

    base = apply_tag_filters(base, include_tags, exclude_tags)
    base = apply_metadata_filter(base, metadata_filter)

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

    count_stmt = (
        select(func.count())
        .select_from(AssetInfo)
        .join(Asset, Asset.id == AssetInfo.asset_id)
        .where(visible_owner_clause(owner_id))
    )
    if name_contains:
        count_stmt = count_stmt.where(AssetInfo.name.ilike(f"%{name_contains}%"))
    count_stmt = apply_tag_filters(count_stmt, include_tags, exclude_tags)
    count_stmt = apply_metadata_filter(count_stmt, metadata_filter)

    total = int((await session.execute(count_stmt)).scalar_one() or 0)

    infos = (await session.execute(base)).unique().scalars().all()

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
        .join(Asset, Asset.id == AssetInfo.asset_id)
        .where(
            AssetInfo.id == asset_info_id,
            visible_owner_clause(owner_id),
        )
        .limit(1)
        .options(noload(AssetInfo.tags))
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
        .join(Asset, Asset.id == AssetInfo.asset_id)
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

    first_info, first_asset, _ = rows[0]
    tags: list[str] = []
    seen: set[str] = set()
    for _info, _asset, tag_name in rows:
        if tag_name and tag_name not in seen:
            seen.add(tag_name)
            tags.append(tag_name)
    return first_info, first_asset, tags


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
    """Create or return an existing AssetInfo for an Asset identified by asset_hash."""
    now = utcnow()
    asset = await get_asset_by_hash(session, asset_hash=asset_hash)
    if not asset:
        raise ValueError(f"Unknown asset hash {asset_hash}")

    info = AssetInfo(
        owner_id=owner_id,
        name=name,
        asset_id=asset.id,
        preview_id=None,
        created_at=now,
        updated_at=now,
        last_access_time=now,
    )
    try:
        async with session.begin_nested():
            session.add(info)
            await session.flush()
    except IntegrityError:
        existing = (
            await session.execute(
                select(AssetInfo)
                .options(noload(AssetInfo.tags))
                .where(
                    AssetInfo.asset_id == asset.id,
                    AssetInfo.name == name,
                    AssetInfo.owner_id == owner_id,
                )
                .limit(1)
            )
        ).unique().scalars().first()
        if not existing:
            raise RuntimeError("AssetInfo upsert failed to find existing row after conflict.")
        return existing

    # metadata["filename"] hack
    new_meta = dict(user_metadata or {})
    computed_filename = None
    try:
        state = await get_cache_state_by_asset_id(session, asset_id=asset.id)
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
    desired = normalize_tags(tags)

    current = set(
        tag_name for (tag_name,) in (
            await session.execute(select(AssetInfoTag.tag_name).where(AssetInfoTag.asset_info_id == asset_info_id))
        ).all()
    )

    to_add = [t for t in desired if t not in current]
    to_remove = [t for t in current if t not in desired]

    if to_add:
        await ensure_tags_exist(session, to_add, tag_type="user")
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

    computed_filename = None
    try:
        state = await get_cache_state_by_asset_id(session, asset_id=info.asset_id)
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


async def replace_asset_info_metadata_projection(
    session: AsyncSession,
    *,
    asset_info_id: str,
    user_metadata: Optional[dict],
) -> None:
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
        for r in project_kv(k, v):
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


async def delete_asset_info_by_id(session: AsyncSession, *, asset_info_id: str, owner_id: str) -> bool:
    res = await session.execute(delete(AssetInfo).where(
        AssetInfo.id == asset_info_id,
        visible_owner_clause(owner_id),
    ))
    return bool(res.rowcount)


async def add_tags_to_asset_info(
    session: AsyncSession,
    *,
    asset_info_id: str,
    tags: Sequence[str],
    origin: str = "manual",
    create_if_missing: bool = True,
    asset_info_row: Any = None,
) -> dict:
    if not asset_info_row:
        info = await session.get(AssetInfo, asset_info_id)
        if not info:
            raise ValueError(f"AssetInfo {asset_info_id} not found")

    norm = normalize_tags(tags)
    if not norm:
        total = await get_asset_tags(session, asset_info_id=asset_info_id)
        return {"added": [], "already_present": [], "total_tags": total}

    if create_if_missing:
        await ensure_tags_exist(session, norm, tag_type="user")

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


async def list_tags_with_usage(
    session: AsyncSession,
    *,
    prefix: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    include_zero: bool = True,
    order: str = "count_desc",
    owner_id: str = "",
) -> tuple[list[tuple[str, str, int]], int]:
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

    q = (
        select(
            Tag.name,
            Tag.tag_type,
            func.coalesce(counts_sq.c.cnt, 0).label("count"),
        )
        .select_from(Tag)
        .join(counts_sq, counts_sq.c.tag_name == Tag.name, isouter=True)
    )

    if prefix:
        q = q.where(Tag.name.like(prefix.strip().lower() + "%"))

    if not include_zero:
        q = q.where(func.coalesce(counts_sq.c.cnt, 0) > 0)

    if order == "name_asc":
        q = q.order_by(Tag.name.asc())
    else:
        q = q.order_by(func.coalesce(counts_sq.c.cnt, 0).desc(), Tag.name.asc())

    total_q = select(func.count()).select_from(Tag)
    if prefix:
        total_q = total_q.where(Tag.name.like(prefix.strip().lower() + "%"))
    if not include_zero:
        total_q = total_q.where(
            Tag.name.in_(select(AssetInfoTag.tag_name).group_by(AssetInfoTag.tag_name))
        )

    rows = (await session.execute(q.limit(limit).offset(offset))).all()
    total = (await session.execute(total_q)).scalar_one()

    rows_norm = [(name, ttype, int(count or 0)) for (name, ttype, count) in rows]
    return rows_norm, int(total or 0)


async def get_asset_tags(session: AsyncSession, *, asset_info_id: str) -> list[str]:
    return [
        tag_name
        for (tag_name,) in (
            await session.execute(
                sa.select(AssetInfoTag.tag_name).where(AssetInfoTag.asset_info_id == asset_info_id)
            )
        ).all()
    ]


async def set_asset_info_preview(
    session: AsyncSession,
    *,
    asset_info_id: str,
    preview_asset_id: Optional[str],
) -> None:
    """Set or clear preview_id and bump updated_at. Raises on unknown IDs."""
    info = await session.get(AssetInfo, asset_info_id)
    if not info:
        raise ValueError(f"AssetInfo {asset_info_id} not found")

    if preview_asset_id is None:
        info.preview_id = None
    else:
        # validate preview asset exists
        if not await session.get(Asset, preview_asset_id):
            raise ValueError(f"Preview Asset {preview_asset_id} not found")
        info.preview_id = preview_asset_id

    info.updated_at = utcnow()
    await session.flush()
