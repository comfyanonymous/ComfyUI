import os
import logging
import sqlalchemy as sa
from collections import defaultdict
from datetime import datetime
from typing import Iterable, Any
from sqlalchemy import select, delete, exists, func
from sqlalchemy.dialects import sqlite
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, contains_eager, noload
from app.assets.database.models import Asset, AssetInfo, AssetCacheState, AssetInfoMeta, AssetInfoTag, Tag
from app.assets.helpers import (
    compute_relative_filename, escape_like_prefix, normalize_tags, project_kv, utcnow
)
from typing import Sequence


def visible_owner_clause(owner_id: str) -> sa.sql.ClauseElement:
    """Build owner visibility predicate for reads. Owner-less rows are visible to everyone."""
    owner_id = (owner_id or "").strip()
    if owner_id == "":
        return AssetInfo.owner_id == ""
    return AssetInfo.owner_id.in_(["", owner_id])


def pick_best_live_path(states: Sequence[AssetCacheState]) -> str:
    """
    Return the best on-disk path among cache states:
      1) Prefer a path that exists with needs_verify == False (already verified).
      2) Otherwise, pick the first path that exists.
      3) Otherwise return empty string.
    """
    alive = [s for s in states if getattr(s, "file_path", None) and os.path.isfile(s.file_path)]
    if not alive:
        return ""
    for s in alive:
        if not getattr(s, "needs_verify", False):
            return s.file_path
    return alive[0].file_path


def apply_tag_filters(
    stmt: sa.sql.Select,
    include_tags: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
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


def apply_metadata_filter(
    stmt: sa.sql.Select,
    metadata_filter: dict | None = None,
) -> sa.sql.Select:
    """Apply filters using asset_info_meta projection table."""
    if not metadata_filter:
        return stmt

    def _exists_for_pred(key: str, *preds) -> sa.sql.ClauseElement:
        return sa.exists().where(
            AssetInfoMeta.asset_info_id == AssetInfo.id,
            AssetInfoMeta.key == key,
            *preds,
        )

    def _exists_clause_for_value(key: str, value) -> sa.sql.ClauseElement:
        if value is None:
            no_row_for_key = sa.not_(
                sa.exists().where(
                    AssetInfoMeta.asset_info_id == AssetInfo.id,
                    AssetInfoMeta.key == key,
                )
            )
            null_row = _exists_for_pred(
                key,
                AssetInfoMeta.val_json.is_(None),
                AssetInfoMeta.val_str.is_(None),
                AssetInfoMeta.val_num.is_(None),
                AssetInfoMeta.val_bool.is_(None),
            )
            return sa.or_(no_row_for_key, null_row)

        if isinstance(value, bool):
            return _exists_for_pred(key, AssetInfoMeta.val_bool == bool(value))
        if isinstance(value, (int, float)):
            from decimal import Decimal
            num = value if isinstance(value, Decimal) else Decimal(str(value))
            return _exists_for_pred(key, AssetInfoMeta.val_num == num)
        if isinstance(value, str):
            return _exists_for_pred(key, AssetInfoMeta.val_str == value)
        return _exists_for_pred(key, AssetInfoMeta.val_json == value)

    for k, v in metadata_filter.items():
        if isinstance(v, list):
            ors = [_exists_clause_for_value(k, elem) for elem in v]
            if ors:
                stmt = stmt.where(sa.or_(*ors))
        else:
            stmt = stmt.where(_exists_clause_for_value(k, v))
    return stmt


def asset_exists_by_hash(
    session: Session,
    *,
    asset_hash: str,
) -> bool:
    """
    Check if an asset with a given hash exists in database.
    """
    row = (
        session.execute(
            select(sa.literal(True)).select_from(Asset).where(Asset.hash == asset_hash).limit(1)
        )
    ).first()
    return row is not None


def asset_info_exists_for_asset_id(
    session: Session,
    *,
    asset_id: str,
) -> bool:
    q = (
        select(sa.literal(True))
        .select_from(AssetInfo)
        .where(AssetInfo.asset_id == asset_id)
        .limit(1)
    )
    return (session.execute(q)).first() is not None


def get_asset_by_hash(
    session: Session,
    *,
    asset_hash: str,
) -> Asset | None:
    return (
        session.execute(select(Asset).where(Asset.hash == asset_hash).limit(1))
    ).scalars().first()


def get_asset_info_by_id(
    session: Session,
    *,
    asset_info_id: str,
) -> AssetInfo | None:
    return session.get(AssetInfo, asset_info_id)


def list_asset_infos_page(
    session: Session,
    owner_id: str = "",
    include_tags: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    name_contains: str | None = None,
    metadata_filter: dict | None = None,
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
        escaped, esc = escape_like_prefix(name_contains)
        base = base.where(AssetInfo.name.ilike(f"%{escaped}%", escape=esc))

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
        select(sa.func.count())
        .select_from(AssetInfo)
        .join(Asset, Asset.id == AssetInfo.asset_id)
        .where(visible_owner_clause(owner_id))
    )
    if name_contains:
        escaped, esc = escape_like_prefix(name_contains)
        count_stmt = count_stmt.where(AssetInfo.name.ilike(f"%{escaped}%", escape=esc))
    count_stmt = apply_tag_filters(count_stmt, include_tags, exclude_tags)
    count_stmt = apply_metadata_filter(count_stmt, metadata_filter)

    total = int((session.execute(count_stmt)).scalar_one() or 0)

    infos = (session.execute(base)).unique().scalars().all()

    id_list: list[str] = [i.id for i in infos]
    tag_map: dict[str, list[str]] = defaultdict(list)
    if id_list:
        rows = session.execute(
            select(AssetInfoTag.asset_info_id, Tag.name)
            .join(Tag, Tag.name == AssetInfoTag.tag_name)
            .where(AssetInfoTag.asset_info_id.in_(id_list))
            .order_by(AssetInfoTag.added_at)
        )
        for aid, tag_name in rows.all():
            tag_map[aid].append(tag_name)

    return infos, tag_map, total


def fetch_asset_info_asset_and_tags(
    session: Session,
    asset_info_id: str,
    owner_id: str = "",
) -> tuple[AssetInfo, Asset, list[str]] | None:
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

    rows = (session.execute(stmt)).all()
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


def fetch_asset_info_and_asset(
    session: Session,
    *,
    asset_info_id: str,
    owner_id: str = "",
) -> tuple[AssetInfo, Asset] | None:
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
    row = session.execute(stmt)
    pair = row.first()
    if not pair:
        return None
    return pair[0], pair[1]

def list_cache_states_by_asset_id(
    session: Session, *, asset_id: str
) -> Sequence[AssetCacheState]:
    return (
        session.execute(
            select(AssetCacheState)
            .where(AssetCacheState.asset_id == asset_id)
            .order_by(AssetCacheState.id.asc())
        )
    ).scalars().all()


def touch_asset_info_by_id(
    session: Session,
    *,
    asset_info_id: str,
    ts: datetime | None = None,
    only_if_newer: bool = True,
) -> None:
    ts = ts or utcnow()
    stmt = sa.update(AssetInfo).where(AssetInfo.id == asset_info_id)
    if only_if_newer:
        stmt = stmt.where(
            sa.or_(AssetInfo.last_access_time.is_(None), AssetInfo.last_access_time < ts)
        )
    session.execute(stmt.values(last_access_time=ts))


def create_asset_info_for_existing_asset(
    session: Session,
    *,
    asset_hash: str,
    name: str,
    user_metadata: dict | None = None,
    tags: Sequence[str] | None = None,
    tag_origin: str = "manual",
    owner_id: str = "",
) -> AssetInfo:
    """Create or return an existing AssetInfo for an Asset identified by asset_hash."""
    now = utcnow()
    asset = get_asset_by_hash(session, asset_hash=asset_hash)
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
        with session.begin_nested():
            session.add(info)
            session.flush()
    except IntegrityError:
        existing = (
            session.execute(
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
        p = pick_best_live_path(list_cache_states_by_asset_id(session, asset_id=asset.id))
        if p:
            computed_filename = compute_relative_filename(p)
    except Exception:
        computed_filename = None
    if computed_filename:
        new_meta["filename"] = computed_filename
    if new_meta:
        replace_asset_info_metadata_projection(
            session,
            asset_info_id=info.id,
            user_metadata=new_meta,
        )

    if tags is not None:
        set_asset_info_tags(
            session,
            asset_info_id=info.id,
            tags=tags,
            origin=tag_origin,
        )
    return info


def set_asset_info_tags(
    session: Session,
    *,
    asset_info_id: str,
    tags: Sequence[str],
    origin: str = "manual",
) -> dict:
    desired = normalize_tags(tags)

    current = set(
        tag_name for (tag_name,) in (
            session.execute(select(AssetInfoTag.tag_name).where(AssetInfoTag.asset_info_id == asset_info_id))
        ).all()
    )

    to_add = [t for t in desired if t not in current]
    to_remove = [t for t in current if t not in desired]

    if to_add:
        ensure_tags_exist(session, to_add, tag_type="user")
        session.add_all([
            AssetInfoTag(asset_info_id=asset_info_id, tag_name=t, origin=origin, added_at=utcnow())
            for t in to_add
        ])
        session.flush()

    if to_remove:
        session.execute(
            delete(AssetInfoTag)
            .where(AssetInfoTag.asset_info_id == asset_info_id, AssetInfoTag.tag_name.in_(to_remove))
        )
        session.flush()

    return {"added": to_add, "removed": to_remove, "total": desired}


def replace_asset_info_metadata_projection(
    session: Session,
    *,
    asset_info_id: str,
    user_metadata: dict | None = None,
) -> None:
    info = session.get(AssetInfo, asset_info_id)
    if not info:
        raise ValueError(f"AssetInfo {asset_info_id} not found")

    info.user_metadata = user_metadata or {}
    info.updated_at = utcnow()
    session.flush()

    session.execute(delete(AssetInfoMeta).where(AssetInfoMeta.asset_info_id == asset_info_id))
    session.flush()

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
        session.flush()


def ingest_fs_asset(
    session: Session,
    *,
    asset_hash: str,
    abs_path: str,
    size_bytes: int,
    mtime_ns: int,
    mime_type: str | None = None,
    info_name: str | None = None,
    owner_id: str = "",
    preview_id: str | None = None,
    user_metadata: dict | None = None,
    tags: Sequence[str] = (),
    tag_origin: str = "manual",
    require_existing_tags: bool = False,
) -> dict:
    """
    Idempotently upsert:
      - Asset by content hash (create if missing)
      - AssetCacheState(file_path) pointing to asset_id
      - Optionally AssetInfo + tag links and metadata projection
    Returns flags and ids.
    """
    locator = os.path.abspath(abs_path)
    now = utcnow()

    if preview_id:
        if not session.get(Asset, preview_id):
            preview_id = None

    out: dict[str, Any] = {
        "asset_created": False,
        "asset_updated": False,
        "state_created": False,
        "state_updated": False,
        "asset_info_id": None,
    }

    # 1) Asset by hash
    asset = (
        session.execute(select(Asset).where(Asset.hash == asset_hash).limit(1))
    ).scalars().first()
    if not asset:
        vals = {
            "hash": asset_hash,
            "size_bytes": int(size_bytes),
            "mime_type": mime_type,
            "created_at": now,
        }
        res = session.execute(
            sqlite.insert(Asset)
            .values(**vals)
            .on_conflict_do_nothing(index_elements=[Asset.hash])
        )
        if int(res.rowcount or 0) > 0:
            out["asset_created"] = True
        asset = (
            session.execute(
                select(Asset).where(Asset.hash == asset_hash).limit(1)
            )
        ).scalars().first()
        if not asset:
            raise RuntimeError("Asset row not found after upsert.")
    else:
        changed = False
        if asset.size_bytes != int(size_bytes) and int(size_bytes) > 0:
            asset.size_bytes = int(size_bytes)
            changed = True
        if mime_type and asset.mime_type != mime_type:
            asset.mime_type = mime_type
            changed = True
        if changed:
            out["asset_updated"] = True

    # 2) AssetCacheState upsert by file_path (unique)
    vals = {
        "asset_id": asset.id,
        "file_path": locator,
        "mtime_ns": int(mtime_ns),
    }
    ins = (
        sqlite.insert(AssetCacheState)
        .values(**vals)
        .on_conflict_do_nothing(index_elements=[AssetCacheState.file_path])
    )

    res = session.execute(ins)
    if int(res.rowcount or 0) > 0:
        out["state_created"] = True
    else:
        upd = (
            sa.update(AssetCacheState)
            .where(AssetCacheState.file_path == locator)
            .where(
                sa.or_(
                    AssetCacheState.asset_id != asset.id,
                    AssetCacheState.mtime_ns.is_(None),
                    AssetCacheState.mtime_ns != int(mtime_ns),
                )
            )
            .values(asset_id=asset.id, mtime_ns=int(mtime_ns))
        )
        res2 = session.execute(upd)
        if int(res2.rowcount or 0) > 0:
            out["state_updated"] = True

    # 3) Optional AssetInfo + tags + metadata
    if info_name:
        try:
            with session.begin_nested():
                info = AssetInfo(
                    owner_id=owner_id,
                    name=info_name,
                    asset_id=asset.id,
                    preview_id=preview_id,
                    created_at=now,
                    updated_at=now,
                    last_access_time=now,
                )
                session.add(info)
                session.flush()
                out["asset_info_id"] = info.id
        except IntegrityError:
            pass

        existing_info = (
            session.execute(
                select(AssetInfo)
                .where(
                    AssetInfo.asset_id == asset.id,
                    AssetInfo.name == info_name,
                    (AssetInfo.owner_id == owner_id),
                )
                .limit(1)
            )
        ).unique().scalar_one_or_none()
        if not existing_info:
            raise RuntimeError("Failed to update or insert AssetInfo.")

        if preview_id and existing_info.preview_id != preview_id:
            existing_info.preview_id = preview_id

        existing_info.updated_at = now
        if existing_info.last_access_time < now:
            existing_info.last_access_time = now
        session.flush()
        out["asset_info_id"] = existing_info.id

        norm = [t.strip().lower() for t in (tags or []) if (t or "").strip()]
        if norm and out["asset_info_id"] is not None:
            if not require_existing_tags:
                ensure_tags_exist(session, norm, tag_type="user")

            existing_tag_names = set(
                name for (name,) in (session.execute(select(Tag.name).where(Tag.name.in_(norm)))).all()
            )
            missing = [t for t in norm if t not in existing_tag_names]
            if missing and require_existing_tags:
                raise ValueError(f"Unknown tags: {missing}")

            existing_links = set(
                tag_name
                for (tag_name,) in (
                    session.execute(
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
                            added_at=now,
                        )
                        for t in to_add
                    ]
                )
                session.flush()

        # metadata["filename"] hack
        if out["asset_info_id"] is not None:
            primary_path = pick_best_live_path(list_cache_states_by_asset_id(session, asset_id=asset.id))
            computed_filename = compute_relative_filename(primary_path) if primary_path else None

            current_meta = existing_info.user_metadata or {}
            new_meta = dict(current_meta)
            if user_metadata is not None:
                for k, v in user_metadata.items():
                    new_meta[k] = v
            if computed_filename:
                new_meta["filename"] = computed_filename

            if new_meta != current_meta:
                replace_asset_info_metadata_projection(
                    session,
                    asset_info_id=out["asset_info_id"],
                    user_metadata=new_meta,
                )

    try:
        remove_missing_tag_for_asset_id(session, asset_id=asset.id)
    except Exception:
        logging.exception("Failed to clear 'missing' tag for asset %s", asset.id)
    return out


def update_asset_info_full(
    session: Session,
    *,
    asset_info_id: str,
    name: str | None = None,
    tags: Sequence[str] | None = None,
    user_metadata: dict | None = None,
    tag_origin: str = "manual",
    asset_info_row: Any = None,
) -> AssetInfo:
    if not asset_info_row:
        info = session.get(AssetInfo, asset_info_id)
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
        p = pick_best_live_path(list_cache_states_by_asset_id(session, asset_id=info.asset_id))
        if p:
            computed_filename = compute_relative_filename(p)
    except Exception:
        computed_filename = None

    if user_metadata is not None:
        new_meta = dict(user_metadata)
        if computed_filename:
            new_meta["filename"] = computed_filename
        replace_asset_info_metadata_projection(
            session, asset_info_id=asset_info_id, user_metadata=new_meta
        )
        touched = True
    else:
        if computed_filename:
            current_meta = info.user_metadata or {}
            if current_meta.get("filename") != computed_filename:
                new_meta = dict(current_meta)
                new_meta["filename"] = computed_filename
                replace_asset_info_metadata_projection(
                    session, asset_info_id=asset_info_id, user_metadata=new_meta
                )
                touched = True

    if tags is not None:
        set_asset_info_tags(
            session,
            asset_info_id=asset_info_id,
            tags=tags,
            origin=tag_origin,
        )
        touched = True

    if touched and user_metadata is None:
        info.updated_at = utcnow()
        session.flush()

    return info


def delete_asset_info_by_id(
    session: Session,
    *,
    asset_info_id: str,
    owner_id: str,
) -> bool:
    stmt = sa.delete(AssetInfo).where(
        AssetInfo.id == asset_info_id,
        visible_owner_clause(owner_id),
    )
    return int((session.execute(stmt)).rowcount or 0) > 0


def list_tags_with_usage(
    session: Session,
    prefix: str | None = None,
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
        escaped, esc = escape_like_prefix(prefix.strip().lower())
        q = q.where(Tag.name.like(escaped + "%", escape=esc))

    if not include_zero:
        q = q.where(func.coalesce(counts_sq.c.cnt, 0) > 0)

    if order == "name_asc":
        q = q.order_by(Tag.name.asc())
    else:
        q = q.order_by(func.coalesce(counts_sq.c.cnt, 0).desc(), Tag.name.asc())

    total_q = select(func.count()).select_from(Tag)
    if prefix:
        escaped, esc = escape_like_prefix(prefix.strip().lower())
        total_q = total_q.where(Tag.name.like(escaped + "%", escape=esc))
    if not include_zero:
        total_q = total_q.where(
            Tag.name.in_(select(AssetInfoTag.tag_name).group_by(AssetInfoTag.tag_name))
        )

    rows = (session.execute(q.limit(limit).offset(offset))).all()
    total = (session.execute(total_q)).scalar_one()

    rows_norm = [(name, ttype, int(count or 0)) for (name, ttype, count) in rows]
    return rows_norm, int(total or 0)


def ensure_tags_exist(session: Session, names: Iterable[str], tag_type: str = "user") -> None:
    wanted = normalize_tags(list(names))
    if not wanted:
        return
    rows = [{"name": n, "tag_type": tag_type} for n in list(dict.fromkeys(wanted))]
    ins = (
        sqlite.insert(Tag)
        .values(rows)
        .on_conflict_do_nothing(index_elements=[Tag.name])
    )
    session.execute(ins)


def get_asset_tags(session: Session, *, asset_info_id: str) -> list[str]:
    return [
        tag_name for (tag_name,) in (
            session.execute(
                select(AssetInfoTag.tag_name).where(AssetInfoTag.asset_info_id == asset_info_id)
            )
        ).all()
    ]


def add_tags_to_asset_info(
    session: Session,
    *,
    asset_info_id: str,
    tags: Sequence[str],
    origin: str = "manual",
    create_if_missing: bool = True,
    asset_info_row: Any = None,
) -> dict:
    if not asset_info_row:
        info = session.get(AssetInfo, asset_info_id)
        if not info:
            raise ValueError(f"AssetInfo {asset_info_id} not found")

    norm = normalize_tags(tags)
    if not norm:
        total = get_asset_tags(session, asset_info_id=asset_info_id)
        return {"added": [], "already_present": [], "total_tags": total}

    if create_if_missing:
        ensure_tags_exist(session, norm, tag_type="user")

    current = {
        tag_name
        for (tag_name,) in (
            session.execute(
                sa.select(AssetInfoTag.tag_name).where(AssetInfoTag.asset_info_id == asset_info_id)
            )
        ).all()
    }

    want = set(norm)
    to_add = sorted(want - current)

    if to_add:
        with session.begin_nested() as nested:
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
                session.flush()
            except IntegrityError:
                nested.rollback()

    after = set(get_asset_tags(session, asset_info_id=asset_info_id))
    return {
        "added": sorted(((after - current) & want)),
        "already_present": sorted(want & current),
        "total_tags": sorted(after),
    }


def remove_tags_from_asset_info(
    session: Session,
    *,
    asset_info_id: str,
    tags: Sequence[str],
) -> dict:
    info = session.get(AssetInfo, asset_info_id)
    if not info:
        raise ValueError(f"AssetInfo {asset_info_id} not found")

    norm = normalize_tags(tags)
    if not norm:
        total = get_asset_tags(session, asset_info_id=asset_info_id)
        return {"removed": [], "not_present": [], "total_tags": total}

    existing = {
        tag_name
        for (tag_name,) in (
            session.execute(
                sa.select(AssetInfoTag.tag_name).where(AssetInfoTag.asset_info_id == asset_info_id)
            )
        ).all()
    }

    to_remove = sorted(set(t for t in norm if t in existing))
    not_present = sorted(set(t for t in norm if t not in existing))

    if to_remove:
        session.execute(
            delete(AssetInfoTag)
            .where(
                AssetInfoTag.asset_info_id == asset_info_id,
                AssetInfoTag.tag_name.in_(to_remove),
            )
        )
        session.flush()

    total = get_asset_tags(session, asset_info_id=asset_info_id)
    return {"removed": to_remove, "not_present": not_present, "total_tags": total}


def remove_missing_tag_for_asset_id(
    session: Session,
    *,
    asset_id: str,
) -> None:
    session.execute(
        sa.delete(AssetInfoTag).where(
            AssetInfoTag.asset_info_id.in_(sa.select(AssetInfo.id).where(AssetInfo.asset_id == asset_id)),
            AssetInfoTag.tag_name == "missing",
        )
    )


def set_asset_info_preview(
    session: Session,
    *,
    asset_info_id: str,
    preview_asset_id: str | None = None,
) -> None:
    """Set or clear preview_id and bump updated_at. Raises on unknown IDs."""
    info = session.get(AssetInfo, asset_info_id)
    if not info:
        raise ValueError(f"AssetInfo {asset_info_id} not found")

    if preview_asset_id is None:
        info.preview_id = None
    else:
        # validate preview asset exists
        if not session.get(Asset, preview_asset_id):
            raise ValueError(f"Preview Asset {preview_asset_id} not found")
        info.preview_id = preview_asset_id

    info.updated_at = utcnow()
    session.flush()
