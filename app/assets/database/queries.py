import sqlalchemy as sa
from collections import defaultdict
from sqlalchemy import select, exists, func
from sqlalchemy.orm import Session, contains_eager, noload
from app.assets.database.models import Asset, AssetInfo, AssetInfoMeta, AssetInfoTag, Tag
from app.assets.helpers import escape_like_prefix, normalize_tags
from typing import Sequence


def visible_owner_clause(owner_id: str) -> sa.sql.ClauseElement:
    """Build owner visibility predicate for reads. Owner-less rows are visible to everyone."""
    owner_id = (owner_id or "").strip()
    if owner_id == "":
        return AssetInfo.owner_id == ""
    return AssetInfo.owner_id.in_(["", owner_id])


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


def asset_exists_by_hash(session: Session, asset_hash: str) -> bool:
    """
    Check if an asset with a given hash exists in database.
    """
    row = (
        session.execute(
            select(sa.literal(True)).select_from(Asset).where(Asset.hash == asset_hash).limit(1)
        )
    ).first()
    return row is not None

def get_asset_info_by_id(session: Session, asset_info_id: str) -> AssetInfo | None:
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
