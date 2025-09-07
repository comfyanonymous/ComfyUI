from decimal import Decimal
from typing import Any, Sequence, Optional, Iterable

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, exists

from .models import AssetInfo, AssetInfoTag, Tag, AssetInfoMeta
from .._assets_helpers import normalize_tags


async def ensure_tags_exist(session: AsyncSession, names: Iterable[str], tag_type: str = "user") -> list[Tag]:
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


def apply_tag_filters(
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


def apply_metadata_filter(
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
            ors = [_exists_clause_for_value(k, elem) for elem in v]
            if ors:
                stmt = stmt.where(sa.or_(*ors))
        else:
            stmt = stmt.where(_exists_clause_for_value(k, v))
    return stmt


def is_scalar(v: Any) -> bool:
    if v is None:  # treat None as a value (explicit null) so it can be indexed for "is null" queries
        return True
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float, Decimal, str)):
        return True
    return False


def project_kv(key: str, value: Any) -> list[dict]:
    """
    Turn a metadata key/value into one or more projection rows:
    - scalar -> one row (ordinal=0) in the proper typed column
    - list of scalars -> one row per element with ordinal=i
    - dict or list with non-scalars -> single row with val_json (or one per element w/ val_json if list)
    - None -> single row with val_json = None
    Each row: {"key": key, "ordinal": i, "val_str"/"val_num"/"val_bool"/"val_json": ...}
    """
    rows: list[dict] = []

    if value is None:
        rows.append({"key": key, "ordinal": 0, "val_json": None})
        return rows

    if is_scalar(value):
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

    if isinstance(value, list):
        if all(is_scalar(x) for x in value):
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
