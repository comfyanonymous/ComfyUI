from typing import Optional, Sequence

import sqlalchemy as sa
from sqlalchemy import exists

from ..._helpers import normalize_tags
from ..models import AssetInfo, AssetInfoMeta, AssetInfoTag


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
