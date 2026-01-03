from typing import Iterable

import sqlalchemy
from sqlalchemy.orm import Session
from sqlalchemy.dialects import sqlite

from app.assets.helpers import normalize_tags, utcnow
from app.assets.database.models import Tag, AssetInfoTag, AssetInfo


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
    return session.execute(ins)

def add_missing_tag_for_asset_id(
    session: Session,
    *,
    asset_id: str,
    origin: str = "automatic",
) -> None:
    select_rows = (
        sqlalchemy.select(
            AssetInfo.id.label("asset_info_id"),
            sqlalchemy.literal("missing").label("tag_name"),
            sqlalchemy.literal(origin).label("origin"),
            sqlalchemy.literal(utcnow()).label("added_at"),
        )
        .where(AssetInfo.asset_id == asset_id)
        .where(
            sqlalchemy.not_(
                sqlalchemy.exists().where((AssetInfoTag.asset_info_id == AssetInfo.id) & (AssetInfoTag.tag_name == "missing"))
            )
        )
    )
    session.execute(
        sqlite.insert(AssetInfoTag)
        .from_select(
            ["asset_info_id", "tag_name", "origin", "added_at"],
            select_rows,
        )
        .on_conflict_do_nothing(index_elements=[AssetInfoTag.asset_info_id, AssetInfoTag.tag_name])
    )

def remove_missing_tag_for_asset_id(
    session: Session,
    *,
    asset_id: str,
) -> None:
    session.execute(
        sqlalchemy.delete(AssetInfoTag).where(
            AssetInfoTag.asset_info_id.in_(sqlalchemy.select(AssetInfo.id).where(AssetInfo.asset_id == asset_id)),
            AssetInfoTag.tag_name == "missing",
        )
    )
