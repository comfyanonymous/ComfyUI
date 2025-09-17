from typing import Iterable

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql as d_pg
from sqlalchemy.dialects import sqlite as d_sqlite
from sqlalchemy.ext.asyncio import AsyncSession

from ..._assets_helpers import normalize_tags
from ..models import AssetInfo, AssetInfoTag, Tag
from ..timeutil import utcnow


async def ensure_tags_exist(session: AsyncSession, names: Iterable[str], tag_type: str = "user") -> None:
    wanted = normalize_tags(list(names))
    if not wanted:
        return
    rows = [{"name": n, "tag_type": tag_type} for n in list(dict.fromkeys(wanted))]
    dialect = session.bind.dialect.name
    if dialect == "sqlite":
        ins = (
            d_sqlite.insert(Tag)
            .values(rows)
            .on_conflict_do_nothing(index_elements=[Tag.name])
        )
    elif dialect == "postgresql":
        ins = (
            d_pg.insert(Tag)
            .values(rows)
            .on_conflict_do_nothing(index_elements=[Tag.name])
        )
    else:
        raise NotImplementedError(f"Unsupported database dialect: {dialect}")
    await session.execute(ins)


async def add_missing_tag_for_asset_id(
    session: AsyncSession,
    *,
    asset_id: str,
    origin: str = "automatic",
) -> None:
    select_rows = (
        sa.select(
            AssetInfo.id.label("asset_info_id"),
            sa.literal("missing").label("tag_name"),
            sa.literal(origin).label("origin"),
            sa.literal(utcnow()).label("added_at"),
        )
        .where(AssetInfo.asset_id == asset_id)
        .where(
            sa.not_(
                sa.exists().where((AssetInfoTag.asset_info_id == AssetInfo.id) & (AssetInfoTag.tag_name == "missing"))
            )
        )
    )
    dialect = session.bind.dialect.name
    if dialect == "sqlite":
        ins = (
            d_sqlite.insert(AssetInfoTag)
            .from_select(
                ["asset_info_id", "tag_name", "origin", "added_at"],
                select_rows,
            )
            .on_conflict_do_nothing(index_elements=[AssetInfoTag.asset_info_id, AssetInfoTag.tag_name])
        )
    elif dialect == "postgresql":
        ins = (
            d_pg.insert(AssetInfoTag)
            .from_select(
                ["asset_info_id", "tag_name", "origin", "added_at"],
                select_rows,
            )
            .on_conflict_do_nothing(index_elements=[AssetInfoTag.asset_info_id, AssetInfoTag.tag_name])
        )
    else:
        raise NotImplementedError(f"Unsupported database dialect: {dialect}")
    await session.execute(ins)


async def remove_missing_tag_for_asset_id(
    session: AsyncSession,
    *,
    asset_id: str,
) -> None:
    await session.execute(
        sa.delete(AssetInfoTag).where(
            AssetInfoTag.asset_info_id.in_(sa.select(AssetInfo.id).where(AssetInfo.asset_id == asset_id)),
            AssetInfoTag.tag_name == "missing",
        )
    )
