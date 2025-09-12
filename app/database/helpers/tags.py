from typing import Iterable

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..._assets_helpers import normalize_tags
from ..models import Asset, AssetInfo, AssetInfoTag, Tag
from ..timeutil import utcnow


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


async def add_missing_tag_for_asset_id(
    session: AsyncSession,
    *,
    asset_id: str,
    origin: str = "automatic",
) -> int:
    """Ensure every AssetInfo for asset_id has 'missing' tag."""
    ids = (await session.execute(select(AssetInfo.id).where(AssetInfo.asset_id == asset_id))).scalars().all()
    if not ids:
        return 0

    existing = {
        asset_info_id
        for (asset_info_id,) in (
            await session.execute(
                select(AssetInfoTag.asset_info_id).where(
                    AssetInfoTag.asset_info_id.in_(ids),
                    AssetInfoTag.tag_name == "missing",
                )
            )
        ).all()
    }
    to_add = [i for i in ids if i not in existing]
    if not to_add:
        return 0

    now = utcnow()
    session.add_all(
        [
            AssetInfoTag(asset_info_id=i, tag_name="missing", origin=origin, added_at=now)
            for i in to_add
        ]
    )
    await session.flush()
    return len(to_add)


async def add_missing_tag_for_asset_hash(
    session: AsyncSession,
    *,
    asset_hash: str,
    origin: str = "automatic",
) -> int:
    asset = (await session.execute(select(Asset).where(Asset.hash == asset_hash).limit(1))).scalars().first()
    if not asset:
        return 0
    return await add_missing_tag_for_asset_id(session, asset_id=asset.id, origin=origin)


async def remove_missing_tag_for_asset_id(
    session: AsyncSession,
    *,
    asset_id: str,
) -> int:
    """Remove the 'missing' tag from all AssetInfos for asset_id."""
    ids = (await session.execute(select(AssetInfo.id).where(AssetInfo.asset_id == asset_id))).scalars().all()
    if not ids:
        return 0

    res = await session.execute(
        delete(AssetInfoTag).where(
            AssetInfoTag.asset_info_id.in_(ids),
            AssetInfoTag.tag_name == "missing",
        )
    )
    await session.flush()
    return int(res.rowcount or 0)


async def remove_missing_tag_for_asset_hash(
    session: AsyncSession,
    *,
    asset_hash: str,
) -> int:
    asset = (await session.execute(select(Asset).where(Asset.hash == asset_hash).limit(1))).scalars().first()
    if not asset:
        return 0
    return await remove_missing_tag_for_asset_id(session, asset_id=asset.id)
