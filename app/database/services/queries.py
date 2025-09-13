from typing import Optional, Sequence, Union

import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Asset, AssetCacheState, AssetInfo


async def asset_exists_by_hash(session: AsyncSession, *, asset_hash: str) -> bool:
    row = (
        await session.execute(
            select(sa.literal(True)).select_from(Asset).where(Asset.hash == asset_hash).limit(1)
        )
    ).first()
    return row is not None


async def get_asset_by_hash(session: AsyncSession, *, asset_hash: str) -> Optional[Asset]:
    return (
        await session.execute(select(Asset).where(Asset.hash == asset_hash).limit(1))
    ).scalars().first()


async def get_asset_info_by_id(session: AsyncSession, *, asset_info_id: str) -> Optional[AssetInfo]:
    return await session.get(AssetInfo, asset_info_id)


async def asset_info_exists_for_asset_id(session: AsyncSession, *, asset_id: str) -> bool:
    q = (
        select(sa.literal(True))
        .select_from(AssetInfo)
        .where(AssetInfo.asset_id == asset_id)
        .limit(1)
    )
    return (await session.execute(q)).first() is not None


async def get_cache_state_by_asset_id(session: AsyncSession, *, asset_id: str) -> Optional[AssetCacheState]:
    return (
        await session.execute(
            select(AssetCacheState)
            .where(AssetCacheState.asset_id == asset_id)
            .order_by(AssetCacheState.id.asc())
            .limit(1)
        )
    ).scalars().first()


async def list_cache_states_by_asset_id(
    session: AsyncSession, *, asset_id: str
) -> Union[list[AssetCacheState], Sequence[AssetCacheState]]:
    return (
        await session.execute(
            select(AssetCacheState)
            .where(AssetCacheState.asset_id == asset_id)
            .order_by(AssetCacheState.id.asc())
        )
    ).scalars().all()
