import os
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


def pick_best_live_path(states: Union[list[AssetCacheState], Sequence[AssetCacheState]]) -> str:
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
