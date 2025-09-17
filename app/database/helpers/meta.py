from sqlalchemy.dialects import postgresql as d_pg
from sqlalchemy.dialects import sqlite as d_sqlite
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import AssetInfoMeta


async def insert_meta_from_batch(session: AsyncSession, *, rows: list[dict]) -> None:
    """Bulk insert rows into asset_info_meta with ON CONFLICT DO NOTHING.
    Each row should contain: asset_info_id, key, ordinal, val_str, val_num, val_bool, val_json
    """
    if session.bind.dialect.name == "sqlite":
        ins = (
            d_sqlite.insert(AssetInfoMeta)
            .values(rows)
            .on_conflict_do_nothing(
                index_elements=[AssetInfoMeta.asset_info_id, AssetInfoMeta.key, AssetInfoMeta.ordinal]
            )
        )
    elif session.bind.dialect.name == "postgresql":
        ins = (
            d_pg.insert(AssetInfoMeta)
            .values(rows)
            .on_conflict_do_nothing(
                index_elements=[AssetInfoMeta.asset_info_id, AssetInfoMeta.key, AssetInfoMeta.ordinal]
            )
        )
    else:
        raise NotImplementedError(f"Unsupported database dialect: {session.bind.dialect.name}")
    await session.execute(ins)
