import os
from datetime import datetime, timezone
from typing import Optional, Sequence

from comfy.cli_args import args
from comfy_api.internal import async_to_sync

from .database.db import create_session
from .storage import hashing
from .database.services import (
    check_fs_asset_exists_quick,
    ingest_fs_asset,
    touch_asset_infos_by_fs_path,
    list_asset_infos_page,
    update_asset_info_full,
    get_asset_tags,
)


def populate_db_with_asset(tags: list[str], file_name: str, file_path: str) -> None:
    if not args.disable_model_processing:
        async_to_sync.AsyncToSyncConverter.run_async_in_thread(
            add_local_asset, tags=tags, file_name=file_name, file_path=file_path
        )


async def add_local_asset(tags: list[str], file_name: str, file_path: str) -> None:
    """Adds a local asset to the DB. If already present and unchanged, does nothing.

    Notes:
    - Uses absolute path as the canonical locator for the 'fs' backend.
    - Computes BLAKE3 only when the fast existence check indicates it's needed.
    - This function ensures the identity row and seeds mtime in asset_locator_state.
    """
    abs_path = os.path.abspath(file_path)
    size_bytes, mtime_ns = _get_size_mtime_ns(abs_path)
    if not size_bytes:
        return

    async with await create_session() as session:
        if await check_fs_asset_exists_quick(session, file_path=abs_path, size_bytes=size_bytes, mtime_ns=mtime_ns):
            await touch_asset_infos_by_fs_path(session, abs_path=abs_path, ts=datetime.now(timezone.utc))
            await session.commit()
            return

    asset_hash = hashing.blake3_hash_sync(abs_path)

    async with await create_session() as session:
        await ingest_fs_asset(
            session,
            asset_hash="blake3:" + asset_hash,
            abs_path=abs_path,
            size_bytes=size_bytes,
            mtime_ns=mtime_ns,
            mime_type=None,
            info_name=file_name,
            tag_origin="automatic",
            tags=tags,
        )
        await session.commit()


async def list_assets(
    *,
    include_tags: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    name_contains: Optional[str] = None,
    metadata_filter: Optional[dict] = None,
    limit: int = 20,
    offset: int = 0,
    sort: str | None = "created_at",
    order: str | None = "desc",
) -> dict:
    sort = _safe_sort_field(sort)
    order = "desc" if (order or "desc").lower() not in {"asc", "desc"} else order.lower()

    async with await create_session() as session:
        infos, tag_map, total = await list_asset_infos_page(
            session,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            name_contains=name_contains,
            metadata_filter=metadata_filter,
            limit=limit,
            offset=offset,
            sort=sort,
            order=order,
        )

    assets_json = []
    for info in infos:
        asset = info.asset  # populated via contains_eager
        tags = tag_map.get(info.id, [])
        assets_json.append(
            {
                "id": info.id,
                "name": info.name,
                "asset_hash": info.asset_hash,
                "size": int(asset.size_bytes) if asset else None,
                "mime_type": asset.mime_type if asset else None,
                "tags": tags,
                "preview_url": f"/api/v1/assets/{info.id}/content",  # TODO: implement actual content endpoint later
                "created_at": info.created_at.isoformat() if info.created_at else None,
                "updated_at": info.updated_at.isoformat() if info.updated_at else None,
                "last_access_time": info.last_access_time.isoformat() if info.last_access_time else None,
            }
        )

    return {
        "assets": assets_json,
        "total": total,
        "has_more": (offset + len(assets_json)) < total,
    }


async def update_asset(
    *,
    asset_info_id: int,
    name: str | None = None,
    tags: list[str] | None = None,
    user_metadata: dict | None = None,
) -> dict:
    async with await create_session() as session:
        info = await update_asset_info_full(
            session,
            asset_info_id=asset_info_id,
            name=name,
            tags=tags,
            user_metadata=user_metadata,
            tag_origin="manual",
            added_by=None,
        )

        tag_names = await get_asset_tags(session, asset_info_id=asset_info_id)
        await session.commit()

    return {
        "id": info.id,
        "name": info.name,
        "asset_hash": info.asset_hash,
        "tags": tag_names,
        "user_metadata": info.user_metadata or {},
        "updated_at": info.updated_at.isoformat() if info.updated_at else None,
    }


def _safe_sort_field(requested: str | None) -> str:
    if not requested:
        return "created_at"
    v = requested.lower()
    if v in {"name", "created_at", "updated_at", "size", "last_access_time"}:
        return v
    return "created_at"


def _get_size_mtime_ns(path: str) -> tuple[int, int]:
    st = os.stat(path, follow_symlinks=True)
    return st.st_size, getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))
