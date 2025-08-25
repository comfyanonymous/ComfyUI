import logging
import mimetypes
import os
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
    list_tags_with_usage,
    add_tags_to_asset_info,
    remove_tags_from_asset_info,
    fetch_asset_info_and_asset,
    touch_asset_info_by_id,
    delete_asset_info_by_id,
    asset_exists_by_hash,
    get_asset_by_hash,
    create_asset_info_for_existing_asset,
)
from .api import schemas_out
from ._assets_helpers import get_name_and_tags_from_asset_path


async def asset_exists(*, asset_hash: str) -> bool:
    async with await create_session() as session:
        return await asset_exists_by_hash(session, asset_hash=asset_hash)


def populate_db_with_asset(file_path: str, tags: Optional[list[str]] = None) -> None:
    if not args.disable_model_processing:
        if tags is None:
            tags = []
        try:
            asset_name, path_tags = get_name_and_tags_from_asset_path(file_path)
            async_to_sync.AsyncToSyncConverter.run_async_in_thread(
                add_local_asset,
                tags=list(dict.fromkeys([*path_tags, *tags])),
                file_name=asset_name,
                file_path=file_path,
            )
        except ValueError:
            logging.exception("Cant parse '%s' as an asset file path.", file_path)


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
            await touch_asset_infos_by_fs_path(session, abs_path=abs_path)
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
    include_tags: Optional[Sequence[str]] = None,
    exclude_tags: Optional[Sequence[str]] = None,
    name_contains: Optional[str] = None,
    metadata_filter: Optional[dict] = None,
    limit: int = 20,
    offset: int = 0,
    sort: str = "created_at",
    order: str = "desc",
) -> schemas_out.AssetsList:
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

    summaries: list[schemas_out.AssetSummary] = []
    for info in infos:
        asset = info.asset
        tags = tag_map.get(info.id, [])
        summaries.append(
            schemas_out.AssetSummary(
                id=info.id,
                name=info.name,
                asset_hash=info.asset_hash,
                size=int(asset.size_bytes) if asset else None,
                mime_type=asset.mime_type if asset else None,
                tags=tags,
                preview_url=f"/api/v1/assets/{info.id}/content",
                created_at=info.created_at,
                updated_at=info.updated_at,
                last_access_time=info.last_access_time,
            )
        )

    return schemas_out.AssetsList(
        assets=summaries,
        total=total,
        has_more=(offset + len(summaries)) < total,
    )


async def resolve_asset_content_for_download(
    *, asset_info_id: int
) -> tuple[str, str, str]:
    """
    Returns (abs_path, content_type, download_name) for the given AssetInfo id.
    Also touches last_access_time (only_if_newer).
    Raises:
      ValueError if AssetInfo not found
      NotImplementedError for unsupported backend
      FileNotFoundError if underlying file does not exist (fs backend)
    """
    async with await create_session() as session:
        pair = await fetch_asset_info_and_asset(session, asset_info_id=asset_info_id)
        if not pair:
            raise ValueError(f"AssetInfo {asset_info_id} not found")

        info, asset = pair

        if asset.storage_backend != "fs":
            # Future: support http/s3/gcs/...
            raise NotImplementedError(f"backend {asset.storage_backend!r} not supported yet")

        abs_path = os.path.abspath(asset.storage_locator)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(abs_path)

        await touch_asset_info_by_id(session, asset_info_id=asset_info_id)
        await session.commit()

        ctype = asset.mime_type or mimetypes.guess_type(info.name or abs_path)[0] or "application/octet-stream"
        download_name = info.name or os.path.basename(abs_path)
        return abs_path, ctype, download_name


async def update_asset(
    *,
    asset_info_id: int,
    name: Optional[str] = None,
    tags: Optional[list[str]] = None,
    user_metadata: Optional[dict] = None,
) -> schemas_out.AssetUpdated:
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

    return schemas_out.AssetUpdated(
        id=info.id,
        name=info.name,
        asset_hash=info.asset_hash,
        tags=tag_names,
        user_metadata=info.user_metadata or {},
        updated_at=info.updated_at,
    )


async def delete_asset_reference(*, asset_info_id: int) -> bool:
    async with await create_session() as session:
        r = await delete_asset_info_by_id(session, asset_info_id=asset_info_id)
        await session.commit()
    return r


async def create_asset_from_hash(
    *,
    hash_str: str,
    name: str,
    tags: Optional[list[str]] = None,
    user_metadata: Optional[dict] = None,
) -> Optional[schemas_out.AssetCreated]:
    canonical = hash_str.strip().lower()
    async with await create_session() as session:
        asset = await get_asset_by_hash(session, asset_hash=canonical)
        if not asset:
            return None

        info = await create_asset_info_for_existing_asset(
            session,
            asset_hash=canonical,
            name=_safe_filename(name, fallback=canonical.split(":", 1)[1]),
            user_metadata=user_metadata or {},
            tags=tags or [],
            tag_origin="manual",
            added_by=None,
        )
        tag_names = await get_asset_tags(session, asset_info_id=info.id)
        await session.commit()

    return schemas_out.AssetCreated(
        id=info.id,
        name=info.name,
        asset_hash=info.asset_hash,
        size=int(asset.size_bytes),
        mime_type=asset.mime_type,
        tags=tag_names,
        user_metadata=info.user_metadata or {},
        preview_hash=info.preview_hash,
        created_at=info.created_at,
        last_access_time=info.last_access_time,
        created_new=False,
    )


async def list_tags(
    *,
    prefix: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    order: str = "count_desc",
    include_zero: bool = True,
) -> schemas_out.TagsList:
    limit = max(1, min(1000, limit))
    offset = max(0, offset)

    async with await create_session() as session:
        rows, total = await list_tags_with_usage(
            session,
            prefix=prefix,
            limit=limit,
            offset=offset,
            include_zero=include_zero,
            order=order,
        )

    tags = [schemas_out.TagUsage(name=name, count=count, type=tag_type) for (name, tag_type, count) in rows]
    return schemas_out.TagsList(tags=tags, total=total, has_more=(offset + len(tags)) < total)


async def add_tags_to_asset(
    *,
    asset_info_id: int,
    tags: list[str],
    origin: str = "manual",
    added_by: Optional[str] = None,
) -> schemas_out.TagsAdd:
    async with await create_session() as session:
        data = await add_tags_to_asset_info(
            session,
            asset_info_id=asset_info_id,
            tags=tags,
            origin=origin,
            added_by=added_by,
            create_if_missing=True,
        )
        await session.commit()
    return schemas_out.TagsAdd(**data)


async def remove_tags_from_asset(
    *,
    asset_info_id: int,
    tags: list[str],
) -> schemas_out.TagsRemove:
    async with await create_session() as session:
        data = await remove_tags_from_asset_info(
            session,
            asset_info_id=asset_info_id,
            tags=tags,
        )
        await session.commit()
    return schemas_out.TagsRemove(**data)


def _safe_sort_field(requested: Optional[str]) -> str:
    if not requested:
        return "created_at"
    v = requested.lower()
    if v in {"name", "created_at", "updated_at", "size", "last_access_time"}:
        return v
    return "created_at"


def _get_size_mtime_ns(path: str) -> tuple[int, int]:
    st = os.stat(path, follow_symlinks=True)
    return st.st_size, getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))


def _safe_filename(name: Optional[str] , fallback: str) -> str:
    n = os.path.basename((name or "").strip() or fallback)
    if n:
        return n
    return fallback
