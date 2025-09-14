import contextlib
import logging
import mimetypes
import os
from typing import Optional, Sequence

from comfy_api.internal import async_to_sync

from ._assets_helpers import (
    ensure_within_base,
    get_name_and_tags_from_asset_path,
    resolve_destination_from_tags,
)
from .api import schemas_in, schemas_out
from .database.db import create_session
from .database.models import Asset
from .database.services import (
    add_tags_to_asset_info,
    asset_exists_by_hash,
    asset_info_exists_for_asset_id,
    check_fs_asset_exists_quick,
    create_asset_info_for_existing_asset,
    delete_asset_info_by_id,
    fetch_asset_info_and_asset,
    fetch_asset_info_asset_and_tags,
    get_asset_by_hash,
    get_asset_info_by_id,
    get_asset_tags,
    ingest_fs_asset,
    list_asset_infos_page,
    list_cache_states_by_asset_id,
    list_tags_with_usage,
    pick_best_live_path,
    remove_tags_from_asset_info,
    set_asset_info_preview,
    touch_asset_info_by_id,
    touch_asset_infos_by_fs_path,
    update_asset_info_full,
)
from .storage import hashing


async def asset_exists(*, asset_hash: str) -> bool:
    async with await create_session() as session:
        return await asset_exists_by_hash(session, asset_hash=asset_hash)


def populate_db_with_asset(file_path: str, tags: Optional[list[str]] = None) -> None:
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
    except ValueError as e:
        logging.warning("Skipping non-asset path %s: %s", file_path, e)


async def add_local_asset(tags: list[str], file_name: str, file_path: str) -> None:
    abs_path = os.path.abspath(file_path)
    size_bytes, mtime_ns = _get_size_mtime_ns(abs_path)
    if not size_bytes:
        return

    async with await create_session() as session:
        if await check_fs_asset_exists_quick(session, file_path=abs_path, size_bytes=size_bytes, mtime_ns=mtime_ns):
            await touch_asset_infos_by_fs_path(session, file_path=abs_path)
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
    owner_id: str = "",
) -> schemas_out.AssetsList:
    sort = _safe_sort_field(sort)
    order = "desc" if (order or "desc").lower() not in {"asc", "desc"} else order.lower()

    async with await create_session() as session:
        infos, tag_map, total = await list_asset_infos_page(
            session,
            owner_id=owner_id,
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
                asset_hash=asset.hash if asset else None,
                size=int(asset.size_bytes) if asset else None,
                mime_type=asset.mime_type if asset else None,
                tags=tags,
                preview_url=f"/api/assets/{info.id}/content",
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


async def get_asset(*, asset_info_id: str, owner_id: str = "") -> schemas_out.AssetDetail:
    async with await create_session() as session:
        res = await fetch_asset_info_asset_and_tags(session, asset_info_id=asset_info_id, owner_id=owner_id)
        if not res:
            raise ValueError(f"AssetInfo {asset_info_id} not found")
        info, asset, tag_names = res
        preview_id = info.preview_id

    return schemas_out.AssetDetail(
        id=info.id,
        name=info.name,
        asset_hash=asset.hash if asset else None,
        size=int(asset.size_bytes) if asset and asset.size_bytes is not None else None,
        mime_type=asset.mime_type if asset else None,
        tags=tag_names,
        user_metadata=info.user_metadata or {},
        preview_id=preview_id,
        created_at=info.created_at,
        last_access_time=info.last_access_time,
    )


async def resolve_asset_content_for_download(
    *,
    asset_info_id: str,
    owner_id: str = "",
) -> tuple[str, str, str]:
    async with await create_session() as session:
        pair = await fetch_asset_info_and_asset(session, asset_info_id=asset_info_id, owner_id=owner_id)
        if not pair:
            raise ValueError(f"AssetInfo {asset_info_id} not found")

        info, asset = pair
        states = await list_cache_states_by_asset_id(session, asset_id=asset.id)
        abs_path = pick_best_live_path(states)
        if not abs_path:
            raise FileNotFoundError

        await touch_asset_info_by_id(session, asset_info_id=asset_info_id)
        await session.commit()

    ctype = asset.mime_type or mimetypes.guess_type(info.name or abs_path)[0] or "application/octet-stream"
    download_name = info.name or os.path.basename(abs_path)
    return abs_path, ctype, download_name


async def upload_asset_from_temp_path(
    spec: schemas_in.UploadAssetSpec,
    *,
    temp_path: str,
    client_filename: Optional[str] = None,
    owner_id: str = "",
    expected_asset_hash: Optional[str] = None,
) -> schemas_out.AssetCreated:
    try:
        digest = await hashing.blake3_hash(temp_path)
    except Exception as e:
        raise RuntimeError(f"failed to hash uploaded file: {e}")
    asset_hash = "blake3:" + digest

    if expected_asset_hash and asset_hash != expected_asset_hash.strip().lower():
        raise ValueError("HASH_MISMATCH")

    async with await create_session() as session:
        existing = await get_asset_by_hash(session, asset_hash=asset_hash)
        if existing is not None:
            with contextlib.suppress(Exception):
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

            desired_name = _safe_filename(spec.name or (client_filename or ""), fallback=digest)
            info = await create_asset_info_for_existing_asset(
                session,
                asset_hash=asset_hash,
                name=desired_name,
                user_metadata=spec.user_metadata or {},
                tags=spec.tags or [],
                tag_origin="manual",
                owner_id=owner_id,
            )
            tag_names = await get_asset_tags(session, asset_info_id=info.id)
            await session.commit()

            return schemas_out.AssetCreated(
                id=info.id,
                name=info.name,
                asset_hash=existing.hash,
                size=int(existing.size_bytes) if existing.size_bytes is not None else None,
                mime_type=existing.mime_type,
                tags=tag_names,
                user_metadata=info.user_metadata or {},
                preview_id=info.preview_id,
                created_at=info.created_at,
                last_access_time=info.last_access_time,
                created_new=False,
            )

    base_dir, subdirs = resolve_destination_from_tags(spec.tags)
    dest_dir = os.path.join(base_dir, *subdirs) if subdirs else base_dir
    os.makedirs(dest_dir, exist_ok=True)

    desired_name = _safe_filename(spec.name or (client_filename or ""), fallback=digest)
    dest_abs = os.path.abspath(os.path.join(dest_dir, desired_name))
    ensure_within_base(dest_abs, base_dir)

    content_type = mimetypes.guess_type(desired_name, strict=False)[0] or "application/octet-stream"

    try:
        os.replace(temp_path, dest_abs)
    except Exception as e:
        raise RuntimeError(f"failed to move uploaded file into place: {e}")

    try:
        size_bytes, mtime_ns = _get_size_mtime_ns(dest_abs)
    except OSError as e:
        raise RuntimeError(f"failed to stat destination file: {e}")

    async with await create_session() as session:
        result = await ingest_fs_asset(
            session,
            asset_hash=asset_hash,
            abs_path=dest_abs,
            size_bytes=size_bytes,
            mtime_ns=mtime_ns,
            mime_type=content_type,
            info_name=os.path.basename(dest_abs),
            owner_id=owner_id,
            preview_id=None,
            user_metadata=spec.user_metadata or {},
            tags=spec.tags,
            tag_origin="manual",
            require_existing_tags=False,
        )
        info_id = result["asset_info_id"]
        if not info_id:
            raise RuntimeError("failed to create asset metadata")

        pair = await fetch_asset_info_and_asset(session, asset_info_id=info_id, owner_id=owner_id)
        if not pair:
            raise RuntimeError("inconsistent DB state after ingest")
        info, asset = pair
        tag_names = await get_asset_tags(session, asset_info_id=info.id)
        await session.commit()

    return schemas_out.AssetCreated(
        id=info.id,
        name=info.name,
        asset_hash=asset.hash,
        size=int(asset.size_bytes),
        mime_type=asset.mime_type,
        tags=tag_names,
        user_metadata=info.user_metadata or {},
        preview_id=info.preview_id,
        created_at=info.created_at,
        last_access_time=info.last_access_time,
        created_new=result["asset_created"],
    )


async def update_asset(
    *,
    asset_info_id: str,
    name: Optional[str] = None,
    tags: Optional[list[str]] = None,
    user_metadata: Optional[dict] = None,
    owner_id: str = "",
) -> schemas_out.AssetUpdated:
    async with await create_session() as session:
        info_row = await get_asset_info_by_id(session, asset_info_id=asset_info_id)
        if not info_row:
            raise ValueError(f"AssetInfo {asset_info_id} not found")
        if info_row.owner_id and info_row.owner_id != owner_id:
            raise PermissionError("not owner")

        info = await update_asset_info_full(
            session,
            asset_info_id=asset_info_id,
            name=name,
            tags=tags,
            user_metadata=user_metadata,
            tag_origin="manual",
            asset_info_row=info_row,
        )

        tag_names = await get_asset_tags(session, asset_info_id=asset_info_id)
        await session.commit()

    return schemas_out.AssetUpdated(
        id=info.id,
        name=info.name,
        asset_hash=info.asset.hash if info.asset else None,
        tags=tag_names,
        user_metadata=info.user_metadata or {},
        updated_at=info.updated_at,
    )


async def set_asset_preview(
    *,
    asset_info_id: str,
    preview_asset_id: Optional[str],
    owner_id: str = "",
) -> schemas_out.AssetDetail:
    async with await create_session() as session:
        info_row = await get_asset_info_by_id(session, asset_info_id=asset_info_id)
        if not info_row:
            raise ValueError(f"AssetInfo {asset_info_id} not found")
        if info_row.owner_id and info_row.owner_id != owner_id:
            raise PermissionError("not owner")

        await set_asset_info_preview(
            session,
            asset_info_id=asset_info_id,
            preview_asset_id=preview_asset_id,
        )

        res = await fetch_asset_info_asset_and_tags(session, asset_info_id=asset_info_id, owner_id=owner_id)
        if not res:
            raise RuntimeError("State changed during preview update")
        info, asset, tags = res
        await session.commit()

    return schemas_out.AssetDetail(
        id=info.id,
        name=info.name,
        asset_hash=asset.hash if asset else None,
        size=int(asset.size_bytes) if asset and asset.size_bytes is not None else None,
        mime_type=asset.mime_type if asset else None,
        tags=tags,
        user_metadata=info.user_metadata or {},
        preview_id=info.preview_id,
        created_at=info.created_at,
        last_access_time=info.last_access_time,
    )


async def delete_asset_reference(*, asset_info_id: str, owner_id: str, delete_content_if_orphan: bool = True) -> bool:
    async with await create_session() as session:
        info_row = await get_asset_info_by_id(session, asset_info_id=asset_info_id)
        asset_id = info_row.asset_id if info_row else None
        deleted = await delete_asset_info_by_id(session, asset_info_id=asset_info_id, owner_id=owner_id)
        if not deleted:
            await session.commit()
            return False

        if not delete_content_if_orphan or not asset_id:
            await session.commit()
            return True

        still_exists = await asset_info_exists_for_asset_id(session, asset_id=asset_id)
        if still_exists:
            await session.commit()
            return True

        states = await list_cache_states_by_asset_id(session, asset_id=asset_id)
        file_paths = [s.file_path for s in (states or []) if getattr(s, "file_path", None)]

        asset_row = await session.get(Asset, asset_id)
        if asset_row is not None:
            await session.delete(asset_row)

        await session.commit()
        for p in file_paths:
            with contextlib.suppress(Exception):
                if p and os.path.isfile(p):
                    os.remove(p)
    return True


async def create_asset_from_hash(
    *,
    hash_str: str,
    name: str,
    tags: Optional[list[str]] = None,
    user_metadata: Optional[dict] = None,
    owner_id: str = "",
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
            owner_id=owner_id,
        )
        tag_names = await get_asset_tags(session, asset_info_id=info.id)
        await session.commit()

    return schemas_out.AssetCreated(
        id=info.id,
        name=info.name,
        asset_hash=asset.hash,
        size=int(asset.size_bytes),
        mime_type=asset.mime_type,
        tags=tag_names,
        user_metadata=info.user_metadata or {},
        preview_id=info.preview_id,
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
    owner_id: str = "",
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
            owner_id=owner_id,
        )

    tags = [schemas_out.TagUsage(name=name, count=count, type=tag_type) for (name, tag_type, count) in rows]
    return schemas_out.TagsList(tags=tags, total=total, has_more=(offset + len(tags)) < total)


async def add_tags_to_asset(
    *,
    asset_info_id: str,
    tags: list[str],
    origin: str = "manual",
    owner_id: str = "",
) -> schemas_out.TagsAdd:
    async with await create_session() as session:
        info_row = await get_asset_info_by_id(session, asset_info_id=asset_info_id)
        if not info_row:
            raise ValueError(f"AssetInfo {asset_info_id} not found")
        if info_row.owner_id and info_row.owner_id != owner_id:
            raise PermissionError("not owner")
        data = await add_tags_to_asset_info(
            session,
            asset_info_id=asset_info_id,
            tags=tags,
            origin=origin,
            create_if_missing=True,
            asset_info_row=info_row,
        )
        await session.commit()
    return schemas_out.TagsAdd(**data)


async def remove_tags_from_asset(
    *,
    asset_info_id: str,
    tags: list[str],
    owner_id: str = "",
) -> schemas_out.TagsRemove:
    async with await create_session() as session:
        info_row = await get_asset_info_by_id(session, asset_info_id=asset_info_id)
        if not info_row:
            raise ValueError(f"AssetInfo {asset_info_id} not found")
        if info_row.owner_id and info_row.owner_id != owner_id:
            raise PermissionError("not owner")

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


def _safe_filename(name: Optional[str], fallback: str) -> str:
    n = os.path.basename((name or "").strip() or fallback)
    if n:
        return n
    return fallback
