import contextlib
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
    fetch_asset_info_asset_and_tags,
    get_asset_info_by_id,
)
from .api import schemas_in, schemas_out
from ._assets_helpers import (
    get_name_and_tags_from_asset_path,
    ensure_within_base,
    resolve_destination_from_tags,
)
from .assets_fetcher import ensure_asset_cached


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
        except ValueError as e:
            logging.warning("Skipping non-asset path %s: %s", file_path, e)


async def add_local_asset(tags: list[str], file_name: str, file_path: str) -> None:
    """Adds a local asset to the DB. If already present and unchanged, does nothing.

    Notes:
    - Uses absolute path as the canonical locator for the cache backend.
    - Computes BLAKE3 only when the fast existence check indicates it's needed.
    - This function ensures the identity row and seeds mtime in asset_cache_state.
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
                asset_hash=info.asset_hash,
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


async def get_asset(*, asset_info_id: int, owner_id: str = "") -> schemas_out.AssetDetail:
    async with await create_session() as session:
        res = await fetch_asset_info_asset_and_tags(session, asset_info_id=asset_info_id, owner_id=owner_id)
        if not res:
            raise ValueError(f"AssetInfo {asset_info_id} not found")
        info, asset, tag_names = res

    return schemas_out.AssetDetail(
        id=info.id,
        name=info.name,
        asset_hash=info.asset_hash,
        size=int(asset.size_bytes) if asset and asset.size_bytes is not None else None,
        mime_type=asset.mime_type if asset else None,
        tags=tag_names,
        preview_hash=info.preview_hash,
        user_metadata=info.user_metadata or {},
        created_at=info.created_at,
        last_access_time=info.last_access_time,
    )


async def resolve_asset_content_for_download(
    *,
    asset_info_id: int,
    owner_id: str = "",
) -> tuple[str, str, str]:
    """
    Returns (abs_path, content_type, download_name) for the given AssetInfo id and touches last_access_time.
    Also touches last_access_time (only_if_newer).
    Ensures the local cache is present (uses resolver if needed).
    Raises:
      ValueError if AssetInfo cannot be found
    """
    async with await create_session() as session:
        pair = await fetch_asset_info_and_asset(session, asset_info_id=asset_info_id, owner_id=owner_id)
        if not pair:
            raise ValueError(f"AssetInfo {asset_info_id} not found")

        info, asset = pair
        tag_names = await get_asset_tags(session, asset_info_id=info.id)

    # Ensure cached (download if missing)
    preferred_name = info.name or info.asset_hash.split(":", 1)[-1]
    abs_path = await ensure_asset_cached(info.asset_hash, preferred_name=preferred_name, tags_hint=tag_names)

    async with await create_session() as session:
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
    """
    Finalize an uploaded temp file:
      - compute blake3 hash
      - if expected_asset_hash provided, verify equality (400 on mismatch at caller)
      - if an Asset with the same hash exists: discard temp, create AssetInfo only (no write)
      - else resolve destination from tags and atomically move into place
      - ingest into DB (assets, locator state, asset_info + tags)
    Returns a populated AssetCreated payload.
    """

    try:
        digest = await hashing.blake3_hash(temp_path)
    except Exception as e:
        raise RuntimeError(f"failed to hash uploaded file: {e}")
    asset_hash = "blake3:" + digest

    if expected_asset_hash and asset_hash != expected_asset_hash.strip().lower():
        raise ValueError("HASH_MISMATCH")

    # Fast path: content already known --> no writes, just create a reference
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
                asset_hash=info.asset_hash,
                size=int(existing.size_bytes) if existing.size_bytes is not None else None,
                mime_type=existing.mime_type,
                tags=tag_names,
                user_metadata=info.user_metadata or {},
                preview_hash=info.preview_hash,
                created_at=info.created_at,
                last_access_time=info.last_access_time,
                created_new=False,
            )

    # Resolve destination (only for truly new content)
    base_dir, subdirs = resolve_destination_from_tags(spec.tags)
    dest_dir = os.path.join(base_dir, *subdirs) if subdirs else base_dir
    os.makedirs(dest_dir, exist_ok=True)

    # Decide filename
    desired_name = _safe_filename(spec.name or (client_filename or ""), fallback=digest)
    dest_abs = os.path.abspath(os.path.join(dest_dir, desired_name))
    ensure_within_base(dest_abs, base_dir)

    # Content type based on final name
    content_type = mimetypes.guess_type(desired_name, strict=False)[0] or "application/octet-stream"

    # Atomic move into place
    try:
        os.replace(temp_path, dest_abs)
    except Exception as e:
        raise RuntimeError(f"failed to move uploaded file into place: {e}")

    # Stat final file
    try:
        size_bytes, mtime_ns = _get_size_mtime_ns(dest_abs)
    except OSError as e:
        raise RuntimeError(f"failed to stat destination file: {e}")

    # Ingest + build response
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
            preview_hash=None,
            user_metadata=spec.user_metadata or {},
            tags=spec.tags,
            tag_origin="manual",
            require_existing_tags=False,
        )
        info_id = result["asset_info_id"]
        if not info_id:
            raise RuntimeError("failed to create asset metadata")

        pair = await fetch_asset_info_and_asset(session, asset_info_id=int(info_id), owner_id=owner_id)
        if not pair:
            raise RuntimeError("inconsistent DB state after ingest")
        info, asset = pair
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
        created_new=result["asset_created"],
    )


async def update_asset(
    *,
    asset_info_id: int,
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
        asset_hash=info.asset_hash,
        tags=tag_names,
        user_metadata=info.user_metadata or {},
        updated_at=info.updated_at,
    )


async def delete_asset_reference(*, asset_info_id: int, owner_id: str) -> bool:
    async with await create_session() as session:
        r = await delete_asset_info_by_id(session, asset_info_id=asset_info_id, owner_id=owner_id)
        await session.commit()
    return r


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
    asset_info_id: int,
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
    asset_info_id: int,
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
