import os
import mimetypes
import contextlib
from typing import Sequence

from app.database.db import create_session
from app.assets.api import schemas_out, schemas_in
from app.assets.database.queries import (
    asset_exists_by_hash,
    asset_info_exists_for_asset_id,
    get_asset_by_hash,
    get_asset_info_by_id,
    fetch_asset_info_asset_and_tags,
    fetch_asset_info_and_asset,
    create_asset_info_for_existing_asset,
    touch_asset_info_by_id,
    update_asset_info_full,
    delete_asset_info_by_id,
    list_cache_states_by_asset_id,
    list_asset_infos_page,
    list_tags_with_usage,
    get_asset_tags,
    add_tags_to_asset_info,
    remove_tags_from_asset_info,
    pick_best_live_path,
    ingest_fs_asset,
    set_asset_info_preview,
)
from app.assets.helpers import resolve_destination_from_tags, ensure_within_base
from app.assets.database.models import Asset


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


def _safe_filename(name: str | None, fallback: str) -> str:
    n = os.path.basename((name or "").strip() or fallback)
    if n:
        return n
    return fallback


def asset_exists(*, asset_hash: str) -> bool:
    """
    Check if an asset with a given hash exists in database.
    """
    with create_session() as session:
        return asset_exists_by_hash(session, asset_hash=asset_hash)


def list_assets(
    *,
    include_tags: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    name_contains: str | None = None,
    metadata_filter: dict | None = None,
    limit: int = 20,
    offset: int = 0,
    sort: str = "created_at",
    order: str = "desc",
    owner_id: str = "",
) -> schemas_out.AssetsList:
    sort = _safe_sort_field(sort)
    order = "desc" if (order or "desc").lower() not in {"asc", "desc"} else order.lower()

    with create_session() as session:
        infos, tag_map, total = list_asset_infos_page(
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


def get_asset(
    *,
    asset_info_id: str,
    owner_id: str = "",
) -> schemas_out.AssetDetail:
    with create_session() as session:
        res = fetch_asset_info_asset_and_tags(session, asset_info_id=asset_info_id, owner_id=owner_id)
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


def resolve_asset_content_for_download(
    *,
    asset_info_id: str,
    owner_id: str = "",
) -> tuple[str, str, str]:
    with create_session() as session:
        pair = fetch_asset_info_and_asset(session, asset_info_id=asset_info_id, owner_id=owner_id)
        if not pair:
            raise ValueError(f"AssetInfo {asset_info_id} not found")

        info, asset = pair
        states = list_cache_states_by_asset_id(session, asset_id=asset.id)
        abs_path = pick_best_live_path(states)
        if not abs_path:
            raise FileNotFoundError

        touch_asset_info_by_id(session, asset_info_id=asset_info_id)
        session.commit()

        ctype = asset.mime_type or mimetypes.guess_type(info.name or abs_path)[0] or "application/octet-stream"
        download_name = info.name or os.path.basename(abs_path)
        return abs_path, ctype, download_name


def upload_asset_from_temp_path(
    spec: schemas_in.UploadAssetSpec,
    *,
    temp_path: str,
    client_filename: str | None = None,
    owner_id: str = "",
    expected_asset_hash: str | None = None,
) -> schemas_out.AssetCreated:
    """
    Create new asset or update existing asset from a temporary file path.
    """
    try:
        # NOTE: blake3 is not required right now, so this will fail if blake3 is not installed in local environment
        import app.assets.hashing as hashing
        digest = hashing.blake3_hash(temp_path)
    except Exception as e:
        raise RuntimeError(f"failed to hash uploaded file: {e}")
    asset_hash = "blake3:" + digest

    if expected_asset_hash and asset_hash != expected_asset_hash.strip().lower():
        raise ValueError("HASH_MISMATCH")

    with create_session() as session:
        existing = get_asset_by_hash(session, asset_hash=asset_hash)
        if existing is not None:
            with contextlib.suppress(Exception):
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

            display_name = _safe_filename(spec.name or (client_filename or ""), fallback=digest)
            info = create_asset_info_for_existing_asset(
                session,
                asset_hash=asset_hash,
                name=display_name,
                user_metadata=spec.user_metadata or {},
                tags=spec.tags or [],
                tag_origin="manual",
                owner_id=owner_id,
            )
            tag_names = get_asset_tags(session, asset_info_id=info.id)
            session.commit()

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

    src_for_ext = (client_filename or spec.name or "").strip()
    _ext = os.path.splitext(os.path.basename(src_for_ext))[1] if src_for_ext else ""
    ext = _ext if 0 < len(_ext) <= 16 else ""
    hashed_basename = f"{digest}{ext}"
    dest_abs = os.path.abspath(os.path.join(dest_dir, hashed_basename))
    ensure_within_base(dest_abs, base_dir)

    content_type = (
        mimetypes.guess_type(os.path.basename(src_for_ext), strict=False)[0]
        or mimetypes.guess_type(hashed_basename, strict=False)[0]
        or "application/octet-stream"
    )

    try:
        os.replace(temp_path, dest_abs)
    except Exception as e:
        raise RuntimeError(f"failed to move uploaded file into place: {e}")

    try:
        size_bytes, mtime_ns = _get_size_mtime_ns(dest_abs)
    except OSError as e:
        raise RuntimeError(f"failed to stat destination file: {e}")

    with create_session() as session:
        result = ingest_fs_asset(
            session,
            asset_hash=asset_hash,
            abs_path=dest_abs,
            size_bytes=size_bytes,
            mtime_ns=mtime_ns,
            mime_type=content_type,
            info_name=_safe_filename(spec.name or (client_filename or ""), fallback=digest),
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

        pair = fetch_asset_info_and_asset(session, asset_info_id=info_id, owner_id=owner_id)
        if not pair:
            raise RuntimeError("inconsistent DB state after ingest")
        info, asset = pair
        tag_names = get_asset_tags(session, asset_info_id=info.id)
        created_result = schemas_out.AssetCreated(
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
        session.commit()

    return created_result


def update_asset(
    *,
    asset_info_id: str,
    name: str | None = None,
    tags: list[str] | None = None,
    user_metadata: dict | None = None,
    owner_id: str = "",
) -> schemas_out.AssetUpdated:
    with create_session() as session:
        info_row = get_asset_info_by_id(session, asset_info_id=asset_info_id)
        if not info_row:
            raise ValueError(f"AssetInfo {asset_info_id} not found")
        if info_row.owner_id and info_row.owner_id != owner_id:
            raise PermissionError("not owner")

        info = update_asset_info_full(
            session,
            asset_info_id=asset_info_id,
            name=name,
            tags=tags,
            user_metadata=user_metadata,
            tag_origin="manual",
            asset_info_row=info_row,
        )

        tag_names = get_asset_tags(session, asset_info_id=asset_info_id)
        result = schemas_out.AssetUpdated(
            id=info.id,
            name=info.name,
            asset_hash=info.asset.hash if info.asset else None,
            tags=tag_names,
            user_metadata=info.user_metadata or {},
            updated_at=info.updated_at,
        )
        session.commit()

    return result


def set_asset_preview(
    *,
    asset_info_id: str,
    preview_asset_id: str | None = None,
    owner_id: str = "",
) -> schemas_out.AssetDetail:
    with create_session() as session:
        info_row = get_asset_info_by_id(session, asset_info_id=asset_info_id)
        if not info_row:
            raise ValueError(f"AssetInfo {asset_info_id} not found")
        if info_row.owner_id and info_row.owner_id != owner_id:
            raise PermissionError("not owner")

        set_asset_info_preview(
            session,
            asset_info_id=asset_info_id,
            preview_asset_id=preview_asset_id,
        )

        res = fetch_asset_info_asset_and_tags(session, asset_info_id=asset_info_id, owner_id=owner_id)
        if not res:
            raise RuntimeError("State changed during preview update")
        info, asset, tags = res
        result = schemas_out.AssetDetail(
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
        session.commit()

    return result


def delete_asset_reference(*, asset_info_id: str, owner_id: str, delete_content_if_orphan: bool = True) -> bool:
    with create_session() as session:
        info_row = get_asset_info_by_id(session, asset_info_id=asset_info_id)
        asset_id = info_row.asset_id if info_row else None
        deleted = delete_asset_info_by_id(session, asset_info_id=asset_info_id, owner_id=owner_id)
        if not deleted:
            session.commit()
            return False

        if not delete_content_if_orphan or not asset_id:
            session.commit()
            return True

        still_exists = asset_info_exists_for_asset_id(session, asset_id=asset_id)
        if still_exists:
            session.commit()
            return True

        states = list_cache_states_by_asset_id(session, asset_id=asset_id)
        file_paths = [s.file_path for s in (states or []) if getattr(s, "file_path", None)]

        asset_row = session.get(Asset, asset_id)
        if asset_row is not None:
            session.delete(asset_row)

        session.commit()
        for p in file_paths:
            with contextlib.suppress(Exception):
                if p and os.path.isfile(p):
                    os.remove(p)
    return True


def create_asset_from_hash(
    *,
    hash_str: str,
    name: str,
    tags: list[str] | None = None,
    user_metadata: dict | None = None,
    owner_id: str = "",
) -> schemas_out.AssetCreated | None:
    canonical = hash_str.strip().lower()
    with create_session() as session:
        asset = get_asset_by_hash(session, asset_hash=canonical)
        if not asset:
            return None

        info = create_asset_info_for_existing_asset(
            session,
            asset_hash=canonical,
            name=_safe_filename(name, fallback=canonical.split(":", 1)[1]),
            user_metadata=user_metadata or {},
            tags=tags or [],
            tag_origin="manual",
            owner_id=owner_id,
        )
        tag_names = get_asset_tags(session, asset_info_id=info.id)
        result = schemas_out.AssetCreated(
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
        session.commit()

    return result


def add_tags_to_asset(
    *,
    asset_info_id: str,
    tags: list[str],
    origin: str = "manual",
    owner_id: str = "",
) -> schemas_out.TagsAdd:
    with create_session() as session:
        info_row = get_asset_info_by_id(session, asset_info_id=asset_info_id)
        if not info_row:
            raise ValueError(f"AssetInfo {asset_info_id} not found")
        if info_row.owner_id and info_row.owner_id != owner_id:
            raise PermissionError("not owner")
        data = add_tags_to_asset_info(
            session,
            asset_info_id=asset_info_id,
            tags=tags,
            origin=origin,
            create_if_missing=True,
            asset_info_row=info_row,
        )
        session.commit()
    return schemas_out.TagsAdd(**data)


def remove_tags_from_asset(
    *,
    asset_info_id: str,
    tags: list[str],
    owner_id: str = "",
) -> schemas_out.TagsRemove:
    with create_session() as session:
        info_row = get_asset_info_by_id(session, asset_info_id=asset_info_id)
        if not info_row:
            raise ValueError(f"AssetInfo {asset_info_id} not found")
        if info_row.owner_id and info_row.owner_id != owner_id:
            raise PermissionError("not owner")

        data = remove_tags_from_asset_info(
            session,
            asset_info_id=asset_info_id,
            tags=tags,
        )
        session.commit()
    return schemas_out.TagsRemove(**data)


def list_tags(
    prefix: str | None = None,
    limit: int = 100,
    offset: int = 0,
    order: str = "count_desc",
    include_zero: bool = True,
    owner_id: str = "",
) -> schemas_out.TagsList:
    limit = max(1, min(1000, limit))
    offset = max(0, offset)

    with create_session() as session:
        rows, total = list_tags_with_usage(
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
