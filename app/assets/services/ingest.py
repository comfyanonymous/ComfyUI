import contextlib
import logging
import mimetypes
import os
from typing import Any, Sequence

from sqlalchemy.orm import Session

import app.assets.services.hashing as hashing
from app.assets.database.queries import (
    add_tags_to_reference,
    fetch_reference_and_asset,
    get_asset_by_hash,
    get_existing_asset_ids,
    get_reference_by_file_path,
    get_reference_tags,
    get_or_create_reference,
    remove_missing_tag_for_asset_id,
    set_reference_metadata,
    set_reference_tags,
    upsert_asset,
    upsert_reference,
    validate_tags_exist,
)
from app.assets.helpers import normalize_tags
from app.assets.services.file_utils import get_size_and_mtime_ns
from app.assets.services.path_utils import (
    compute_relative_filename,
    resolve_destination_from_tags,
    validate_path_within_base,
)
from app.assets.services.schemas import (
    IngestResult,
    RegisterAssetResult,
    UploadResult,
    UserMetadata,
    extract_asset_data,
    extract_reference_data,
)
from app.database.db import create_session


def _ingest_file_from_path(
    abs_path: str,
    asset_hash: str,
    size_bytes: int,
    mtime_ns: int,
    mime_type: str | None = None,
    info_name: str | None = None,
    owner_id: str = "",
    preview_id: str | None = None,
    user_metadata: UserMetadata = None,
    tags: Sequence[str] = (),
    tag_origin: str = "manual",
    require_existing_tags: bool = False,
) -> IngestResult:
    locator = os.path.abspath(abs_path)
    user_metadata = user_metadata or {}

    asset_created = False
    asset_updated = False
    ref_created = False
    ref_updated = False
    reference_id: str | None = None

    with create_session() as session:
        if preview_id:
            if preview_id not in get_existing_asset_ids(session, [preview_id]):
                preview_id = None

        asset, asset_created, asset_updated = upsert_asset(
            session,
            asset_hash=asset_hash,
            size_bytes=size_bytes,
            mime_type=mime_type,
        )

        ref_created, ref_updated = upsert_reference(
            session,
            asset_id=asset.id,
            file_path=locator,
            name=info_name or os.path.basename(locator),
            mtime_ns=mtime_ns,
            owner_id=owner_id,
        )

        # Get the reference we just created/updated
        ref = get_reference_by_file_path(session, locator)
        if ref:
            reference_id = ref.id

            if preview_id and ref.preview_id != preview_id:
                ref.preview_id = preview_id

            norm = normalize_tags(list(tags))
            if norm:
                if require_existing_tags:
                    validate_tags_exist(session, norm)
                add_tags_to_reference(
                    session,
                    reference_id=reference_id,
                    tags=norm,
                    origin=tag_origin,
                    create_if_missing=not require_existing_tags,
                )

            _update_metadata_with_filename(
                session,
                reference_id=reference_id,
                file_path=ref.file_path,
                current_metadata=ref.user_metadata,
                user_metadata=user_metadata,
            )

        try:
            remove_missing_tag_for_asset_id(session, asset_id=asset.id)
        except Exception:
            logging.exception("Failed to clear 'missing' tag for asset %s", asset.id)

        session.commit()

    return IngestResult(
        asset_created=asset_created,
        asset_updated=asset_updated,
        ref_created=ref_created,
        ref_updated=ref_updated,
        reference_id=reference_id,
    )


def _register_existing_asset(
    asset_hash: str,
    name: str,
    user_metadata: UserMetadata = None,
    tags: list[str] | None = None,
    tag_origin: str = "manual",
    owner_id: str = "",
) -> RegisterAssetResult:
    user_metadata = user_metadata or {}

    with create_session() as session:
        asset = get_asset_by_hash(session, asset_hash=asset_hash)
        if not asset:
            raise ValueError(f"No asset with hash {asset_hash}")

        ref, ref_created = get_or_create_reference(
            session,
            asset_id=asset.id,
            owner_id=owner_id,
            name=name,
        )

        if not ref_created:
            tag_names = get_reference_tags(session, reference_id=ref.id)
            result = RegisterAssetResult(
                ref=extract_reference_data(ref),
                asset=extract_asset_data(asset),
                tags=tag_names,
                created=False,
            )
            session.commit()
            return result

        new_meta = dict(user_metadata)
        computed_filename = compute_relative_filename(ref.file_path) if ref.file_path else None
        if computed_filename:
            new_meta["filename"] = computed_filename

        if new_meta:
            set_reference_metadata(
                session,
                reference_id=ref.id,
                user_metadata=new_meta,
            )

        if tags is not None:
            set_reference_tags(
                session,
                reference_id=ref.id,
                tags=tags,
                origin=tag_origin,
            )

        tag_names = get_reference_tags(session, reference_id=ref.id)
        session.refresh(ref)
        result = RegisterAssetResult(
            ref=extract_reference_data(ref),
            asset=extract_asset_data(asset),
            tags=tag_names,
            created=True,
        )
        session.commit()

        return result



def _update_metadata_with_filename(
    session: Session,
    reference_id: str,
    file_path: str | None,
    current_metadata: dict | None,
    user_metadata: dict[str, Any],
) -> None:
    computed_filename = compute_relative_filename(file_path) if file_path else None

    current_meta = current_metadata or {}
    new_meta = dict(current_meta)
    for k, v in user_metadata.items():
        new_meta[k] = v
    if computed_filename:
        new_meta["filename"] = computed_filename

    if new_meta != current_meta:
        set_reference_metadata(
            session,
            reference_id=reference_id,
            user_metadata=new_meta,
        )


def _sanitize_filename(name: str | None, fallback: str) -> str:
    n = os.path.basename((name or "").strip() or fallback)
    return n if n else fallback


class HashMismatchError(Exception):
    pass


class DependencyMissingError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


def upload_from_temp_path(
    temp_path: str,
    name: str | None = None,
    tags: list[str] | None = None,
    user_metadata: dict | None = None,
    client_filename: str | None = None,
    owner_id: str = "",
    expected_hash: str | None = None,
) -> UploadResult:
    try:
        digest, _ = hashing.compute_blake3_hash(temp_path)
    except ImportError as e:
        raise DependencyMissingError(str(e))
    except Exception as e:
        raise RuntimeError(f"failed to hash uploaded file: {e}")
    asset_hash = "blake3:" + digest

    if expected_hash and asset_hash != expected_hash.strip().lower():
        raise HashMismatchError("Uploaded file hash does not match provided hash.")

    with create_session() as session:
        existing = get_asset_by_hash(session, asset_hash=asset_hash)

    if existing is not None:
        with contextlib.suppress(Exception):
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        display_name = _sanitize_filename(name or client_filename, fallback=digest)
        result = _register_existing_asset(
            asset_hash=asset_hash,
            name=display_name,
            user_metadata=user_metadata or {},
            tags=tags or [],
            tag_origin="manual",
            owner_id=owner_id,
        )
        return UploadResult(
            ref=result.ref,
            asset=result.asset,
            tags=result.tags,
            created_new=False,
        )

    if not tags:
        raise ValueError("tags are required for new asset uploads")
    base_dir, subdirs = resolve_destination_from_tags(tags)
    dest_dir = os.path.join(base_dir, *subdirs) if subdirs else base_dir
    os.makedirs(dest_dir, exist_ok=True)

    src_for_ext = (client_filename or name or "").strip()
    _ext = os.path.splitext(os.path.basename(src_for_ext))[1] if src_for_ext else ""
    ext = _ext if 0 < len(_ext) <= 16 else ""
    hashed_basename = f"{digest}{ext}"
    dest_abs = os.path.abspath(os.path.join(dest_dir, hashed_basename))
    validate_path_within_base(dest_abs, base_dir)

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
        size_bytes, mtime_ns = get_size_and_mtime_ns(dest_abs)
    except OSError as e:
        raise RuntimeError(f"failed to stat destination file: {e}")

    ingest_result = _ingest_file_from_path(
        asset_hash=asset_hash,
        abs_path=dest_abs,
        size_bytes=size_bytes,
        mtime_ns=mtime_ns,
        mime_type=content_type,
        info_name=_sanitize_filename(name or client_filename, fallback=digest),
        owner_id=owner_id,
        preview_id=None,
        user_metadata=user_metadata or {},
        tags=tags,
        tag_origin="manual",
        require_existing_tags=False,
    )
    reference_id = ingest_result.reference_id
    if not reference_id:
        raise RuntimeError("failed to create asset reference")

    with create_session() as session:
        pair = fetch_reference_and_asset(
            session, reference_id=reference_id, owner_id=owner_id
        )
        if not pair:
            raise RuntimeError("inconsistent DB state after ingest")
        ref, asset = pair
        tag_names = get_reference_tags(session, reference_id=ref.id)

    return UploadResult(
        ref=extract_reference_data(ref),
        asset=extract_asset_data(asset),
        tags=tag_names,
        created_new=ingest_result.asset_created,
    )


def create_from_hash(
    hash_str: str,
    name: str,
    tags: list[str] | None = None,
    user_metadata: dict | None = None,
    owner_id: str = "",
) -> UploadResult | None:
    canonical = hash_str.strip().lower()

    with create_session() as session:
        asset = get_asset_by_hash(session, asset_hash=canonical)
        if not asset:
            return None

    result = _register_existing_asset(
        asset_hash=canonical,
        name=_sanitize_filename(
            name, fallback=canonical.split(":", 1)[1] if ":" in canonical else canonical
        ),
        user_metadata=user_metadata or {},
        tags=tags or [],
        tag_origin="manual",
        owner_id=owner_id,
    )

    return UploadResult(
        ref=result.ref,
        asset=result.asset,
        tags=result.tags,
        created_new=False,
    )
