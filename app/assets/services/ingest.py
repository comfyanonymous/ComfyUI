import contextlib
import logging
import mimetypes
import os
from typing import Any, Sequence

from sqlalchemy.orm import Session

from app.assets.database.models import Asset
from app.assets.database.queries import (
    add_tags_to_reference,
    count_active_siblings,
    create_stub_asset,
    ensure_tags_exist,
    get_asset_by_hash,
    get_reference_by_file_path,
    get_reference_tags,
    get_or_create_reference,
    list_references_by_asset_id,
    reference_exists,
    remove_missing_tag_for_asset_id,
    set_reference_metadata,
    set_reference_system_metadata,
    set_reference_tags,
    update_asset_hash_and_mime,
    upsert_asset,
    upsert_reference as _legacy_upsert_reference,  # wave-3-fixes: replaced by B-schema write paths in Wave 3
    validate_tags_exist,
)
from app.assets.helpers import get_utc_now, normalize_tags, to_stored_hash
from app.assets.services.bulk_ingest import batch_insert_seed_assets
from app.assets.services.file_utils import get_mtime_ns, get_size_and_mtime_ns
from app.assets.services.image_dimensions import extract_image_dimensions
from app.assets.services.metadata_extract import extract_file_metadata
from app.assets.services.path_utils import (
    compute_loader_path,
    get_name_and_tags_from_asset_path,
    get_path_derived_tags_from_path,
    resolve_destination_from_tags,
    validate_path_within_base,
)
from app.assets.services.schemas import (
    AssetData,
    IngestResult,
    ReferenceData,
    RegisterAssetResult,
    UploadResult,
    UserMetadata,
    extract_asset_data,
    extract_reference_data,
)
from app.database.db import create_session


def _normalize_hash_input(hash_str: str) -> str:
    """Canonicalise a client-supplied hash to stored form."""
    if not hash_str:
        return hash_str
    normalized = hash_str.strip().lower()
    digest = normalized.partition(":")[2] or normalized
    return to_stored_hash(digest)


def _extract_system_metadata_sync(
    locator: str,
    mime_type: str | None,
    stat_result: os.stat_result | None = None,
) -> dict[str, Any]:
    """Extract ``system_metadata`` at registration time (S29/D8).

    Mirrors the ``scanner.enrich`` pass: tier-1/tier-2 file metadata plus image
    dimensions for image MIME types, so records carry metadata at creation
    instead of waiting for the background enrich pass to fill it.
    """
    metadata = extract_file_metadata(
        locator,
        stat_result=stat_result,
        relative_filename=compute_loader_path(locator),
    )
    system_metadata = metadata.to_user_metadata()
    if mime_type and mime_type.startswith("image/"):
        dims = extract_image_dimensions(locator, mime_type=mime_type)
        if dims:
            system_metadata.update(dims)
    return system_metadata


def _discard_unreferenced_content(session: Session, content_id: str) -> None:
    """Remove a content row left orphaned by a failed registration.

    ``create_content`` inserts inside a SAVEPOINT (``begin_nested``); under
    pysqlite that insert survives the enclosing ``rollback`` because pysqlite
    has no real nested transaction. When the follow-on ``create_record`` fails
    we would otherwise leak an unreferenced content row, so delete it explicitly
    once we have confirmed no record points at it. Best-effort: cleanup errors
    are logged and swallowed so the original failure is what surfaces.
    """
    from sqlalchemy import func, select

    from app.assets.database.models import Asset, AssetContent

    try:
        ref_count = session.scalar(
            select(func.count())
            .select_from(Asset)
            .where(Asset.content_id == content_id)
        )
        if ref_count:
            return
        content = session.get(AssetContent, content_id)
        if content is not None:
            session.delete(content)
            session.commit()
    except Exception:
        logging.exception("Failed to discard orphan content %s", content_id)


def _ingest_file_from_path(
    abs_path: str,
    asset_hash: str,
    size_bytes: int,
    mtime_ns: int,
    mime_type: str | None = None,
    info_name: str | None = None,
    tenant_id: str = "",
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
            if not reference_exists(session, preview_id):
                preview_id = None

        asset, asset_created, asset_updated = upsert_asset(
            session,
            asset_hash=asset_hash,
            size_bytes=size_bytes,
            mime_type=mime_type,
        )

        ref_created, ref_updated = _legacy_upsert_reference(  # wave-3-fixes: replaced in Wave 3
            session,
            asset_id=asset.id,
            file_path=locator,
            name=info_name or os.path.basename(locator),
            mtime_ns=mtime_ns,
            tenant_id=tenant_id,
            loader_path=compute_loader_path(locator),
        )

        # Get the reference we just created/updated
        ref = get_reference_by_file_path(session, locator)
        if ref:
            reference_id = ref.id

            if preview_id and ref.preview_id != preview_id:
                ref.preview_id = preview_id

            try:
                backend_tags = get_path_derived_tags_from_path(locator)
            except ValueError:
                backend_tags = []
            caller_tags = normalize_tags(tags)
            backend_tags = normalize_tags(backend_tags)
            all_tags = normalize_tags([*caller_tags, *backend_tags])
            if all_tags:
                if require_existing_tags:
                    validate_tags_exist(session, all_tags)
                if backend_tags:
                    add_tags_to_reference(
                        session,
                        reference_id=reference_id,
                        tags=backend_tags,
                        origin="automatic",
                        create_if_missing=not require_existing_tags,
                    )
                if caller_tags:
                    add_tags_to_reference(
                        session,
                        reference_id=reference_id,
                        tags=caller_tags,
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

            _maybe_store_image_dimensions(
                session,
                reference_id=reference_id,
                file_path=locator,
                mime_type=mime_type,
                current_system_metadata=ref.system_metadata,
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


def ingest_existing_file(
    abs_path: str,
    user_metadata: UserMetadata = None,
    extra_tags: Sequence[str] = (),
    tenant_id: str = "",
    job_id: str | None = None,
) -> bool:
    """Register an existing on-disk file as an asset stub.

    If a reference already exists for this path, updates mtime_ns, job_id,
    size_bytes, and resets enrichment so the enricher will re-hash it.

    For brand-new paths, inserts a stub record (hash=NULL) for immediate
    UX visibility.

    Returns True if a row was inserted or updated, False otherwise.
    """
    locator = os.path.abspath(abs_path)
    size_bytes, mtime_ns = get_size_and_mtime_ns(abs_path)
    mime_type = mimetypes.guess_type(abs_path, strict=False)[0]
    name, path_tags = get_name_and_tags_from_asset_path(abs_path)
    tags = list(dict.fromkeys(path_tags + list(extra_tags)))

    with create_session() as session:
        existing_ref = get_reference_by_file_path(session, locator)
        if existing_ref is not None:
            now = get_utc_now()
            existing_ref.mtime_ns = mtime_ns
            existing_ref.job_id = job_id
            existing_ref.is_missing = False
            existing_ref.updated_at = now
            existing_ref.hash_state = 0

            asset = existing_ref.asset
            if asset:
                # If other refs share this asset, detach to a new stub
                # instead of mutating the shared row.
                siblings = count_active_siblings(session, asset.id, existing_ref.id)
                if siblings > 0:
                    new_asset = create_stub_asset(
                        session,
                        size_bytes=size_bytes,
                        mime_type=mime_type or asset.mime_type,
                    )
                    existing_ref.asset_id = new_asset.id
                else:
                    asset.hash = None
                    asset.size_bytes = size_bytes
                    if mime_type:
                        asset.mime_type = mime_type
            session.commit()
            return True

        spec = {
            "abs_path": abs_path,
            "size_bytes": size_bytes,
            "mtime_ns": mtime_ns,
            "info_name": name,
            "tags": tags,
            "fname": compute_loader_path(abs_path),
            "metadata": None,
            "hash": None,
            "mime_type": mime_type,
            "job_id": job_id,
        }
        if tags:
            ensure_tags_exist(session, tags)
        result = batch_insert_seed_assets(session, [spec], tenant_id=tenant_id)
        session.commit()
        return result.won_paths > 0


def _register_existing_asset(
    asset_hash: str,
    name: str,
    user_metadata: UserMetadata = None,
    tags: list[str] | None = None,
    tag_origin: str = "manual",
    tenant_id: str = "",
    mime_type: str | None = None,
    preview_id: str | None = None,
) -> RegisterAssetResult:
    user_metadata = user_metadata or {}

    with create_session() as session:
        asset = get_asset_by_hash(session, asset_hash=asset_hash)
        if not asset:
            raise ValueError(f"No asset with hash {asset_hash}")

        if mime_type and not asset.mime_type:
            update_asset_hash_and_mime(session, asset_id=asset.id, mime_type=mime_type)

        if preview_id:
            if not reference_exists(session, preview_id):
                preview_id = None

        ref, ref_created = get_or_create_reference(
            session,
            asset_id=asset.id,
            tenant_id=tenant_id,
            name=name,
            preview_id=preview_id,
        )

        if not ref_created:
            if preview_id and ref.preview_id != preview_id:
                ref.preview_id = preview_id

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
        computed_filename = compute_loader_path(ref.file_path) if ref.file_path else None
        if computed_filename:
            new_meta["filename"] = computed_filename

        if new_meta:
            set_reference_metadata(
                session,
                reference_id=ref.id,
                user_metadata=new_meta,
            )

        _backfill_image_dimensions_from_siblings(
            session,
            asset_id=asset.id,
            new_reference_id=ref.id,
            current_system_metadata=ref.system_metadata,
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
    computed_filename = compute_loader_path(file_path) if file_path else None

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


_IMAGE_DIMENSION_KEYS = ("kind", "width", "height")


def _maybe_store_image_dimensions(
    session: Session,
    reference_id: str,
    file_path: str,
    mime_type: str | None,
    current_system_metadata: dict | None,
) -> None:
    """Populate ``kind``/``width``/``height`` on system_metadata for image refs.

    Non-image MIME types are a no-op. Pre-existing keys (e.g. enricher-written
    safetensors metadata, download provenance) are preserved by merge.
    """
    if not mime_type or not mime_type.startswith("image/"):
        return

    dims = extract_image_dimensions(file_path, mime_type=mime_type)
    if not dims:
        return

    current = current_system_metadata or {}
    merged = dict(current)
    merged.update(dims)
    if merged != current:
        set_reference_system_metadata(
            session,
            reference_id=reference_id,
            system_metadata=merged,
        )


def _backfill_image_dimensions_from_siblings(
    session: Session,
    asset_id: str,
    new_reference_id: str,
    current_system_metadata: dict | None,
) -> None:
    """Copy image dimension keys from any sibling reference of the same asset.

    The from-hash path doesn't read the file bytes, so dimensions can't be
    extracted there directly. When another reference of the same asset already
    carries image dimensions, copy them onto the new reference so consumers
    see consistent metadata regardless of how the asset was registered.

    Best-effort: missing siblings, non-image siblings, or absent dimension
    keys leave the target reference unchanged.
    """
    current = current_system_metadata or {}
    if current.get("kind") == "image" and "width" in current and "height" in current:
        return

    for sibling in list_references_by_asset_id(session, asset_id):
        if sibling.id == new_reference_id:
            continue
        meta = sibling.system_metadata or {}
        if meta.get("kind") != "image":
            continue
        width = meta.get("width")
        height = meta.get("height")
        if (
            type(width) is not int
            or type(height) is not int
            or width <= 0
            or height <= 0
        ):
            continue
        merged = dict(current)
        merged["kind"] = "image"
        merged["width"] = width
        merged["height"] = height
        if merged != current:
            set_reference_system_metadata(
                session,
                reference_id=new_reference_id,
                system_metadata=merged,
            )
        return


def _sanitize_filename(name: str | None, fallback: str) -> str:
    n = os.path.basename((name or "").strip() or fallback)
    return n if n else fallback


class HashMismatchError(Exception):
    pass


class UploadUnstableError(Exception):
    pass


class DependencyMissingError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


_UPLOAD_HASH_ATTEMPTS = 3


def _remove_temp_path(temp_path: str | None) -> None:
    if not temp_path or not os.path.exists(temp_path):
        return
    with contextlib.suppress(OSError):
        os.remove(temp_path)
    parent = os.path.dirname(temp_path)
    with contextlib.suppress(OSError):
        if parent and os.path.isdir(parent):
            os.rmdir(parent)


def _snapshot_hash_with_retry(path: str) -> str:
    from app.assets.services.snapshot_hash import snapshot_hash

    for _ in range(_UPLOAD_HASH_ATTEMPTS):
        digest = snapshot_hash(path)
        if digest is not None:
            return digest
    raise UploadUnstableError("upload file changed during hashing")


def _hash_mode_dest_path(
    tags: list[str],
    digest: str,
    client_filename: str | None,
    name: str | None,
) -> str:
    base_dir, subdirs = resolve_destination_from_tags(tags)
    dest_dir = os.path.join(base_dir, *subdirs) if subdirs else base_dir
    os.makedirs(dest_dir, exist_ok=True)
    src_for_ext = (client_filename or name or "").strip()
    _ext = os.path.splitext(os.path.basename(src_for_ext))[1] if src_for_ext else ""
    ext = _ext if 0 < len(_ext) <= 16 else ""
    hashed_basename = f"{digest}{ext}"
    dest_abs = os.path.abspath(os.path.join(dest_dir, hashed_basename))
    validate_path_within_base(dest_abs, base_dir)
    return dest_abs


def _guess_upload_mime_type(
    mime_type: str | None,
    client_filename: str | None,
    name: str | None,
    fallback_basename: str,
) -> str:
    src_for_ext = (client_filename or name or "").strip()
    if mime_type:
        return mime_type
    guessed = mimetypes.guess_type(os.path.basename(src_for_ext), strict=False)[0]
    if guessed:
        return guessed
    guessed = mimetypes.guess_type(fallback_basename, strict=False)[0]
    return guessed or "application/octet-stream"


def _move_temp_to_dest(temp_path: str, dest_abs: str) -> None:
    os.makedirs(os.path.dirname(dest_abs), exist_ok=True)
    try:
        os.replace(temp_path, dest_abs)
    except Exception as e:
        raise RuntimeError(f"failed to move uploaded file into place: {e}") from e


def _create_upload_record(
    session: Session,
    content_id: str,
    name: str,
    abs_path: str,
    tags: Sequence[str],
    mime_type: str | None,
    user_metadata: UserMetadata,
    preview_id: str | None,
) -> Asset:
    from app.assets.database.queries.records import create_record

    record = create_record(
        session,
        content_id,
        name,
        mime_type=mime_type,
        loader_path=compute_loader_path(abs_path),
        tags=list(tags),
        system_metadata=_extract_system_metadata_sync(abs_path, mime_type),
    )
    if user_metadata:
        record.user_metadata = dict(user_metadata)
    if preview_id:
        record.preview_id = preview_id
    session.flush()
    return record


def _record_to_upload_result(
    session: Session, record: Asset, *, created_new: bool
) -> UploadResult:
    from sqlalchemy import select

    from app.assets.database.models import AssetContent, AssetTag

    content = session.get(AssetContent, record.content_id)
    tag_names = list(
        session.scalars(
            select(AssetTag.tag_name)
            .where(AssetTag.asset_id == record.id)
            .order_by(AssetTag.tag_name)
        )
    )
    api_hash = content.hash if content else None
    ref = ReferenceData(
        id=record.id,
        name=record.name,
        file_path=content.path if content else None,
        loader_path=record.loader_path,
        user_metadata=record.user_metadata,
        preview_id=record.preview_id,
        system_metadata=record.system_metadata,
        job_id=record.job_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
        last_access_time=record.last_access_time,
    )
    asset = AssetData(
        hash=api_hash,
        size_bytes=content.size_bytes if content else None,
        mime_type=record.mime_type,
    )
    return UploadResult(ref=ref, asset=asset, tags=tag_names, created_new=created_new)


def upload_from_temp_path(
    temp_path: str,
    name: str | None = None,
    tags: list[str] | None = None,
    user_metadata: dict | None = None,
    client_filename: str | None = None,
    tenant_id: str = "",
    expected_hash: str | None = None,
    mime_type: str | None = None,
    preview_id: str | None = None,
) -> UploadResult:
    from app.assets.database.models import Asset, AssetContent
    from app.assets.database.queries.records import create_content
    from app.assets.services.lookup import lookup_for_upload_dedup

    display_name = _sanitize_filename(name or client_filename, fallback="upload")
    user_metadata = user_metadata or {}

    try:
        digest = _snapshot_hash_with_retry(temp_path)
    except UploadUnstableError:
        _remove_temp_path(temp_path)
        raise
    stored_hash = to_stored_hash(digest)
    if expected_hash and stored_hash != _normalize_hash_input(expected_hash):
        _remove_temp_path(temp_path)
        raise HashMismatchError("Uploaded file hash does not match provided hash.")

    with create_session() as session:
        dedup = lookup_for_upload_dedup(session, stored_hash, display_name)

    if isinstance(dedup, Asset):
        _remove_temp_path(temp_path)
        with create_session() as session:
            record = session.get(Asset, dedup.id)
            if record is None:
                raise RuntimeError("inconsistent DB state after dedup")
            return _record_to_upload_result(session, record, created_new=False)

    if isinstance(dedup, AssetContent):
        _remove_temp_path(temp_path)
        with create_session() as session:
            record = _create_upload_record(
                session,
                dedup.id,
                display_name,
                dedup.path,
                [*(tags or []), "uploaded"],
                mime_type,
                user_metadata,
                preview_id,
            )
            session.commit()
            return _record_to_upload_result(session, record, created_new=True)

    if not tags:
        _remove_temp_path(temp_path)
        raise ValueError("tags are required for new asset uploads")

    dest_abs = _hash_mode_dest_path(tags, digest, client_filename, name)
    content_type = _guess_upload_mime_type(
        mime_type, client_filename, name, os.path.basename(dest_abs)
    )
    _move_temp_to_dest(temp_path, dest_abs)
    size_bytes, mtime_ns = get_size_and_mtime_ns(dest_abs)
    with create_session() as session:
        content = create_content(
            session, dest_abs, stored_hash, size_bytes, mtime_ns
        )
        record = _create_upload_record(
            session,
            content.id,
            display_name,
            dest_abs,
            [*(tags or []), "uploaded"],
            content_type,
            user_metadata,
            preview_id,
        )
        session.commit()
        return _record_to_upload_result(session, record, created_new=True)


def register_file_in_place(
    abs_path: str,
    name: str,
    tags: list[str],
    tenant_id: str = "",
    mime_type: str | None = None,
) -> UploadResult:
    """Register an already-saved file in the asset database without moving it.

    Used by ``/upload/image`` after the handler writes (or reuses) bytes on disk.
    ``compare_image_hash`` same-name dedup is legacy physical behavior and stays
    outside hash-mode policy (parallel to path-form ``/view`` semantics).
    """
    from app.assets.database.models import Asset, AssetContent
    from app.assets.database.queries.records import create_content
    from app.assets.services.lookup import lookup_for_upload_dedup

    locator = os.path.abspath(abs_path)
    display_name = _sanitize_filename(name, fallback=os.path.basename(locator))
    try:
        _, path_tags = get_name_and_tags_from_asset_path(locator)
    except ValueError:
        path_tags = []
    merged_tags = normalize_tags([*path_tags, *tags, "uploaded"])
    content_type = _guess_upload_mime_type(
        mime_type, name, name, os.path.basename(locator)
    )
    size_bytes, mtime_ns = get_size_and_mtime_ns(locator)

    digest = _snapshot_hash_with_retry(locator)
    stored_hash = to_stored_hash(digest)
    with create_session() as session:
        dedup = lookup_for_upload_dedup(session, stored_hash, display_name)

    if isinstance(dedup, Asset):
        with create_session() as session:
            record = session.get(Asset, dedup.id)
            if record is None:
                raise RuntimeError("inconsistent DB state after dedup")
            return _record_to_upload_result(session, record, created_new=False)

    if isinstance(dedup, AssetContent):
        with create_session() as session:
            record = _create_upload_record(
                session,
                dedup.id,
                display_name,
                dedup.path,
                merged_tags,
                content_type,
                None,
                None,
            )
            session.commit()
            return _record_to_upload_result(session, record, created_new=True)

    with create_session() as session:
        content = create_content(
            session, locator, stored_hash, size_bytes, mtime_ns
        )
        record = _create_upload_record(
            session,
            content.id,
            display_name,
            locator,
            merged_tags,
            content_type,
            None,
            None,
        )
        session.commit()
        return _record_to_upload_result(session, record, created_new=True)


def create_from_hash(
    hash_str: str,
    name: str,
    tags: list[str] | None = None,
    user_metadata: dict | None = None,
    tenant_id: str = "",
    mime_type: str | None = None,
    preview_id: str | None = None,
) -> UploadResult | None:
    from app.assets import mode
    from app.assets.services.lookup import lookup_for_from_hash

    if not mode.hashing_enabled():
        return None

    stored_hash = _normalize_hash_input(hash_str)
    bare_digest = stored_hash.partition(":")[2] or stored_hash
    display_name = _sanitize_filename(
        name, fallback=bare_digest
    )

    with create_session() as session:
        content = lookup_for_from_hash(session, stored_hash)
        if content is None:
            logging.warning("create_from_hash: no asset found for hash %s", hash_str)
            return None
        record = _create_upload_record(
            session,
            content.id,
            display_name,
            content.path,
            tags or [],
            mime_type,
            user_metadata,
            preview_id,
        )
        session.commit()
        return _record_to_upload_result(session, record, created_new=True)


def register_cached_output(abs_path: str, job_id: str | None = None):
    """Register a replayed output as a new delivery record without mutations.

    S10.4: missing live content is a logged non-event - it creates nothing and
    returns ``None``, and is never re-registered as a fresh executed output.
    S29: the new record copies ``system_metadata`` from the earliest sibling
    record of the same content (``created_at`` ascending, ``id`` ascending as
    the tiebreak); when the content is orphaned (no sibling record survives,
    e.g. after ``delete_asset_reference``) metadata is extracted fresh from the
    file instead - the only behaviour satisfying both S10.4 and S29.
    """
    from sqlalchemy import event, select

    from app.assets.database.models import Asset, AssetContent
    from app.assets.database.queries.records import create_record

    locator = os.path.abspath(abs_path)
    try:
        with create_session() as session:
            update_count = [0]

            @event.listens_for(session, "after_bulk_update")
            def _count_update(update_context):
                update_count[0] += 1

            existing = session.scalars(
                select(AssetContent).where(
                    AssetContent.path == locator, AssetContent.is_missing.is_(False)
                )
            ).first()
            if existing is None:
                logging.info(
                    "Cached output registration is a non-event; no live content "
                    "for %s",
                    locator,
                )
                return None

            name, path_tags = get_name_and_tags_from_asset_path(locator)
            mime_type = mimetypes.guess_type(locator, strict=False)[0]

            sibling = session.scalars(
                select(Asset)
                .where(Asset.content_id == existing.id)
                .order_by(Asset.created_at.asc(), Asset.id.asc())
                .limit(1)
            ).first()
            if sibling is not None:
                system_metadata = (
                    dict(sibling.system_metadata)
                    if sibling.system_metadata is not None
                    else None
                )
            else:
                system_metadata = _extract_system_metadata_sync(locator, mime_type)

            try:
                record = create_record(
                    session,
                    existing.id,
                    name,
                    mime_type=mime_type,
                    job_id=job_id,
                    loader_path=compute_loader_path(locator),
                    tags=path_tags,
                    system_metadata=system_metadata,
                )
                if update_count[0] != 0:
                    logging.error(
                        "Cached save must not UPDATE any row; got %d for %s; discarding",
                        update_count[0],
                        locator,
                    )
                    session.rollback()
                    return None
                session.commit()
            except Exception:
                session.rollback()
                raise
            record_id = record.id
            record_content_id = record.content_id
            record_job_id = record.job_id
            record_name = record.name
    except Exception:
        logging.exception("Failed to register cached output: %s", locator)
        return None

    from types import SimpleNamespace

    return SimpleNamespace(
        id=record_id,
        content_id=record_content_id,
        job_id=record_job_id,
        name=record_name,
    )


def register_executed_output(abs_path: str, job_id: str | None = None):
    """Register a freshly-executed workflow output as a new delivery record.

    D14a: the content hash is left ``None`` at registration; the background
    enrich pass fills it later - outputs are never force-hashed inline. S29/D8:
    ``system_metadata`` is extracted synchronously and stored at creation.
    S10.4: any failure is logged and swallowed (returns ``None``) so a save
    error never propagates into the execution pipeline, and no partial rows are
    left behind.
    """
    from sqlalchemy import select

    from app.assets.database.models import AssetContent
    from app.assets.database.queries.records import (
        create_content,
        create_record,
        mark_content_missing,
    )

    locator = os.path.abspath(abs_path)
    try:
        stat_result = os.stat(locator, follow_symlinks=True)
        size_bytes = stat_result.st_size
        mtime_ns = get_mtime_ns(stat_result)
        mime_type = mimetypes.guess_type(locator, strict=False)[0]
        name, path_tags = get_name_and_tags_from_asset_path(locator)
        system_metadata = _extract_system_metadata_sync(
            locator, mime_type, stat_result
        )
        with create_session() as session:
            created_content_id: str | None = None
            try:
                existing = session.scalars(
                    select(AssetContent).where(
                        AssetContent.path == locator,
                        AssetContent.is_missing.is_(False),
                    )
                ).first()
                if existing is not None:
                    mark_content_missing(session, existing.id)
                content = create_content(session, locator, None, size_bytes, mtime_ns)
                created_content_id = content.id
                record = create_record(
                    session,
                    content.id,
                    name,
                    mime_type=mime_type,
                    job_id=job_id,
                    loader_path=compute_loader_path(locator),
                    tags=path_tags,
                    system_metadata=system_metadata,
                )
                session.commit()
            except Exception:
                session.rollback()
                if created_content_id is not None:
                    _discard_unreferenced_content(session, created_content_id)
                raise
            record_id = record.id
            record_content_id = record.content_id
            record_job_id = record.job_id
            record_name = record.name
    except Exception:
        logging.exception("Failed to register executed output: %s", locator)
        return None

    from types import SimpleNamespace
    return SimpleNamespace(
        id=record_id,
        content_id=record_content_id,
        job_id=record_job_id,
        name=record_name,
    )
