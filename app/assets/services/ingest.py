import contextlib
import logging
import mimetypes
import os
from types import SimpleNamespace
from typing import Any, Sequence

from sqlalchemy import event, func, select
from sqlalchemy.orm import Session

from app.assets import mode
from app.assets.database.models import Asset, AssetContent, AssetTag
from app.assets.database.queries.records import create_content, create_record, mark_content_missing
from app.assets.helpers import normalize_tags, to_stored_hash
from app.assets.services.file_utils import get_mtime_ns, get_size_and_mtime_ns
from app.assets.services.image_dimensions import extract_image_dimensions
from app.assets.services.lookup import lookup_for_from_hash, lookup_for_upload_dedup
from app.assets.services.metadata_extract import extract_file_metadata
from app.assets.services.path_utils import (
    compute_loader_path,
    get_name_and_tags_from_asset_path,
    resolve_destination_from_tags,
    validate_path_within_base,
)
from app.assets.services.schemas import (
    AssetData,
    ReferenceData,
    UploadResult,
    UserMetadata,
)
from app.assets.services.snapshot_hash import snapshot_hash
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
    for _ in range(_UPLOAD_HASH_ATTEMPTS):
        snapshot = snapshot_hash(path)
        if snapshot is not None:
            digest, _ = snapshot
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
    if preview_id is not None and session.get(Asset, preview_id) is None:
        raise ValueError(f"preview_id {preview_id!r} does not reference an existing asset")
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
        created_content_id = content.id
        try:
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
        except Exception:
            session.rollback()
            _discard_unreferenced_content(session, created_content_id)
            raise
        return _record_to_upload_result(session, record, created_new=True)


def _retire_stale_live_content(
    session: Session, locator: str, new_hash: str | None
) -> None:
    """Retire a live content row at ``locator`` whose bytes no longer match.

    ``create_content`` resolves a live-path uniqueness conflict
    (``uq_asset_contents_path_live``) by returning the existing row, so
    re-registering a path in place without first retiring the previous content
    would hand the caller the OLD file's hash/size. When the on-disk bytes have
    changed (hash differs) we mark the stale row missing so the fresh
    ``create_content`` inserts a new live row; when the hash matches we leave the
    row untouched and let ``create_content``'s dedup path reuse it (same-bytes
    dedup is unchanged).
    """
    existing = session.scalars(
        select(AssetContent).where(
            AssetContent.path == locator,
            AssetContent.is_missing.is_(False),
        )
    ).first()
    if existing is not None and existing.hash != new_hash:
        mark_content_missing(session, existing.id)


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
        _retire_stale_live_content(session, locator, stored_hash)
        content = create_content(
            session, locator, stored_hash, size_bytes, mtime_ns
        )
        created_content_id = content.id
        try:
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
        except Exception:
            session.rollback()
            _discard_unreferenced_content(session, created_content_id)
            raise
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

    return SimpleNamespace(
        id=record_id,
        content_id=record_content_id,
        job_id=record_job_id,
        name=record_name,
    )
