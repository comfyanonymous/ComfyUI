import mimetypes
import os
from datetime import timezone
from typing import Sequence

from app.assets.services.cursor import (
    CursorPayload,
    InvalidCursorError,
    decode_cursor,
    decode_cursor_int,
    decode_cursor_time,
    encode_cursor,
    encode_cursor_from_time,
)


from app.assets.database.models import AssetContent
from app.assets.database.queries import (
    delete_record,
    fetch_record_tags,
    get_record_by_id,
    fetch_reference_asset_and_tags,
    get_asset_by_hash as queries_get_asset_by_hash,
    get_reference_with_owner_check,
    list_references_page,
    set_reference_preview,
    update_record_access_time,
)
from app.assets.database.queries.records import get_preview_file_paths_by_ids
from app.assets.helpers import normalize_tags
from app.assets.services.schemas import (
    AssetData,
    AssetDetailResult,
    AssetSummaryData,
    DownloadResolutionResult,
    ListAssetsResult,
    ReferenceData,
    UserMetadata,
    extract_asset_data,
    extract_reference_data,
)
from app.database.db import create_session


def _record_to_detail_result(session, record) -> AssetDetailResult:
    content = session.get(AssetContent, record.content_id)
    tags = fetch_record_tags(session, record.id)
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
    return AssetDetailResult(ref=ref, asset=asset, tags=tags)


def get_asset_detail(
    reference_id: str,
    tenant_id: str = "",
) -> AssetDetailResult | None:
    del tenant_id
    with create_session() as session:
        record = get_record_by_id(session, reference_id)
        if record is None:
            return None
        return _record_to_detail_result(session, record)


def update_asset_metadata(
    reference_id: str,
    name: str | None = None,
    tags: Sequence[str] | None = None,
    user_metadata: UserMetadata = None,
    tag_origin: str = "manual",
    tenant_id: str = "",
    mime_type: str | None = None,
    preview_id: str | None = None,
) -> AssetDetailResult:
    from sqlalchemy import delete, update

    from app.assets.database.models import Asset, AssetTag, Tag
    from app.assets.database.queries.records import get_record_by_id, rename_record

    del tenant_id
    with create_session() as session:
        record = get_record_by_id(session, reference_id)
        if record is None:
            raise ValueError(f"Asset {reference_id} not found")

        if name is not None:
            rename_record(session, reference_id, name)
        if user_metadata is not None:
            session.execute(
                update(Asset)
                .where(Asset.id == reference_id)
                .values(user_metadata=dict(user_metadata))
            )
        if tags is not None:
            session.execute(
                delete(AssetTag).where(
                    AssetTag.asset_id == reference_id,
                    AssetTag.origin != "automatic",
                )
            )
        if mime_type is not None:
            session.execute(
                update(Asset)
                .where(Asset.id == reference_id)
                .values(mime_type=mime_type)
            )
        if preview_id is not None:
            session.execute(
                update(Asset)
                .where(Asset.id == reference_id)
                .values(preview_id=preview_id)
            )
        if tags is not None:
            for tag_name in normalize_tags(list(tags)):
                if session.get(Tag, tag_name) is None:
                    session.add(Tag(name=tag_name))
                    session.flush()
                if session.get(AssetTag, (reference_id, tag_name)) is None:
                    session.add(
                        AssetTag(
                            asset_id=reference_id,
                            tag_name=tag_name,
                            origin=tag_origin,
                        )
                    )
        session.commit()

    detail = get_asset_detail(reference_id)
    if detail is None:
        raise RuntimeError("Asset deleted during update")
    return detail


def delete_asset_reference(
    reference_id: str,
    tenant_id: str = "",
    delete_content_if_orphan: bool = True,
) -> bool:
    """Hard-delete an asset record. Content rows and files are untouched (D-3 floor)."""
    del tenant_id, delete_content_if_orphan
    with create_session() as session:
        if get_record_by_id(session, reference_id) is None:
            return False
        delete_record(session, reference_id)
        session.commit()
        return True


def set_asset_preview(
    reference_id: str,
    preview_reference_id: str | None = None,
    tenant_id: str = "",
) -> AssetDetailResult:
    with create_session() as session:
        get_reference_with_owner_check(session, reference_id, tenant_id)

        set_reference_preview(
            session,
            reference_id=reference_id,
            preview_reference_id=preview_reference_id,
        )

        result = fetch_reference_asset_and_tags(
            session, reference_id=reference_id, tenant_id=tenant_id
        )
        if not result:
            raise RuntimeError("State changed during preview update")

        ref, asset, tags = result
        detail = AssetDetailResult(
            ref=extract_reference_data(ref),
            asset=extract_asset_data(asset),
            tags=tags,
        )
        session.commit()

        return detail


def asset_exists(asset_hash: str) -> bool:
    from app.assets.helpers import validate_blake3_hash
    from app.assets.services.lookup import lookup_for_view

    try:
        canonical = validate_blake3_hash(asset_hash)
    except ValueError:
        return False
    digest = canonical.partition(":")[2]
    with create_session() as session:
        return lookup_for_view(session, digest) is not None


def get_asset_by_hash(asset_hash: str) -> AssetData | None:
    with create_session() as session:
        asset = queries_get_asset_by_hash(session, asset_hash=asset_hash)
        return extract_asset_data(asset)


# Sort fields that support cursor pagination. `last_access_time` is not
# in this list — it falls back to offset/limit.
_CURSOR_SORT_FIELDS = ("created_at", "updated_at", "name", "size")


def list_assets_page(
    tenant_id: str = "",
    include_tags: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    name_contains: str | None = None,
    metadata_filter: dict | None = None,
    limit: int = 20,
    offset: int = 0,
    sort: str = "created_at",
    order: str = "desc",
    after: str | None = None,
    # Appended last so pre-existing positional callers keep binding correctly.
    any_tags: Sequence[str] | None = None,
) -> ListAssetsResult:
    """List assets with optional cursor pagination.

    When ``after`` is supplied it overrides ``offset``. The cursor's sort field
    must match ``sort`` and be in the cursor-supported allowlist; mismatches
    raise InvalidCursorError so the handler can map to 400 INVALID_CURSOR.
    """
    cursor_value: object | None = None
    cursor_id: str | None = None
    # Mint next_cursor on every page where the sort is cursor-supported, not
    # only when the request itself arrived with a cursor. Otherwise a first
    # request (no `after`) returns next_cursor=None and the client can never
    # enter cursor mode.
    mint_cursor = sort in _CURSOR_SORT_FIELDS

    if after is not None:
        if sort not in _CURSOR_SORT_FIELDS:
            raise InvalidCursorError(
                f"cursor pagination is not supported for sort={sort!r}"
            )
        payload = decode_cursor(after, _CURSOR_SORT_FIELDS, expected_order=order)
        if payload.sort_field != sort:
            raise InvalidCursorError(
                f"cursor sort field {payload.sort_field!r} does not match request sort {sort!r}"
            )
        cursor_value, cursor_id = _resolve_cursor_value(payload), payload.id

    # Over-fetch by one row so we can distinguish "exactly `limit` rows total
    # remaining" from "more rows past this page" without a second query. Drop
    # the sentinel before returning.
    fetch_limit = limit + 1 if mint_cursor else limit

    with create_session() as session:
        refs, tag_map, total = list_references_page(
            session,
            tenant_id=tenant_id,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            any_tags=any_tags,
            name_contains=name_contains,
            metadata_filter=metadata_filter,
            limit=fetch_limit,
            offset=offset,
            sort=sort,
            order=order,
            after_cursor_value=cursor_value,
            after_cursor_id=cursor_id,
        )

        next_cursor: str | None = None
        if mint_cursor and len(refs) > limit:
            # There's at least one more row past this page — mint a cursor from
            # the last row of the page (i.e. index `limit - 1`, since we
            # over-fetched), and drop the sentinel.
            next_cursor = _encode_next_cursor(refs[limit - 1], sort, order)
            refs = refs[:limit]

        items: list[AssetSummaryData] = []
        for ref in refs:
            items.append(
                AssetSummaryData(
                    ref=extract_reference_data(ref),
                    asset=extract_asset_data(ref.asset),
                    tags=tag_map.get(ref.id, []),
                )
            )

        return ListAssetsResult(items=items, total=total, next_cursor=next_cursor)


def _resolve_cursor_value(payload: CursorPayload) -> object:
    """Map a decoded cursor payload to a column-typed Python value."""
    if payload.sort_field in ("created_at", "updated_at"):
        # DB stores naive UTC; strip tzinfo so the comparison binds against a
        # `TIMESTAMP WITHOUT TIME ZONE` column without an offset shift.
        return decode_cursor_time(payload).replace(tzinfo=None)
    if payload.sort_field == "size":
        return decode_cursor_int(payload)
    return payload.value  # name, str-typed


def _encode_next_cursor(ref, sort: str, order: str) -> str | None:
    """Mint a cursor pointing at *ref* for the given sort dimension.

    Returns None when the boundary row carries a NULL sort value (e.g. an asset
    record whose size_bytes hasn't been backfilled). Continuing pagination
    across a NULL boundary is undefined under keyset ordering — better to
    truncate cleanly here than to mint a cursor that mis-positions.
    """
    if sort == "name":
        return encode_cursor("name", ref.name, ref.id, order=order)
    if sort == "size":
        if ref.asset is None or ref.asset.size_bytes is None:
            return None
        return encode_cursor("size", str(ref.asset.size_bytes), ref.id, order=order)
    # created_at / updated_at — DB datetimes are naive UTC; attach tz before encoding.
    value = ref.created_at if sort == "created_at" else ref.updated_at
    if value is None:
        return None
    return encode_cursor_from_time(sort, value.replace(tzinfo=timezone.utc), ref.id, order=order)


def resolve_hash_to_path(
    asset_hash: str,
    tenant_id: str = "",
) -> DownloadResolutionResult | None:
    """Resolve a blake3 hash to an on-disk file path via lookup_for_view.

    Uses the first qualified live content row. Temp paths are excluded from all
    hash lookups inside qualified_content_iterator, so a hash resolving only to
    temp content returns None. Updates last_access_time on every record pointing
    at the served content.
    """
    from sqlalchemy import select

    from app.assets.database.models import Asset
    from app.assets.services.lookup import lookup_for_view

    del tenant_id
    digest = asset_hash.partition(":")[2] or asset_hash
    with create_session() as session:
        content = lookup_for_view(session, digest)
        if content is None:
            return None

        records = list(
            session.scalars(
                select(Asset)
                .where(Asset.content_id == content.id)
                .order_by(Asset.created_at, Asset.id)
            )
        )
        display_name = os.path.basename(content.path)
        mime_type = None
        for record in records:
            if record.name:
                display_name = record.name
            if mime_type is None and record.mime_type:
                mime_type = record.mime_type
            update_record_access_time(session, record.id)
        abs_path = content.path
        session.commit()

        ctype = (
            mime_type
            or mimetypes.guess_type(display_name)[0]
            or mimetypes.guess_type(abs_path)[0]
            or "application/octet-stream"
        )
    return DownloadResolutionResult(
        abs_path=abs_path,
        content_type=ctype,
        download_name=display_name,
    )


def get_preview_file_paths(preview_ids: list[str]) -> dict[str, str]:
    """Map preview reference id -> file_path, in one query for the whole page."""
    if not preview_ids:
        return {}
    with create_session() as session:
        return get_preview_file_paths_by_ids(session, preview_ids=preview_ids)


def resolve_asset_for_download(
    reference_id: str,
    tenant_id: str = "",
) -> DownloadResolutionResult:
    del tenant_id
    with create_session() as session:
        record = get_record_by_id(session, reference_id)
        if record is None:
            raise ValueError(f"AssetReference {reference_id} not found")

        content = session.get(AssetContent, record.content_id)
        if (
            content is None
            or content.is_missing
            or not os.path.isfile(content.path)
        ):
            raise FileNotFoundError(
                f"No live content for AssetReference {reference_id} "
                f"(content id={record.content_id}, name={record.name})"
            )

        ref_name = record.name
        asset_mime = record.mime_type
        abs_path = content.path

        update_record_access_time(session, reference_id)
        session.commit()

        ctype = (
            asset_mime
            or mimetypes.guess_type(ref_name or abs_path)[0]
            or "application/octet-stream"
        )
        download_name = ref_name or os.path.basename(abs_path)
        return DownloadResolutionResult(
            abs_path=abs_path,
            content_type=ctype,
            download_name=download_name,
        )
