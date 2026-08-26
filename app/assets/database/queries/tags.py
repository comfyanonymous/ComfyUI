from __future__ import annotations

from dataclasses import dataclass

from typing import Iterable, Sequence

import sqlalchemy as sa
from sqlalchemy import delete, func, select
from sqlalchemy.dialects import sqlite
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.assets.database.models import (
    Asset,
    AssetContent,
    AssetReference,
    AssetReferenceMeta,
    AssetReferenceTag,
    AssetTag,
    Tag,
)
from app.assets.database.queries.common import iter_row_chunks
from app.assets.database.queries.records import (
    build_record_tag_filter_clauses,
    live_asset_content_clause,
)
from app.assets.helpers import escape_sql_like_string, get_utc_now, normalize_tags


@dataclass(frozen=True)
class AddTagsResult:
    added: list[str]
    already_present: list[str]
    total_tags: list[str]


@dataclass(frozen=True)
class RemoveTagsResult:
    removed: list[str]
    not_present: list[str]
    total_tags: list[str]


@dataclass(frozen=True)
class SetTagsResult:
    added: list[str]
    removed: list[str]
    total: list[str]


def validate_tags_exist(session: Session, tags: list[str]) -> None:
    """Raise ValueError if any of the given tag names do not exist."""
    existing_tag_names = set(
        name
        for (name,) in session.execute(select(Tag.name).where(Tag.name.in_(tags))).all()
    )
    missing = [t for t in tags if t not in existing_tag_names]
    if missing:
        raise ValueError(f"Unknown tags: {missing}")


def ensure_tags_exist(session: Session, names: Iterable[str]) -> None:
    wanted = normalize_tags(list(names))
    if not wanted:
        return
    rows = [{"name": n} for n in list(dict.fromkeys(wanted))]
    ins = (
        sqlite.insert(Tag)
        .values(rows)
        .on_conflict_do_nothing(index_elements=[Tag.name])
    )
    session.execute(ins)


def get_reference_tags(session: Session, reference_id: str) -> list[str]:
    return [
        tag_name
        for (tag_name,) in (
            session.execute(
                select(AssetReferenceTag.tag_name)
                .where(AssetReferenceTag.asset_reference_id == reference_id)
                .order_by(AssetReferenceTag.tag_name.asc())
            )
        ).all()
    ]


def set_reference_tags(
    session: Session,
    reference_id: str,
    tags: Sequence[str],
    origin: str = "manual",
) -> SetTagsResult:
    desired = normalize_tags(tags)

    current = set(get_reference_tags(session, reference_id))

    to_add = [t for t in desired if t not in current]
    to_remove = [t for t in current if t not in desired]

    if to_add:
        ensure_tags_exist(session, to_add)
        session.add_all(
            [
                AssetReferenceTag(
                    asset_reference_id=reference_id,
                    tag_name=t,
                    origin=origin,
                    added_at=get_utc_now(),
                )
                for t in to_add
            ]
        )
        session.flush()

    if to_remove:
        session.execute(
            delete(AssetReferenceTag).where(
                AssetReferenceTag.asset_reference_id == reference_id,
                AssetReferenceTag.tag_name.in_(to_remove),
            )
        )
        session.flush()

    return SetTagsResult(added=sorted(to_add), removed=sorted(to_remove), total=sorted(desired))


def add_tags_to_reference(
    session: Session,
    reference_id: str,
    tags: Sequence[str],
    origin: str = "manual",
    create_if_missing: bool = True,
    reference_row: AssetReference | None = None,
) -> AddTagsResult:
    if not reference_row:
        ref = session.get(AssetReference, reference_id)
        if not ref:
            raise ValueError(f"AssetReference {reference_id} not found")

    norm = normalize_tags(tags)
    if not norm:
        total = get_reference_tags(session, reference_id=reference_id)
        return AddTagsResult(added=[], already_present=[], total_tags=total)

    if create_if_missing:
        ensure_tags_exist(session, norm)

    current = set(get_reference_tags(session, reference_id))

    want = set(norm)
    to_add = sorted(want - current)

    if to_add:
        with session.begin_nested() as nested:
            try:
                session.add_all(
                    [
                        AssetReferenceTag(
                            asset_reference_id=reference_id,
                            tag_name=t,
                            origin=origin,
                            added_at=get_utc_now(),
                        )
                        for t in to_add
                    ]
                )
                session.flush()
            except IntegrityError:
                nested.rollback()

    after = set(get_reference_tags(session, reference_id=reference_id))
    return AddTagsResult(
        added=sorted(((after - current) & want)),
        already_present=sorted(want & current),
        total_tags=sorted(after),
    )


def remove_missing_tag_for_asset_id(
    session: Session,
    asset_id: str,
) -> None:
    session.execute(
        sa.delete(AssetReferenceTag).where(
            AssetReferenceTag.asset_reference_id.in_(
                sa.select(AssetReference.id).where(AssetReference.asset_id == asset_id)
            ),
            AssetReferenceTag.tag_name == "missing",
        )
    )


def list_tags_with_usage(
    session: Session,
    prefix: str | None = None,
    limit: int = 100,
    offset: int = 0,
    include_zero: bool = True,
    order: str = "count_desc",
) -> tuple[list[tuple[str, int]], int]:
    prefix_filter = prefix.strip() if prefix else ""

    # An asset counts toward a tag when its content is live, EXCEPT the "missing"
    # tag, which stays visible precisely because the content went missing.
    usage_visibility = sa.or_(
        AssetContent.is_missing.is_(False),
        AssetTag.tag_name == "missing",
    )

    counts_sq = (
        select(
            AssetTag.tag_name.label("tag_name"),
            func.count(AssetTag.asset_id).label("cnt"),
        )
        .select_from(AssetTag)
        .join(Asset, Asset.id == AssetTag.asset_id)
        .join(AssetContent, Asset.content_id == AssetContent.id)
        .where(usage_visibility)
        .group_by(AssetTag.tag_name)
        .subquery()
    )

    q = (
        select(
            Tag.name,
            func.coalesce(counts_sq.c.cnt, 0).label("count"),
        )
        .select_from(Tag)
        .join(counts_sq, counts_sq.c.tag_name == Tag.name, isouter=True)
    )

    if prefix_filter:
        q = q.where(func.substr(Tag.name, 1, len(prefix_filter)) == prefix_filter)

    if not include_zero:
        q = q.where(func.coalesce(counts_sq.c.cnt, 0) > 0)

    if order == "name_asc":
        q = q.order_by(Tag.name.asc())
    else:
        q = q.order_by(func.coalesce(counts_sq.c.cnt, 0).desc(), Tag.name.asc())

    total_q = select(func.count()).select_from(Tag)
    if prefix_filter:
        total_q = total_q.where(func.substr(Tag.name, 1, len(prefix_filter)) == prefix_filter)
    if not include_zero:
        visible_tags_sq = (
            select(AssetTag.tag_name)
            .join(Asset, Asset.id == AssetTag.asset_id)
            .join(AssetContent, Asset.content_id == AssetContent.id)
            .where(usage_visibility)
            .group_by(AssetTag.tag_name)
        )
        total_q = total_q.where(Tag.name.in_(visible_tags_sq))

    rows = (session.execute(q.limit(limit).offset(offset))).all()
    total = (session.execute(total_q)).scalar_one()

    rows_norm = [(name, int(count or 0)) for (name, count) in rows]
    return rows_norm, int(total or 0)


def list_tag_counts_for_filtered_assets(
    session: Session,
    include_tags: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    name_contains: str | None = None,
    limit: int = 100,
    # Appended last so pre-existing positional callers keep binding correctly.
    any_tags: Sequence[str] | None = None,
) -> dict[str, int]:
    """Return {tag_name: count} for the live assets matching the given filters.

    Reuses build_record_tag_filter_clauses + live_asset_content_clause from the
    record query layer verbatim, so /api/assets and /api/assets/tags/refine agree
    on which assets a given all/any/none + name_contains filter selects.
    """
    filters = list(
        build_record_tag_filter_clauses(
            tuple(include_tags or ()),
            tuple(any_tags or ()),
            tuple(exclude_tags or ()),
        )
    )
    if name_contains:
        escaped, esc = escape_sql_like_string(name_contains)
        filters.append(Asset.name.ilike(f"%{escaped}%", escape=esc))

    asset_sq = (
        select(Asset.id)
        .join(AssetContent, live_asset_content_clause())
        .where(*filters)
        .subquery()
    )

    # Count every tag carried by the matching assets.
    q = (
        select(
            AssetTag.tag_name,
            func.count(AssetTag.asset_id).label("cnt"),
        )
        .where(AssetTag.asset_id.in_(select(asset_sq.c.id)))
        .group_by(AssetTag.tag_name)
        .order_by(func.count(AssetTag.asset_id).desc(), AssetTag.tag_name.asc())
        .limit(limit)
    )

    rows = session.execute(q).all()
    return {tag_name: int(cnt) for tag_name, cnt in rows}


def bulk_insert_tags_and_meta(
    session: Session,
    tag_rows: list[dict],
    meta_rows: list[dict],
) -> None:
    """Batch insert into asset_reference_tags and asset_reference_meta.

    Uses ON CONFLICT DO NOTHING.

    Args:
        session: Database session
        tag_rows: Dicts with: asset_reference_id, tag_name, origin, added_at
        meta_rows: Dicts with: asset_reference_id, key, ordinal, val_*
    """
    if tag_rows:
        ins_tags = sqlite.insert(AssetReferenceTag).on_conflict_do_nothing(
            index_elements=[
                AssetReferenceTag.asset_reference_id,
                AssetReferenceTag.tag_name,
            ]
        )
        for chunk in iter_row_chunks(tag_rows, cols_per_row=4):
            session.execute(ins_tags, chunk)

    if meta_rows:
        ins_meta = sqlite.insert(AssetReferenceMeta).on_conflict_do_nothing(
            index_elements=[
                AssetReferenceMeta.asset_reference_id,
                AssetReferenceMeta.key,
                AssetReferenceMeta.ordinal,
            ]
        )
        for chunk in iter_row_chunks(meta_rows, cols_per_row=7):
            session.execute(ins_meta, chunk)
