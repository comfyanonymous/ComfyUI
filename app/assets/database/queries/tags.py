from __future__ import annotations

from dataclasses import dataclass, field

from typing import Iterable, Sequence

from sqlalchemy import func, select
from sqlalchemy.dialects import sqlite
from sqlalchemy.orm import Session

from app.assets.database.models import (
    Asset,
    AssetContent,
    AssetTag,
    Tag,
)
from app.assets.database.queries.records import (
    build_record_tag_filter_clauses,
)
from app.assets.helpers import escape_sql_like_string, normalize_tags


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
    # Tags that ARE present on the record but carry origin="automatic", so they
    # cannot be removed via this API. Kept distinct from ``not_present`` so a
    # caller can tell "the tag wasn't there" apart from "the tag is protected".
    protected: list[str] = field(default_factory=list)


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


def list_tags_with_usage(
    session: Session,
    prefix: str | None = None,
    limit: int = 100,
    offset: int = 0,
    include_zero: bool = True,
    order: str = "count_desc",
) -> tuple[list[tuple[str, int]], int]:
    prefix_filter = prefix.strip() if prefix else ""

    # Every asset counts toward each tag it carries, missing content included, so
    # /api/tags agrees with the catalog list/refine surfaces on tag counts. A
    # record whose content went missing keeps ALL its tags countable here, not
    # just the automatic "missing" one.
    counts_sq = (
        select(
            AssetTag.tag_name.label("tag_name"),
            func.count(AssetTag.asset_id).label("cnt"),
        )
        .select_from(AssetTag)
        .join(Asset, Asset.id == AssetTag.asset_id)
        .join(AssetContent, Asset.content_id == AssetContent.id)
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
    """Return {tag_name: count} for the assets matching the given filters.

    Reuses build_record_tag_filter_clauses and the same Asset->AssetContent inner
    join as list_records_page, so /api/assets and /api/assets/tags/refine agree on
    which assets a given all/any/none + name_contains filter selects — including
    missing-content records, which stay catalog-visible.
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
        .join(AssetContent, Asset.content_id == AssetContent.id)
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
