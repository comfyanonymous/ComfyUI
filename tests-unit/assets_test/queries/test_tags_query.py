"""B-schema coverage for the two tag-listing query functions.

`list_tags_with_usage` backs `GET /api/tags`; `list_tag_counts_for_filtered_assets`
backs `GET /api/assets/tags/refine`. Both were rewritten over
`AssetTag`/`Asset`/`AssetContent` (todo 12 / D3+D4).
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from sqlalchemy.orm import Session

from app.assets.database.models import Asset, Tag
from app.assets.database.queries.records import (
    RecordPageSpec,
    create_content,
    create_record,
    list_records_page,
    mark_content_missing,
)
from app.assets.database.queries.tags import (
    list_tag_counts_for_filtered_assets,
    list_tags_with_usage,
)


def _seed_asset(
    session: Session, path: str, name: str, tags: Sequence[str]
) -> Asset:
    content = create_content(session, path=path)
    return create_record(session, content_id=content.id, name=name, tags=list(tags))


# --------------------------------------------------------------------------
# list_tags_with_usage
# --------------------------------------------------------------------------


def test_usage_counts_match_seeded_asset_tag_rows(session: Session) -> None:
    # Given three live assets carrying overlapping tags
    _seed_asset(session, "/models/a", "a", ["models", "checkpoint"])
    _seed_asset(session, "/models/b", "b", ["models"])
    _seed_asset(session, "/models/c", "c", ["lora"])
    session.commit()

    # When listing tag usage
    rows, total = list_tags_with_usage(session)

    # Then each tag's count is the number of assets that carry it
    assert dict(rows) == {"models": 2, "checkpoint": 1, "lora": 1}
    assert total == 3


def test_prefix_filters_tags_by_leading_substring(session: Session) -> None:
    # Given tags with distinct leading substrings
    _seed_asset(session, "/m/1", "1", ["model", "mask", "zebra"])
    session.commit()

    # When a prefix is supplied
    rows, total = list_tags_with_usage(session, prefix="ma")

    # Then only tags starting with that prefix survive, and total reflects it
    assert [name for name, _ in rows] == ["mask"]
    assert total == 1


def test_limit_and_offset_paginate_over_real_total(session: Session) -> None:
    # Given five distinct single-use tags
    for index in range(5):
        _seed_asset(session, f"/p/{index}", str(index), [f"tag{index}"])
    session.commit()

    # When paginating a name-ordered window
    rows, total = list_tags_with_usage(session, limit=2, offset=1, order="name_asc")

    # Then the window is the offset slice and total counts every tag, not the page
    assert [name for name, _ in rows] == ["tag1", "tag2"]
    assert total == 5


def test_include_zero_false_drops_unused_tags_and_shrinks_total(session: Session) -> None:
    # Given one used tag and one orphan tag with no asset
    _seed_asset(session, "/z/1", "1", ["used"])
    session.add(Tag(name="unused"))
    session.commit()

    # When include_zero toggles
    with_zero, total_with = list_tags_with_usage(session, include_zero=True)
    without_zero, total_without = list_tags_with_usage(session, include_zero=False)

    # Then the orphan appears only when zero counts are included
    assert ("unused", 0) in with_zero
    assert total_with == 2
    assert [name for name, _ in without_zero] == ["used"]
    assert total_without == 1


def test_order_is_count_desc_then_name_asc_or_name_asc(session: Session) -> None:
    # Given three tags with strictly increasing usage
    _seed_asset(session, "/o/1", "1", ["gamma"])
    _seed_asset(session, "/o/2", "2", ["gamma", "beta"])
    _seed_asset(session, "/o/3", "3", ["gamma", "beta", "alpha"])
    session.commit()

    # When ordered each way
    by_count, _ = list_tags_with_usage(session, order="count_desc")
    by_name, _ = list_tags_with_usage(session, order="name_asc")

    # Then count_desc ranks by usage; name_asc ranks alphabetically
    assert [name for name, _ in by_count] == ["gamma", "beta", "alpha"]
    assert [name for name, _ in by_name] == ["alpha", "beta", "gamma"]


def test_missing_content_hides_ordinary_tags_but_keeps_the_missing_tag(
    session: Session,
) -> None:
    # Given a live asset and one whose content went missing (auto-tagged "missing")
    _seed_asset(session, "/live", "live", ["foo"])
    gone = create_content(session, path="/gone")
    create_record(session, content_id=gone.id, name="gone", tags=["foo"])
    mark_content_missing(session, gone.id)
    session.commit()

    # When listing usage
    counts = dict(list_tags_with_usage(session)[0])

    # Then the missing asset's ordinary "foo" is suppressed, but its "missing" shows
    assert counts["foo"] == 1
    assert counts["missing"] == 1


# --------------------------------------------------------------------------
# list_tag_counts_for_filtered_assets
# --------------------------------------------------------------------------


def test_histogram_counts_every_tag_on_the_filtered_assets(session: Session) -> None:
    # Given assets, only some of which carry "models"
    _seed_asset(session, "/h/a", "a", ["models", "checkpoint"])
    _seed_asset(session, "/h/b", "b", ["models", "lora"])
    _seed_asset(session, "/h/c", "c", ["other"])
    session.commit()

    # When refining on "models"
    histogram = list_tag_counts_for_filtered_assets(session, include_tags=["models"])

    # Then every tag present on the matching assets is tallied
    assert histogram == {"models": 2, "checkpoint": 1, "lora": 1}


def test_histogram_excludes_missing_content_assets(session: Session) -> None:
    # Given a live and a missing-content asset that share a tag
    _seed_asset(session, "/hm/live", "live", ["models"])
    gone = create_content(session, path="/hm/gone")
    create_record(session, content_id=gone.id, name="gone", tags=["models"])
    mark_content_missing(session, gone.id)
    session.commit()

    # When refining on "models"
    histogram = list_tag_counts_for_filtered_assets(session, include_tags=["models"])

    # Then only the live asset contributes (missing content is dropped by live clause)
    assert histogram == {"models": 1}


def test_refine_and_list_share_the_same_filtered_asset_set(session: Session) -> None:
    # Given assets stressing all/none/name_contains plus a missing-content decoy
    kept = _seed_asset(session, "/s/a", "checkpoint-a", ["models", "checkpoint"])
    _seed_asset(session, "/s/b", "checkpoint-b", ["models", "archived"])
    _seed_asset(session, "/s/c", "lora-c", ["models"])
    decoy = create_content(session, path="/s/d")
    create_record(
        session, content_id=decoy.id, name="check-d", tags=["models", "checkpoint"]
    )
    mark_content_missing(session, decoy.id)
    session.commit()

    # When the identical filter is applied through /api/assets and refine
    spec = RecordPageSpec(
        all_tags=("models",),
        none_tags=("archived",),
        name_contains="check",
        limit=100,
    )
    records, tag_map, _ = list_records_page(session, spec)
    expected = Counter(
        tag for record in records for tag in tag_map.get(record.id, [])
    )
    histogram = list_tag_counts_for_filtered_assets(
        session,
        include_tags=["models"],
        exclude_tags=["archived"],
        name_contains="check",
        limit=100,
    )

    # Then refine's histogram is exactly the tag tally over the /api/assets result set
    # (proves both reuse build_record_tag_filter_clauses + live_asset_content_clause)
    assert {record.id for record in records} == {kept.id}
    assert histogram == dict(expected)
    assert histogram == {"models": 1, "checkpoint": 1}


def test_histogram_any_tags_matches_the_list_union(session: Session) -> None:
    # Given three assets with disjoint tags
    red = _seed_asset(session, "/an/a", "a", ["red"])
    blue = _seed_asset(session, "/an/b", "b", ["blue"])
    _seed_asset(session, "/an/c", "c", ["green"])
    session.commit()

    # When refining with any_tags mirroring a list_records_page any-query
    records, _, _ = list_records_page(
        session, RecordPageSpec(any_tags=("red", "blue"), limit=100)
    )
    histogram = list_tag_counts_for_filtered_assets(session, any_tags=["red", "blue"])

    # Then both select the union of the two tags
    assert {record.id for record in records} == {red.id, blue.id}
    assert histogram == {"red": 1, "blue": 1}
