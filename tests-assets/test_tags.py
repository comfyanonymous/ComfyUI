import json
import uuid

import aiohttp
import pytest


@pytest.mark.asyncio
async def test_tags_present(http: aiohttp.ClientSession, api_base: str, seeded_asset: dict):
    # Include zero-usage tags by default
    async with http.get(api_base + "/api/tags", params={"limit": "50"}) as r1:
        body1 = await r1.json()
        assert r1.status == 200
        names = [t["name"] for t in body1["tags"]]
        # A few system tags from migration should exist:
        assert "models" in names
        assert "checkpoints" in names

    # Only used tags before we add anything new from this test cycle
    async with http.get(api_base + "/api/tags", params={"include_zero": "false"}) as r2:
        body2 = await r2.json()
        assert r2.status == 200
        # We already seeded one asset via fixture, so used tags must be non-empty
        used_names = [t["name"] for t in body2["tags"]]
        assert "models" in used_names
        assert "checkpoints" in used_names

    # Prefix filter should refine the list
    async with http.get(api_base + "/api/tags", params={"include_zero": "false", "prefix": "uni"}) as r3:
        b3 = await r3.json()
        assert r3.status == 200
        names3 = [t["name"] for t in b3["tags"]]
        assert "unit-tests" in names3
        assert "models" not in names3  # filtered out by prefix

    # Order by name ascending should be stable
    async with http.get(api_base + "/api/tags", params={"include_zero": "false", "order": "name_asc"}) as r4:
        b4 = await r4.json()
        assert r4.status == 200
        names4 = [t["name"] for t in b4["tags"]]
        assert names4 == sorted(names4)


@pytest.mark.asyncio
async def test_tags_empty_usage(http: aiohttp.ClientSession, api_base: str, asset_factory, make_asset_bytes):
    # Baseline: system tags exist when include_zero (default) is true
    async with http.get(api_base + "/api/tags", params={"limit": "500"}) as r1:
        body1 = await r1.json()
        assert r1.status == 200
        names = [t["name"] for t in body1["tags"]]
        assert "models" in names and "checkpoints" in names

    # Create a short-lived asset under input with a unique custom tag
    scope = f"tags-empty-usage-{uuid.uuid4().hex[:6]}"
    custom_tag = f"temp-{uuid.uuid4().hex[:8]}"
    name = "tag_seed.bin"
    _asset = await asset_factory(
        name,
        ["input", "unit-tests", scope, custom_tag],
        {},
        make_asset_bytes(name, 512),
    )

    # While the asset exists, the custom tag must appear when include_zero=false
    async with http.get(
        api_base + "/api/tags",
        params={"include_zero": "false", "prefix": custom_tag, "limit": "50"},
    ) as r2:
        body2 = await r2.json()
        assert r2.status == 200
        used_names = [t["name"] for t in body2["tags"]]
        assert custom_tag in used_names

    # Delete the asset so the tag usage drops to zero
    async with http.delete(f"{api_base}/api/assets/{_asset['id']}") as rd:
        assert rd.status == 204

    # Now the custom tag must not be returned when include_zero=false
    async with http.get(
        api_base + "/api/tags",
        params={"include_zero": "false", "prefix": custom_tag, "limit": "50"},
    ) as r3:
        body3 = await r3.json()
        assert r3.status == 200
        names_after = [t["name"] for t in body3["tags"]]
        assert custom_tag not in names_after
        assert not names_after  # filtered view should be empty now


@pytest.mark.asyncio
async def test_add_and_remove_tags(http: aiohttp.ClientSession, api_base: str, seeded_asset: dict):
    aid = seeded_asset["id"]

    # Add tags with duplicates and mixed case
    payload_add = {"tags": ["NewTag", "unit-tests", "newtag", "BETA"]}
    async with http.post(f"{api_base}/api/assets/{aid}/tags", json=payload_add) as r1:
        b1 = await r1.json()
        assert r1.status == 200, b1
        # normalized, deduplicated; 'unit-tests' was already present from the seed
        assert set(b1["added"]) == {"newtag", "beta"}
        assert set(b1["already_present"]) == {"unit-tests"}
        assert "newtag" in b1["total_tags"] and "beta" in b1["total_tags"]

    async with http.get(f"{api_base}/api/assets/{aid}") as rg:
        g = await rg.json()
        assert rg.status == 200
        tags_now = set(g["tags"])
        assert {"newtag", "beta"}.issubset(tags_now)

    # Remove a tag and a non-existent tag
    payload_del = {"tags": ["newtag", "does-not-exist"]}
    async with http.delete(f"{api_base}/api/assets/{aid}/tags", json=payload_del) as r2:
        b2 = await r2.json()
        assert r2.status == 200
        assert set(b2["removed"]) == {"newtag"}
        assert set(b2["not_present"]) == {"does-not-exist"}

    # Verify remaining tags after deletion
    async with http.get(f"{api_base}/api/assets/{aid}") as rg2:
        g2 = await rg2.json()
        assert rg2.status == 200
        tags_later = set(g2["tags"])
        assert "newtag" not in tags_later
        assert "beta" in tags_later  # still present


@pytest.mark.asyncio
async def test_tags_list_order_and_prefix(http: aiohttp.ClientSession, api_base: str, seeded_asset: dict):
    aid = seeded_asset["id"]
    h = seeded_asset["asset_hash"]

    # Add both tags to the seeded asset (usage: orderaaa=1, orderbbb=1)
    async with http.post(f"{api_base}/api/assets/{aid}/tags", json={"tags": ["orderaaa", "orderbbb"]}) as r_add:
        add_body = await r_add.json()
        assert r_add.status == 200, add_body

    # Create another AssetInfo from the same content but tagged ONLY with 'orderbbb'.
    payload = {
        "hash": h,
        "name": "order_only_bbb.safetensors",
        "tags": ["input", "unit-tests", "orderbbb"],
        "user_metadata": {},
    }
    async with http.post(f"{api_base}/api/assets/from-hash", json=payload) as r_copy:
        copy_body = await r_copy.json()
        assert r_copy.status == 201, copy_body

    # 1) Default order (count_desc): 'orderbbb' should come before 'orderaaa'
    #    because it has higher usage (2 vs 1).
    async with http.get(api_base + "/api/tags", params={"prefix": "order", "include_zero": "false"}) as r1:
        b1 = await r1.json()
        assert r1.status == 200, b1
        names1 = [t["name"] for t in b1["tags"]]
        counts1 = {t["name"]: t["count"] for t in b1["tags"]}
        # Both must be present within the prefix subset
        assert "orderaaa" in names1 and "orderbbb" in names1
        # Usage of 'orderbbb' must be >= 'orderaaa'; in our setup it's 2 vs 1
        assert counts1["orderbbb"] >= counts1["orderaaa"]
        # And with count_desc, 'orderbbb' appears earlier than 'orderaaa'
        assert names1.index("orderbbb") < names1.index("orderaaa")

    # 2) name_asc: lexical order should flip the relative order
    async with http.get(
        api_base + "/api/tags",
        params={"prefix": "order", "include_zero": "false", "order": "name_asc"},
    ) as r2:
        b2 = await r2.json()
        assert r2.status == 200, b2
        names2 = [t["name"] for t in b2["tags"]]
        assert "orderaaa" in names2 and "orderbbb" in names2
        assert names2.index("orderaaa") < names2.index("orderbbb")

    # 3) invalid limit rejected (existing negative case retained)
    async with http.get(api_base + "/api/tags", params={"limit": "1001"}) as r3:
        b3 = await r3.json()
        assert r3.status == 400
        assert b3["error"]["code"] == "INVALID_QUERY"


@pytest.mark.asyncio
async def test_tags_endpoints_invalid_bodies(http: aiohttp.ClientSession, api_base: str, seeded_asset: dict):
    aid = seeded_asset["id"]

    # Add with empty list
    async with http.post(f"{api_base}/api/assets/{aid}/tags", json={"tags": []}) as r1:
        b1 = await r1.json()
        assert r1.status == 400
        assert b1["error"]["code"] == "INVALID_BODY"

    # Remove with wrong type
    async with http.delete(f"{api_base}/api/assets/{aid}/tags", json={"tags": [123]}) as r2:
        b2 = await r2.json()
        assert r2.status == 400
        assert b2["error"]["code"] == "INVALID_BODY"

    # metadata_filter provided as JSON array should be rejected (must be object)
    async with http.get(
        api_base + "/api/assets",
        params={"metadata_filter": json.dumps([{"x": 1}])},
    ) as r3:
        b3 = await r3.json()
        assert r3.status == 400
        assert b3["error"]["code"] == "INVALID_QUERY"
