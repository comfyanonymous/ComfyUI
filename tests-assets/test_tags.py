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
async def test_tags_empty_usage(http: aiohttp.ClientSession, api_base: str):
    # Include zero-usage tags by default
    async with http.get(api_base + "/api/tags", params={"limit": "50"}) as r1:
        body1 = await r1.json()
        assert r1.status == 200
        names = [t["name"] for t in body1["tags"]]
        # A few system tags from migration should exist:
        assert "models" in names
        assert "checkpoints" in names

    # With include_zero=False there should be no tags returned for the database without Assets.
    async with http.get(api_base + "/api/tags", params={"include_zero": "false"}) as r2:
        body2 = await r2.json()
        assert r2.status == 200
        assert not [t["name"] for t in body2["tags"]]


@pytest.mark.asyncio
async def test_add_and_remove_tags(http: aiohttp.ClientSession, api_base: str, seeded_asset: dict):
    aid = seeded_asset["id"]

    # Add tags with duplicates and mixed case
    payload_add = {"tags": ["NewTag", "unit-tests", "newtag", "BETA"]}
    async with http.post(f"{api_base}/api/assets/{aid}/tags", json=payload_add) as r1:
        b1 = await r1.json()
        assert r1.status == 200, b1
        # normalized and deduplicated
        assert "newtag" in b1["added"] or "beta" in b1["added"] or "unit-tests" not in b1["added"]

    async with http.get(f"{api_base}/api/assets/{aid}") as rg:
        g = await rg.json()
        assert rg.status == 200
        tags_now = set(g["tags"])
        assert "newtag" in tags_now
        assert "beta" in tags_now

    # Remove a tag and a non-existent tag
    payload_del = {"tags": ["newtag", "does-not-exist"]}
    async with http.delete(f"{api_base}/api/assets/{aid}/tags", json=payload_del) as r2:
        b2 = await r2.json()
        assert r2.status == 200
        assert "newtag" in b2["removed"]
        assert "does-not-exist" in b2["not_present"]


@pytest.mark.asyncio
async def test_tags_list_order_and_prefix(http: aiohttp.ClientSession, api_base: str, seeded_asset: dict):
    # name ascending
    async with http.get(api_base + "/api/tags", params={"order": "name_asc", "limit": "100"}) as r1:
        b1 = await r1.json()
        assert r1.status == 200
        names = [t["name"] for t in b1["tags"]]
        assert names == sorted(names)

    # invalid limit rejected
    async with http.get(api_base + "/api/tags", params={"limit": "1001"}) as r2:
        b2 = await r2.json()
        assert r2.status == 400
        assert b2["error"]["code"] == "INVALID_QUERY"
