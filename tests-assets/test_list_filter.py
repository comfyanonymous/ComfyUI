import json
import aiohttp
import pytest


@pytest.mark.asyncio
async def test_list_assets_paging_and_sort(http: aiohttp.ClientSession, api_base: str, asset_factory, make_asset_bytes):
    names = ["a1_u.safetensors", "a2_u.safetensors", "a3_u.safetensors"]
    for n in names:
        await asset_factory(
            n,
            ["models", "checkpoints", "unit-tests", "paging"],
            {"epoch": 1},
            make_asset_bytes(n, size=2048),
        )

    # name ascending for stable order
    async with http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,paging", "sort": "name", "order": "asc", "limit": "2", "offset": "0"},
    ) as r1:
        b1 = await r1.json()
        assert r1.status == 200
        got1 = [a["name"] for a in b1["assets"]]
        assert got1 == sorted(names)[:2]
        assert b1["has_more"] is True

    async with http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,paging", "sort": "name", "order": "asc", "limit": "2", "offset": "2"},
    ) as r2:
        b2 = await r2.json()
        assert r2.status == 200
        got2 = [a["name"] for a in b2["assets"]]
        assert got2 == sorted(names)[2:]
        assert b2["has_more"] is False


@pytest.mark.asyncio
async def test_list_assets_include_exclude_and_name_contains(http: aiohttp.ClientSession, api_base: str, asset_factory):
    a = await asset_factory("inc_a.safetensors", ["models", "checkpoints", "unit-tests", "alpha"], {}, b"X" * 1024)
    b = await asset_factory("inc_b.safetensors", ["models", "checkpoints", "unit-tests", "beta"], {}, b"Y" * 1024)

    async with http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,alpha", "exclude_tags": "beta", "limit": "50"},
    ) as r:
        body = await r.json()
        assert r.status == 200
        names = [x["name"] for x in body["assets"]]
        assert a["name"] in names
        assert b["name"] not in names

    async with http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests", "name_contains": "inc_"},
    ) as r2:
        body2 = await r2.json()
        assert r2.status == 200
        names2 = [x["name"] for x in body2["assets"]]
        assert a["name"] in names2
        assert b["name"] in names2

    async with http.get(
        api_base + "/api/assets",
        params={"include_tags": "non-existing-tag"},
    ) as r2:
        body3 = await r2.json()
        assert r2.status == 200
        assert not body3["assets"]


@pytest.mark.asyncio
async def test_list_assets_invalid_query_rejected(http: aiohttp.ClientSession, api_base: str):
    # limit too small
    async with http.get(api_base + "/api/assets", params={"limit": "0"}) as r1:
        b1 = await r1.json()
        assert r1.status == 400
        assert b1["error"]["code"] == "INVALID_QUERY"

    # bad metadata JSON
    async with http.get(api_base + "/api/assets", params={"metadata_filter": "{not json"}) as r2:
        b2 = await r2.json()
        assert r2.status == 400
        assert b2["error"]["code"] == "INVALID_QUERY"
