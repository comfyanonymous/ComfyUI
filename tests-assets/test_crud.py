import aiohttp
import pytest


@pytest.mark.asyncio
async def test_get_and_delete_asset(http: aiohttp.ClientSession, api_base: str, seeded_asset: dict):
    aid = seeded_asset["id"]

    # GET detail
    async with http.get(f"{api_base}/api/assets/{aid}") as rg:
        detail = await rg.json()
        assert rg.status == 200, detail
        assert detail["id"] == aid
        assert "user_metadata" in detail
        assert "filename" in detail["user_metadata"]

    # DELETE
    async with http.delete(f"{api_base}/api/assets/{aid}") as rd:
        assert rd.status == 204

    # GET again -> 404
    async with http.get(f"{api_base}/api/assets/{aid}") as rg2:
        body = await rg2.json()
        assert rg2.status == 404
        assert body["error"]["code"] == "ASSET_NOT_FOUND"


@pytest.mark.asyncio
async def test_update_asset_fields(http: aiohttp.ClientSession, api_base: str, seeded_asset: dict):
    aid = seeded_asset["id"]

    payload = {
        "name": "unit_1_renamed.safetensors",
        "tags": ["models", "checkpoints", "unit-tests", "beta"],
        "user_metadata": {"purpose": "updated", "epoch": 2},
    }
    async with http.put(f"{api_base}/api/assets/{aid}", json=payload) as ru:
        body = await ru.json()
        assert ru.status == 200, body
        assert body["name"] == payload["name"]
        assert "beta" in body["tags"]
        assert body["user_metadata"]["purpose"] == "updated"
        # filename should still be present and normalized by server
        assert "filename" in body["user_metadata"]


@pytest.mark.asyncio
async def test_head_asset_by_hash_and_invalids(http: aiohttp.ClientSession, api_base: str, seeded_asset: dict):
    h = seeded_asset["asset_hash"]

    # Existing
    async with http.head(f"{api_base}/api/assets/hash/{h}") as rh1:
        assert rh1.status == 200

    # Non-existent
    async with http.head(f"{api_base}/api/assets/hash/blake3:{'0'*64}") as rh2:
        assert rh2.status == 404

    # Invalid format
    async with http.head(f"{api_base}/api/assets/hash/not_a_hash") as rh3:
        jb = await rh3.json()
        assert rh3.status == 400
        assert jb is None  # HEAD request should not include "body" in response
