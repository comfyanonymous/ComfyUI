import aiohttp
import pytest


@pytest.mark.asyncio
async def test_get_update_download_bad_ids(http: aiohttp.ClientSession, api_base: str):
    # All endpoints should be not found, as we UUID regex directly in the route definition.
    bad_id = "not-a-uuid"

    async with http.get(f"{api_base}/api/assets/{bad_id}") as r1:
        assert r1.status == 404

    async with http.get(f"{api_base}/api/assets/{bad_id}/content") as r3:
        assert r3.status == 404


@pytest.mark.asyncio
async def test_update_requires_at_least_one_field(http: aiohttp.ClientSession, api_base: str, seeded_asset: dict):
    aid = seeded_asset["id"]
    async with http.put(f"{api_base}/api/assets/{aid}", json={}) as r:
        body = await r.json()
        assert r.status == 400
        assert body["error"]["code"] == "INVALID_BODY"
