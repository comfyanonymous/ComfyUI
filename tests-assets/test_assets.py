import aiohttp
import pytest


@pytest.mark.asyncio
async def test_tags_listing_endpoint(http: aiohttp.ClientSession, api_base: str):
    # Include zero-usage tags by default
    async with http.get(api_base + "/api/tags", params={"limit": "50"}) as r1:
        body1 = await r1.json()
        assert r1.status == 200
        names = [t["name"] for t in body1["tags"]]
        # A few system tags from migration should exist:
        assert "models" in names
        assert "checkpoints" in names

    # Only used tags
    async with http.get(api_base + "/api/tags", params={"include_zero": "false"}) as r2:
        body2 = await r2.json()
        assert r2.status == 200
        # Should contain no tags
        assert not [t["name"] for t in body2["tags"]]

        # TODO-1: add some asset
        # TODO-2: check that "used" tags are now non zero amount

    # TODO-3: do a global teardown, so the state of ComfyUI is clear after each test, and all test can be run solo or one-by-one without any problems.
