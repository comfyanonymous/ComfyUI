import uuid
import aiohttp
import pytest


@pytest.mark.asyncio
async def test_create_from_hash_success(
    http: aiohttp.ClientSession, api_base: str, seeded_asset: dict
):
    h = seeded_asset["asset_hash"]
    payload = {
        "hash": h,
        "name": "from_hash_ok.safetensors",
        "tags": ["models", "checkpoints", "unit-tests", "from-hash"],
        "user_metadata": {"k": "v"},
    }
    async with http.post(f"{api_base}/api/assets/from-hash", json=payload) as r1:
        b1 = await r1.json()
        assert r1.status == 201, b1
        assert b1["asset_hash"] == h
        assert b1["created_new"] is False
        aid = b1["id"]

    # Calling again with the same name should return the same AssetInfo id
    async with http.post(f"{api_base}/api/assets/from-hash", json=payload) as r2:
        b2 = await r2.json()
        assert r2.status == 201, b2
        assert b2["id"] == aid


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
async def test_delete_upon_reference_count(
    http: aiohttp.ClientSession, api_base: str, seeded_asset: dict
):
    # Create a second reference to the same asset via from-hash
    src_hash = seeded_asset["asset_hash"]
    payload = {
        "hash": src_hash,
        "name": "unit_ref_copy.safetensors",
        "tags": ["models", "checkpoints", "unit-tests", "del-flow"],
        "user_metadata": {"note": "copy"},
    }
    async with http.post(f"{api_base}/api/assets/from-hash", json=payload) as r2:
        copy = await r2.json()
        assert r2.status == 201, copy
        assert copy["asset_hash"] == src_hash
        assert copy["created_new"] is False

    # Delete original reference -> asset identity must remain
    aid1 = seeded_asset["id"]
    async with http.delete(f"{api_base}/api/assets/{aid1}") as rd1:
        assert rd1.status == 204

    async with http.head(f"{api_base}/api/assets/hash/{src_hash}") as rh1:
        assert rh1.status == 200  # identity still present

    # Delete the last reference with default semantics -> identity and cached files removed
    aid2 = copy["id"]
    async with http.delete(f"{api_base}/api/assets/{aid2}") as rd2:
        assert rd2.status == 204

    async with http.head(f"{api_base}/api/assets/hash/{src_hash}") as rh2:
        assert rh2.status == 404  # orphan content removed


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
async def test_head_asset_by_hash(http: aiohttp.ClientSession, api_base: str, seeded_asset: dict):
    h = seeded_asset["asset_hash"]

    # Existing
    async with http.head(f"{api_base}/api/assets/hash/{h}") as rh1:
        assert rh1.status == 200

    # Non-existent
    async with http.head(f"{api_base}/api/assets/hash/blake3:{'0'*64}") as rh2:
        assert rh2.status == 404


@pytest.mark.asyncio
async def test_head_asset_bad_hash_returns_400_and_no_body(http: aiohttp.ClientSession, api_base: str):
    # Invalid format; handler returns a JSON error, but HEAD responses must not carry a payload.
    # aiohttp exposes an empty body for HEAD, so validate status and that there is no payload.
    async with http.head(f"{api_base}/api/assets/hash/not_a_hash") as rh:
        assert rh.status == 400
        body = await rh.read()
        assert body == b""


@pytest.mark.asyncio
async def test_delete_nonexistent_returns_404(http: aiohttp.ClientSession, api_base: str):
    bogus = str(uuid.uuid4())
    async with http.delete(f"{api_base}/api/assets/{bogus}") as r:
        body = await r.json()
        assert r.status == 404
        assert body["error"]["code"] == "ASSET_NOT_FOUND"


@pytest.mark.asyncio
async def test_create_from_hash_invalids(http: aiohttp.ClientSession, api_base: str):
    # Bad hash algorithm
    bad = {
        "hash": "sha256:" + "0" * 64,
        "name": "x.bin",
        "tags": ["models", "checkpoints", "unit-tests"],
    }
    async with http.post(f"{api_base}/api/assets/from-hash", json=bad) as r1:
        b1 = await r1.json()
        assert r1.status == 400
        assert b1["error"]["code"] == "INVALID_BODY"

    # Invalid JSON body
    async with http.post(f"{api_base}/api/assets/from-hash", data=b"{not json}") as r2:
        b2 = await r2.json()
        assert r2.status == 400
        assert b2["error"]["code"] == "INVALID_JSON"


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
