import json
import aiohttp
import pytest


@pytest.mark.asyncio
async def test_upload_requires_multipart(http: aiohttp.ClientSession, api_base: str):
    async with http.post(api_base + "/api/assets", json={"foo": "bar"}) as r:
        body = await r.json()
        assert r.status == 415
        assert body["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


@pytest.mark.asyncio
async def test_upload_missing_file_and_hash(http: aiohttp.ClientSession, api_base: str):
    form = aiohttp.FormData(default_to_multipart=True)
    form.add_field("tags", json.dumps(["models", "checkpoints", "unit-tests"]))
    form.add_field("name", "x.safetensors")
    async with http.post(api_base + "/api/assets", data=form) as r:
        body = await r.json()
        assert r.status == 400
        assert body["error"]["code"] == "MISSING_FILE"


@pytest.mark.asyncio
async def test_upload_models_unknown_category(http: aiohttp.ClientSession, api_base: str):
    form = aiohttp.FormData()
    form.add_field("file", b"A" * 128, filename="m.safetensors", content_type="application/octet-stream")
    form.add_field("tags", json.dumps(["models", "no_such_category", "unit-tests"]))
    form.add_field("name", "m.safetensors")
    async with http.post(api_base + "/api/assets", data=form) as r:
        body = await r.json()
        assert r.status == 400
        assert body["error"]["code"] == "INVALID_BODY"
        assert "unknown models category" in body["error"]["message"] or "unknown model category" in body["error"]["message"]


@pytest.mark.asyncio
async def test_upload_tags_traversal_guard(http: aiohttp.ClientSession, api_base: str):
    form = aiohttp.FormData()
    form.add_field("file", b"A" * 256, filename="evil.safetensors", content_type="application/octet-stream")
    # '..' should be rejected by destination resolver
    form.add_field("tags", json.dumps(["models", "checkpoints", "unit-tests", "..", "zzz"]))
    form.add_field("name", "evil.safetensors")
    async with http.post(api_base + "/api/assets", data=form) as r:
        body = await r.json()
        assert r.status == 400
        assert body["error"]["code"] in ("BAD_REQUEST", "INVALID_BODY")


@pytest.mark.asyncio
async def test_upload_ok_duplicate_reference(http: aiohttp.ClientSession, api_base: str, make_asset_bytes):
    name = "dup_a.safetensors"
    tags = ["models", "checkpoints", "unit-tests", "alpha"]
    meta = {"purpose": "dup"}
    data = make_asset_bytes(name)
    form1 = aiohttp.FormData()
    form1.add_field("file", data, filename=name, content_type="application/octet-stream")
    form1.add_field("tags", json.dumps(tags))
    form1.add_field("name", name)
    form1.add_field("user_metadata", json.dumps(meta))
    async with http.post(api_base + "/api/assets", data=form1) as r1:
        a1 = await r1.json()
        assert r1.status == 201, a1
        assert a1["created_new"] is True

    # Second upload with the same data and name should return created_new == False and the same asset
    form2 = aiohttp.FormData()
    form2.add_field("file", data, filename=name, content_type="application/octet-stream")
    form2.add_field("tags", json.dumps(tags))
    form2.add_field("name", name)
    form2.add_field("user_metadata", json.dumps(meta))
    async with http.post(api_base + "/api/assets", data=form2) as r2:
        a2 = await r2.json()
        assert r2.status == 200, a2
        assert a2["created_new"] is False
        assert a2["asset_hash"] == a1["asset_hash"]
        assert a2["id"] == a1["id"]  # old reference

    # Third upload with the same data but new name should return created_new == False and the new AssetReference
    form3 = aiohttp.FormData()
    form3.add_field("file", data, filename=name, content_type="application/octet-stream")
    form3.add_field("tags", json.dumps(tags))
    form3.add_field("name", name + "_d")
    form3.add_field("user_metadata", json.dumps(meta))
    async with http.post(api_base + "/api/assets", data=form3) as r2:
        a3 = await r2.json()
        assert r2.status == 200, a3
        assert a3["created_new"] is False
        assert a3["asset_hash"] == a1["asset_hash"]
        assert a3["id"] != a1["id"]  # old reference


@pytest.mark.asyncio
async def test_upload_fastpath_from_existing_hash_no_file(http: aiohttp.ClientSession, api_base: str):
    # Seed a small file first
    name = "fastpath_seed.safetensors"
    tags = ["models", "checkpoints", "unit-tests"]
    meta = {}
    form1 = aiohttp.FormData()
    form1.add_field("file", b"B" * 1024, filename=name, content_type="application/octet-stream")
    form1.add_field("tags", json.dumps(tags))
    form1.add_field("name", name)
    form1.add_field("user_metadata", json.dumps(meta))
    async with http.post(api_base + "/api/assets", data=form1) as r1:
        b1 = await r1.json()
        assert r1.status == 201, b1
        h = b1["asset_hash"]

    # Now POST /api/assets with only hash and no file
    form2 = aiohttp.FormData()
    form2.add_field("hash", h)
    form2.add_field("tags", json.dumps(tags))
    form2.add_field("name", "fastpath_copy.safetensors")
    form2.add_field("user_metadata", json.dumps({"purpose": "copy"}))
    async with http.post(api_base + "/api/assets", data=form2) as r2:
        b2 = await r2.json()
        assert r2.status == 200, b2  # fast path returns 200 with created_new == False
        assert b2["created_new"] is False
        assert b2["asset_hash"] == h


@pytest.mark.asyncio
async def test_create_from_hash_endpoint_404(http: aiohttp.ClientSession, api_base: str):
    payload = {
        "hash": "blake3:" + "0" * 64,
        "name": "nonexistent.bin",
        "tags": ["models", "checkpoints", "unit-tests"],
    }
    async with http.post(api_base + "/api/assets/from-hash", json=payload) as r:
        body = await r.json()
        assert r.status == 404
        assert body["error"]["code"] == "ASSET_NOT_FOUND"
