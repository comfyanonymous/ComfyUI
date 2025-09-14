import asyncio
import json
import uuid

import aiohttp
import pytest


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
    form2 = aiohttp.FormData(default_to_multipart=True)
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
async def test_upload_fastpath_with_known_hash_and_file(
    http: aiohttp.ClientSession, api_base: str
):
    # Seed
    form1 = aiohttp.FormData()
    form1.add_field("file", b"C" * 128, filename="seed.safetensors", content_type="application/octet-stream")
    form1.add_field("tags", json.dumps(["models", "checkpoints", "unit-tests", "fp"]))
    form1.add_field("name", "seed.safetensors")
    form1.add_field("user_metadata", json.dumps({}))
    async with http.post(api_base + "/api/assets", data=form1) as r1:
        b1 = await r1.json()
        assert r1.status == 201, b1
        h = b1["asset_hash"]

    # Send both file and hash of existing content -> server must drain file and create from hash (200)
    form2 = aiohttp.FormData()
    form2.add_field("file", b"ignored" * 10, filename="ignored.bin", content_type="application/octet-stream")
    form2.add_field("hash", h)
    form2.add_field("tags", json.dumps(["models", "checkpoints", "unit-tests", "fp"]))
    form2.add_field("name", "copy_from_hash.safetensors")
    form2.add_field("user_metadata", json.dumps({}))
    async with http.post(api_base + "/api/assets", data=form2) as r2:
        b2 = await r2.json()
        assert r2.status == 200, b2
        assert b2["created_new"] is False
        assert b2["asset_hash"] == h


@pytest.mark.asyncio
async def test_upload_multiple_tags_fields_are_merged(http: aiohttp.ClientSession, api_base: str):
    form = aiohttp.FormData()
    form.add_field("file", b"B" * 256, filename="merge.safetensors", content_type="application/octet-stream")
    form.add_field("tags", "models,checkpoints")  # CSV
    form.add_field("tags", json.dumps(["unit-tests", "alpha"]))  # JSON array in second field
    form.add_field("name", "merge.safetensors")
    form.add_field("user_metadata", json.dumps({"u": 1}))
    async with http.post(api_base + "/api/assets", data=form) as r1:
        created = await r1.json()
        assert r1.status in (200, 201), created
        aid = created["id"]

    # Verify all tags are present on the resource
    async with http.get(f"{api_base}/api/assets/{aid}") as rg:
        detail = await rg.json()
        assert rg.status == 200, detail
        tags = set(detail["tags"])
        assert {"models", "checkpoints", "unit-tests", "alpha"}.issubset(tags)


@pytest.mark.asyncio
@pytest.mark.parametrize("root", ["input", "output"])
async def test_concurrent_upload_identical_bytes_different_names(
    root: str,
    http: aiohttp.ClientSession,
    api_base: str,
    make_asset_bytes,
):
    """
    Two concurrent uploads of identical bytes but different names.
    Expect a single Asset (same hash), two AssetInfo rows, and exactly one created_new=True.
    """
    scope = f"concupload-{uuid.uuid4().hex[:6]}"
    name1, name2 = "cu_a.bin", "cu_b.bin"
    data = make_asset_bytes("concurrent", 4096)
    tags = [root, "unit-tests", scope]

    def _form(name: str) -> aiohttp.FormData:
        f = aiohttp.FormData()
        f.add_field("file", data, filename=name, content_type="application/octet-stream")
        f.add_field("tags", json.dumps(tags))
        f.add_field("name", name)
        f.add_field("user_metadata", json.dumps({}))
        return f

    r1, r2 = await asyncio.gather(
        http.post(api_base + "/api/assets", data=_form(name1)),
        http.post(api_base + "/api/assets", data=_form(name2)),
    )
    b1, b2 = await r1.json(), await r2.json()
    assert r1.status in (200, 201), b1
    assert r2.status in (200, 201), b2
    assert b1["asset_hash"] == b2["asset_hash"]
    assert b1["id"] != b2["id"]

    created_flags = sorted([bool(b1.get("created_new")), bool(b2.get("created_new"))])
    assert created_flags == [False, True]

    async with http.get(
        api_base + "/api/assets",
        params={"include_tags": f"unit-tests,{scope}", "sort": "name"},
    ) as rl:
        bl = await rl.json()
        assert rl.status == 200, bl
        names = [a["name"] for a in bl.get("assets", [])]
        assert set([name1, name2]).issubset(names)


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


@pytest.mark.asyncio
async def test_upload_zero_byte_rejected(http: aiohttp.ClientSession, api_base: str):
    form = aiohttp.FormData()
    form.add_field("file", b"", filename="empty.safetensors", content_type="application/octet-stream")
    form.add_field("tags", json.dumps(["models", "checkpoints", "unit-tests", "edge"]))
    form.add_field("name", "empty.safetensors")
    form.add_field("user_metadata", json.dumps({}))
    async with http.post(api_base + "/api/assets", data=form) as r:
        body = await r.json()
        assert r.status == 400
        assert body["error"]["code"] == "EMPTY_UPLOAD"


@pytest.mark.asyncio
async def test_upload_invalid_root_tag_rejected(http: aiohttp.ClientSession, api_base: str):
    form = aiohttp.FormData()
    form.add_field("file", b"A" * 64, filename="badroot.bin", content_type="application/octet-stream")
    form.add_field("tags", json.dumps(["not-a-root", "whatever"]))
    form.add_field("name", "badroot.bin")
    form.add_field("user_metadata", json.dumps({}))
    async with http.post(api_base + "/api/assets", data=form) as r:
        body = await r.json()
        assert r.status == 400
        assert body["error"]["code"] == "INVALID_BODY"


@pytest.mark.asyncio
async def test_upload_user_metadata_must_be_json(http: aiohttp.ClientSession, api_base: str):
    form = aiohttp.FormData()
    form.add_field("file", b"A" * 128, filename="badmeta.bin", content_type="application/octet-stream")
    form.add_field("tags", json.dumps(["models", "checkpoints", "unit-tests", "edge"]))
    form.add_field("name", "badmeta.bin")
    form.add_field("user_metadata", "{not json}")  # invalid
    async with http.post(api_base + "/api/assets", data=form) as r:
        body = await r.json()
        assert r.status == 400
        assert body["error"]["code"] == "INVALID_BODY"


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
        assert body["error"]["message"].startswith("unknown models category")


@pytest.mark.asyncio
async def test_upload_models_requires_category(http: aiohttp.ClientSession, api_base: str):
    form = aiohttp.FormData()
    form.add_field("file", b"A" * 64, filename="nocat.safetensors", content_type="application/octet-stream")
    form.add_field("tags", json.dumps(["models"]))  # missing category
    form.add_field("name", "nocat.safetensors")
    form.add_field("user_metadata", json.dumps({}))
    async with http.post(api_base + "/api/assets", data=form) as r:
        body = await r.json()
        assert r.status == 400
        assert body["error"]["code"] == "INVALID_BODY"


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
