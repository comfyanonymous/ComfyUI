import json

import aiohttp
import pytest


@pytest.mark.asyncio
async def test_meta_and_across_keys_and_types(http: aiohttp.ClientSession, api_base: str, asset_factory, make_asset_bytes):
    name = "mf_and_mix.safetensors"
    tags = ["models", "checkpoints", "unit-tests", "mf-and"]
    meta = {"purpose": "mix", "epoch": 1, "active": True, "score": 1.23}
    await asset_factory(name, tags, meta, make_asset_bytes(name, 4096))

    # All keys must match (AND semantics)
    f_ok = {"purpose": "mix", "epoch": 1, "active": True, "score": 1.23}
    async with http.get(
        api_base + "/api/assets",
        params={
            "include_tags": "unit-tests,mf-and",
            "metadata_filter": json.dumps(f_ok),
        },
    ) as r1:
        b1 = await r1.json()
        assert r1.status == 200
        names = [a["name"] for a in b1["assets"]]
        assert name in names

    # One key mismatched -> no result
    f_bad = {"purpose": "mix", "epoch": 2, "active": True}
    async with http.get(
        api_base + "/api/assets",
        params={
            "include_tags": "unit-tests,mf-and",
            "metadata_filter": json.dumps(f_bad),
        },
    ) as r2:
        b2 = await r2.json()
        assert r2.status == 200
        assert not b2["assets"]


@pytest.mark.asyncio
async def test_meta_type_strictness_int_vs_str_and_bool(http, api_base, asset_factory, make_asset_bytes):
    name = "mf_types.safetensors"
    tags = ["models", "checkpoints", "unit-tests", "mf-types"]
    meta = {"epoch": 1, "active": True}
    await asset_factory(name, tags, meta, make_asset_bytes(name))

    # int filter matches numeric
    async with http.get(
        api_base + "/api/assets",
        params={
            "include_tags": "unit-tests,mf-types",
            "metadata_filter": json.dumps({"epoch": 1}),
        },
    ) as r1:
        b1 = await r1.json()
        assert r1.status == 200 and any(a["name"] == name for a in b1["assets"])

    # string "1" must NOT match numeric 1
    async with http.get(
        api_base + "/api/assets",
        params={
            "include_tags": "unit-tests,mf-types",
            "metadata_filter": json.dumps({"epoch": "1"}),
        },
    ) as r2:
        b2 = await r2.json()
        assert r2.status == 200 and not b2["assets"]

    # bool True matches, string "true" must NOT match
    async with http.get(
        api_base + "/api/assets",
        params={
            "include_tags": "unit-tests,mf-types",
            "metadata_filter": json.dumps({"active": True}),
        },
    ) as r3:
        b3 = await r3.json()
        assert r3.status == 200 and any(a["name"] == name for a in b3["assets"])

    async with http.get(
        api_base + "/api/assets",
        params={
            "include_tags": "unit-tests,mf-types",
            "metadata_filter": json.dumps({"active": "true"}),
        },
    ) as r4:
        b4 = await r4.json()
        assert r4.status == 200 and not b4["assets"]


@pytest.mark.asyncio
async def test_meta_any_of_list_of_scalars(http, api_base, asset_factory, make_asset_bytes):
    name = "mf_list_scalars.safetensors"
    tags = ["models", "checkpoints", "unit-tests", "mf-list"]
    meta = {"flags": ["red", "green"]}
    await asset_factory(name, tags, meta, make_asset_bytes(name, 3000))

    # Any-of should match because "green" is present
    filt_ok = {"flags": ["blue", "green"]}
    async with http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,mf-list", "metadata_filter": json.dumps(filt_ok)},
    ) as r1:
        b1 = await r1.json()
        assert r1.status == 200 and any(a["name"] == name for a in b1["assets"])

    # None of provided flags present -> no match
    filt_miss = {"flags": ["blue", "yellow"]}
    async with http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,mf-list", "metadata_filter": json.dumps(filt_miss)},
    ) as r2:
        b2 = await r2.json()
        assert r2.status == 200 and not b2["assets"]

    # Duplicates in list should not break matching
    filt_dup = {"flags": ["green", "green", "green"]}
    async with http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,mf-list", "metadata_filter": json.dumps(filt_dup)},
    ) as r3:
        b3 = await r3.json()
        assert r3.status == 200 and any(a["name"] == name for a in b3["assets"])


@pytest.mark.asyncio
async def test_meta_none_semantics_missing_or_null_and_any_of_with_none(http, api_base, asset_factory, make_asset_bytes):
    # a1: key missing; a2: explicit null; a3: concrete value
    t = ["models", "checkpoints", "unit-tests", "mf-none"]
    a1 = await asset_factory("mf_none_missing.safetensors", t, {"x": 1}, make_asset_bytes("a1"))
    a2 = await asset_factory("mf_none_null.safetensors", t, {"maybe": None}, make_asset_bytes("a2"))
    a3 = await asset_factory("mf_none_value.safetensors", t, {"maybe": "x"}, make_asset_bytes("a3"))

    # Filter {maybe: None} must match a1 and a2, not a3
    filt = {"maybe": None}
    async with http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,mf-none", "metadata_filter": json.dumps(filt), "sort": "name"},
    ) as r1:
        b1 = await r1.json()
        assert r1.status == 200
        got = [a["name"] for a in b1["assets"]]
        assert a1["name"] in got and a2["name"] in got and a3["name"] not in got

    # Any-of with None should include missing/null plus value matches
    filt_any = {"maybe": [None, "x"]}
    async with http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,mf-none", "metadata_filter": json.dumps(filt_any), "sort": "name"},
    ) as r2:
        b2 = await r2.json()
        assert r2.status == 200
        got2 = [a["name"] for a in b2["assets"]]
        assert a1["name"] in got2 and a2["name"] in got2 and a3["name"] in got2


@pytest.mark.asyncio
async def test_meta_nested_json_object_equality(http, api_base, asset_factory, make_asset_bytes):
    name = "mf_nested_json.safetensors"
    tags = ["models", "checkpoints", "unit-tests", "mf-nested"]
    cfg = {"optimizer": "adam", "lr": 0.001, "schedule": {"type": "cosine", "warmup": 100}}
    await asset_factory(name, tags, {"config": cfg}, make_asset_bytes(name, 2200))

    # Exact JSON object equality (same structure)
    async with http.get(
        api_base + "/api/assets",
        params={
            "include_tags": "unit-tests,mf-nested",
            "metadata_filter": json.dumps({"config": cfg}),
        },
    ) as r1:
        b1 = await r1.json()
        assert r1.status == 200 and any(a["name"] == name for a in b1["assets"])

    # Different JSON object should not match
    async with http.get(
        api_base + "/api/assets",
        params={
            "include_tags": "unit-tests,mf-nested",
            "metadata_filter": json.dumps({"config": {"optimizer": "sgd"}}),
        },
    ) as r2:
        b2 = await r2.json()
        assert r2.status == 200 and not b2["assets"]


@pytest.mark.asyncio
async def test_meta_list_of_objects_any_of(http, api_base, asset_factory, make_asset_bytes):
    name = "mf_list_objects.safetensors"
    tags = ["models", "checkpoints", "unit-tests", "mf-objlist"]
    transforms = [{"type": "crop", "size": 128}, {"type": "flip", "p": 0.5}]
    await asset_factory(name, tags, {"transforms": transforms}, make_asset_bytes(name, 2048))

    # Any-of for list of objects should match when one element equals the filter object
    async with http.get(
        api_base + "/api/assets",
        params={
            "include_tags": "unit-tests,mf-objlist",
            "metadata_filter": json.dumps({"transforms": {"type": "flip", "p": 0.5}}),
        },
    ) as r1:
        b1 = await r1.json()
        assert r1.status == 200 and any(a["name"] == name for a in b1["assets"])

    # Non-matching object -> no match
    async with http.get(
        api_base + "/api/assets",
        params={
            "include_tags": "unit-tests,mf-objlist",
            "metadata_filter": json.dumps({"transforms": {"type": "rotate", "deg": 90}}),
        },
    ) as r2:
        b2 = await r2.json()
        assert r2.status == 200 and not b2["assets"]


@pytest.mark.asyncio
async def test_meta_with_special_and_unicode_keys(http, api_base, asset_factory, make_asset_bytes):
    name = "mf_keys_unicode.safetensors"
    tags = ["models", "checkpoints", "unit-tests", "mf-keys"]
    meta = {
        "weird.key": "v1",
        "path/like": 7,
        "with:colon": True,
        "ключ": "значение",
        "emoji": "🐍",
    }
    await asset_factory(name, tags, meta, make_asset_bytes(name, 1500))

    # Match all the special keys
    filt = {"weird.key": "v1", "path/like": 7, "with:colon": True, "emoji": "🐍"}
    async with http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,mf-keys", "metadata_filter": json.dumps(filt)},
    ) as r1:
        b1 = await r1.json()
        assert r1.status == 200 and any(a["name"] == name for a in b1["assets"])

    # Unicode key match
    async with http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,mf-keys", "metadata_filter": json.dumps({"ключ": "значение"})},
    ) as r2:
        b2 = await r2.json()
        assert r2.status == 200 and any(a["name"] == name for a in b2["assets"])


@pytest.mark.asyncio
async def test_meta_with_zero_and_boolean_lists(http, api_base, asset_factory, make_asset_bytes):
    t = ["models", "checkpoints", "unit-tests", "mf-zero-bool"]
    a0 = await asset_factory("mf_zero_count.safetensors", t, {"count": 0}, make_asset_bytes("z", 1025))
    a1 = await asset_factory("mf_bool_list.safetensors", t, {"choices": [True, False]}, make_asset_bytes("b", 1026))

    # count == 0 must match only a0
    async with http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,mf-zero-bool", "metadata_filter": json.dumps({"count": 0})},
    ) as r1:
        b1 = await r1.json()
        assert r1.status == 200
        names1 = [a["name"] for a in b1["assets"]]
        assert a0["name"] in names1 and a1["name"] not in names1

    # Any-of list of booleans: True matches second asset
    async with http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,mf-zero-bool", "metadata_filter": json.dumps({"choices": True})},
    ) as r2:
        b2 = await r2.json()
        assert r2.status == 200 and any(a["name"] == a1["name"] for a in b2["assets"])


@pytest.mark.asyncio
async def test_meta_mixed_list_types_and_strictness(http, api_base, asset_factory, make_asset_bytes):
    name = "mf_mixed_list.safetensors"
    tags = ["models", "checkpoints", "unit-tests", "mf-mixed"]
    meta = {"mix": ["1", 1, True, None]}
    await asset_factory(name, tags, meta, make_asset_bytes(name, 1999))

    # Should match because 1 is present
    async with http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,mf-mixed", "metadata_filter": json.dumps({"mix": [2, 1]})},
    ) as r1:
        b1 = await r1.json()
        assert r1.status == 200 and any(a["name"] == name for a in b1["assets"])

    # Should NOT match for False
    async with http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,mf-mixed", "metadata_filter": json.dumps({"mix": False})},
    ) as r2:
        b2 = await r2.json()
        assert r2.status == 200 and not b2["assets"]


@pytest.mark.asyncio
async def test_meta_unknown_key_and_none_behavior_with_scope_tags(http, api_base, asset_factory, make_asset_bytes):
    # Use a unique scope tag to avoid interference
    t = ["models", "checkpoints", "unit-tests", "mf-unknown-scope"]
    x = await asset_factory("mf_unknown_a.safetensors", t, {"k1": 1}, make_asset_bytes("ua"))
    y = await asset_factory("mf_unknown_b.safetensors", t, {"k2": 2}, make_asset_bytes("ub"))

    # Filtering by unknown key with None should return both (missing key OR null)
    async with http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,mf-unknown-scope", "metadata_filter": json.dumps({"unknown": None})},
    ) as r1:
        b1 = await r1.json()
        assert r1.status == 200
        names = {a["name"] for a in b1["assets"]}
        assert x["name"] in names and y["name"] in names

    # Filtering by unknown key with concrete value should return none
    async with http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,mf-unknown-scope", "metadata_filter": json.dumps({"unknown": "x"})},
    ) as r2:
        b2 = await r2.json()
        assert r2.status == 200 and not b2["assets"]


@pytest.mark.asyncio
async def test_meta_with_tags_include_exclude_and_name_contains(http, api_base, asset_factory, make_asset_bytes):
    # alpha matches epoch=1; beta has epoch=2
    a = await asset_factory(
        "mf_tag_alpha.safetensors",
        ["models", "checkpoints", "unit-tests", "mf-tag", "alpha"],
        {"epoch": 1},
        make_asset_bytes("alpha"),
    )
    b = await asset_factory(
        "mf_tag_beta.safetensors",
        ["models", "checkpoints", "unit-tests", "mf-tag", "beta"],
        {"epoch": 2},
        make_asset_bytes("beta"),
    )

    params = {
        "include_tags": "unit-tests,mf-tag,alpha",
        "exclude_tags": "beta",
        "name_contains": "mf_tag_",
        "metadata_filter": json.dumps({"epoch": 1}),
    }
    async with http.get(api_base + "/api/assets", params=params) as r:
        body = await r.json()
        assert r.status == 200
        names = [x["name"] for x in body["assets"]]
        assert a["name"] in names
        assert b["name"] not in names


@pytest.mark.asyncio
async def test_meta_sort_and_paging_under_filter(http, api_base, asset_factory, make_asset_bytes):
    # Three assets in same scope with different sizes and a common filter key
    t = ["models", "checkpoints", "unit-tests", "mf-sort"]
    n1, n2, n3 = "mf_sort_1.safetensors", "mf_sort_2.safetensors", "mf_sort_3.safetensors"
    await asset_factory(n1, t, {"group": "g"}, make_asset_bytes(n1, 1024))
    await asset_factory(n2, t, {"group": "g"}, make_asset_bytes(n2, 2048))
    await asset_factory(n3, t, {"group": "g"}, make_asset_bytes(n3, 3072))

    # Sort by size ascending with paging
    q = {"include_tags": "unit-tests,mf-sort", "metadata_filter": json.dumps({"group": "g"}), "sort": "size", "order": "asc", "limit": "2"}
    async with http.get(api_base + "/api/assets", params=q) as r1:
        b1 = await r1.json()
        assert r1.status == 200
        got1 = [a["name"] for a in b1["assets"]]
        assert got1 == [n1, n2]
        assert b1["has_more"] is True

    q2 = {**q, "offset": "2"}
    async with http.get(api_base + "/api/assets", params=q2) as r2:
        b2 = await r2.json()
        assert r2.status == 200
        got2 = [a["name"] for a in b2["assets"]]
        assert got2 == [n3]
        assert b2["has_more"] is False
