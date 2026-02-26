import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests
from conftest import get_asset_filename, trigger_sync_seed_assets


def test_create_from_hash_success(
    http: requests.Session, api_base: str, seeded_asset: dict
):
    h = seeded_asset["asset_hash"]
    payload = {
        "hash": h,
        "name": "from_hash_ok.safetensors",
        "tags": ["models", "checkpoints", "unit-tests", "from-hash"],
        "user_metadata": {"k": "v"},
    }
    r1 = http.post(f"{api_base}/api/assets/from-hash", json=payload, timeout=120)
    b1 = r1.json()
    assert r1.status_code == 201, b1
    assert b1["asset_hash"] == h
    assert b1["created_new"] is False
    aid = b1["id"]

    # Calling again with the same name should return the same AssetInfo id
    r2 = http.post(f"{api_base}/api/assets/from-hash", json=payload, timeout=120)
    b2 = r2.json()
    assert r2.status_code == 201, b2
    assert b2["id"] == aid


def test_get_and_delete_asset(http: requests.Session, api_base: str, seeded_asset: dict):
    aid = seeded_asset["id"]

    # GET detail
    rg = http.get(f"{api_base}/api/assets/{aid}", timeout=120)
    detail = rg.json()
    assert rg.status_code == 200, detail
    assert detail["id"] == aid
    assert "user_metadata" in detail
    assert "filename" in detail["user_metadata"]

    # DELETE
    rd = http.delete(f"{api_base}/api/assets/{aid}", timeout=120)
    assert rd.status_code == 204

    # GET again -> 404
    rg2 = http.get(f"{api_base}/api/assets/{aid}", timeout=120)
    body = rg2.json()
    assert rg2.status_code == 404
    assert body["error"]["code"] == "ASSET_NOT_FOUND"


def test_delete_upon_reference_count(
    http: requests.Session, api_base: str, seeded_asset: dict
):
    # Create a second reference to the same asset via from-hash
    src_hash = seeded_asset["asset_hash"]
    payload = {
        "hash": src_hash,
        "name": "unit_ref_copy.safetensors",
        "tags": ["models", "checkpoints", "unit-tests", "del-flow"],
        "user_metadata": {"note": "copy"},
    }
    r2 = http.post(f"{api_base}/api/assets/from-hash", json=payload, timeout=120)
    copy = r2.json()
    assert r2.status_code == 201, copy
    assert copy["asset_hash"] == src_hash
    assert copy["created_new"] is False

    # Delete original reference -> asset identity must remain
    aid1 = seeded_asset["id"]
    rd1 = http.delete(f"{api_base}/api/assets/{aid1}", timeout=120)
    assert rd1.status_code == 204

    rh1 = http.head(f"{api_base}/api/assets/hash/{src_hash}", timeout=120)
    assert rh1.status_code == 200  # identity still present

    # Delete the last reference with default semantics -> identity and cached files removed
    aid2 = copy["id"]
    rd2 = http.delete(f"{api_base}/api/assets/{aid2}", timeout=120)
    assert rd2.status_code == 204

    rh2 = http.head(f"{api_base}/api/assets/hash/{src_hash}", timeout=120)
    assert rh2.status_code == 404  # orphan content removed


def test_update_asset_fields(http: requests.Session, api_base: str, seeded_asset: dict):
    aid = seeded_asset["id"]
    original_tags = seeded_asset["tags"]

    payload = {
        "name": "unit_1_renamed.safetensors",
        "user_metadata": {"purpose": "updated", "epoch": 2},
    }
    ru = http.put(f"{api_base}/api/assets/{aid}", json=payload, timeout=120)
    body = ru.json()
    assert ru.status_code == 200, body
    assert body["name"] == payload["name"]
    assert body["tags"] == original_tags  # tags unchanged
    assert body["user_metadata"]["purpose"] == "updated"
    # filename should still be present and normalized by server
    assert "filename" in body["user_metadata"]


def test_head_asset_by_hash(http: requests.Session, api_base: str, seeded_asset: dict):
    h = seeded_asset["asset_hash"]

    # Existing
    rh1 = http.head(f"{api_base}/api/assets/hash/{h}", timeout=120)
    assert rh1.status_code == 200

    # Non-existent
    rh2 = http.head(f"{api_base}/api/assets/hash/blake3:{'0'*64}", timeout=120)
    assert rh2.status_code == 404


def test_head_asset_bad_hash_returns_400_and_no_body(http: requests.Session, api_base: str):
    # Invalid format; handler returns a JSON error, but HEAD responses must not carry a payload.
    # requests exposes an empty body for HEAD, so validate status and that there is no payload.
    rh = http.head(f"{api_base}/api/assets/hash/not_a_hash", timeout=120)
    assert rh.status_code == 400
    body = rh.content
    assert body == b""


def test_delete_nonexistent_returns_404(http: requests.Session, api_base: str):
    bogus = str(uuid.uuid4())
    r = http.delete(f"{api_base}/api/assets/{bogus}", timeout=120)
    body = r.json()
    assert r.status_code == 404
    assert body["error"]["code"] == "ASSET_NOT_FOUND"


def test_create_from_hash_invalids(http: requests.Session, api_base: str):
    # Bad hash algorithm
    bad = {
        "hash": "sha256:" + "0" * 64,
        "name": "x.bin",
        "tags": ["models", "checkpoints", "unit-tests"],
    }
    r1 = http.post(f"{api_base}/api/assets/from-hash", json=bad, timeout=120)
    b1 = r1.json()
    assert r1.status_code == 400
    assert b1["error"]["code"] == "INVALID_BODY"

    # Invalid JSON body
    r2 = http.post(f"{api_base}/api/assets/from-hash", data=b"{not json}", timeout=120)
    b2 = r2.json()
    assert r2.status_code == 400
    assert b2["error"]["code"] == "INVALID_JSON"


def test_get_update_download_bad_ids(http: requests.Session, api_base: str):
    # All endpoints should be not found, as we UUID regex directly in the route definition.
    bad_id = "not-a-uuid"

    r1 = http.get(f"{api_base}/api/assets/{bad_id}", timeout=120)
    assert r1.status_code == 404

    r3 = http.get(f"{api_base}/api/assets/{bad_id}/content", timeout=120)
    assert r3.status_code == 404


def test_update_requires_at_least_one_field(http: requests.Session, api_base: str, seeded_asset: dict):
    aid = seeded_asset["id"]
    r = http.put(f"{api_base}/api/assets/{aid}", json={}, timeout=120)
    body = r.json()
    assert r.status_code == 400
    assert body["error"]["code"] == "INVALID_BODY"


@pytest.mark.parametrize("root", ["input", "output"])
def test_concurrent_delete_same_asset_info_single_204(
    root: str,
    http: requests.Session,
    api_base: str,
    asset_factory,
    make_asset_bytes,
):
    """
    Many concurrent DELETE for the same AssetInfo should result in:
      - exactly one 204 No Content (the one that actually deleted)
      - all others 404 Not Found (row already gone)
    """
    scope = f"conc-del-{uuid.uuid4().hex[:6]}"
    name = "to_delete.bin"
    data = make_asset_bytes(name, 1536)

    created = asset_factory(name, [root, "unit-tests", scope], {}, data)
    aid = created["id"]

    # Hit the same endpoint N times in parallel.
    n_tests = 4
    url = f"{api_base}/api/assets/{aid}?delete_content=false"

    def _do_delete(delete_url):
        with requests.Session() as s:
            return s.delete(delete_url, timeout=120).status_code

    with ThreadPoolExecutor(max_workers=n_tests) as ex:
        statuses = list(ex.map(_do_delete, [url] * n_tests))

    # Exactly one actual delete, the rest must be 404
    assert statuses.count(204) == 1, f"Expected exactly one 204; got: {statuses}"
    assert statuses.count(404) == n_tests - 1, f"Expected {n_tests-1} 404; got: {statuses}"

    # The resource must be gone.
    rg = http.get(f"{api_base}/api/assets/{aid}", timeout=120)
    assert rg.status_code == 404


@pytest.mark.parametrize("root", ["input", "output"])
def test_metadata_filename_is_set_for_seed_asset_without_hash(
    root: str,
    http: requests.Session,
    api_base: str,
    comfy_tmp_base_dir: Path,
):
    """Seed ingest (no hash yet) must compute user_metadata['filename'] immediately."""
    scope = f"seedmeta-{uuid.uuid4().hex[:6]}"
    name = "seed_filename.bin"

    base = comfy_tmp_base_dir / root / "unit-tests" / scope / "a" / "b"
    base.mkdir(parents=True, exist_ok=True)
    fp = base / name
    fp.write_bytes(b"Z" * 2048)

    trigger_sync_seed_assets(http, api_base)

    r1 = http.get(
        api_base + "/api/assets",
        params={"include_tags": f"unit-tests,{scope}", "name_contains": name},
        timeout=120,
    )
    body = r1.json()
    assert r1.status_code == 200, body
    matches = [a for a in body.get("assets", []) if a.get("name") == name]
    assert matches, "Seed asset should be visible after sync"
    assert matches[0].get("asset_hash") is None  # still a seed
    aid = matches[0]["id"]

    r2 = http.get(f"{api_base}/api/assets/{aid}", timeout=120)
    detail = r2.json()
    assert r2.status_code == 200, detail
    filename = (detail.get("user_metadata") or {}).get("filename")
    expected = str(fp.relative_to(comfy_tmp_base_dir / root)).replace("\\", "/")
    assert filename == expected, f"expected filename={expected}, got {filename!r}"


@pytest.mark.skip(reason="Requires computing hashes of files in directories to retarget cache states")
@pytest.mark.parametrize("root", ["input", "output"])
def test_metadata_filename_computed_and_updated_on_retarget(
    root: str,
    http: requests.Session,
    api_base: str,
    comfy_tmp_base_dir: Path,
    asset_factory,
    make_asset_bytes,
    run_scan_and_wait,
):
    """
    1) Ingest under {root}/unit-tests/<scope>/a/b/<name> -> filename reflects relative path.
    2) Retarget by copying to {root}/unit-tests/<scope>/x/<new_name>, remove old file,
       run fast pass + scan -> filename updates to new relative path.
    """
    scope = f"meta-fn-{uuid.uuid4().hex[:6]}"
    name1 = "compute_metadata_filename.png"
    name2 = "compute_changed_metadata_filename.png"
    data = make_asset_bytes(name1, 2100)

    # Upload into nested path a/b
    a = asset_factory(name1, [root, "unit-tests", scope, "a", "b"], {}, data)
    aid = a["id"]

    root_base = comfy_tmp_base_dir / root
    p1 = (root_base / "unit-tests" / scope / "a" / "b" / get_asset_filename(a["asset_hash"], ".png"))
    assert p1.exists()

    # filename at ingest should be the path relative to root
    rel1 = str(p1.relative_to(root_base)).replace("\\", "/")
    g1 = http.get(f"{api_base}/api/assets/{aid}", timeout=120)
    d1 = g1.json()
    assert g1.status_code == 200, d1
    fn1 = d1["user_metadata"].get("filename")
    assert fn1 == rel1

    # Retarget: copy to x/<name2>, remove old, then sync+scan
    p2 = root_base / "unit-tests" / scope / "x" / name2
    p2.parent.mkdir(parents=True, exist_ok=True)
    p2.write_bytes(data)
    if p1.exists():
        p1.unlink()

    trigger_sync_seed_assets(http, api_base)  # seed the new path
    run_scan_and_wait(root)                   # verify/hash and reconcile

    # filename should now point at x/<name2>
    rel2 = str(p2.relative_to(root_base)).replace("\\", "/")
    g2 = http.get(f"{api_base}/api/assets/{aid}", timeout=120)
    d2 = g2.json()
    assert g2.status_code == 200, d2
    fn2 = d2["user_metadata"].get("filename")
    assert fn2 == rel2
