import os
import uuid
from pathlib import Path

import aiohttp
import pytest
from conftest import get_asset_filename, trigger_sync_seed_assets


@pytest.mark.asyncio
@pytest.mark.parametrize("root", ["input", "output"])
async def test_seed_asset_removed_when_file_is_deleted(
    root: str,
    http: aiohttp.ClientSession,
    api_base: str,
    comfy_tmp_base_dir: Path,
):
    """Asset without hash (seed) whose file disappears:
       after triggering sync_seed_assets, Asset + AssetInfo disappear.
    """
    # Create a file directly under input/unit-tests/<case> so tags include "unit-tests"
    case_dir = comfy_tmp_base_dir / root / "unit-tests" / "syncseed"
    case_dir.mkdir(parents=True, exist_ok=True)
    name = f"seed_{uuid.uuid4().hex[:8]}.bin"
    fp = case_dir / name
    fp.write_bytes(b"Z" * 2048)

    # Trigger a seed sync so DB sees this path (seed asset => hash is NULL)
    await trigger_sync_seed_assets(http, api_base)

    # Verify it is visible via API and carries no hash (seed)
    async with http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,syncseed", "name_contains": name},
    ) as r1:
        body1 = await r1.json()
        assert r1.status == 200
        # there should be exactly one with that name
        matches = [a for a in body1.get("assets", []) if a.get("name") == name]
        assert matches
        assert matches[0].get("asset_hash") is None
        asset_info_id = matches[0]["id"]

    # Remove the underlying file and sync again
    if fp.exists():
        fp.unlink()

    await trigger_sync_seed_assets(http, api_base)

    # It should disappear (AssetInfo and seed Asset gone)
    async with http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests,syncseed", "name_contains": name},
    ) as r2:
        body2 = await r2.json()
        assert r2.status == 200
        matches2 = [a for a in body2.get("assets", []) if a.get("name") == name]
        assert not matches2, f"Seed asset {asset_info_id} should be gone after sync"


@pytest.mark.asyncio
async def test_hashed_asset_missing_tag_added_then_removed_after_scan(
    http: aiohttp.ClientSession,
    api_base: str,
    comfy_tmp_base_dir: Path,
    asset_factory,
    make_asset_bytes,
    run_scan_and_wait,
):
    """Hashed asset with a single cache_state:
       1. delete its file -> sync adds 'missing'
       2. restore file -> scan removes 'missing'
    """
    name = "missing_tag_test.png"
    tags = ["input", "unit-tests", "msync2"]
    data = make_asset_bytes(name, 4096)
    a = await asset_factory(name, tags, {}, data)

    # Compute its on-disk path and remove it
    dest = comfy_tmp_base_dir / "input" / "unit-tests" / "msync2" / get_asset_filename(a["asset_hash"], ".png")
    assert dest.exists(), f"Expected asset file at {dest}"
    dest.unlink()

    # Fast sync should add 'missing' to the AssetInfo
    await trigger_sync_seed_assets(http, api_base)

    async with http.get(f"{api_base}/api/assets/{a['id']}") as g1:
        d1 = await g1.json()
        assert g1.status == 200, d1
        assert "missing" in set(d1.get("tags", [])), "Expected 'missing' tag after deletion"

    # Restore the file with the exact same content and re-hash/verify via scan
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)

    await run_scan_and_wait("input")

    async with http.get(f"{api_base}/api/assets/{a['id']}") as g2:
        d2 = await g2.json()
        assert g2.status == 200, d2
        assert "missing" not in set(d2.get("tags", [])), "Missing tag should be cleared after verify"


@pytest.mark.asyncio
async def test_hashed_asset_two_asset_infos_both_get_missing(
    http: aiohttp.ClientSession,
    api_base: str,
    comfy_tmp_base_dir: Path,
    asset_factory,
):
    """Hashed asset with a single cache_state, but two AssetInfo rows:
       deleting the single file then syncing should add 'missing' to both infos.
    """
    # Upload one hashed asset
    name = "two_infos_one_path.png"
    base_tags = ["input", "unit-tests", "multiinfo"]
    created = await asset_factory(name, base_tags, {}, b"A" * 2048)

    # Create second AssetInfo for the same Asset via from-hash
    payload = {
        "hash": created["asset_hash"],
        "name": "two_infos_one_path_copy.png",
        "tags": base_tags,  # keep it in our unit-tests scope for cleanup
        "user_metadata": {"k": "v"},
    }
    async with http.post(api_base + "/api/assets/from-hash", json=payload) as r2:
        b2 = await r2.json()
        assert r2.status == 201, b2
        second_id = b2["id"]

    # Remove the single underlying file
    p = comfy_tmp_base_dir / "input" / "unit-tests" / "multiinfo" / get_asset_filename(b2["asset_hash"], ".png")
    assert p.exists()
    p.unlink()

    async with http.get(api_base + "/api/tags", params={"limit": "1000", "include_zero": "false"}) as r0:
        tags0 = await r0.json()
        assert r0.status == 200, tags0
        byname0 = {t["name"]: t for t in tags0.get("tags", [])}
        old_missing = int(byname0.get("missing", {}).get("count", 0))

    # Sync -> both AssetInfos for this asset must receive 'missing'
    await trigger_sync_seed_assets(http, api_base)

    async with http.get(f"{api_base}/api/assets/{created['id']}") as ga:
        da = await ga.json()
        assert ga.status == 200, da
        assert "missing" in set(da.get("tags", []))

    async with http.get(f"{api_base}/api/assets/{second_id}") as gb:
        db = await gb.json()
        assert gb.status == 200, db
        assert "missing" in set(db.get("tags", []))

    # Tag usage for 'missing' increased by exactly 2 (two AssetInfos)
    async with http.get(api_base + "/api/tags", params={"limit": "1000", "include_zero": "false"}) as r1:
        tags1 = await r1.json()
        assert r1.status == 200, tags1
        byname1 = {t["name"]: t for t in tags1.get("tags", [])}
        new_missing = int(byname1.get("missing", {}).get("count", 0))
        assert new_missing == old_missing + 2


@pytest.mark.asyncio
async def test_hashed_asset_two_cache_states_partial_delete_then_full_delete(
    http: aiohttp.ClientSession,
    api_base: str,
    comfy_tmp_base_dir: Path,
    asset_factory,
    make_asset_bytes,
    run_scan_and_wait,
):
    """Hashed asset with two cache_state rows:
       1. delete one file -> sync should NOT add 'missing'
       2. delete second file -> sync should add 'missing'
    """
    name = "two_cache_states_partial_delete.png"
    tags = ["input", "unit-tests", "dual"]
    data = make_asset_bytes(name, 3072)

    created = await asset_factory(name, tags, {}, data)
    path1 = comfy_tmp_base_dir / "input" / "unit-tests" / "dual" / get_asset_filename(created["asset_hash"], ".png")
    assert path1.exists()

    # Create a second on-disk copy under the same root but different subfolder
    path2 = comfy_tmp_base_dir / "input" / "unit-tests" / "dual_copy" / name
    path2.parent.mkdir(parents=True, exist_ok=True)
    path2.write_bytes(data)

    # Fast seed so the second path appears (as a seed initially)
    await trigger_sync_seed_assets(http, api_base)

    # Deduplication of AssetInfo-s will not happen as first AssetInfo has owner='default' and second has empty owner.
    await run_scan_and_wait("input")

    # Remove only one file and sync -> asset should still be healthy (no 'missing')
    path1.unlink()
    await trigger_sync_seed_assets(http, api_base)

    async with http.get(f"{api_base}/api/assets/{created['id']}") as g1:
        d1 = await g1.json()
        assert g1.status == 200, d1
        assert "missing" not in set(d1.get("tags", [])), "Should not be missing while one valid path remains"

    # Baseline 'missing' usage count just before last file removal
    async with http.get(api_base + "/api/tags", params={"limit": "1000", "include_zero": "false"}) as r0:
        tags0 = await r0.json()
        assert r0.status == 200, tags0
        old_missing = int({t["name"]: t for t in tags0.get("tags", [])}.get("missing", {}).get("count", 0))

    # Remove the second (last) file and sync -> now we expect 'missing' on this AssetInfo
    path2.unlink()
    await trigger_sync_seed_assets(http, api_base)

    async with http.get(f"{api_base}/api/assets/{created['id']}") as g2:
        d2 = await g2.json()
        assert g2.status == 200, d2
        assert "missing" in set(d2.get("tags", [])), "Missing must be set once no valid paths remain"

    # Tag usage for 'missing' increased by exactly 2 (two AssetInfo for one Asset)
    async with http.get(api_base + "/api/tags", params={"limit": "1000", "include_zero": "false"}) as r1:
        tags1 = await r1.json()
        assert r1.status == 200, tags1
        new_missing = int({t["name"]: t for t in tags1.get("tags", [])}.get("missing", {}).get("count", 0))
        assert new_missing == old_missing + 2


@pytest.mark.asyncio
@pytest.mark.parametrize("root", ["input", "output"])
async def test_missing_tag_clears_on_fastpass_when_mtime_and_size_match(
    root: str,
    http: aiohttp.ClientSession,
    api_base: str,
    comfy_tmp_base_dir: Path,
    asset_factory,
    make_asset_bytes,
):
    """
    Fast pass alone clears 'missing' when size and mtime match exactly:
      1) upload (hashed), record original mtime_ns
      2) delete -> fast pass adds 'missing'
      3) restore same bytes and set mtime back to the original value
      4) run fast pass again -> 'missing' is removed (no slow scan)
    """
    scope = f"fastclear-{uuid.uuid4().hex[:6]}"
    name = "fastpass_clear.bin"
    data = make_asset_bytes(name, 3072)

    a = await asset_factory(name, [root, "unit-tests", scope], {}, data)
    aid = a["id"]
    base = comfy_tmp_base_dir / root / "unit-tests" / scope
    p = base / get_asset_filename(a["asset_hash"], ".bin")
    st0 = p.stat()
    orig_mtime_ns = getattr(st0, "st_mtime_ns", int(st0.st_mtime * 1_000_000_000))

    # Delete -> fast pass adds 'missing'
    p.unlink()
    await trigger_sync_seed_assets(http, api_base)
    async with http.get(f"{api_base}/api/assets/{aid}") as g1:
        d1 = await g1.json()
        assert g1.status == 200, d1
        assert "missing" in set(d1.get("tags", []))

    # Restore same bytes and revert mtime to the original value
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    # set both atime and mtime in ns to ensure exact match
    os.utime(p, ns=(orig_mtime_ns, orig_mtime_ns))

    # Fast pass should clear 'missing' without a scan
    await trigger_sync_seed_assets(http, api_base)
    async with http.get(f"{api_base}/api/assets/{aid}") as g2:
        d2 = await g2.json()
        assert g2.status == 200, d2
        assert "missing" not in set(d2.get("tags", [])), "Fast pass should clear 'missing' when size+mtime match"


@pytest.mark.asyncio
@pytest.mark.parametrize("root", ["input", "output"])
async def test_fastpass_removes_stale_state_row_no_missing(
    root: str,
    http: aiohttp.ClientSession,
    api_base: str,
    comfy_tmp_base_dir: Path,
    asset_factory,
    make_asset_bytes,
    run_scan_and_wait,
):
    """
    Hashed asset with two states:
      - delete one file
      - run fast pass only
    Expect:
      - asset stays healthy (no 'missing')
      - stale AssetCacheState row for the deleted path is removed.
        We verify this behaviorally by recreating the deleted path and running fast pass again:
        a new *seed* AssetInfo is created, which proves the old state row was not reused.
    """
    scope = f"stale-{uuid.uuid4().hex[:6]}"
    name = "two_states.bin"
    data = make_asset_bytes(name, 2048)

    # Upload hashed asset at path1
    a = await asset_factory(name, [root, "unit-tests", scope], {}, data)
    base = comfy_tmp_base_dir / root / "unit-tests" / scope
    a1_filename = get_asset_filename(a["asset_hash"], ".bin")
    p1 = base / a1_filename
    assert p1.exists()

    aid = a["id"]
    h = a["asset_hash"]

    # Create second state path2, seed+scan to dedupe into the same Asset
    p2 = base / "copy" / name
    p2.parent.mkdir(parents=True, exist_ok=True)
    p2.write_bytes(data)
    await trigger_sync_seed_assets(http, api_base)
    await run_scan_and_wait(root)

    # Delete path1 and run fast pass -> no 'missing' and stale state row should be removed
    p1.unlink()
    await trigger_sync_seed_assets(http, api_base)
    async with http.get(f"{api_base}/api/assets/{aid}") as g1:
        d1 = await g1.json()
        assert g1.status == 200, d1
        assert "missing" not in set(d1.get("tags", []))

    # Recreate path1 and run fast pass again.
    # If the stale state row was removed, a NEW seed AssetInfo will appear for this path.
    p1.write_bytes(data)
    await trigger_sync_seed_assets(http, api_base)

    async with http.get(
        api_base + "/api/assets",
        params={"include_tags": f"unit-tests,{scope}"},
    ) as rl:
        bl = await rl.json()
        assert rl.status == 200, bl
        items = bl.get("assets", [])
        # one hashed AssetInfo (asset_hash == h) + one seed AssetInfo (asset_hash == null)
        hashes = [it.get("asset_hash") for it in items if it.get("name") in (name, a1_filename)]
        assert h in hashes
        assert any(x is None for x in hashes), "Expected a new seed AssetInfo for the recreated path"

    # Asset identity still healthy
    async with http.head(f"{api_base}/api/assets/hash/{h}") as rh:
        assert rh.status == 200
