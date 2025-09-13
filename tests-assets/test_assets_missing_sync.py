from pathlib import Path
import uuid

import aiohttp
import pytest

from conftest import trigger_sync_seed_assets


@pytest.mark.asyncio
async def test_seed_asset_removed_when_file_is_deleted(
    http: aiohttp.ClientSession,
    api_base: str,
    comfy_tmp_base_dir: Path,
):
    """Asset without hash (seed) whose file disappears:
       after triggering sync_seed_assets, Asset + AssetInfo disappear.
    """
    # Create a file directly under input/unit-tests/<case> so tags include "unit-tests"
    case_dir = comfy_tmp_base_dir / "input" / "unit-tests" / "syncseed"
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
    dest = comfy_tmp_base_dir / "input" / "unit-tests" / "msync2" / name
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
async def test_hashed_asset_two_assetinfos_both_get_missing(
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
    p = comfy_tmp_base_dir / "input" / "unit-tests" / "multiinfo" / name
    assert p.exists()
    p.unlink()

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
    path1 = comfy_tmp_base_dir / "input" / "unit-tests" / "dual" / name
    assert path1.exists()

    # Create a second on-disk copy under the same root but different subfolder
    path2 = comfy_tmp_base_dir / "input" / "unit-tests" / "dual_copy" / name
    path2.parent.mkdir(parents=True, exist_ok=True)
    path2.write_bytes(data)

    # Fast seed so the second path appears (as a seed initially)
    await trigger_sync_seed_assets(http, api_base)

    # Now run a 'models' scan so the seed copy is hashed and deduped
    await run_scan_and_wait("input")

    # Remove only one file and sync -> asset should still be healthy (no 'missing')
    path1.unlink()
    await trigger_sync_seed_assets(http, api_base)

    async with http.get(f"{api_base}/api/assets/{created['id']}") as g1:
        d1 = await g1.json()
        assert g1.status == 200, d1
        assert "missing" not in set(d1.get("tags", [])), "Should not be missing while one valid path remains"

    # Remove the second (last) file and sync -> now we expect 'missing'
    path2.unlink()
    await trigger_sync_seed_assets(http, api_base)

    async with http.get(f"{api_base}/api/assets/{created['id']}") as g2:
        d2 = await g2.json()
        assert g2.status == 200, d2
        assert "missing" in set(d2.get("tags", [])), "Missing must be set once no valid paths remain"
