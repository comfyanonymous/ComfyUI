import asyncio
import os
import uuid
from pathlib import Path

import aiohttp
import pytest
from conftest import get_asset_filename, trigger_sync_seed_assets


def _base_for(root: str, comfy_tmp_base_dir: Path) -> Path:
    assert root in ("input", "output")
    return comfy_tmp_base_dir / root


def _mkbytes(label: str, size: int) -> bytes:
    seed = sum(label.encode("utf-8")) % 251
    return bytes((i * 31 + seed) % 256 for i in range(size))


@pytest.mark.asyncio
@pytest.mark.parametrize("root", ["input", "output"])
async def test_scan_schedule_idempotent_while_running(
    root: str,
    http,
    api_base: str,
    comfy_tmp_base_dir: Path,
    run_scan_and_wait,
):
    """Idempotent schedule while running."""
    scope = f"idem-{uuid.uuid4().hex[:6]}"
    base = _base_for(root, comfy_tmp_base_dir) / "unit-tests" / scope
    base.mkdir(parents=True, exist_ok=True)

    # Create several seed files (non-zero) to ensure the scan runs long enough
    for i in range(8):
        (base / f"f{i}.bin").write_bytes(_mkbytes(f"{scope}-{i}", 2 * 1024 * 1024))  # ~2 MiB each

    # Seed -> states with hash=NULL
    await trigger_sync_seed_assets(http, api_base)

    # Schedule once
    async with http.post(api_base + "/api/assets/scan/schedule", json={"roots": [root]}) as r1:
        b1 = await r1.json()
        assert r1.status == 202, b1
        scans1 = {s["root"]: s for s in b1.get("scans", [])}
        s1 = scans1.get(root)
        assert s1 and s1["status"] in {"scheduled", "running"}
        sid1 = s1["scan_id"]

    # Schedule again immediately — must return the same scan entry (no new worker)
    async with http.post(api_base + "/api/assets/scan/schedule", json={"roots": [root]}) as r2:
        b2 = await r2.json()
        assert r2.status == 202, b2
        scans2 = {s["root"]: s for s in b2.get("scans", [])}
        s2 = scans2.get(root)
        assert s2 and s2["scan_id"] == sid1

    # Filtered GET must show exactly one scan for this root
    async with http.get(api_base + "/api/assets/scan", params={"root": root}) as gs:
        bs = await gs.json()
        assert gs.status == 200, bs
        scans = bs.get("scans", [])
        assert len(scans) == 1 and scans[0]["scan_id"] == sid1

    # Let it finish to avoid cross-test interference
    await run_scan_and_wait(root)


@pytest.mark.asyncio
async def test_scan_status_filter_by_root_and_file_errors(
    http,
    api_base: str,
    comfy_tmp_base_dir: Path,
    run_scan_and_wait,
    asset_factory,
):
    """Filtering get scan status by root (schedule for both input and output) + file_errors presence."""
    # Create one hashed asset in input under a dir we will chmod to 000 to force PermissionError in reconcile stage
    in_scope = f"filter-in-{uuid.uuid4().hex[:6]}"
    protected_dir = _base_for("input", comfy_tmp_base_dir) / "unit-tests" / in_scope / "deny"
    protected_dir.mkdir(parents=True, exist_ok=True)
    name_in = "protected.bin"

    data = b"A" * 4096
    await asset_factory(name_in, ["input", "unit-tests", in_scope, "deny"], {}, data)
    try:
        os.chmod(protected_dir, 0x000)

        # Also schedule a scan for output root (no errors there)
        out_scope = f"filter-out-{uuid.uuid4().hex[:6]}"
        out_dir = _base_for("output", comfy_tmp_base_dir) / "unit-tests" / out_scope
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "ok.bin").write_bytes(b"B" * 1024)
        await trigger_sync_seed_assets(http, api_base)  # seed output file

        # Schedule both roots
        async with http.post(api_base + "/api/assets/scan/schedule", json={"roots": ["input"]}) as r_in:
            assert r_in.status == 202
        async with http.post(api_base + "/api/assets/scan/schedule", json={"roots": ["output"]}) as r_out:
            assert r_out.status == 202

        # Wait both to complete, input last (we want its errors)
        await run_scan_and_wait("output")
        await run_scan_and_wait("input")

        # Filter by root=input: only input scan listed and must have file_errors
        async with http.get(api_base + "/api/assets/scan", params={"root": "input"}) as gs:
            body = await gs.json()
            assert gs.status == 200, body
            scans = body.get("scans", [])
            assert len(scans) == 1
            errs = scans[0].get("file_errors", [])
            # Must contain at least one error with a message
            assert errs and any(e.get("message") for e in errs)
    finally:
        # Restore perms so cleanup can remove files/dirs
        try:
            os.chmod(protected_dir, 0o755)
        except Exception:
            pass


@pytest.mark.asyncio
@pytest.mark.parametrize("root", ["input", "output"])
@pytest.mark.skipif(os.name == "nt", reason="Permission-based file_errors are unreliable on Windows")
async def test_scan_records_file_errors_permission_denied(
    root: str,
    http,
    api_base: str,
    comfy_tmp_base_dir: Path,
    asset_factory,
    run_scan_and_wait,
):
    """file_errors recording (permission denied) for input/output"""
    scope = f"errs-{uuid.uuid4().hex[:6]}"
    deny_dir = _base_for(root, comfy_tmp_base_dir) / "unit-tests" / scope / "deny"
    deny_dir.mkdir(parents=True, exist_ok=True)
    name = "deny.bin"

    a1 = await asset_factory(name, [root, "unit-tests", scope, "deny"], {}, b"X" * 2048)
    asset_filename = get_asset_filename(a1["asset_hash"], ".bin")
    try:
        os.chmod(deny_dir, 0x000)
        async with http.post(api_base + "/api/assets/scan/schedule", json={"roots": [root]}) as r:
            assert r.status == 202
        await run_scan_and_wait(root)

        async with http.get(api_base + "/api/assets/scan", params={"root": root}) as gs:
            body = await gs.json()
            assert gs.status == 200, body
            scans = body.get("scans", [])
            assert len(scans) == 1
            errs = scans[0].get("file_errors", [])
            # Should contain at least one PermissionError-like record
            assert errs
            assert any(e.get("path", "").endswith(asset_filename) and e.get("message") for e in errs)
    finally:
        try:
            os.chmod(deny_dir, 0x755)
        except Exception:
            pass


@pytest.mark.asyncio
@pytest.mark.parametrize("root", ["input", "output"])
async def test_missing_tag_created_and_visible_in_tags(
    root: str,
    http,
    api_base: str,
    comfy_tmp_base_dir: Path,
    asset_factory,
):
    """Missing tag appears in tags list and increments count (input/output)"""
    # Baseline count of 'missing' tag (may be absent)
    async with http.get(api_base + "/api/tags", params={"limit": "1000"}) as r0:
        t0 = await r0.json()
        assert r0.status == 200, t0
        byname = {t["name"]: t for t in t0.get("tags", [])}
        old_count = int(byname.get("missing", {}).get("count", 0))

    scope = f"miss-{uuid.uuid4().hex[:6]}"
    name = "missing_me.bin"
    created = await asset_factory(name, [root, "unit-tests", scope], {}, b"Y" * 4096)

    # Remove the only file and trigger fast pass
    p = _base_for(root, comfy_tmp_base_dir) / "unit-tests" / scope / get_asset_filename(created["asset_hash"], ".bin")
    assert p.exists()
    p.unlink()
    await trigger_sync_seed_assets(http, api_base)

    # Asset has 'missing' tag
    async with http.get(f"{api_base}/api/assets/{created['id']}") as g1:
        d1 = await g1.json()
        assert g1.status == 200, d1
        assert "missing" in set(d1.get("tags", []))

    # Tag list now contains 'missing' with increased count
    async with http.get(api_base + "/api/tags", params={"limit": "1000", "include_zero": "false"}) as r1:
        t1 = await r1.json()
        assert r1.status == 200, t1
        byname1 = {t["name"]: t for t in t1.get("tags", [])}
        assert "missing" in byname1
        assert int(byname1["missing"]["count"]) >= old_count + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("root", ["input", "output"])
async def test_missing_reapplies_after_manual_removal(
    root: str,
    http,
    api_base: str,
    comfy_tmp_base_dir: Path,
    asset_factory,
):
    """Manual removal of 'missing' does not block automatic re-apply (input/output)"""
    scope = f"reapply-{uuid.uuid4().hex[:6]}"
    name = "reapply.bin"
    created = await asset_factory(name, [root, "unit-tests", scope], {}, b"Z" * 1024)

    # Make it missing
    p = _base_for(root, comfy_tmp_base_dir) / "unit-tests" / scope / get_asset_filename(created["asset_hash"], ".bin")
    p.unlink()
    await trigger_sync_seed_assets(http, api_base)

    # Remove the 'missing' tag manually
    async with http.delete(f"{api_base}/api/assets/{created['id']}/tags", json={"tags": ["missing"]}) as rdel:
        b = await rdel.json()
        assert rdel.status == 200, b
        assert "missing" in set(b.get("removed", []))

    # Next sync must re-add it
    await trigger_sync_seed_assets(http, api_base)
    async with http.get(f"{api_base}/api/assets/{created['id']}") as g2:
        d2 = await g2.json()
        assert g2.status == 200, d2
        assert "missing" in set(d2.get("tags", []))


@pytest.mark.asyncio
@pytest.mark.parametrize("root", ["input", "output"])
async def test_delete_one_asset_info_of_missing_asset_keeps_identity(
    root: str,
    http,
    api_base: str,
    comfy_tmp_base_dir: Path,
    asset_factory,
):
    """Delete one AssetInfo of a missing asset while another exists (input/output)"""
    scope = f"twoinfos-{uuid.uuid4().hex[:6]}"
    name = "twoinfos.bin"
    a1 = await asset_factory(name, [root, "unit-tests", scope], {}, b"W" * 2048)

    # Second AssetInfo for the same content under same root (different name to avoid collision)
    a2 = await asset_factory("copy_" + name, [root, "unit-tests", scope], {}, b"W" * 2048)

    # Remove file of the first (both point to the same Asset, but we know on-disk path name for a1)
    p1 = _base_for(root, comfy_tmp_base_dir) / "unit-tests" / scope / get_asset_filename(a1["asset_hash"], ".bin")
    p1.unlink()
    await trigger_sync_seed_assets(http, api_base)

    # Both infos should be marked missing
    async with http.get(f"{api_base}/api/assets/{a1['id']}") as g1:
        d1 = await g1.json()
        assert "missing" in set(d1.get("tags", []))
    async with http.get(f"{api_base}/api/assets/{a2['id']}") as g2:
        d2 = await g2.json()
        assert "missing" in set(d2.get("tags", []))

    # Delete one info
    async with http.delete(f"{api_base}/api/assets/{a1['id']}") as rd:
        assert rd.status == 204

    # Asset identity still exists (by hash)
    h = a1["asset_hash"]
    async with http.head(f"{api_base}/api/assets/hash/{h}") as rh:
        assert rh.status == 200

    # Remaining info still reflects 'missing'
    async with http.get(f"{api_base}/api/assets/{a2['id']}") as g3:
        d3 = await g3.json()
        assert g3.status == 200 and "missing" in set(d3.get("tags", []))


@pytest.mark.asyncio
@pytest.mark.parametrize("keep_root", ["input", "output"])
async def test_delete_last_asset_info_false_keeps_asset_and_states_multiroot(
    keep_root: str,
    http,
    api_base: str,
    comfy_tmp_base_dir: Path,
    make_asset_bytes,
    asset_factory,
):
    """Delete last AssetInfo with delete_content_if_orphan=false keeps asset and the underlying on-disk content."""
    other_root = "output" if keep_root == "input" else "input"
    scope = f"delfalse-{uuid.uuid4().hex[:6]}"
    data = make_asset_bytes(scope, 3072)

    # First upload creates the physical file
    a1 = await asset_factory("keep1.bin", [keep_root, "unit-tests", scope], {}, data)
    # Second upload (other root) is deduped to the same content; no new file on disk
    a2 = await asset_factory("keep2.bin", [other_root, "unit-tests", scope], {}, data)

    h = a1["asset_hash"]
    p1 = _base_for(keep_root, comfy_tmp_base_dir) / "unit-tests" / scope / get_asset_filename(h, ".bin")

    # De-dup semantics: only the first physical file exists
    assert p1.exists(), "Expected the first physical file to exist"

    # Delete both AssetInfos; keep content on the very last delete
    async with http.delete(f"{api_base}/api/assets/{a2['id']}") as rfirst:
        assert rfirst.status == 204
    async with http.delete(f"{api_base}/api/assets/{a1['id']}?delete_content=false") as rlast:
        assert rlast.status == 204

    # Asset identity remains and physical content is still present
    async with http.head(f"{api_base}/api/assets/hash/{h}") as rh:
        assert rh.status == 200
    assert p1.exists(), "Content file should remain after keep-content delete"

    # Cleanup: re-create a reference by hash and then delete to purge content
    payload = {
        "hash": h,
        "name": "cleanup.bin",
        "tags": [keep_root, "unit-tests", scope, "cleanup"],
        "user_metadata": {},
    }
    async with http.post(f"{api_base}/api/assets/from-hash", json=payload) as rfh:
        ref = await rfh.json()
        assert rfh.status == 201, ref
        cid = ref["id"]
    async with http.delete(f"{api_base}/api/assets/{cid}") as rdel:
        assert rdel.status == 204


@pytest.mark.asyncio
@pytest.mark.parametrize("root", ["input", "output"])
async def test_sync_seed_ignores_zero_byte_files(
    root: str,
    http,
    api_base: str,
    comfy_tmp_base_dir: Path,
):
    scope = f"zero-{uuid.uuid4().hex[:6]}"
    base = _base_for(root, comfy_tmp_base_dir) / "unit-tests" / scope
    base.mkdir(parents=True, exist_ok=True)
    z = base / "empty.dat"
    z.write_bytes(b"")  # zero bytes

    await trigger_sync_seed_assets(http, api_base)

    # No AssetInfo created for this zero-byte file
    async with http.get(
        api_base + "/api/assets",
        params={"include_tags": "unit-tests," + scope, "name_contains": "empty.dat"},
    ) as r:
        body = await r.json()
        assert r.status == 200, body
        assert not [a for a in body.get("assets", []) if a.get("name") == "empty.dat"]


@pytest.mark.asyncio
@pytest.mark.parametrize("root", ["input", "output"])
async def test_sync_seed_idempotency(
    root: str,
    http,
    api_base: str,
    comfy_tmp_base_dir: Path,
):
    scope = f"idemseed-{uuid.uuid4().hex[:6]}"
    base = _base_for(root, comfy_tmp_base_dir) / "unit-tests" / scope
    base.mkdir(parents=True, exist_ok=True)
    files = [f"f{i}.dat" for i in range(3)]
    for i, n in enumerate(files):
        (base / n).write_bytes(_mkbytes(n, 1500 + i * 10))

    await trigger_sync_seed_assets(http, api_base)
    async with http.get(api_base + "/api/assets", params={"include_tags": "unit-tests," + scope}) as r1:
        b1 = await r1.json()
        assert r1.status == 200, b1
        c1 = len(b1.get("assets", []))

    # Seed again -> count must stay the same
    await trigger_sync_seed_assets(http, api_base)
    async with http.get(api_base + "/api/assets", params={"include_tags": "unit-tests," + scope}) as r2:
        b2 = await r2.json()
        assert r2.status == 200, b2
        c2 = len(b2.get("assets", []))
        assert c1 == c2 == len(files)


@pytest.mark.asyncio
@pytest.mark.parametrize("root", ["input", "output"])
async def test_sync_seed_nested_dirs_produce_parent_tags(
    root: str,
    http,
    api_base: str,
    comfy_tmp_base_dir: Path,
):
    scope = f"nest-{uuid.uuid4().hex[:6]}"
    # nested: unit-tests / scope / a / b / c / deep.txt
    deep_dir = _base_for(root, comfy_tmp_base_dir) / "unit-tests" / scope / "a" / "b" / "c"
    deep_dir.mkdir(parents=True, exist_ok=True)
    (deep_dir / "deep.txt").write_bytes(b"content")

    await trigger_sync_seed_assets(http, api_base)

    async with http.get(
        api_base + "/api/assets",
        params={"include_tags": f"unit-tests,{scope}", "name_contains": "deep.txt"},
    ) as r:
        body = await r.json()
        assert r.status == 200, body
        assets = body.get("assets", [])
        assert assets, "seeded asset not found"
        tags = set(assets[0].get("tags", []))
        # Must include all parent parts as tags + the root
        for must in {root, "unit-tests", scope, "a", "b", "c"}:
            assert must in tags


@pytest.mark.asyncio
@pytest.mark.parametrize("root", ["input", "output"])
async def test_concurrent_seed_hashing_same_file_no_dupes(
    root: str,
    http: aiohttp.ClientSession,
    api_base: str,
    comfy_tmp_base_dir: Path,
    run_scan_and_wait,
):
    """
    Create a single seed file, then schedule two scans back-to-back.
    Expect: no duplicate AssetInfos, a single hashed asset, and no scan failure.
    """
    scope = f"conc-seed-{uuid.uuid4().hex[:6]}"
    name = "seed_concurrent.bin"

    base =  _base_for(root, comfy_tmp_base_dir) / "unit-tests" / scope
    base.mkdir(parents=True, exist_ok=True)
    (base / name).write_bytes(b"Z" * 2048)

    await trigger_sync_seed_assets(http, api_base)

    s1, s2 = await asyncio.gather(
        http.post(api_base + "/api/assets/scan/schedule", json={"roots": [root]}),
        http.post(api_base + "/api/assets/scan/schedule", json={"roots": [root]}),
    )
    await s1.read()
    await s2.read()
    assert s1.status in (200, 202)
    assert s2.status in (200, 202)

    await run_scan_and_wait(root)

    async with http.get(
        api_base + "/api/assets",
        params={"include_tags": f"unit-tests,{scope}", "name_contains": name},
    ) as r:
        b = await r.json()
        assert r.status == 200, b
        matches = [a for a in b.get("assets", []) if a.get("name") == name]
        assert len(matches) == 1
        assert matches[0].get("asset_hash"), "Seed should have been hashed into an Asset"


@pytest.mark.asyncio
@pytest.mark.parametrize("root", ["input", "output"])
async def test_cache_state_retarget_on_content_change_asset_info_stays(
    root: str,
    http,
    api_base: str,
    comfy_tmp_base_dir: Path,
    asset_factory,
    make_asset_bytes,
    run_scan_and_wait,
):
    """
    Start with hashed H1 (AssetInfo A1). Replace file bytes on disk to become H2.
    After scan: AssetCacheState points to H2; A1 still references H1; downloading A1 -> 404.
    """
    scope = f"retarget-{uuid.uuid4().hex[:6]}"
    name = "content_change.bin"
    d1 = make_asset_bytes("v1-" + scope, 2048)

    a1 = await asset_factory(name, [root, "unit-tests", scope], {}, d1)
    aid = a1["id"]
    h1 = a1["asset_hash"]

    p = comfy_tmp_base_dir / root / "unit-tests" / scope / get_asset_filename(a1["asset_hash"], ".bin")
    assert p.exists()

    # Change the bytes in place to force a new content hash (H2)
    d2 = make_asset_bytes("v2-" + scope, 3072)
    p.write_bytes(d2)

    # Scan to verify and retarget the state; reconcilers run after scan
    await run_scan_and_wait(root)

    # AssetInfo still on the old content identity (H1)
    async with http.get(f"{api_base}/api/assets/{aid}") as rg:
        g = await rg.json()
        assert rg.status == 200, g
        assert g.get("asset_hash") == h1

    # Download must fail until a state exists for H1 (we removed the only one by retarget)
    async with http.get(f"{api_base}/api/assets/{aid}/content") as dl:
        body = await dl.json()
        assert dl.status == 404, body
        assert body["error"]["code"] == "FILE_NOT_FOUND"
