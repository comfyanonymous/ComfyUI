import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiohttp
import pytest
from conftest import trigger_sync_seed_assets


@pytest.mark.asyncio
async def test_download_attachment_and_inline(http: aiohttp.ClientSession, api_base: str, seeded_asset: dict):
    aid = seeded_asset["id"]

    # default attachment
    async with http.get(f"{api_base}/api/assets/{aid}/content") as r1:
        data = await r1.read()
        assert r1.status == 200
        cd = r1.headers.get("Content-Disposition", "")
        assert "attachment" in cd
        assert data and len(data) == 4096

    # inline requested
    async with http.get(f"{api_base}/api/assets/{aid}/content?disposition=inline") as r2:
        await r2.read()
        assert r2.status == 200
        cd2 = r2.headers.get("Content-Disposition", "")
        assert "inline" in cd2


@pytest.mark.asyncio
@pytest.mark.parametrize("root", ["input", "output"])
async def test_download_chooses_existing_state_and_updates_access_time(
    root: str,
    http: aiohttp.ClientSession,
    api_base: str,
    comfy_tmp_base_dir: Path,
    asset_factory,
    make_asset_bytes,
    run_scan_and_wait,
):
    """
    Hashed asset with two state paths: if the first one disappears,
    GET /content still serves from the remaining path and bumps last_access_time.
    """
    scope = f"dl-first-{uuid.uuid4().hex[:6]}"
    name = "first_existing_state.bin"
    data = make_asset_bytes(name, 3072)

    # Upload -> path1
    a = await asset_factory(name, [root, "unit-tests", scope], {}, data)
    aid = a["id"]

    base = comfy_tmp_base_dir / root / "unit-tests" / scope
    path1 = base / name
    assert path1.exists()

    # Seed path2 by copying, then scan to dedupe into a second state
    path2 = base / "alt" / name
    path2.parent.mkdir(parents=True, exist_ok=True)
    path2.write_bytes(data)
    await trigger_sync_seed_assets(http, api_base)
    await run_scan_and_wait(root)

    # Remove path1 so server must fall back to path2
    path1.unlink()

    # last_access_time before
    async with http.get(f"{api_base}/api/assets/{aid}") as rg0:
        d0 = await rg0.json()
        assert rg0.status == 200, d0
        ts0 = d0.get("last_access_time")

    await asyncio.sleep(0.05)
    async with http.get(f"{api_base}/api/assets/{aid}/content") as r:
        blob = await r.read()
        assert r.status == 200
        assert blob == data  # must serve from the surviving state (same bytes)

    async with http.get(f"{api_base}/api/assets/{aid}") as rg1:
        d1 = await rg1.json()
        assert rg1.status == 200, d1
        ts1 = d1.get("last_access_time")

    def _parse_iso8601(s: Optional[str]) -> Optional[float]:
        if not s:
            return None
        s = s[:-1] if s.endswith("Z") else s
        return datetime.fromisoformat(s).timestamp()

    t0 = _parse_iso8601(ts0)
    t1 = _parse_iso8601(ts1)
    assert t1 is not None
    if t0 is not None:
        assert t1 > t0


@pytest.mark.asyncio
@pytest.mark.parametrize("seeded_asset", [{"tags": ["models", "checkpoints"]}], indirect=True)
async def test_download_missing_file_returns_404(
    http: aiohttp.ClientSession, api_base: str, comfy_tmp_base_dir: Path, seeded_asset: dict
):
    # Remove the underlying file then attempt download.
    # We initialize fixture without additional tags to know exactly the asset file path.
    try:
        aid = seeded_asset["id"]
        async with http.get(f"{api_base}/api/assets/{aid}") as rg:
            detail = await rg.json()
            assert rg.status == 200
            rel_inside_category = detail["name"]
            abs_path = comfy_tmp_base_dir / "models" / "checkpoints" / rel_inside_category
            if abs_path.exists():
                abs_path.unlink()

        async with http.get(f"{api_base}/api/assets/{aid}/content") as r2:
            body = await r2.json()
            assert r2.status == 404
            assert body["error"]["code"] == "FILE_NOT_FOUND"
    finally:
        # We created asset without the "unit-tests" tag(see `autoclean_unit_test_assets`), we need to clear it manually.
        async with http.delete(f"{api_base}/api/assets/{aid}") as dr:
            await dr.read()


@pytest.mark.asyncio
@pytest.mark.parametrize("root", ["input", "output"])
async def test_download_404_if_all_states_missing(
    root: str,
    http: aiohttp.ClientSession,
    api_base: str,
    comfy_tmp_base_dir: Path,
    asset_factory,
    make_asset_bytes,
    run_scan_and_wait,
):
    """Multi-state asset: after the last remaining on-disk file is removed, download must return 404."""
    scope = f"dl-404-{uuid.uuid4().hex[:6]}"
    name = "missing_all_states.bin"
    data = make_asset_bytes(name, 2048)

    # Upload -> path1
    a = await asset_factory(name, [root, "unit-tests", scope], {}, data)
    aid = a["id"]

    base = comfy_tmp_base_dir / root / "unit-tests" / scope
    p1 = base / name
    assert p1.exists()

    # Seed a second state and dedupe
    p2 = base / "copy" / name
    p2.parent.mkdir(parents=True, exist_ok=True)
    p2.write_bytes(data)
    await trigger_sync_seed_assets(http, api_base)
    await run_scan_and_wait(root)

    # Remove first file -> download should still work via the second state
    p1.unlink()
    async with http.get(f"{api_base}/api/assets/{aid}/content") as ok1:
        b1 = await ok1.read()
        assert ok1.status == 200 and b1 == data

    # Remove the last file -> download must 404
    p2.unlink()
    async with http.get(f"{api_base}/api/assets/{aid}/content") as r2:
        body = await r2.json()
        assert r2.status == 404
        assert body["error"]["code"] == "FILE_NOT_FOUND"
