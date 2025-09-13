from pathlib import Path

import aiohttp
import pytest


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
