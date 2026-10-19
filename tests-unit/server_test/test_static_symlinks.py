from unittest.mock import MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from app.frontend_management import FrontendManager
from comfy.cli_args import DEFAULT_VERSION_STRING, args
import server


@pytest.mark.asyncio
async def test_package_static_assets_follow_symlinks(tmp_path, monkeypatch):
    cache_file = tmp_path / "cache" / "asset.js"
    cache_file.parent.mkdir()
    cache_file.write_text("package asset")

    paths = {}
    for prefix in ("frontend", "templates", "docs"):
        directory = tmp_path / prefix
        directory.mkdir()
        (directory / "asset.js").symlink_to(cache_file)
        paths[prefix] = str(directory)

    monkeypatch.setattr(args, "front_end_root", None)
    monkeypatch.setattr(args, "front_end_version", DEFAULT_VERSION_STRING)
    monkeypatch.setattr(FrontendManager, "init_frontend", lambda _version: paths["frontend"])
    monkeypatch.setattr(FrontendManager, "default_frontend_path", lambda: paths["frontend"])
    monkeypatch.setattr(FrontendManager, "get_installed_templates_version", lambda: None)
    monkeypatch.setattr(FrontendManager, "legacy_templates_path", lambda: paths["templates"])
    monkeypatch.setattr(FrontendManager, "embedded_docs_path", lambda: paths["docs"])

    prompt_server = server.PromptServer(None, MagicMock(enabled=False))
    prompt_server.add_routes()
    async with TestClient(TestServer(prompt_server.app)) as client:
        for path in ("/asset.js", "/templates/asset.js", "/docs/asset.js"):
            response = await client.get(path)
            assert response.status == 200
            assert await response.text() == "package asset"


@pytest.mark.asyncio
async def test_custom_frontend_root_does_not_follow_symlinks(tmp_path, monkeypatch):
    cache_file = tmp_path / "outside.js"
    cache_file.write_text("outside")
    root = tmp_path / "custom"
    root.mkdir()
    (root / "asset.js").symlink_to(cache_file)

    monkeypatch.setattr(args, "front_end_root", str(root))
    monkeypatch.setattr(FrontendManager, "get_installed_templates_version", lambda: None)
    monkeypatch.setattr(FrontendManager, "legacy_templates_path", lambda: None)
    monkeypatch.setattr(FrontendManager, "embedded_docs_path", lambda: None)

    prompt_server = server.PromptServer(None, MagicMock(enabled=False))
    prompt_server.add_routes()
    async with TestClient(TestServer(prompt_server.app)) as client:
        response = await client.get("/asset.js")
        assert response.status == 404


@pytest.mark.asyncio
async def test_selected_frontend_does_not_require_default_package(tmp_path, monkeypatch):
    root = tmp_path / "selected"
    root.mkdir()
    (root / "asset.js").write_text("selected asset")

    def unavailable_default():
        raise SystemExit(-1)

    monkeypatch.setattr(args, "front_end_root", None)
    monkeypatch.setattr(args, "front_end_version", "owner/frontend@v1.0.0")
    monkeypatch.setattr(FrontendManager, "init_frontend", lambda _version: str(root))
    monkeypatch.setattr(FrontendManager, "default_frontend_path", unavailable_default)
    monkeypatch.setattr(FrontendManager, "get_installed_templates_version", lambda: None)
    monkeypatch.setattr(FrontendManager, "legacy_templates_path", lambda: None)
    monkeypatch.setattr(FrontendManager, "embedded_docs_path", lambda: None)

    prompt_server = server.PromptServer(None, MagicMock(enabled=False))
    prompt_server.add_routes()
    async with TestClient(TestServer(prompt_server.app)) as client:
        response = await client.get("/asset.js")
        assert response.status == 200
        assert await response.text() == "selected asset"
