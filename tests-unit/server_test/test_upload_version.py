"""/upload/image returns a content version that changes when the file is rewritten."""

import os
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from aiohttp import FormData
from aiohttp.test_utils import TestClient, TestServer

import folder_paths
import server


@pytest.fixture
def input_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(folder_paths, "input_directory", str(tmp_path))
    return tmp_path


@pytest_asyncio.fixture
async def client():
    prompt_server = server.PromptServer(None, MagicMock(enabled=False, register_upload=MagicMock(return_value=None)))
    prompt_server.app.add_routes(prompt_server.routes)
    async with TestClient(TestServer(prompt_server.app)) as c:
        yield c


async def upload(client, content: bytes) -> dict:
    form = FormData()
    form.add_field("image", content, filename="same.png", content_type="image/png")
    form.add_field("overwrite", "true")
    resp = await client.post("/upload/image", data=form)
    assert resp.status == 200
    return await resp.json()


@pytest.mark.asyncio
async def test_upload_version_matches_the_file_and_changes_on_overwrite(input_dir, client):
    first = await upload(client, b"first")
    st = os.stat(input_dir / "same.png")
    assert first["version"] == f"{st.st_mtime_ns:x}-{st.st_size:x}"

    second = await upload(client, b"second, longer")
    assert second["name"] == "same.png"
    assert second["version"] != first["version"]
