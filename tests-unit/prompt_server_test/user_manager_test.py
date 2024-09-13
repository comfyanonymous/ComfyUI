import pytest
import os
from aiohttp import web
from app.user_manager import UserManager
from unittest.mock import patch

pytestmark = (
    pytest.mark.asyncio
)  # This applies the asyncio mark to all test functions in the module


@pytest.fixture
def user_manager(tmp_path):
    um = UserManager()
    um.get_request_user_filepath = lambda req, file, **kwargs: os.path.join(
        tmp_path, file
    )
    return um


@pytest.fixture
def app(user_manager):
    app = web.Application()
    routes = web.RouteTableDef()
    user_manager.add_routes(routes)
    app.add_routes(routes)
    return app


async def test_listuserdata_empty_directory(aiohttp_client, app, tmp_path):
    client = await aiohttp_client(app)
    resp = await client.get("/userdata?dir=test_dir")
    assert resp.status == 404


async def test_listuserdata_with_files(aiohttp_client, app, tmp_path):
    os.makedirs(tmp_path / "test_dir")
    with open(tmp_path / "test_dir" / "file1.txt", "w") as f:
        f.write("test content")

    client = await aiohttp_client(app)
    resp = await client.get("/userdata?dir=test_dir")
    assert resp.status == 200
    assert await resp.json() == ["file1.txt"]


async def test_listuserdata_recursive(aiohttp_client, app, tmp_path):
    os.makedirs(tmp_path / "test_dir" / "subdir")
    with open(tmp_path / "test_dir" / "file1.txt", "w") as f:
        f.write("test content")
    with open(tmp_path / "test_dir" / "subdir" / "file2.txt", "w") as f:
        f.write("test content")

    client = await aiohttp_client(app)
    resp = await client.get("/userdata?dir=test_dir&recurse=true")
    assert resp.status == 200
    assert set(await resp.json()) == {"file1.txt", "subdir/file2.txt"}


async def test_listuserdata_full_info(aiohttp_client, app, tmp_path):
    os.makedirs(tmp_path / "test_dir")
    with open(tmp_path / "test_dir" / "file1.txt", "w") as f:
        f.write("test content")

    client = await aiohttp_client(app)
    resp = await client.get("/userdata?dir=test_dir&full_info=true")
    assert resp.status == 200
    result = await resp.json()
    assert len(result) == 1
    assert result[0]["path"] == "file1.txt"
    assert "size" in result[0]
    assert "modified" in result[0]


async def test_listuserdata_split_path(aiohttp_client, app, tmp_path):
    os.makedirs(tmp_path / "test_dir" / "subdir")
    with open(tmp_path / "test_dir" / "subdir" / "file1.txt", "w") as f:
        f.write("test content")

    client = await aiohttp_client(app)
    resp = await client.get("/userdata?dir=test_dir&recurse=true&split=true")
    assert resp.status == 200
    assert await resp.json() == [
        ["subdir/file1.txt", "subdir", "file1.txt"]
    ]


async def test_listuserdata_invalid_directory(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.get("/userdata?dir=")
    assert resp.status == 400


async def test_listuserdata_normalized_separator(aiohttp_client, app, tmp_path):
    os_sep = "\\"
    with patch("os.sep", os_sep):
        with patch("os.path.sep", os_sep):
            os.makedirs(tmp_path / "test_dir" / "subdir")
            with open(tmp_path / "test_dir" / "subdir" / "file1.txt", "w") as f:
                f.write("test content")

            client = await aiohttp_client(app)
            resp = await client.get("/userdata?dir=test_dir&recurse=true")
            assert resp.status == 200
            result = await resp.json()
            assert len(result) == 1
            assert "/" in result[0]  # Ensure forward slash is used
            assert "\\" not in result[0]  # Ensure backslash is not present
            assert result[0] == "subdir/file1.txt"

            # Test with full_info
            resp = await client.get(
                "/userdata?dir=test_dir&recurse=true&full_info=true"
            )
            assert resp.status == 200
            result = await resp.json()
            assert len(result) == 1
            assert "/" in result[0]["path"]  # Ensure forward slash is used
            assert "\\" not in result[0]["path"]  # Ensure backslash is not present
            assert result[0]["path"] == "subdir/file1.txt"
