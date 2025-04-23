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
    ) if file else tmp_path
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
    assert await resp.json() == [["subdir/file1.txt", "subdir", "file1.txt"]]


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

async def test_listuserdata_include_empty_dirs(aiohttp_client, app, tmp_path):
    # Arrange
    test_dir = tmp_path / "test_dir"
    empty_subdir = test_dir / "empty_subdir"
    file1 = test_dir / "file1.txt"

    os.makedirs(test_dir)
    os.makedirs(empty_subdir)
    with open(file1, "w") as f:
        f.write("test")

    client = await aiohttp_client(app)

    # Act
    resp = await client.get("/userdata?dir=test_dir&emptyDirs=true")

    # Assert
    assert resp.status == 200
    result = await resp.json()
    assert set(result) == {"file1.txt", "empty_subdir"}

async def test_listuserdata_exclude_empty_dirs_default(aiohttp_client, app, tmp_path):
    # Arrange
    test_dir = tmp_path / "test_dir"
    empty_subdir = test_dir / "empty_subdir"
    file1 = test_dir / "file1.txt"

    os.makedirs(test_dir)
    os.makedirs(empty_subdir)
    with open(file1, "w") as f:
        f.write("test")

    client = await aiohttp_client(app)

    # Act
    resp = await client.get("/userdata?dir=test_dir") # emptyDirs defaults to false

    # Assert
    assert resp.status == 200
    result = await resp.json()
    assert result == ["file1.txt"]

async def test_listuserdata_recursive_include_empty_dirs(aiohttp_client, app, tmp_path):
    # Arrange
    base_dir = tmp_path / "test_dir"
    occupied = base_dir / "occupied_directory"
    empty = base_dir / "empty_directory"
    file1 = occupied / "file1.txt"

    os.makedirs(occupied)
    os.makedirs(empty)
    with open(file1, "w") as f:
        f.write("content")

    client = await aiohttp_client(app)

    # Act
    resp = await client.get("/userdata?dir=test_dir&recurse=true&emptyDirs=true")

    # Assert
    assert resp.status == 200
    result = await resp.json()
    assert set(result) == {"occupied_directory/file1.txt", "empty_directory"}

async def test_listuserdata_full_info_include_empty_dirs(aiohttp_client, app, tmp_path):
    # Arrange
    test_dir = tmp_path / "test_dir"
    file1 = test_dir / "file1.txt"
    empty = test_dir / "empty_subdir"
    os.makedirs(test_dir)
    os.makedirs(empty)
    with open(file1, "w") as f:
        f.write("content")

    client = await aiohttp_client(app)

    # Act
    resp = await client.get("/userdata?dir=test_dir&full_info=true&emptyDirs=true")

    # Assert
    assert resp.status == 200
    result = await resp.json()
    paths = {info["path"] for info in result}
    assert paths == {"file1.txt", "empty_subdir"}
    for info in result:
        assert "size" in info
        assert "modified" in info

async def test_listuserdata_recurse_split_include_empty_dirs(aiohttp_client, app, tmp_path):
    # Arrange
    test_dir = tmp_path / "test_dir"
    file1 = test_dir / "file1.txt"
    empty = test_dir / "empty_subdir"
    occupying_dir = test_dir / "occupied_directory"
    another_occupying_dir = occupying_dir / "another_occupied_directory"
    file2 = another_occupying_dir / "file2.txt"
    os.makedirs(test_dir)
    os.makedirs(occupying_dir)
    os.makedirs(another_occupying_dir)
    os.makedirs(empty)
    with open(file1, "w") as f:
        f.write("content")
    with open(file2, "w") as f:
        f.write("nested content")

    client = await aiohttp_client(app)

    # Act
    resp = await client.get("/userdata?dir=test_dir&split=true&emptyDirs=true&recurse=true")

    # Assert
    assert resp.status == 200
    result = await resp.json()
    assert set(tuple(r) for r in result) == {
        ("file1.txt", "", "file1.txt"),
        ("empty_subdir", "empty_subdir", ""),
        ("occupied_directory/another_occupied_directory/file2.txt", "occupied_directory/another_occupied_directory", "file2.txt"),
    }

async def test_post_userdata_new_file(aiohttp_client, app, tmp_path):
    client = await aiohttp_client(app)
    content = b"test content"
    resp = await client.post("/userdata/test.txt", data=content)

    assert resp.status == 200
    assert await resp.text() == '"test.txt"'

    # Verify file was created with correct content
    with open(tmp_path / "test.txt", "rb") as f:
        assert f.read() == content


async def test_post_userdata_overwrite_existing(aiohttp_client, app, tmp_path):
    # Create initial file
    with open(tmp_path / "test.txt", "w") as f:
        f.write("initial content")

    client = await aiohttp_client(app)
    new_content = b"updated content"
    resp = await client.post("/userdata/test.txt", data=new_content)

    assert resp.status == 200
    assert await resp.text() == '"test.txt"'

    # Verify file was overwritten
    with open(tmp_path / "test.txt", "rb") as f:
        assert f.read() == new_content


async def test_post_userdata_no_overwrite(aiohttp_client, app, tmp_path):
    # Create initial file
    with open(tmp_path / "test.txt", "w") as f:
        f.write("initial content")

    client = await aiohttp_client(app)
    resp = await client.post("/userdata/test.txt?overwrite=false", data=b"new content")

    assert resp.status == 409

    # Verify original content unchanged
    with open(tmp_path / "test.txt", "r") as f:
        assert f.read() == "initial content"


async def test_post_userdata_full_info(aiohttp_client, app, tmp_path):
    client = await aiohttp_client(app)
    content = b"test content"
    resp = await client.post("/userdata/test.txt?full_info=true", data=content)

    assert resp.status == 200
    result = await resp.json()
    assert result["path"] == "test.txt"
    assert result["size"] == len(content)
    assert "modified" in result


async def test_move_userdata(aiohttp_client, app, tmp_path):
    # Create initial file
    with open(tmp_path / "source.txt", "w") as f:
        f.write("test content")

    client = await aiohttp_client(app)
    resp = await client.post("/userdata/source.txt/move/dest.txt")

    assert resp.status == 200
    assert await resp.text() == '"dest.txt"'

    # Verify file was moved
    assert not os.path.exists(tmp_path / "source.txt")
    with open(tmp_path / "dest.txt", "r") as f:
        assert f.read() == "test content"


async def test_move_userdata_no_overwrite(aiohttp_client, app, tmp_path):
    # Create source and destination files
    with open(tmp_path / "source.txt", "w") as f:
        f.write("source content")
    with open(tmp_path / "dest.txt", "w") as f:
        f.write("destination content")

    client = await aiohttp_client(app)
    resp = await client.post("/userdata/source.txt/move/dest.txt?overwrite=false")

    assert resp.status == 409

    # Verify files remain unchanged
    with open(tmp_path / "source.txt", "r") as f:
        assert f.read() == "source content"
    with open(tmp_path / "dest.txt", "r") as f:
        assert f.read() == "destination content"


async def test_move_userdata_full_info(aiohttp_client, app, tmp_path):
    # Create initial file
    with open(tmp_path / "source.txt", "w") as f:
        f.write("test content")

    client = await aiohttp_client(app)
    resp = await client.post("/userdata/source.txt/move/dest.txt?full_info=true")

    assert resp.status == 200
    result = await resp.json()
    assert result["path"] == "dest.txt"
    assert result["size"] == len("test content")
    assert "modified" in result

    # Verify file was moved
    assert not os.path.exists(tmp_path / "source.txt")
    with open(tmp_path / "dest.txt", "r") as f:
        assert f.read() == "test content"
