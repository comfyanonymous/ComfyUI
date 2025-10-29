import pytest
import fsspec
from comfy.component_model import package_filesystem
import os


# Ensure the filesystem is registered once for all tests
@pytest.fixture(scope="module", autouse=True)
def setup_package_filesystem():
    if "pkg" not in fsspec.available_protocols():
        fsspec.register_implementation(
            package_filesystem.PkgResourcesFileSystem.protocol,
            package_filesystem.PkgResourcesFileSystem,
        )
    # Yield to allow tests to run, then teardown if necessary (though not needed here)
    yield


@pytest.fixture
def pkg_fs():
    return fsspec.filesystem("pkg")


def test_open_file_in_package(pkg_fs):
    """Test opening a file directly within a package."""
    with pkg_fs.open("pkg://tests.unit.fsspec_tests.files/b.txt", "rb") as f:
        content = f.read()
    assert content == b"OK"


def test_open_file_in_text_mode(pkg_fs):
    """Test opening a file in text mode."""
    with pkg_fs.open("pkg://tests.unit.fsspec_tests.files/b.txt", "r") as f:
        content = f.read()
    assert content == "OK"


def test_open_file_in_subdir(pkg_fs):
    """Test opening a file in a subdirectory of a package."""
    with pkg_fs.open("pkg://tests.unit.fsspec_tests.files/subdir/a.txt", "rb") as f:
        content = f.read()
    assert content == b"OK"


def test_file_not_found(pkg_fs):
    """Test that opening a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        pkg_fs.open("pkg://tests.unit.fsspec_tests.files/nonexistent.txt")


def test_package_not_found(pkg_fs):
    """Test that using a non-existent package raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        pkg_fs.open("pkg://non.existent.package/resource.txt")


def test_ls_package_root(pkg_fs):
    """Test listing the contents of a package."""
    contents = pkg_fs.ls("pkg://tests.unit.fsspec_tests.files", detail=False)
    expected_items = {
        "pkg://tests.unit.fsspec_tests.files/b.txt",
        "pkg://tests.unit.fsspec_tests.files/subdir",
        "pkg://tests.unit.fsspec_tests.files/__init__.py",
    }
    # Use a subset assertion to be resilient to __pycache__
    normalized_contents = {os.path.normpath(p.split('@')[0]) for p in contents}
    normalized_expected = {os.path.normpath(p) for p in expected_items}
    assert normalized_expected.issubset(normalized_contents)


def test_ls_subdir(pkg_fs):
    """Test listing the contents of a subdirectory."""
    contents = pkg_fs.ls("pkg://tests.unit.fsspec_tests.files/subdir", detail=False)
    normalized_contents = [os.path.normpath(p.split('@')[0]) for p in contents]
    assert os.path.normpath("pkg://tests.unit.fsspec_tests.files/subdir/a.txt") in normalized_contents


def test_info_file(pkg_fs):
    """Test getting info for a file."""
    info = pkg_fs.info("pkg://tests.unit.fsspec_tests.files/b.txt")
    assert info["type"] == "file"
    assert info["name"] == "pkg://tests.unit.fsspec_tests.files/b.txt"
    assert info["size"] == 2


def test_info_directory(pkg_fs):
    """Test getting info for a directory."""
    info = pkg_fs.info("pkg://tests.unit.fsspec_tests.files/subdir")
    assert info["type"] == "directory"
    assert info["name"] == "pkg://tests.unit.fsspec_tests.files/subdir"
    # Directories typically don't have a size in this context, or it might be 0
    assert "size" in info  # Ensure size key exists
    assert info["size"] is None or info["size"] == 0


def test_load_font_with_upath(pkg_fs):
    """Test that a font can be loaded from the pkg filesystem using UPath."""
    from upath import UPath
    from PIL import ImageFont, features

    # This test requires Pillow with FreeType support
    if not features.check("freetype2"):
        pytest.skip("Pillow FreeType support not available")

    # UPath will use the registered fsspec filesystem for "pkg"
    font_path = UPath("pkg://comfy.fonts/Tiny5-Regular.ttf")

    with font_path.open("rb") as f:
        font = ImageFont.truetype(f, 10)

    assert font is not None
    assert isinstance(font, ImageFont.FreeTypeFont)
