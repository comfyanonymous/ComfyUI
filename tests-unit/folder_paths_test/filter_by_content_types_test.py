import pytest
import os
import tempfile
from folder_paths import filter_files_content_types

@pytest.fixture(scope="module")
def file_extensions():
    return {
        'image': ['gif', 'heif', 'ico', 'jpeg', 'jpg', 'png', 'pnm', 'ppm', 'svg', 'tiff', 'webp', 'xbm', 'xpm'], 
        'audio': ['aif', 'aifc', 'aiff', 'au', 'flac', 'm4a', 'mp2', 'mp3', 'ogg', 'snd', 'wav'], 
        'video': ['avi', 'm2v', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ogv', 'qt', 'webm', 'wmv']
    }


@pytest.fixture(scope="module")
def mock_dir(file_extensions):
    with tempfile.TemporaryDirectory() as directory:
        for content_type, extensions in file_extensions.items():
            for extension in extensions:
                with open(f"{directory}/sample_{content_type}.{extension}", "w") as f:
                    f.write(f"Sample {content_type} file in {extension} format")
        yield directory


def test_categorizes_all_correctly(mock_dir, file_extensions):
    files = os.listdir(mock_dir)
    for content_type, extensions in file_extensions.items():
        filtered_files = filter_files_content_types(files, [content_type])
        for extension in extensions:
            assert f"sample_{content_type}.{extension}" in filtered_files


def test_categorizes_all_uniquely(mock_dir, file_extensions):
    files = os.listdir(mock_dir)
    for content_type, extensions in file_extensions.items():
        filtered_files = filter_files_content_types(files, [content_type])
        assert len(filtered_files) == len(extensions)


def test_handles_bad_extensions():
    files = ["file1.txt", "file2.py", "file3.example", "file4.pdf", "file5.ini", "file6.doc", "file7.md"]
    assert filter_files_content_types(files, ["image", "audio", "video"]) == []


def test_handles_no_extension():
    files = ["file1", "file2", "file3", "file4", "file5", "file6", "file7"]
    assert filter_files_content_types(files, ["image", "audio", "video"]) == []


def test_handles_no_files():
    files = []
    assert filter_files_content_types(files, ["image", "audio", "video"]) == []