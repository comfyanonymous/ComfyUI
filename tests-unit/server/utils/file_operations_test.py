import pytest
from typing import List
from api_server.utils.file_operations import FileSystemOperations, FileSystemItem, is_file_info

@pytest.fixture
def temp_directory(tmp_path):
    # Create a temporary directory structure
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    dir1.mkdir()
    dir2.mkdir()
    (dir1 / "file1.txt").write_text("content1")
    (dir2 / "file2.txt").write_text("content2")
    (tmp_path / "file3.txt").write_text("content3")
    return tmp_path

def test_walk_directory(temp_directory):
    result: List[FileSystemItem] = FileSystemOperations.walk_directory(str(temp_directory))

    assert len(result) == 5  # 2 directories and 3 files

    files = [item for item in result if item['type'] == 'file']
    dirs = [item for item in result if item['type'] == 'directory']

    assert len(files) == 3
    assert len(dirs) == 2

    file_names = {file['name'] for file in files}
    assert file_names == {'file1.txt', 'file2.txt', 'file3.txt'}

    dir_names = {dir['name'] for dir in dirs}
    assert dir_names == {'dir1', 'dir2'}

def test_walk_directory_empty(tmp_path):
    result = FileSystemOperations.walk_directory(str(tmp_path))
    assert len(result) == 0

def test_walk_directory_file_size(temp_directory):
    result: List[FileSystemItem] = FileSystemOperations.walk_directory(str(temp_directory))
    files = [item for item in result if is_file_info(item)]
    for file in files:
        assert file['size'] > 0  # Assuming all files have some content
