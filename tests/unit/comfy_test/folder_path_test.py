### 🗻 This file is created through the spirit of Mount Fuji at its peak
# TODO(yoland): clean up this after I get back down
import os
import tempfile
from pathlib import Path

import pytest

from comfy.cmd import folder_paths
from comfy.component_model.folder_path_types import FolderNames, ModelPaths
from comfy.execution_context import context_folder_names_and_paths


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


def test_get_directory_by_type():
    test_dir = "/test/dir"
    folder_paths.set_output_directory(test_dir)
    assert folder_paths.get_directory_by_type("output") == test_dir
    assert folder_paths.get_directory_by_type("invalid") is None


def test_annotated_filepath():
    assert folder_paths.annotated_filepath("test.txt") == ("test.txt", None)
    assert folder_paths.annotated_filepath("test.txt [output]") == ("test.txt", folder_paths.get_output_directory())
    assert folder_paths.annotated_filepath("test.txt [input]") == ("test.txt", folder_paths.get_input_directory())
    assert folder_paths.annotated_filepath("test.txt [temp]") == ("test.txt", folder_paths.get_temp_directory())


def test_get_annotated_filepath():
    default_dir = "/default/dir"
    assert folder_paths.get_annotated_filepath("test.txt", default_dir) == os.path.join(default_dir, "test.txt")
    assert folder_paths.get_annotated_filepath("test.txt [output]") == os.path.join(folder_paths.get_output_directory(), "test.txt")


def test_add_model_folder_path():
    folder_paths.add_model_folder_path("test_folder", "/test/path")
    assert "/test/path" in folder_paths.get_folder_paths("test_folder")


def test_filter_files_extensions():
    files = ["file1.txt", "file2.jpg", "file3.png", "file4.txt"]
    assert folder_paths.filter_files_extensions(files, [".txt"]) == ["file1.txt", "file4.txt"]
    assert folder_paths.filter_files_extensions(files, [".jpg", ".png"]) == ["file2.jpg", "file3.png"]
    assert folder_paths.filter_files_extensions(files, []) == files


def test_get_filename_list(temp_dir):
    base_path = Path(temp_dir)
    fn = FolderNames(base_paths=[base_path])
    rel_path = Path("test/path")
    fn.add(ModelPaths(["test_folder"], additional_relative_directory_paths={rel_path}, supported_extensions={".txt"}))
    dir_path = base_path / rel_path
    Path.mkdir(dir_path, parents=True, exist_ok=True)
    files = ["file1.txt", "file2.jpg"]

    for file in files:
        Path.touch(dir_path / file, exist_ok=True)

    with context_folder_names_and_paths(fn):
        assert folder_paths.get_filename_list("test_folder") == ["file1.txt"]


def test_get_save_image_path(temp_dir):
    with context_folder_names_and_paths(FolderNames(base_paths=[Path(temp_dir)])):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path("test", temp_dir, 100, 100)
        assert os.path.samefile(full_output_folder, temp_dir)
        assert filename == "test"
        assert counter == 1
        assert subfolder == ""
        assert filename_prefix == "test"
