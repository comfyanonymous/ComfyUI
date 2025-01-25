### 🗻 This file is created through the spirit of Mount Fuji at its peak
# TODO(yoland): clean up this after I get back down
import pytest
import os
import tempfile
from unittest.mock import patch

import folder_paths

@pytest.fixture()
def clear_folder_paths():
    # Clear the global dictionary before each test to ensure isolation
    original = folder_paths.folder_names_and_paths.copy()
    folder_paths.folder_names_and_paths.clear()
    yield
    folder_paths.folder_names_and_paths = original

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

def test_add_model_folder_path_append(clear_folder_paths):
    folder_paths.add_model_folder_path("test_folder", "/default/path", is_default=True)
    folder_paths.add_model_folder_path("test_folder", "/test/path", is_default=False)
    assert folder_paths.get_folder_paths("test_folder") == ["/default/path", "/test/path"]


def test_add_model_folder_path_insert(clear_folder_paths):
    folder_paths.add_model_folder_path("test_folder", "/test/path", is_default=False)
    folder_paths.add_model_folder_path("test_folder", "/default/path", is_default=True)
    assert folder_paths.get_folder_paths("test_folder") == ["/default/path", "/test/path"]


def test_add_model_folder_path_re_add_existing_default(clear_folder_paths):
    folder_paths.add_model_folder_path("test_folder", "/test/path", is_default=False)
    folder_paths.add_model_folder_path("test_folder", "/old_default/path", is_default=True)
    assert folder_paths.get_folder_paths("test_folder") == ["/old_default/path", "/test/path"]
    folder_paths.add_model_folder_path("test_folder", "/test/path", is_default=True)
    assert folder_paths.get_folder_paths("test_folder") == ["/test/path", "/old_default/path"]


def test_add_model_folder_path_re_add_existing_non_default(clear_folder_paths):
    folder_paths.add_model_folder_path("test_folder", "/test/path", is_default=False)
    folder_paths.add_model_folder_path("test_folder", "/default/path", is_default=True)
    assert folder_paths.get_folder_paths("test_folder") == ["/default/path", "/test/path"]
    folder_paths.add_model_folder_path("test_folder", "/test/path", is_default=False)
    assert folder_paths.get_folder_paths("test_folder") == ["/default/path", "/test/path"]


def test_recursive_search(temp_dir):
    os.makedirs(os.path.join(temp_dir, "subdir"))
    open(os.path.join(temp_dir, "file1.txt"), "w").close()
    open(os.path.join(temp_dir, "subdir", "file2.txt"), "w").close()

    files, dirs = folder_paths.recursive_search(temp_dir)
    assert set(files) == {"file1.txt", os.path.join("subdir", "file2.txt")}
    assert len(dirs) == 2  # temp_dir and subdir

def test_filter_files_extensions():
    files = ["file1.txt", "file2.jpg", "file3.png", "file4.txt"]
    assert folder_paths.filter_files_extensions(files, [".txt"]) == ["file1.txt", "file4.txt"]
    assert folder_paths.filter_files_extensions(files, [".jpg", ".png"]) == ["file2.jpg", "file3.png"]
    assert folder_paths.filter_files_extensions(files, []) == files

@patch("folder_paths.recursive_search")
@patch("folder_paths.folder_names_and_paths")
def test_get_filename_list(mock_folder_names_and_paths, mock_recursive_search):
    mock_folder_names_and_paths.__getitem__.return_value = (["/test/path"], {".txt"})
    mock_recursive_search.return_value = (["file1.txt", "file2.jpg"], {})
    assert folder_paths.get_filename_list("test_folder") == ["file1.txt"]

def test_get_save_image_path(temp_dir):
    with patch("folder_paths.output_directory", temp_dir):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path("test", temp_dir, 100, 100)
        assert os.path.samefile(full_output_folder, temp_dir)
        assert filename == "test"
        assert counter == 1
        assert subfolder == ""
        assert filename_prefix == "test"


def test_base_path_changes():
    test_dir = "/test/dir"
    folder_paths.reset_all_paths(test_dir)
    assert folder_paths.base_path == test_dir
    assert folder_paths.models_dir == os.path.join(test_dir, "models")
    assert folder_paths.input_directory == os.path.join(test_dir, "input")
    assert folder_paths.output_directory == os.path.join(test_dir, "output")
    assert folder_paths.temp_directory == os.path.join(test_dir, "temp")
    assert folder_paths.user_directory == os.path.join(test_dir, "user")

    assert os.path.join(test_dir, "custom_nodes") in folder_paths.get_folder_paths("custom_nodes")

    for name in ["checkpoints", "loras", "vae", "configs", "embeddings", "controlnet", "classifiers", "configs"]:
        assert folder_paths.get_folder_paths(name)[0] == os.path.join(test_dir, name)


def test_add_default_paths_preseves_dirs():
    test_dir = os.path.abspath(os.path.join(os.path.curdir, "..", ".."))
    base_path = folder_paths.base_path
    models_dir = folder_paths.models_dir
    input_directory = folder_paths.input_directory

    folder_paths.add_default_model_paths(test_dir)
    assert folder_paths.base_path == base_path
    assert folder_paths.models_dir == models_dir
    assert folder_paths.input_directory == input_directory


def test_add_default_paths_preseves_paths():
    folder_paths.reset_all_paths("/test/dir")
    test_dir = os.path.abspath(os.path.join(os.path.curdir, "invalid"))
    checkpoints = folder_paths.get_folder_paths("checkpoints")[0]
    text_encoders = folder_paths.get_folder_paths("text_encoders")[0]
    clip = folder_paths.get_folder_paths("clip")[1]

    folder_paths.add_default_model_paths(test_dir)
    assert not os.path.join(test_dir, "custom_nodes") in folder_paths.get_folder_paths("custom_nodes")
    assert folder_paths.get_folder_paths("checkpoints")[0] == checkpoints
    assert folder_paths.get_folder_paths("text_encoders")[0] == text_encoders
    assert folder_paths.get_folder_paths("clip")[1] == clip


def test_add_default_model_paths():
    folder_paths.reset_all_paths("/test/dir")
    test_dir = os.path.abspath(os.path.join(os.path.curdir, "bad_path"))
    folder_paths.add_default_model_paths(test_dir)

    for name in ["checkpoints", "loras", "vae", "configs", "embeddings", "controlnet", "classifiers", "configs"]:
        paths = folder_paths.get_folder_paths(name)
        # Handle multiple default paths
        assert len(paths) % 2 == 0
        index = int(len(paths) / 2)
        assert folder_paths.get_folder_paths(name)[index] == os.path.join(test_dir, name)

    assert folder_paths.get_folder_paths("clip")[2] == os.path.join(test_dir, "text_encoders")
    assert folder_paths.get_folder_paths("clip")[3] == os.path.join(test_dir, "clip")
