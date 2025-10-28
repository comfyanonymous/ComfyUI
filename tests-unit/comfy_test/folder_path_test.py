### 🗻 This file is created through the spirit of Mount Fuji at its peak
# TODO(yoland): clean up this after I get back down
import sys
import pytest
import os
import tempfile
from unittest.mock import patch
from importlib import reload

import folder_paths
import comfy.cli_args
from comfy.options import enable_args_parsing
enable_args_parsing()


@pytest.fixture()
def clear_folder_paths():
    # Reload the module after each test to ensure isolation
    yield
    reload(folder_paths)

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


@pytest.fixture
def set_base_dir():
    def _set_base_dir(base_dir):
        # Mock CLI args
        with patch.object(sys, 'argv', ["main.py", "--base-directory", base_dir]):
            reload(comfy.cli_args)
            reload(folder_paths)
    yield _set_base_dir
    # Reload the modules after each test to ensure isolation
    with patch.object(sys, 'argv', ["main.py"]):
        reload(comfy.cli_args)
        reload(folder_paths)


def test_get_directory_by_type(clear_folder_paths):
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


def test_base_path_changes(set_base_dir):
    test_dir = os.path.abspath("/test/dir")
    set_base_dir(test_dir)

    assert folder_paths.base_path == test_dir
    assert folder_paths.models_dir == os.path.join(test_dir, "models")
    assert folder_paths.input_directory == os.path.join(test_dir, "input")
    assert folder_paths.output_directory == os.path.join(test_dir, "output")
    assert folder_paths.temp_directory == os.path.join(test_dir, "temp")
    assert folder_paths.user_directory == os.path.join(test_dir, "user")

    assert os.path.join(test_dir, "custom_nodes") in folder_paths.get_folder_paths("custom_nodes")

    for name in ["checkpoints", "loras", "vae", "configs", "embeddings", "controlnet", "classifiers"]:
        assert folder_paths.get_folder_paths(name)[0] == os.path.join(test_dir, "models", name)


def test_base_path_change_clears_old(set_base_dir):
    test_dir = os.path.abspath("/test/dir")
    set_base_dir(test_dir)

    assert len(folder_paths.get_folder_paths("custom_nodes")) == 1

    single_model_paths = [
        "checkpoints",
        "loras",
        "vae",
        "configs",
        "clip_vision",
        "style_models",
        "diffusers",
        "vae_approx",
        "gligen",
        "upscale_models",
        "embeddings",
        "hypernetworks",
        "photomaker",
        "classifiers",
    ]
    for name in single_model_paths:
        assert len(folder_paths.get_folder_paths(name)) == 1

    for name in ["controlnet", "diffusion_models", "text_encoders"]:
        assert len(folder_paths.get_folder_paths(name)) == 2
