### 🗻 This file is created through the spirit of Mount Fuji at its peak
# TODO(yoland): clean up this after I get back down
import sys
import logging
import pytest
import os
import subprocess
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
    # get_annotated_filepath now normalizes with os.path.abspath (part of the
    # GHSA-779p traversal hardening), so compare against the normalized form —
    # on Windows abspath also prepends the current drive letter.
    assert folder_paths.get_annotated_filepath("test.txt", default_dir) == os.path.abspath(os.path.join(default_dir, "test.txt"))
    assert folder_paths.get_annotated_filepath("test.txt [output]") == os.path.abspath(os.path.join(folder_paths.get_output_directory(), "test.txt"))

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


def _link_dir(target, link):
    """Create a directory link the way the platform allows.

    Windows restricts symlink creation to elevated processes by default, so fall
    back to a junction there. os.walk follows both, so either exercises the same
    cycle hazard.
    """
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except (OSError, NotImplementedError, AttributeError):
        pass
    if sys.platform != "win32":
        pytest.skip("creating a directory link is not permitted in this environment")
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", link, target], capture_output=True, text=True
    )
    if completed.returncode != 0 or not os.path.isdir(link):
        pytest.skip(f"creating a directory junction failed: {completed.stderr.strip()}")


def test_recursive_search_enters_a_directory_cycle_once(temp_dir):
    """A link back to an ancestor must not list the same file over and over.

    `followlinks=True` is needed because model directories are routinely linked
    in through extra_model_paths.yaml, but os.walk does not detect a link that
    points at an ancestor. Before the visited-set guard, the single checkpoint
    below came back once per level of recursion.
    """
    checkpoints = os.path.join(temp_dir, "checkpoints")
    os.makedirs(checkpoints)
    open(os.path.join(checkpoints, "model.safetensors"), "w").close()
    _link_dir(temp_dir, os.path.join(checkpoints, "all"))

    files, _dirs = folder_paths.recursive_search(temp_dir)

    assert files == [os.path.join("checkpoints", "model.safetensors")], (
        f"the one real file must be listed once, got {files}"
    )


def test_recursive_search_still_follows_a_link_to_a_separate_tree(temp_dir, tmp_path):
    """The guard must not stop following links, only stop revisiting."""
    external = tmp_path / "external"
    external.mkdir()
    (external / "linked.safetensors").write_text("")
    _link_dir(str(external), os.path.join(temp_dir, "extra"))

    files, _dirs = folder_paths.recursive_search(temp_dir)

    assert files == [os.path.join("extra", "linked.safetensors")]


def test_recursive_search_skips_a_directory_it_cannot_resolve(temp_dir, caplog):
    """The warn-and-skip branch has to be reachable, not decorative.

    `os.path.realpath` only raises with `strict=True`; with the default it
    invents a path for anything it cannot resolve. Forcing the failure here
    keeps that branch covered and proves one unresolvable directory does not
    abort the rest of the walk.
    """
    os.makedirs(os.path.join(temp_dir, "good"))
    os.makedirs(os.path.join(temp_dir, "bad"))
    open(os.path.join(temp_dir, "good", "a.txt"), "w").close()
    open(os.path.join(temp_dir, "bad", "b.txt"), "w").close()

    real_realpath = os.path.realpath
    strict_by_name = {}

    def realpath(path, *args, **kwargs):
        name = os.path.basename(path)
        strict_by_name[name] = kwargs.get("strict")
        if name == "bad":
            raise OSError(2, "No such file or directory")
        return real_realpath(path, *args, **kwargs)

    with patch("folder_paths.os.path.realpath", side_effect=realpath):
        with caplog.at_level(logging.WARNING):
            files, _dirs = folder_paths.recursive_search(temp_dir)

    assert files == [os.path.join("good", "a.txt")]
    # strict=True is what makes the raise reachable at all; without it realpath
    # invents a path for anything it cannot resolve and this branch is dead.
    # Only the subdirectories are asserted: the root is resolved once before the
    # walk and is already known to exist from the os.path.isdir guard.
    assert strict_by_name["bad"] is True
    assert strict_by_name["good"] is True
    assert "Unable to resolve bad" in caplog.text


def test_recursive_search_still_excludes_named_directories(temp_dir):
    os.makedirs(os.path.join(temp_dir, "keep"))
    os.makedirs(os.path.join(temp_dir, "skipme"))
    open(os.path.join(temp_dir, "keep", "a.txt"), "w").close()
    open(os.path.join(temp_dir, "skipme", "b.txt"), "w").close()

    files, dirs = folder_paths.recursive_search(temp_dir, excluded_dir_names=["skipme"])

    assert files == [os.path.join("keep", "a.txt")]
    assert not any("skipme" in d for d in dirs)


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


def test_models_directory_cli_and_getters(temp_dir):
    try:
        with patch.object(sys, 'argv', ["main.py", "--models-directory", temp_dir]):
            reload(comfy.cli_args)
            reload(folder_paths)

        assert folder_paths.models_dir == os.path.abspath(temp_dir)

        with pytest.raises(Exception):
            comfy.cli_args.is_valid_directory(os.path.join(temp_dir, "non_existent_folder_path"))
    finally:
        with patch.object(sys, 'argv', ["main.py"]):
            reload(comfy.cli_args)
            reload(folder_paths)

