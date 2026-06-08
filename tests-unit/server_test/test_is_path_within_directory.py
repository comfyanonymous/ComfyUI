"""Tests for server.is_path_within_directory"""

import os
from unittest.mock import patch

# Importing `nodes` (a transitive import of `server`) prepends the `comfy/`
# directory to sys.path, which shadows the top-level `utils` package with
# `comfy/utils.py`. Importing `utils` first caches the real package so later
# `server` imports (e.g. app.frontend_management) resolve it correctly.
import utils  # noqa: F401
from server import is_path_within_directory


def test_subpath_is_within_directory(tmp_path):
    directory = str(tmp_path)
    sub = os.path.join(directory, "subfolder")
    assert is_path_within_directory(sub, directory) is True


def test_directory_itself_is_within_directory(tmp_path):
    directory = str(tmp_path)
    assert is_path_within_directory(directory, directory) is True


def test_sibling_path_is_not_within_directory(tmp_path):
    directory = os.path.join(str(tmp_path), "output")
    sibling = os.path.join(str(tmp_path), "other")
    assert is_path_within_directory(sibling, directory) is False


def test_parent_path_is_not_within_directory(tmp_path):
    directory = os.path.join(str(tmp_path), "output")
    parent = str(tmp_path)
    assert is_path_within_directory(parent, directory) is False


def test_different_drive_returns_false_instead_of_raising():
    # On Windows, os.path.commonpath raises ValueError when the paths don't
    # share a drive (e.g. comparing "C:\\output" with "D:\\secret"). That
    # should be treated as "not within", not bubble up as an exception.
    with patch("os.path.commonpath", side_effect=ValueError("Paths don't have the same drive")):
        assert is_path_within_directory("D:\\secret", "C:\\output") is False
