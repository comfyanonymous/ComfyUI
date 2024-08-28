import os
import tempfile
from unittest.mock import patch

import pytest

from comfy_extras.nodes.nodes_language import SaveString


@pytest.fixture
def save_string_node():
    return SaveString()


@pytest.fixture
def mock_get_save_path(save_string_node):
    with patch.object(save_string_node, 'get_save_path') as mock_method:
        mock_method.return_value = (tempfile.gettempdir(), "test", 0, "", "test")
        yield mock_method


def test_save_string_single(save_string_node, mock_get_save_path):
    test_string = "Test string content"
    result = save_string_node.execute(test_string, "test_prefix", ".txt")

    assert result == {"ui": {"string": [test_string]}}
    mock_get_save_path.assert_called_once_with("test_prefix")

    saved_file_path = os.path.join(tempfile.gettempdir(), "test_00000_.txt")
    assert os.path.exists(saved_file_path)
    with open(saved_file_path, "r") as f:
        assert f.read() == test_string


def test_save_string_list(save_string_node, mock_get_save_path):
    test_strings = ["First string", "Second string", "Third string"]
    result = save_string_node.execute(test_strings, "test_prefix", ".txt")

    assert result == {"ui": {"string": test_strings}}
    mock_get_save_path.assert_called_once_with("test_prefix")

    for i, test_string in enumerate(test_strings):
        saved_file_path = os.path.join(tempfile.gettempdir(), f"test_00000_{i:02d}_.txt")
        assert os.path.exists(saved_file_path)
        with open(saved_file_path, "r") as f:
            assert f.read() == test_string


def test_save_string_default_extension(save_string_node, mock_get_save_path):
    test_string = "Test string content"
    result = save_string_node.execute(test_string, "test_prefix")

    assert result == {"ui": {"string": [test_string]}}
    mock_get_save_path.assert_called_once_with("test_prefix")

    saved_file_path = os.path.join(tempfile.gettempdir(), "test_00000_.json")
    assert os.path.exists(saved_file_path)
    with open(saved_file_path, "r") as f:
        assert f.read() == test_string
