import pytest
from unittest.mock import patch, mock_open
import os

from main import load_extra_path_config

@pytest.fixture
def mock_add_path():
    with patch('folder_paths.add_model_folder_path') as mock:
        yield mock

@pytest.mark.parametrize("os_type,base_path,expected_paths", [
    ("windows", "~\\AppData\\Local\\ComfyUI", {
        'checkpoints': os.path.expanduser('~\\AppData\\Local\\ComfyUI\\models\\checkpoints'),
        'clip': os.path.expanduser('~\\AppData\\Local\\ComfyUI\\models\\clip')
    }),
    ("linux", "~/comfyui", {
        'checkpoints': os.path.expanduser('~/comfyui/models/checkpoints'),
        'clip': os.path.expanduser('~/comfyui/models/clip')
    }),
    ("mac", "~/Library/Application Support/ComfyUI", {
        'checkpoints': os.path.expanduser('~/Library/Application Support/ComfyUI/models/checkpoints'),
        'clip': os.path.expanduser('~/Library/Application Support/ComfyUI/models/clip')
    }),
])
def test_load_config_with_base_path(mock_add_path, os_type, base_path, expected_paths, monkeypatch):
    if os_type == "windows":
        monkeypatch.setattr(os, 'name', 'nt')
        monkeypatch.setattr(os.path, 'sep', '\\')
    else:
        monkeypatch.setattr(os, 'name', 'posix')
        monkeypatch.setattr(os.path, 'sep', '/')

    yaml_content = f"""
    comfyui:
        base_path: "{base_path}"
        checkpoints: models{os.path.sep}checkpoints{os.path.sep}
        clip: models{os.path.sep}clip{os.path.sep}
    """
    mock_open_file = mock_open(read_data=yaml_content)

    with patch('builtins.open', mock_open_file):
        load_extra_path_config('dummy_path.yaml')

    for model_type, expected_path in expected_paths.items():
        mock_add_path.assert_any_call(model_type, expected_path)
