import pytest
import yaml
import os
from unittest.mock import Mock, patch, mock_open

from utils.extra_config import load_extra_path_config
import folder_paths

@pytest.fixture
def mock_yaml_content():
    return {
        'test_config': {
            'base_path': '~/App/',
            'checkpoints': 'subfolder1',
        }
    }

@pytest.fixture
def mock_expanded_home():
    return '/home/user'

@pytest.fixture
def mock_add_model_folder_path():
    return Mock()

@pytest.fixture
def mock_expanduser(mock_expanded_home):
    def _expanduser(path):
        if path.startswith('~/'):
            return os.path.join(mock_expanded_home, path[2:])
        return path
    return _expanduser

@pytest.fixture
def mock_yaml_safe_load(mock_yaml_content):
    return Mock(return_value=mock_yaml_content)

@patch('builtins.open', new_callable=mock_open, read_data="dummy file content")
def test_load_extra_model_paths_expands_userpath(
    mock_file,
    monkeypatch,
    mock_add_model_folder_path, 
    mock_expanduser, 
    mock_yaml_safe_load,
    mock_expanded_home
):
    # Attach mocks used by load_extra_path_config
    monkeypatch.setattr(folder_paths, 'add_model_folder_path', mock_add_model_folder_path)
    monkeypatch.setattr(os.path, 'expanduser', mock_expanduser)
    monkeypatch.setattr(yaml, 'safe_load', mock_yaml_safe_load)

    dummy_yaml_file_name = 'dummy_path.yaml'
    load_extra_path_config(dummy_yaml_file_name)

    expected_calls = [
        ('checkpoints', os.path.join(mock_expanded_home, 'App', 'subfolder1')),
    ]

    assert mock_add_model_folder_path.call_count == len(expected_calls)
    
    # Check if add_model_folder_path was called with the correct arguments
    for actual_call, expected_call in zip(mock_add_model_folder_path.call_args_list, expected_calls):
        assert actual_call.args == expected_call

    # Check if yaml.safe_load was called
    mock_yaml_safe_load.assert_called_once()

    # Check if open was called with the correct file path
    mock_file.assert_called_once_with(dummy_yaml_file_name, 'r')
