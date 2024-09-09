import pytest
import os
import logging
from unittest.mock import patch

# Import the function we're testing
from main import load_extra_path_config

@pytest.fixture
def mock_yaml_file():
    yaml_content = """
    test_config:
        base_path: ~/App/
        checkpoint:
            subfolder1
            subfolder2
        lora: otherfolder
    """
    return yaml_content

@pytest.fixture
def mock_expanded_home():
    return '/home/user'

@patch('os.path.expanduser')
@patch('folder_paths.add_model_folder_path')
def test_load_extra_path_config(mock_add_model_folder_path, mock_expanduser, mock_yaml_file, mock_expanded_home, tmp_path):
    # Setup
    mock_expanduser.return_value = os.path.join(mock_expanded_home, 'App')
    yaml_path = tmp_path / "test_config.yaml"
    with open(yaml_path, 'w') as f:
        f.write(mock_yaml_file)

    # Call the function
    load_extra_path_config(yaml_path)

    # Assertions
    expected_calls = [
        ('checkpoint', os.path.join(mock_expanded_home, 'App', 'subfolder1 subfolder2')),
        ('lora', os.path.join(mock_expanded_home, 'App', 'otherfolder'))
    ]

    assert mock_add_model_folder_path.call_count == len(expected_calls)
    for call in mock_add_model_folder_path.call_args_list:
        assert call.args in expected_calls

    # Check if expanduser was called with the correct path
    mock_expanduser.assert_called_once_with('~/App/')

@pytest.fixture
def caplog(caplog):
    caplog.set_level(logging.INFO)
    return caplog

def test_load_extra_path_config_logging(mock_yaml_file, tmp_path, caplog):
    # Setup
    yaml_path = tmp_path / "test_config.yaml"
    with open(yaml_path, 'w') as f:
        f.write(mock_yaml_file)

    # Call the function
    with patch('folder_paths.add_model_folder_path'):
        load_extra_path_config(yaml_path)

    # Check logged messages
    assert "Adding extra search path checkpoint " in caplog.text
    assert "Adding extra search path lora " in caplog.text
