import pytest
import os
import sys
from unittest.mock import Mock, patch, mock_open
import yaml

from utils.extra_custom_node_config import load_extra_custom_node_path_config, get_current_custom_node_paths
import folder_paths

@pytest.fixture()
def clear_folder_paths():
    # Save original state
    original_paths = folder_paths.folder_names_and_paths.copy()
    original_custom_nodes = folder_paths.get_custom_nodes_directories()
    
    # Clear the custom nodes directories
    for path in original_custom_nodes:
        folder_paths.folder_names_and_paths["custom_nodes"][0].remove(path)
    
    yield
    
    # Restore original state
    folder_paths.folder_names_and_paths = original_paths

@pytest.fixture
def mock_yaml_content():
    return {
        'custom_nodes_config': {
            'base_path': '~/custom_nodes_dir',
        },
        'another_config': {
            'base_path': '/absolute/path/to/nodes'
        }
    }

@pytest.fixture
def yaml_config_with_vars():
    return """
    custom_nodes_config:
      base_path: '%APPDATA%/ComfyUI/custom_nodes'
    """

@patch('builtins.open', new_callable=mock_open, read_data="dummy file content")
@patch('os.path.expanduser')
@patch('yaml.safe_load')
def test_load_extra_custom_node_paths(
    mock_yaml_load, mock_expanduser, mock_file, mock_yaml_content, clear_folder_paths
):
    # Setup mocks
    mock_yaml_load.return_value = mock_yaml_content
    mock_expanduser.side_effect = lambda x: x.replace('~/', '/home/user/')
    
    # Mock add_custom_node_directory
    with patch('folder_paths.add_custom_node_directory') as mock_add_dir:
        load_extra_custom_node_path_config('dummy_path.yaml')
        
        # Verify the directories were added
        expected_paths = [
            '/home/user/custom_nodes_dir/custom_nodes',
            '/absolute/path/to/nodes/custom_nodes'
        ]
        
        # Check that add_custom_node_directory was called with the expected paths
        assert mock_add_dir.call_count == 2
        actual_paths = [call[0][0] for call in mock_add_dir.call_args_list]
        assert set(actual_paths) == set(expected_paths)

@patch('builtins.open', new_callable=mock_open, read_data="dummy file content")
@patch('os.path.expandvars')
@patch('yaml.safe_load')
def test_load_extra_custom_node_paths_with_env_vars(
    mock_yaml_load, mock_expandvars, mock_file, yaml_config_with_vars, clear_folder_paths
):
    # Setup mocks
    mock_yaml_load.return_value = yaml.safe_load(yaml_config_with_vars)
    
    def expandvars_side_effect(path):
        if '%APPDATA%' in path:
            if sys.platform == 'win32':
                return path.replace('%APPDATA%', 'C:\\Users\\TestUser\\AppData\\Roaming')
            else:
                return path.replace('%APPDATA%', '/Users/TestUser/AppData/Roaming')
        return path
    
    mock_expandvars.side_effect = expandvars_side_effect
    
    # Mock add_custom_node_directory
    with patch('folder_paths.add_custom_node_directory') as mock_add_dir:
        load_extra_custom_node_path_config('dummy_path.yaml')
        
        # Verify the directory was added with expanded path
        expected_path = os.path.join(
            expandvars_side_effect('%APPDATA%/ComfyUI/custom_nodes'),
            'custom_nodes'
        )
        mock_add_dir.assert_called_once_with(expected_path)

@patch('builtins.open', new_callable=mock_open, read_data="dummy file content")
@patch('logging.warning')
@patch('os.path.exists', return_value=False)
def test_load_extra_custom_node_paths_nonexistent(
    mock_exists, mock_warning, mock_file, mock_yaml_content, clear_folder_paths
):
    # Setup mocks
    with patch('yaml.safe_load', return_value=mock_yaml_content):
        with patch('os.path.expanduser', side_effect=lambda x: x.replace('~/', '/home/user/')):
            with patch('folder_paths.add_custom_node_directory') as mock_add_dir:
                load_extra_custom_node_path_config('dummy_path.yaml')
                
                # Verify warning was logged for non-existent paths
                assert mock_warning.call_count == 2
                for call in mock_warning.call_args_list:
                    assert "does not exist, skipping" in call[0][0]
                
                # Verify no directories were added
                mock_add_dir.assert_not_called()

def test_get_current_custom_node_paths(clear_folder_paths):
    # Add some test paths
    test_paths = ['/path/one', '/path/two']
    for path in test_paths:
        folder_paths.add_custom_node_directory(path)
    
    # Test getting the paths
    result = get_current_custom_node_paths()
    assert set(result) == set(test_paths)

@patch('builtins.open', side_effect=Exception("Test error"))
@patch('logging.error')
def test_load_extra_custom_node_paths_error(mock_error, mock_file, clear_folder_paths):
    # Test error handling when loading YAML fails
    load_extra_custom_node_path_config('invalid.yaml')
    mock_error.assert_called_once()
    assert "Failed to load extra custom node paths config from invalid.yaml" in str(mock_error.call_args[0][0])

@patch('builtins.open', new_callable=mock_open, read_data="dummy file content")
@patch('yaml.safe_load', return_value={})
@patch('logging.info')
def test_load_extra_custom_node_paths_empty_config(mock_info, mock_yaml_load, mock_file, clear_folder_paths):
    # Test with empty config
    load_extra_custom_node_path_config('empty.yaml')
    mock_info.assert_called_with("No custom node paths found in configuration")
