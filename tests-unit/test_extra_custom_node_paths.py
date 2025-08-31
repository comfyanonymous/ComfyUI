import os
import sys
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.extra_custom_node_config import load_extra_custom_node_path_config, get_current_custom_node_paths
import folder_paths

@pytest.fixture(autouse=True)
def setup_teardown():
    """Fixture to save and restore the original folder paths state."""
    # Save original state
    original_paths = folder_paths.folder_names_and_paths.copy()
    original_custom_nodes = folder_paths.get_custom_nodes_directories()
    
    # Reset custom nodes for each test
    if "custom_nodes" in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths["custom_nodes"] = (list(original_custom_nodes), set())
    
    yield  # Test runs here
    
    # Restore original state
    folder_paths.folder_names_and_paths = original_paths


def test_load_extra_custom_node_paths():
    """Test loading custom node paths from YAML config."""
    # Mock YAML content
    yaml_content = """
    custom_nodes_config:
      base_path: /custom/nodes/path
    another_config:
      base_path: /another/nodes/path
    """
    
    # Set up mocks
    with patch('builtins.open', mock_open(read_data=yaml_content)) as mock_file, \
         patch('os.path.exists', return_value=True), \
         patch('os.path.isdir', return_value=True), \
         patch('os.path.abspath', side_effect=lambda x: x) as mock_abspath:
        
        # Call the function
        load_extra_custom_node_path_config("dummy_path.yaml")
        
        # Verify the file was opened
        mock_file.assert_called_once_with("dummy_path.yaml", 'r', encoding='utf-8')
        
        # Verify the paths were added
        custom_paths = folder_paths.get_custom_nodes_directories()
        # Normalize paths for consistent comparison
        normalized_paths = [os.path.normpath(p).replace('\\', '/').lower() for p in custom_paths]
        expected_paths = ["/custom/nodes/path/custom_nodes", 
                         "/another/nodes/path/custom_nodes"]
        
        for path in expected_paths:
            assert path.lower() in normalized_paths, f"Expected path {path} not found in {normalized_paths}"


def test_load_extra_custom_node_paths_with_tilde():
    """Test that paths with tilde are expanded."""
    yaml_content = """
    custom_nodes_config:
      base_path: ~/custom_nodes
    """
    
    with patch('builtins.open', mock_open(read_data=yaml_content)) as mock_file, \
         patch('os.path.expanduser', side_effect=lambda x: x.replace('~', '/home/user')), \
         patch('os.path.exists', return_value=True), \
         patch('os.path.isdir', return_value=True), \
         patch('os.path.abspath', side_effect=lambda x: x):
        
        load_extra_custom_node_path_config("dummy_path.yaml")
        
        custom_paths = folder_paths.get_custom_nodes_directories()
        # Normalize paths for consistent comparison
        normalized_paths = [os.path.normpath(p).replace('\\', '/').lower() for p in custom_paths]
        expected_path = "/home/user/custom_nodes/custom_nodes"
        assert expected_path.lower() in normalized_paths, f"Expected path {expected_path} not found in {normalized_paths}"


def test_load_extra_custom_node_paths_with_env_vars():
    """Test that environment variables in paths are expanded."""
    yaml_content = """
    custom_nodes_config:
      base_path: ${MY_CUSTOM_NODES_PATH}/custom_nodes
    """
    
    with patch('builtins.open', mock_open(read_data=yaml_content)), \
         patch.dict('os.environ', {'MY_CUSTOM_NODES_PATH': '/env/path'}), \
         patch('os.path.exists', return_value=True), \
         patch('os.path.isdir', return_value=True), \
         patch('os.path.abspath', side_effect=lambda x: x):
        
        load_extra_custom_node_path_config("dummy_path.yaml")
        
        custom_paths = folder_paths.get_custom_nodes_directories()
        # Normalize paths for consistent comparison
        normalized_paths = [os.path.normpath(p).replace('\\', '/').lower() for p in custom_paths]
        expected_path = "/env/path/custom_nodes/custom_nodes"
        assert expected_path.lower() in normalized_paths, f"Expected path {expected_path} not found in {normalized_paths}"


def test_nonexistent_paths_are_skipped():
    """Test that non-existent paths are skipped with a warning."""
    yaml_content = """
    custom_nodes_config:
      base_path: /nonexistent/path
    """
    
    with patch('builtins.open', mock_open(read_data=yaml_content)), \
         patch('os.path.exists', return_value=False), \
         patch('os.path.isdir', return_value=False), \
         patch('logging.warning') as mock_warning:
        
        load_extra_custom_node_path_config("dummy_path.yaml")
        
        # Verify warning was logged
        mock_warning.assert_called_once()
        assert "does not exist" in mock_warning.call_args[0][0]
        
        # Verify path was not added
        assert "/nonexistent/path/custom_nodes" not in folder_paths.get_custom_nodes_directories()


def test_invalid_yaml_handling():
    """Test that invalid YAML is handled gracefully."""
    with patch('builtins.open', mock_open(read_data='invalid: yaml: : :')), \
         patch('logging.error') as mock_error:
        
        load_extra_custom_node_path_config("invalid.yaml")
        
        # Verify error was logged
        mock_error.assert_called_once()
        assert "Failed to load extra custom node paths" in mock_error.call_args[0][0]


def test_empty_config():
    """Test that empty config doesn't cause errors."""
    with patch('builtins.open', mock_open(read_data='')), \
         patch('yaml.safe_load', return_value={'custom_nodes_config': {}}), \
         patch('logging.info') as mock_info:
        
        load_extra_custom_node_path_config("empty.yaml")
        
        # Verify info was logged
        mock_info.assert_called_once()
        assert "No custom node paths found in configuration" in mock_info.call_args[0][0]
