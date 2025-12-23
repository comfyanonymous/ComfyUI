"""
Test for preventing duplicate custom node imports.

This test verifies that when a custom node is imported by another custom node,
the module loading mechanism correctly detects and reuses the already-loaded module
instead of loading it again.
"""
import sys
import os
import pytest
from unittest.mock import MagicMock, patch

# Mock the required modules before importing nodes
mock_comfy_api = MagicMock()
mock_comfy_api.latest.io.ComfyNode = MagicMock
mock_comfy_api.latest.ComfyExtension = MagicMock

sys.modules['comfy_api'] = mock_comfy_api
sys.modules['comfy_api.latest'] = mock_comfy_api.latest
sys.modules['comfy_api.latest.io'] = mock_comfy_api.latest.io

# Mock folder_paths
mock_folder_paths = MagicMock()
sys.modules['folder_paths'] = mock_folder_paths

# Mock comfy modules
sys.modules['comfy'] = MagicMock()
sys.modules['comfy.model_management'] = MagicMock()

# Now we can import nodes
import nodes


@pytest.mark.asyncio
async def test_no_duplicate_import_when_already_loaded():
    """
    Test that load_custom_node detects and reuses already-loaded modules.
    
    Scenario:
    1. Custom node A is loaded by another custom node (e.g., via direct import)
    2. ComfyUI's custom node scanner encounters custom node A again
    3. The scanner should detect that A is already loaded and reuse it
    """
    # Create a mock module
    mock_module = MagicMock()
    mock_module.NODE_CLASS_MAPPINGS = {}
    mock_module.WEB_DIRECTORY = None
    
    # Simulate that the module was already imported with standard naming
    module_name = "custom_nodes.test_node"
    sys.modules[module_name] = mock_module
    
    # Track if exec_module is called (should not be called for already-loaded modules)
    exec_called = False
    
    def mock_exec_module(module):
        nonlocal exec_called
        exec_called = True
    
    # Patch the importlib methods
    with patch('importlib.util.spec_from_file_location') as mock_spec_func, \
         patch('importlib.util.module_from_spec') as mock_module_func:
        
        mock_spec = MagicMock()
        mock_spec.loader.exec_module = mock_exec_module
        mock_spec_func.return_value = mock_spec
        mock_module_func.return_value = MagicMock()
        
        # Create a temporary test directory to simulate the custom node path
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            test_node_dir = os.path.join(tmpdir, "test_node")
            os.makedirs(test_node_dir)
            
            # Create an __init__.py file
            init_file = os.path.join(test_node_dir, "__init__.py")
            with open(init_file, 'w') as f:
                f.write("NODE_CLASS_MAPPINGS = {}\n")
            
            # Attempt to load the custom node
            # Since we mocked sys.modules with 'custom_nodes.test_node',
            # the function should detect it and not execute the module again
            result = await nodes.load_custom_node(test_node_dir)
            
            # The function should return True (successful load)
            assert result == True
            
            # exec_module should NOT have been called because module was already loaded
            assert exec_called == False, "exec_module should not be called for already-loaded modules"


@pytest.mark.asyncio
async def test_load_new_module_when_not_loaded():
    """
    Test that load_custom_node properly loads new modules that haven't been imported yet.
    """
    import tempfile
    
    # Create a temporary test directory
    with tempfile.TemporaryDirectory() as tmpdir:
        test_node_dir = os.path.join(tmpdir, "new_test_node")
        os.makedirs(test_node_dir)
        
        # Create an __init__.py file with required attributes
        init_file = os.path.join(test_node_dir, "__init__.py")
        with open(init_file, 'w') as f:
            f.write("NODE_CLASS_MAPPINGS = {}\n")
        
        # Clear any existing module with this name
        sys_module_name = test_node_dir.replace(".", "_x_")
        if sys_module_name in sys.modules:
            del sys.modules[sys_module_name]
        
        # Load the custom node
        result = await nodes.load_custom_node(test_node_dir)
        
        # Should return True for successful load
        assert result == True
        
        # Module should now be in sys.modules
        assert sys_module_name in sys.modules


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

