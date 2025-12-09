
import pytest
from comfy.cmd import folder_paths

def test_folder_paths_interface_sanity():
    """
    Basic sanity check to ensure functions added to folder_paths.pyi exist in folder_paths.py at runtime.
    """
    # Check for functions recently added/modified
    assert hasattr(folder_paths, "get_system_user_directory"), "get_system_user_directory missing from runtime"
    assert hasattr(folder_paths, "get_public_user_directory"), "get_public_user_directory missing from runtime"
    assert hasattr(folder_paths, "get_input_directory"), "get_input_directory missing from runtime"
    
    # Check variables
    assert hasattr(folder_paths, "extension_mimetypes_cache"), "extension_mimetypes_cache missing from runtime"

    # Minimal signature check (can call them with defaults if possible, but some might require setup)
    # get_input_directory has a default now
    # We might not be able to call it if it depends on execution context not being set up, 
    # but we can check if it is callable.
    assert callable(folder_paths.get_input_directory)
    assert callable(folder_paths.get_system_user_directory)
    assert callable(folder_paths.get_public_user_directory)
