import os
import pytest
from unittest.mock import patch, MagicMock
from comfy import folder_paths

def test_map_legacy():
    """Test the legacy path mapping function."""
    assert folder_paths.map_legacy("unet") == "diffusion_models"
    assert folder_paths.map_legacy("clip") == "text_encoders"
    assert folder_paths.map_legacy("unknown") == "unknown"

def test_annotated_filepath():
    """Test parsing of annotated file paths."""
    # Test with no annotation
    assert folder_paths.annotated_filepath("test.txt") == ("test.txt", None)
    
    # Test with annotation
    assert folder_paths.annotated_filepath("test.txt [output]") == \
           ("test.txt", folder_paths.get_output_directory())
    
    # Test with annotation and spaces
    assert folder_paths.annotated_filepath("test file.txt [output]") == \
           ("test file.txt", folder_paths.get_output_directory())
    
    # Test with invalid annotation
    assert folder_paths.annotated_filepath("test.txt [invalid]") == \
           ("test.txt [invalid]", None)

def test_get_annotated_filepath():
    """Test getting absolute path from annotated filename."""
    with patch('os.path.exists', return_value=True):
        # Test with no annotation
        result = folder_paths.get_annotated_filepath("test.txt", "/default/dir")
        assert result == os.path.join("/default/dir", "test.txt")
        
        # Test with annotation
        result = folder_paths.get_annotated_filepath("test.txt [output]")
        assert result == os.path.join(folder_paths.get_output_directory(), "test.txt")

def test_exists_annotated_filepath():
    """Test checking if an annotated file path exists."""
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        assert folder_paths.exists_annotated_filepath("test.txt [output]")
        mock_exists.assert_called_once()
        
        mock_exists.return_value = False
        assert not folder_paths.exists_annotated_filepath("nonexistent.txt [output]")

def test_filter_files_extensions():
    """Test filtering files by extensions."""
    files = ["test.txt", "image.jpg", "document.pdf", "script.py"]
    
    # Test with single extension
    result = folder_paths.filter_files_extensions(files, [".txt"])
    assert result == ["test.txt"]
    
    # Test with multiple extensions
    result = folder_paths.filter_files_extensions(files, [".jpg", ".pdf"])
    assert set(result) == {"image.jpg", "document.pdf"}
    
    # Test with no matches
    result = folder_paths.filter_files_extensions(files, [".png"])
    assert result == []
    
    # Test with empty file list
    assert folder_paths.filter_files_extensions([], [".txt"]) == []
    
    # Test with empty extensions list
    assert folder_paths.filter_files_extensions(files, []) == []
