import pytest
from unittest.mock import patch, MagicMock
from comfy import folder_paths

class TestContentTypeFiltering:
    def test_filter_files_content_types(self):
        """Test filtering files by content types."""
        test_files = [
            "image.jpg",
            "document.pdf",
            "model.fbx",
            "video.mp4",
            "audio.mp3",
            "unknown.xyz"
        ]
        
        # Test image filtering
        result = folder_paths.filter_files_content_types(test_files, ["image"])
        assert result == ["image.jpg"]
        
        # Test model filtering
        result = folder_paths.filter_files_content_types(test_files, ["model"])
        assert result == ["model.fbx"]
        
        # Test multiple content types
        result = folder_paths.filter_files_content_types(test_files, ["image", "model"])
        assert set(result) == {"image.jpg", "model.fbx"}
        
        # Test with unknown content type
        result = folder_paths.filter_files_content_types(test_files, ["unknown"])
        assert result == []
        
        # Test with empty file list
        assert folder_paths.filter_files_content_types([], ["image"]) == []
        
        # Test with empty content types
        assert folder_paths.filter_files_content_types(test_files, []) == []
    
    @patch('mimetypes.guess_type')
    def test_filter_files_content_types_mock_mime(self, mock_guess):
        """Test content type filtering with mocked mime types."""
        # Setup mock to return different mime types
        def mock_guess_type(filename):
            if filename == "test.jpg":
                return ("image/jpeg", None)
            elif filename == "test.mp4":
                return ("video/mp4", None)
            return (None, None)
            
        mock_guess.side_effect = mock_guess_type
        
        files = ["test.jpg", "test.mp4", "test.unknown"]
        
        # Test image filtering
        result = folder_paths.filter_files_content_types(files, ["image"])
        assert result == ["test.jpg"]
        
        # Test video filtering
        result = folder_paths.filter_files_content_types(files, ["video"])
        assert result == ["test.mp4"]
        
        # Test unknown type filtering
        result = folder_paths.filter_files_content_types(files, ["audio"])
        assert result == []
    
    def test_extension_mimetypes_cache(self):
        """Test that extension to mimetype cache works correctly."""
        # Test with known extensions in cache
        assert folder_paths.extension_mimetypes_cache["webp"] == "image"
        assert folder_paths.extension_mimetypes_cache["fbx"] == "model"
        
        # Test with unknown extension (should not be in cache)
        assert "xyz" not in folder_paths.extension_mimetypes_cache
        
    @patch('mimetypes.guess_type')
    def test_custom_extension_handling(self, mock_guess):
        """Test handling of custom file extensions."""
        # Setup mock to return None for unknown types
        mock_guess.return_value = (None, None)
        
        # Add a custom extension to the cache
        folder_paths.extension_mimetypes_cache["custom"] = "custom_type"
        
        files = ["file.custom"]
        result = folder_paths.filter_files_content_types(files, ["custom_type"])
        assert result == ["file.custom"]
        
        # Test with different case
        result = folder_paths.filter_files_content_types(["FILE.CUSTOM"], ["custom_type"])
        assert result == ["FILE.CUSTOM"]
