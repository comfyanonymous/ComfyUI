import os
import pytest
import time
from unittest.mock import patch, MagicMock, call
from comfy import folder_paths

class TestFileCaching:
    @pytest.fixture(autouse=True)
    def setup_method(self, tmp_path):
        self.temp_dir = tmp_path
        self.test_dir = self.temp_dir / "test_models"
        self.test_dir.mkdir()
        
        # Create some test files
        (self.test_dir / "model1.ckpt").write_text("test1")
        (self.test_dir / "model2.ckpt").write_text("test2")
        
        # Save original state
        self.original_cache = folder_paths.filename_list_cache.copy()
        self.original_paths = folder_paths.folder_names_and_paths.copy()
        
        # Add test directory to paths
        folder_paths.add_model_folder_path("test_models", str(self.test_dir))
        
        yield
        
        # Restore original state
        folder_paths.filename_list_cache = self.original_cache
        folder_paths.folder_names_and_paths = self.original_paths
    
    def test_get_filename_list_caching(self):
        """Test that file lists are properly cached."""
        # First call should populate cache
        result1 = folder_paths.get_filename_list("test_models")
        assert set(result1) == {"model1.ckpt", "model2.ckpt"}
        
        # Verify cache was populated
        cache_key = str(self.test_dir)
        assert cache_key in folder_paths.filename_list_cache
        
        # Second call should use cache
        with patch('os.path.getmtime') as mock_mtime:
            mock_mtime.return_value = 1000
            result2 = folder_paths.get_filename_list("test_models")
            assert result2 == result1
            # Verify getmtime wasn't called (using cache)
            mock_mtime.assert_not_called()
    
    @patch('os.path.getmtime')
    def test_cache_invalidation(self, mock_mtime):
        """Test that cache is invalidated when files change."""
        # Initial call to populate cache
        mock_mtime.return_value = 1000
        folder_paths.get_filename_list("test_models")
        
        # Change modification time to trigger cache invalidation
        mock_mtime.return_value = 2000
        
        # This should trigger a cache refresh
        result = folder_paths.get_filename_list("test_models")
        assert set(result) == {"model1.ckpt", "model2.ckpt"}
        
        # Verify getmtime was called for each file
        assert mock_mtime.call_count >= 2
    
    def test_cached_filename_list_helper(self):
        """Test the cached filename list helper function."""
        # Test with empty cache
        with patch('os.path.getmtime') as mock_mtime:
            mock_mtime.return_value = 1000
            result = folder_paths.cached_filename_list_("test_models")
            assert set(result[0]) == {"model1.ckpt", "model2.ckpt"}
            assert len(result[1]) == 2  # Should have mtimes for both files
            assert result[2] == 1000    # Should have the current time
        
        # Test with valid cache
        with patch('os.path.getmtime') as mock_mtime:
            mock_mtime.return_value = 1000
            # Call again, should use cache
            result = folder_paths.cached_filename_list_("test_models")
            mock_mtime.assert_not_called()  # Shouldn't check mtimes when using cache
    
    def test_get_filename_list_nonexistent_dir(self):
        """Test behavior with non-existent directory."""
        # Add a non-existent directory to the paths
        non_existent = self.temp_dir / "nonexistent"
        folder_paths.add_model_folder_path("test_models", str(non_existent))
        
        # Should not raise and should return empty list
        result = folder_paths.get_filename_list("test_models")
        assert result == []
