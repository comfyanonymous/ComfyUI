import os
import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock, call
from comfy import folder_paths

class TestDirectoryOperations:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.original_paths = folder_paths.folder_names_and_paths.copy()
        yield
        # Cleanup
        shutil.rmtree(self.temp_dir)
        folder_paths.folder_names_and_paths = self.original_paths

    def test_add_model_folder_path(self):
        """Test adding a new model folder path."""
        test_path = os.path.join(self.temp_dir, "test_models")
        os.makedirs(test_path, exist_ok=True)
        
        # Add a new folder path
        folder_paths.add_model_folder_path("test_models", test_path)
        
        # Verify it was added correctly
        paths, extensions = folder_paths.folder_names_and_paths["test_models"]
        assert test_path in paths
        assert extensions == set()
        
        # Verify it's the last item by default
        assert paths[-1] == test_path

    def test_add_model_folder_path_as_default(self):
        """Test adding a folder path as default (should be first in list)."""
        test_path = os.path.join(self.temp_dir, "test_models")
        os.makedirs(test_path, exist_ok=True)
        
        # Add as default
        folder_paths.add_model_folder_path("test_models", test_path, is_default=True)
        
        # Verify it's first in the list
        paths, _ = folder_paths.folder_names_and_paths["test_models"]
        assert paths[0] == test_path

    def test_get_folder_paths(self):
        """Test retrieving folder paths."""
        test_path = os.path.join(self.temp_dir, "test_models")
        os.makedirs(test_path, exist_ok=True)
        
        # Add a test path
        folder_paths.add_model_folder_path("test_models", test_path)
        
        # Test getting the paths
        paths = folder_paths.get_folder_paths("test_models")
        assert paths == [test_path]
        
        # Test getting non-existent folder
        assert folder_paths.get_folder_paths("non_existent") is None

    def test_get_full_path(self):
        """Test getting full path for a file in a folder."""
        # Create a test file
        test_dir = os.path.join(self.temp_dir, "test_models")
        os.makedirs(test_dir, exist_ok=True)
        test_file = os.path.join(test_dir, "model.ckpt")
        with open(test_file, 'w') as f:
            f.write("test")
        
        # Add the test directory
        folder_paths.add_model_folder_path("test_models", test_dir)
        
        # Test getting full path
        result = folder_paths.get_full_path("test_models", "model.ckpt")
        assert result == test_file
        
        # Test with non-existent file
        result = folder_paths.get_full_path("test_models", "nonexistent.ckpt")
        assert result is None
        
        # Test with non-existent folder
        result = folder_paths.get_full_path("non_existent", "model.ckpt")
        assert result is None

    @patch('os.walk')
    def test_recursive_search(self, mock_walk):
        """Test recursive file search."""
        # Mock os.walk to return test directory structure
        mock_walk.return_value = [
            ('/test', ['subdir'], ['file1.txt']),
            ('/test/subdir', [], ['file2.txt'])
        ]
        
        # Test recursive search
        result = folder_paths.recursive_search("/test")
        assert set(result) == {
            os.path.join('/test', 'file1.txt'),
            os.path.join('/test/subdir', 'file2.txt')
        }
        
        # Test with excluded directories
        result = folder_paths.recursive_search("/test", excluded_dir_names=["subdir"])
        assert result == [os.path.join('/test', 'file1.txt')]
