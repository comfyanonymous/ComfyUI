import pytest
import time
from unittest.mock import patch
from comfy import folder_paths

class TestCacheHelper:
    def test_cache_helper_initialization(self):
        """Test that CacheHelper initializes with empty cache and inactive state."""
        cache = folder_paths.CacheHelper()
        assert not cache.active
        assert cache.cache == {}

    def test_cache_operations(self):
        """Test basic cache operations (get, set, clear)."""
        cache = folder_paths.CacheHelper()
        cache.active = True
        
        # Test setting and getting a value
        test_value = (["file1.txt"], {"file1.txt": 123.45}, 1.0)
        cache.set("test_key", test_value)
        assert cache.get("test_key") == test_value
        
        # Test getting non-existent key
        assert cache.get("non_existent") is None
        
        # Test clearing the cache
        cache.clear()
        assert cache.cache == {}

    def test_context_manager(self):
        """Test that the context manager properly handles activation state."""
        cache = folder_paths.CacheHelper()
        
        with cache:
            assert cache.active
            assert cache.get("test") is None  # Shouldn't raise
            
        assert not cache.active
        assert cache.cache == {}  # Should be cleared after context

    def test_cache_inactive(self):
        """Test that cache doesn't store when inactive."""
        cache = folder_paths.CacheHelper()
        cache.set("test", (["file.txt"], {}, 1.0))
        assert cache.get("test") is None

    @patch('time.time', return_value=100.0)
    def test_cache_helper_with_time(self, mock_time):
        """Test cache helper with mocked time."""
        cache = folder_paths.CacheHelper()
        cache.active = True
        
        test_value = (["file.txt"], {"file.txt": 100.0}, 100.0)
        cache.set("test", test_value)
        assert cache.get("test") == test_value
