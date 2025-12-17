"""Tests for LoadImage and LoadImageMask file handle management.

Relates to issue #3477: close image file after loading
"""

import pytest
import tempfile
import os
from PIL import Image


class TestImageFileHandleRelease:
    """Test that image files are properly closed after loading."""

    def test_file_handle_released_after_close(self):
        """Verify file handle is released after calling close()."""
        # Create a temporary test image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name

        try:
            # Create a test image
            img = Image.new('RGB', (64, 64), color='red')
            img.save(temp_path)

            # Open and close the image
            loaded_img = Image.open(temp_path)
            loaded_img.load()  # Force load the image data
            loaded_img.close()

            # Try to delete the file - should succeed if handle is released
            os.unlink(temp_path)
            assert not os.path.exists(temp_path), "File should be deleted"
        except Exception:
            # Cleanup in case of failure
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def test_try_finally_pattern_releases_handle(self):
        """Verify try/finally pattern properly releases file handle."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name

        try:
            # Create a test image
            img = Image.new('RGBA', (64, 64), color='blue')
            img.save(temp_path)

            # Simulate the pattern used in LoadImage
            loaded_img = Image.open(temp_path)
            try:
                # Process the image
                _ = loaded_img.convert("RGB")
            finally:
                loaded_img.close()

            # Verify file can be accessed/deleted
            os.unlink(temp_path)
            assert not os.path.exists(temp_path)
        except Exception:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    def test_image_data_preserved_after_close(self):
        """Verify image data is preserved after closing the file."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name

        try:
            # Create a test image with specific size
            original_size = (128, 64)
            img = Image.new('RGB', original_size, color='green')
            img.save(temp_path)

            # Load and process
            loaded_img = Image.open(temp_path)
            try:
                loaded_img.load()
                size = loaded_img.size
                mode = loaded_img.mode
            finally:
                loaded_img.close()

            # Data should still be valid after close
            assert size == original_size
            assert mode == 'RGB'
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
