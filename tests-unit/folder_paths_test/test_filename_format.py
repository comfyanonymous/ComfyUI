"""Tests for output filename format.

Relates to issue #1389: Trailing underscore in output file names
"""

import os
import tempfile
import pytest

import folder_paths


class TestFilenameFormat:
    """Test output filename format without trailing underscore."""

    def test_new_filename_format_no_trailing_underscore(self):
        """New files should not have trailing underscore before extension."""
        # Expected format: "prefix_00001.png" not "prefix_00001_.png"
        filename = "ComfyUI"
        counter = 1
        new_format = f"{filename}_{counter:05}.png"

        assert new_format == "ComfyUI_00001.png"
        assert not new_format.endswith("_.png"), "Filename should not have trailing underscore"

    def test_get_save_image_path_backward_compatible(self):
        """get_save_image_path should work with both old and new format files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with old format (trailing underscore)
            old_format_files = [
                "TestPrefix_00001_.png",
                "TestPrefix_00002_.png",
            ]
            for f in old_format_files:
                open(os.path.join(tmpdir, f), 'w').close()

            # get_save_image_path should recognize old format and return next counter
            full_path, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
                "TestPrefix", tmpdir
            )

            # Counter should be 3 (after 00001 and 00002)
            assert counter == 3

    def test_get_save_image_path_new_format(self):
        """get_save_image_path should work with new format files (no trailing underscore)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with new format (no trailing underscore)
            new_format_files = [
                "NewPrefix_00001.png",
                "NewPrefix_00002.png",
                "NewPrefix_00003.png",
            ]
            for f in new_format_files:
                open(os.path.join(tmpdir, f), 'w').close()

            # get_save_image_path should recognize new format
            full_path, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
                "NewPrefix", tmpdir
            )

            # Counter should be 4 (after 00001, 00002, 00003)
            assert counter == 4

    def test_get_save_image_path_mixed_formats(self):
        """get_save_image_path should handle mixed old and new format files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mix of old and new format files
            files = [
                "MixedPrefix_00001_.png",  # old format
                "MixedPrefix_00002.png",   # new format
                "MixedPrefix_00003_.png",  # old format
            ]
            for f in files:
                open(os.path.join(tmpdir, f), 'w').close()

            full_path, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
                "MixedPrefix", tmpdir
            )

            # Counter should be 4 (recognizing both formats)
            assert counter == 4

    def test_get_save_image_path_empty_directory(self):
        """get_save_image_path should return counter 1 for empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            full_path, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
                "EmptyDir", tmpdir
            )

            assert counter == 1
