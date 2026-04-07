"""Unit tests for FolderPathsProxy."""

import pytest
from pathlib import Path

from comfy.isolation.proxies.folder_paths_proxy import FolderPathsProxy
from tests.isolation.singleton_boundary_helpers import capture_sealed_singleton_imports


class TestFolderPathsProxy:
    """Test FolderPathsProxy methods."""

    @pytest.fixture
    def proxy(self):
        """Create a FolderPathsProxy instance for testing."""
        return FolderPathsProxy()

    def test_get_temp_directory_returns_string(self, proxy):
        """Verify get_temp_directory returns a non-empty string."""
        result = proxy.get_temp_directory()
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert len(result) > 0, "Temp directory path is empty"

    def test_get_temp_directory_returns_absolute_path(self, proxy):
        """Verify get_temp_directory returns an absolute path."""
        result = proxy.get_temp_directory()
        path = Path(result)
        assert path.is_absolute(), f"Path is not absolute: {result}"

    def test_get_input_directory_returns_string(self, proxy):
        """Verify get_input_directory returns a non-empty string."""
        result = proxy.get_input_directory()
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert len(result) > 0, "Input directory path is empty"

    def test_get_input_directory_returns_absolute_path(self, proxy):
        """Verify get_input_directory returns an absolute path."""
        result = proxy.get_input_directory()
        path = Path(result)
        assert path.is_absolute(), f"Path is not absolute: {result}"

    def test_get_annotated_filepath_plain_name(self, proxy):
        """Verify get_annotated_filepath works with plain filename."""
        result = proxy.get_annotated_filepath("test.png")
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert "test.png" in result, f"Filename not in result: {result}"

    def test_get_annotated_filepath_with_output_annotation(self, proxy):
        """Verify get_annotated_filepath handles [output] annotation."""
        result = proxy.get_annotated_filepath("test.png[output]")
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert "test.pn" in result, f"Filename base not in result: {result}"
        # Should resolve to output directory
        assert "output" in result.lower() or Path(result).parent.name == "output"

    def test_get_annotated_filepath_with_input_annotation(self, proxy):
        """Verify get_annotated_filepath handles [input] annotation."""
        result = proxy.get_annotated_filepath("test.png[input]")
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert "test.pn" in result, f"Filename base not in result: {result}"

    def test_get_annotated_filepath_with_temp_annotation(self, proxy):
        """Verify get_annotated_filepath handles [temp] annotation."""
        result = proxy.get_annotated_filepath("test.png[temp]")
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert "test.pn" in result, f"Filename base not in result: {result}"

    def test_exists_annotated_filepath_returns_bool(self, proxy):
        """Verify exists_annotated_filepath returns a boolean."""
        result = proxy.exists_annotated_filepath("nonexistent.png")
        assert isinstance(result, bool), f"Expected bool, got {type(result)}"

    def test_exists_annotated_filepath_nonexistent_file(self, proxy):
        """Verify exists_annotated_filepath returns False for nonexistent file."""
        result = proxy.exists_annotated_filepath("definitely_does_not_exist_12345.png")
        assert result is False, "Expected False for nonexistent file"

    def test_exists_annotated_filepath_with_annotation(self, proxy):
        """Verify exists_annotated_filepath works with annotation suffix."""
        # Even for nonexistent files, should return bool without error
        result = proxy.exists_annotated_filepath("test.png[output]")
        assert isinstance(result, bool), f"Expected bool, got {type(result)}"

    def test_models_dir_property_returns_string(self, proxy):
        """Verify models_dir property returns valid path string."""
        result = proxy.models_dir
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert len(result) > 0, "Models directory path is empty"

    def test_models_dir_is_absolute_path(self, proxy):
        """Verify models_dir returns an absolute path."""
        result = proxy.models_dir
        path = Path(result)
        assert path.is_absolute(), f"Path is not absolute: {result}"

    def test_add_model_folder_path_runs_without_error(self, proxy):
        """Verify add_model_folder_path executes without raising."""
        test_path = "/tmp/test_models_florence2"
        # Should not raise
        proxy.add_model_folder_path("TEST_FLORENCE2", test_path)

    def test_get_folder_paths_returns_list(self, proxy):
        """Verify get_folder_paths returns a list."""
        # Use known folder type that should exist
        result = proxy.get_folder_paths("checkpoints")
        assert isinstance(result, list), f"Expected list, got {type(result)}"

    def test_get_folder_paths_checkpoints_not_empty(self, proxy):
        """Verify checkpoints folder paths list is not empty."""
        result = proxy.get_folder_paths("checkpoints")
        # Should have at least one checkpoint path registered
        assert len(result) > 0, "Checkpoints folder paths is empty"

    def test_sealed_child_safe_uses_rpc_without_importing_folder_paths(self, monkeypatch):
        monkeypatch.setenv("PYISOLATE_CHILD", "1")
        monkeypatch.setenv("PYISOLATE_IMPORT_TORCH", "0")

        payload = capture_sealed_singleton_imports()

        assert payload["temp_dir"] == "/sandbox/temp"
        assert payload["models_dir"] == "/sandbox/models"
        assert "folder_paths" not in payload["modules"]
