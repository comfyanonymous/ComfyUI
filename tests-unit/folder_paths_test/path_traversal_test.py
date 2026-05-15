"""
Tests for path traversal fix in get_annotated_filepath().
Verifies the security fix blocks traversal attempts while allowing legitimate paths.
"""
import pytest
import os
import tempfile
from folder_paths import get_annotated_filepath, set_input_directory


# ─── BLOCKED: Path traversal attempts ─────────────────────────────────

@pytest.mark.parametrize("malicious_path", [
    "../../../../etc/passwd",
    "subdir/../../../etc/shadow",
    "/absolute/path/to/file.txt",
    "..\\..\\..\\windows\\system32\\config",
    "subdir\\..\\..\\..\\etc\\passwd",  # Windows backslash bypass
    "foo/../../../../etc/hostname",
    None,
    "",
    12345,
])
def test_blocks_path_traversal(malicious_path):
    """All these paths should be rejected with ValueError."""
    with tempfile.TemporaryDirectory() as temp_dir:
        set_input_directory(temp_dir)
        with pytest.raises(ValueError, match="Path traversal|Invalid file path"):
            get_annotated_filepath(malicious_path)


# ─── ALLOWED: Legitimate paths ───────────────────────────────────────

@pytest.mark.parametrize("legitimate_path", [
    "normal_image.png",
    "subdir/image.jpg",
    "my_workflow.json",
    "path/to/file.safetensors",
    "image (1).png",
    "deeply/nested/path/with/file.txt",
    "data-2026-05-15.csv",
    "a" * 255,
])
def test_allows_legitimate_paths(legitimate_path):
    """All these legitimate paths should be allowed through."""
    with tempfile.TemporaryDirectory() as temp_dir:
        set_input_directory(temp_dir)
        try:
            result = get_annotated_filepath(legitimate_path)
            assert result.startswith(temp_dir)
            assert result.endswith(legitimate_path)
        except ValueError:
            pytest.fail(f"Legitimate path '{legitimate_path}' was blocked!")


# ─── EDGE CASES ─────────────────────────────────────────────────────

def test_respects_annotation_overrides():
    """Paths with [output], [input], [temp] annotations should still work."""
    with tempfile.TemporaryDirectory() as temp_dir:
        set_input_directory(temp_dir)
        try:
            result = get_annotated_filepath("test.png[output]")
            assert "[output]" not in result
        except ValueError as e:
            pytest.fail(f"Annotated path was blocked: {e}")


def test_nested_legitimate_path_with_dots():
    """Path components with legitimate dots should pass."""
    with tempfile.TemporaryDirectory() as temp_dir:
        set_input_directory(temp_dir)
        try:
            result = get_annotated_filepath("images/photo.2026.05.png")
            assert result.endswith("images/photo.2026.05.png")
        except ValueError:
            pytest.fail("Path with dots in filename was blocked!")
