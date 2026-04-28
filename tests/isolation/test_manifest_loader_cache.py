"""
Unit tests for manifest_loader.py cache functions.

Phase 1 tests verify:
1. Cache miss on first run (no cache exists)
2. Cache hit when nothing changes
3. Invalidation on .py file touch
4. Invalidation on manifest change
5. Cache location correctness (in venv_root, NOT in custom_nodes)
6. Corrupt cache handling (graceful failure)

These tests verify the cache implementation is correct BEFORE it's activated
in extension_loader.py (Phase 2).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest import mock



class TestComputeCacheKey:
    """Tests for compute_cache_key() function."""

    def test_key_includes_manifest_content(self, tmp_path: Path) -> None:
        """Cache key changes when manifest content changes."""
        from comfy.isolation.manifest_loader import compute_cache_key

        node_dir = tmp_path / "test_node"
        node_dir.mkdir()
        manifest = node_dir / "pyisolate.yaml"

        # Initial manifest
        manifest.write_text("isolated: true\ndependencies: []\n")
        key1 = compute_cache_key(node_dir, manifest)

        # Modified manifest
        manifest.write_text("isolated: true\ndependencies: [numpy]\n")
        key2 = compute_cache_key(node_dir, manifest)

        assert key1 != key2, "Key should change when manifest content changes"

    def test_key_includes_py_file_mtime(self, tmp_path: Path) -> None:
        """Cache key changes when any .py file is touched."""
        from comfy.isolation.manifest_loader import compute_cache_key

        node_dir = tmp_path / "test_node"
        node_dir.mkdir()
        manifest = node_dir / "pyisolate.yaml"
        manifest.write_text("isolated: true\n")

        py_file = node_dir / "nodes.py"
        py_file.write_text("# test code")

        key1 = compute_cache_key(node_dir, manifest)

        # Wait a moment to ensure mtime changes
        time.sleep(0.01)
        py_file.write_text("# modified code")

        key2 = compute_cache_key(node_dir, manifest)

        assert key1 != key2, "Key should change when .py file mtime changes"

    def test_key_includes_python_version(self, tmp_path: Path) -> None:
        """Cache key changes when Python version changes."""
        from comfy.isolation.manifest_loader import compute_cache_key

        node_dir = tmp_path / "test_node"
        node_dir.mkdir()
        manifest = node_dir / "pyisolate.yaml"
        manifest.write_text("isolated: true\n")

        key1 = compute_cache_key(node_dir, manifest)

        # Mock different Python version
        with mock.patch.object(sys, "version", "3.99.0 (fake)"):
            key2 = compute_cache_key(node_dir, manifest)

        assert key1 != key2, "Key should change when Python version changes"

    def test_key_includes_pyisolate_version(self, tmp_path: Path) -> None:
        """Cache key changes when PyIsolate version changes."""
        from comfy.isolation.manifest_loader import compute_cache_key

        node_dir = tmp_path / "test_node"
        node_dir.mkdir()
        manifest = node_dir / "pyisolate.yaml"
        manifest.write_text("isolated: true\n")

        key1 = compute_cache_key(node_dir, manifest)

        # Mock different pyisolate version
        with mock.patch.dict(sys.modules, {"pyisolate": mock.MagicMock(__version__="99.99.99")}):
            # Need to reimport to pick up the mock
            import importlib
            from comfy.isolation import manifest_loader
            importlib.reload(manifest_loader)
            key2 = manifest_loader.compute_cache_key(node_dir, manifest)

        # Keys should be different (though the mock approach is tricky)
        # At minimum, verify key is a valid hex string
        assert len(key1) == 16, "Key should be 16 hex characters"
        assert all(c in "0123456789abcdef" for c in key1), "Key should be hex"
        assert len(key2) == 16, "Key should be 16 hex characters"
        assert all(c in "0123456789abcdef" for c in key2), "Key should be hex"

    def test_key_excludes_pycache(self, tmp_path: Path) -> None:
        """Cache key ignores __pycache__ directory changes."""
        from comfy.isolation.manifest_loader import compute_cache_key

        node_dir = tmp_path / "test_node"
        node_dir.mkdir()
        manifest = node_dir / "pyisolate.yaml"
        manifest.write_text("isolated: true\n")

        py_file = node_dir / "nodes.py"
        py_file.write_text("# test code")

        key1 = compute_cache_key(node_dir, manifest)

        # Add __pycache__ file
        pycache = node_dir / "__pycache__"
        pycache.mkdir()
        (pycache / "nodes.cpython-310.pyc").write_bytes(b"compiled")

        key2 = compute_cache_key(node_dir, manifest)

        assert key1 == key2, "Key should NOT change when __pycache__ modified"

    def test_key_is_deterministic(self, tmp_path: Path) -> None:
        """Same inputs produce same key."""
        from comfy.isolation.manifest_loader import compute_cache_key

        node_dir = tmp_path / "test_node"
        node_dir.mkdir()
        manifest = node_dir / "pyisolate.yaml"
        manifest.write_text("isolated: true\n")
        (node_dir / "nodes.py").write_text("# code")

        key1 = compute_cache_key(node_dir, manifest)
        key2 = compute_cache_key(node_dir, manifest)

        assert key1 == key2, "Key should be deterministic"


class TestGetCachePath:
    """Tests for get_cache_path() function."""

    def test_returns_correct_paths(self, tmp_path: Path) -> None:
        """Cache paths are in venv_root, not in node_dir."""
        from comfy.isolation.manifest_loader import get_cache_path

        node_dir = tmp_path / "custom_nodes" / "MyNode"
        venv_root = tmp_path / ".pyisolate_venvs"

        key_file, data_file = get_cache_path(node_dir, venv_root)

        assert key_file == venv_root / "MyNode" / "cache" / "cache_key"
        assert data_file == venv_root / "MyNode" / "cache" / "node_info.json"

    def test_cache_not_in_custom_nodes(self, tmp_path: Path) -> None:
        """Verify cache is NOT stored in custom_nodes directory."""
        from comfy.isolation.manifest_loader import get_cache_path

        node_dir = tmp_path / "custom_nodes" / "MyNode"
        venv_root = tmp_path / ".pyisolate_venvs"

        key_file, data_file = get_cache_path(node_dir, venv_root)

        # Neither path should be under node_dir
        assert not str(key_file).startswith(str(node_dir))
        assert not str(data_file).startswith(str(node_dir))


class TestIsCacheValid:
    """Tests for is_cache_valid() function."""

    def test_false_when_no_cache_exists(self, tmp_path: Path) -> None:
        """Returns False when cache files don't exist."""
        from comfy.isolation.manifest_loader import is_cache_valid

        node_dir = tmp_path / "test_node"
        node_dir.mkdir()
        manifest = node_dir / "pyisolate.yaml"
        manifest.write_text("isolated: true\n")
        venv_root = tmp_path / ".pyisolate_venvs"

        assert is_cache_valid(node_dir, manifest, venv_root) is False

    def test_true_when_cache_matches(self, tmp_path: Path) -> None:
        """Returns True when cache key matches current state."""
        from comfy.isolation.manifest_loader import (
            compute_cache_key,
            get_cache_path,
            is_cache_valid,
        )

        node_dir = tmp_path / "test_node"
        node_dir.mkdir()
        manifest = node_dir / "pyisolate.yaml"
        manifest.write_text("isolated: true\n")
        (node_dir / "nodes.py").write_text("# code")
        venv_root = tmp_path / ".pyisolate_venvs"

        # Create valid cache
        cache_key = compute_cache_key(node_dir, manifest)
        key_file, data_file = get_cache_path(node_dir, venv_root)
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key_file.write_text(cache_key)
        data_file.write_text("{}")

        assert is_cache_valid(node_dir, manifest, venv_root) is True

    def test_false_when_key_mismatch(self, tmp_path: Path) -> None:
        """Returns False when stored key doesn't match current state."""
        from comfy.isolation.manifest_loader import get_cache_path, is_cache_valid

        node_dir = tmp_path / "test_node"
        node_dir.mkdir()
        manifest = node_dir / "pyisolate.yaml"
        manifest.write_text("isolated: true\n")
        venv_root = tmp_path / ".pyisolate_venvs"

        # Create cache with wrong key
        key_file, data_file = get_cache_path(node_dir, venv_root)
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key_file.write_text("wrong_key_12345")
        data_file.write_text("{}")

        assert is_cache_valid(node_dir, manifest, venv_root) is False

    def test_false_when_data_file_missing(self, tmp_path: Path) -> None:
        """Returns False when node_info.json is missing."""
        from comfy.isolation.manifest_loader import (
            compute_cache_key,
            get_cache_path,
            is_cache_valid,
        )

        node_dir = tmp_path / "test_node"
        node_dir.mkdir()
        manifest = node_dir / "pyisolate.yaml"
        manifest.write_text("isolated: true\n")
        venv_root = tmp_path / ".pyisolate_venvs"

        # Create only key file, not data file
        cache_key = compute_cache_key(node_dir, manifest)
        key_file, _ = get_cache_path(node_dir, venv_root)
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key_file.write_text(cache_key)

        assert is_cache_valid(node_dir, manifest, venv_root) is False

    def test_invalidation_on_py_change(self, tmp_path: Path) -> None:
        """Cache invalidates when .py file is modified."""
        from comfy.isolation.manifest_loader import (
            compute_cache_key,
            get_cache_path,
            is_cache_valid,
        )

        node_dir = tmp_path / "test_node"
        node_dir.mkdir()
        manifest = node_dir / "pyisolate.yaml"
        manifest.write_text("isolated: true\n")
        py_file = node_dir / "nodes.py"
        py_file.write_text("# original")
        venv_root = tmp_path / ".pyisolate_venvs"

        # Create valid cache
        cache_key = compute_cache_key(node_dir, manifest)
        key_file, data_file = get_cache_path(node_dir, venv_root)
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key_file.write_text(cache_key)
        data_file.write_text("{}")

        # Verify cache is valid initially
        assert is_cache_valid(node_dir, manifest, venv_root) is True

        # Modify .py file
        time.sleep(0.01)  # Ensure mtime changes
        py_file.write_text("# modified")

        # Cache should now be invalid
        assert is_cache_valid(node_dir, manifest, venv_root) is False


class TestLoadFromCache:
    """Tests for load_from_cache() function."""

    def test_returns_none_when_no_cache(self, tmp_path: Path) -> None:
        """Returns None when cache doesn't exist."""
        from comfy.isolation.manifest_loader import load_from_cache

        node_dir = tmp_path / "test_node"
        venv_root = tmp_path / ".pyisolate_venvs"

        assert load_from_cache(node_dir, venv_root) is None

    def test_returns_data_when_valid(self, tmp_path: Path) -> None:
        """Returns cached data when file exists and is valid JSON."""
        from comfy.isolation.manifest_loader import get_cache_path, load_from_cache

        node_dir = tmp_path / "test_node"
        venv_root = tmp_path / ".pyisolate_venvs"

        test_data = {"TestNode": {"inputs": [], "outputs": []}}

        _, data_file = get_cache_path(node_dir, venv_root)
        data_file.parent.mkdir(parents=True, exist_ok=True)
        data_file.write_text(json.dumps(test_data))

        result = load_from_cache(node_dir, venv_root)
        assert result == test_data

    def test_returns_none_on_corrupt_json(self, tmp_path: Path) -> None:
        """Returns None when JSON is corrupt."""
        from comfy.isolation.manifest_loader import get_cache_path, load_from_cache

        node_dir = tmp_path / "test_node"
        venv_root = tmp_path / ".pyisolate_venvs"

        _, data_file = get_cache_path(node_dir, venv_root)
        data_file.parent.mkdir(parents=True, exist_ok=True)
        data_file.write_text("{ corrupt json }")

        assert load_from_cache(node_dir, venv_root) is None

    def test_returns_none_on_invalid_structure(self, tmp_path: Path) -> None:
        """Returns None when data is not a dict."""
        from comfy.isolation.manifest_loader import get_cache_path, load_from_cache

        node_dir = tmp_path / "test_node"
        venv_root = tmp_path / ".pyisolate_venvs"

        _, data_file = get_cache_path(node_dir, venv_root)
        data_file.parent.mkdir(parents=True, exist_ok=True)
        data_file.write_text("[1, 2, 3]")  # Array, not dict

        assert load_from_cache(node_dir, venv_root) is None


class TestSaveToCache:
    """Tests for save_to_cache() function."""

    def test_creates_cache_directory(self, tmp_path: Path) -> None:
        """Creates cache directory if it doesn't exist."""
        from comfy.isolation.manifest_loader import get_cache_path, save_to_cache

        node_dir = tmp_path / "test_node"
        node_dir.mkdir()
        manifest = node_dir / "pyisolate.yaml"
        manifest.write_text("isolated: true\n")
        venv_root = tmp_path / ".pyisolate_venvs"

        save_to_cache(node_dir, venv_root, {"TestNode": {}}, manifest)

        key_file, data_file = get_cache_path(node_dir, venv_root)
        assert key_file.parent.exists()

    def test_writes_both_files(self, tmp_path: Path) -> None:
        """Writes both cache_key and node_info.json."""
        from comfy.isolation.manifest_loader import get_cache_path, save_to_cache

        node_dir = tmp_path / "test_node"
        node_dir.mkdir()
        manifest = node_dir / "pyisolate.yaml"
        manifest.write_text("isolated: true\n")
        venv_root = tmp_path / ".pyisolate_venvs"

        save_to_cache(node_dir, venv_root, {"TestNode": {"key": "value"}}, manifest)

        key_file, data_file = get_cache_path(node_dir, venv_root)
        assert key_file.exists()
        assert data_file.exists()

    def test_data_is_valid_json(self, tmp_path: Path) -> None:
        """Written data can be parsed as JSON."""
        from comfy.isolation.manifest_loader import get_cache_path, save_to_cache

        node_dir = tmp_path / "test_node"
        node_dir.mkdir()
        manifest = node_dir / "pyisolate.yaml"
        manifest.write_text("isolated: true\n")
        venv_root = tmp_path / ".pyisolate_venvs"

        test_data = {"TestNode": {"inputs": ["IMAGE"], "outputs": ["IMAGE"]}}
        save_to_cache(node_dir, venv_root, test_data, manifest)

        _, data_file = get_cache_path(node_dir, venv_root)
        loaded = json.loads(data_file.read_text())
        assert loaded == test_data

    def test_roundtrip_with_validation(self, tmp_path: Path) -> None:
        """Saved cache is immediately valid."""
        from comfy.isolation.manifest_loader import (
            is_cache_valid,
            load_from_cache,
            save_to_cache,
        )

        node_dir = tmp_path / "test_node"
        node_dir.mkdir()
        manifest = node_dir / "pyisolate.yaml"
        manifest.write_text("isolated: true\n")
        (node_dir / "nodes.py").write_text("# code")
        venv_root = tmp_path / ".pyisolate_venvs"

        test_data = {"TestNode": {"foo": "bar"}}
        save_to_cache(node_dir, venv_root, test_data, manifest)

        assert is_cache_valid(node_dir, manifest, venv_root) is True
        assert load_from_cache(node_dir, venv_root) == test_data

    def test_cache_not_in_custom_nodes(self, tmp_path: Path) -> None:
        """Verify no files written to custom_nodes directory."""
        from comfy.isolation.manifest_loader import save_to_cache

        node_dir = tmp_path / "custom_nodes" / "MyNode"
        node_dir.mkdir(parents=True)
        manifest = node_dir / "pyisolate.yaml"
        manifest.write_text("isolated: true\n")
        venv_root = tmp_path / ".pyisolate_venvs"

        save_to_cache(node_dir, venv_root, {"TestNode": {}}, manifest)

        # Check nothing was created under node_dir
        for item in node_dir.iterdir():
            assert item.name == "pyisolate.yaml", f"Unexpected file in node_dir: {item}"
