# pylint: disable=import-outside-toplevel
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import folder_paths

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

LOG_PREFIX = "]["
logger = logging.getLogger(__name__)

CACHE_SUBDIR = "cache"
CACHE_KEY_FILE = "cache_key"
CACHE_DATA_FILE = "node_info.json"
CACHE_KEY_LENGTH = 16
_NESTED_SCAN_ROOT = "packages"
_IGNORED_MANIFEST_DIRS = {".git", ".venv", "__pycache__"}


def _read_manifest(manifest_path: Path) -> dict[str, Any] | None:
    try:
        with manifest_path.open("rb") as f:
            data = tomllib.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def _is_isolation_manifest(data: dict[str, Any]) -> bool:
    return (
        "tool" in data
        and "comfy" in data["tool"]
        and "isolation" in data["tool"]["comfy"]
    )


def _discover_nested_manifests(entry: Path) -> List[Tuple[Path, Path]]:
    packages_root = entry / _NESTED_SCAN_ROOT
    if not packages_root.exists() or not packages_root.is_dir():
        return []

    nested: List[Tuple[Path, Path]] = []
    for manifest in sorted(packages_root.rglob("pyproject.toml")):
        node_dir = manifest.parent
        if any(part in _IGNORED_MANIFEST_DIRS for part in node_dir.parts):
            continue

        data = _read_manifest(manifest)
        if not data or not _is_isolation_manifest(data):
            continue

        isolation = data["tool"]["comfy"]["isolation"]
        if isolation.get("standalone") is True:
            nested.append((node_dir, manifest))

    return nested


def find_manifest_directories() -> List[Tuple[Path, Path]]:
    """Find custom node directories containing a valid pyproject.toml with [tool.comfy.isolation]."""
    manifest_dirs: List[Tuple[Path, Path]] = []

    # Standard custom_nodes paths
    for base_path in folder_paths.get_folder_paths("custom_nodes"):
        base = Path(base_path)
        if not base.exists() or not base.is_dir():
            continue

        for entry in base.iterdir():
            if not entry.is_dir():
                continue

            # Look for pyproject.toml
            manifest = entry / "pyproject.toml"
            if not manifest.exists():
                continue

            data = _read_manifest(manifest)
            if not data or not _is_isolation_manifest(data):
                continue

            manifest_dirs.append((entry, manifest))
            manifest_dirs.extend(_discover_nested_manifests(entry))

    return manifest_dirs


def compute_cache_key(node_dir: Path, manifest_path: Path) -> str:
    """Hash manifest + .py mtimes + Python version + PyIsolate version."""
    hasher = hashlib.sha256()

    try:
        # Hashing the manifest content ensures config changes invalidate cache
        hasher.update(manifest_path.read_bytes())
    except OSError:
        hasher.update(b"__manifest_read_error__")

    try:
        py_files = sorted(node_dir.rglob("*.py"))
        for py_file in py_files:
            rel_path = py_file.relative_to(node_dir)
            if "__pycache__" in str(rel_path) or ".venv" in str(rel_path):
                continue
            hasher.update(str(rel_path).encode("utf-8"))
            try:
                hasher.update(str(py_file.stat().st_mtime).encode("utf-8"))
            except OSError:
                hasher.update(b"__file_stat_error__")
    except OSError:
        hasher.update(b"__dir_scan_error__")

    hasher.update(sys.version.encode("utf-8"))

    try:
        import pyisolate

        hasher.update(pyisolate.__version__.encode("utf-8"))
    except (ImportError, AttributeError):
        hasher.update(b"__pyisolate_unknown__")

    return hasher.hexdigest()[:CACHE_KEY_LENGTH]


def get_cache_path(node_dir: Path, venv_root: Path) -> Tuple[Path, Path]:
    """Return (cache_key_file, cache_data_file) in venv_root/{node}/cache/."""
    cache_dir = venv_root / node_dir.name / CACHE_SUBDIR
    return (cache_dir / CACHE_KEY_FILE, cache_dir / CACHE_DATA_FILE)


def is_cache_valid(node_dir: Path, manifest_path: Path, venv_root: Path) -> bool:
    """Return True only if stored cache key matches current computed key."""
    try:
        cache_key_file, cache_data_file = get_cache_path(node_dir, venv_root)
        if not cache_key_file.exists() or not cache_data_file.exists():
            return False
        current_key = compute_cache_key(node_dir, manifest_path)
        stored_key = cache_key_file.read_text(encoding="utf-8").strip()
        return current_key == stored_key
    except Exception as e:
        logger.debug(
            "%s Cache validation error for %s: %s", LOG_PREFIX, node_dir.name, e
        )
        return False


def load_from_cache(node_dir: Path, venv_root: Path) -> Optional[Dict[str, Any]]:
    """Load node metadata from cache, return None on any error."""
    try:
        _, cache_data_file = get_cache_path(node_dir, venv_root)
        if not cache_data_file.exists():
            return None
        data = json.loads(cache_data_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


def save_to_cache(
    node_dir: Path, venv_root: Path, node_data: Dict[str, Any], manifest_path: Path
) -> None:
    """Save node metadata and cache key atomically."""
    try:
        cache_key_file, cache_data_file = get_cache_path(node_dir, venv_root)
        cache_dir = cache_key_file.parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = compute_cache_key(node_dir, manifest_path)

        # Atomic write: data
        tmp_data_fd, tmp_data_path = tempfile.mkstemp(dir=str(cache_dir), suffix=".tmp")
        try:
            with os.fdopen(tmp_data_fd, "w", encoding="utf-8") as f:
                json.dump(node_data, f, indent=2)
            os.replace(tmp_data_path, cache_data_file)
        except Exception:
            try:
                os.unlink(tmp_data_path)
            except OSError:
                pass
            raise

        # Atomic write: key
        tmp_key_fd, tmp_key_path = tempfile.mkstemp(dir=str(cache_dir), suffix=".tmp")
        try:
            with os.fdopen(tmp_key_fd, "w", encoding="utf-8") as f:
                f.write(cache_key)
            os.replace(tmp_key_path, cache_key_file)
        except Exception:
            try:
                os.unlink(tmp_key_path)
            except OSError:
                pass
            raise

    except Exception as e:
        logger.warning("%s Cache save failed for %s: %s", LOG_PREFIX, node_dir.name, e)


__all__ = [
    "LOG_PREFIX",
    "find_manifest_directories",
    "compute_cache_key",
    "get_cache_path",
    "is_cache_valid",
    "load_from_cache",
    "save_to_cache",
]
