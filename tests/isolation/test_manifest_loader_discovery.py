from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType


def _write_manifest(path: Path, *, standalone: bool = False) -> None:
    lines = [
        "[project]",
        'name = "test-node"',
        'version = "0.1.0"',
        "",
        "[tool.comfy.isolation]",
        "can_isolate = true",
        "share_torch = false",
    ]
    if standalone:
        lines.append("standalone = true")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_manifest_loader(custom_nodes_root: Path):
    folder_paths = ModuleType("folder_paths")
    folder_paths.base_path = str(custom_nodes_root)
    folder_paths.get_folder_paths = lambda kind: [str(custom_nodes_root)] if kind == "custom_nodes" else []
    sys.modules["folder_paths"] = folder_paths

    if "comfy.isolation" not in sys.modules:
        iso_mod = ModuleType("comfy.isolation")
        iso_mod.__path__ = [  # type: ignore[attr-defined]
            str(Path(__file__).resolve().parent.parent.parent / "comfy" / "isolation")
        ]
        iso_mod.__package__ = "comfy.isolation"
        sys.modules["comfy.isolation"] = iso_mod

    sys.modules.pop("comfy.isolation.manifest_loader", None)

    import comfy.isolation.manifest_loader as manifest_loader

    return importlib.reload(manifest_loader)


def test_finds_top_level_isolation_manifest(tmp_path: Path) -> None:
    node_dir = tmp_path / "TopLevelNode"
    node_dir.mkdir(parents=True)
    _write_manifest(node_dir / "pyproject.toml")

    manifest_loader = _load_manifest_loader(tmp_path)
    manifests = manifest_loader.find_manifest_directories()

    assert manifests == [(node_dir, node_dir / "pyproject.toml")]


def test_ignores_nested_manifest_without_standalone_flag(tmp_path: Path) -> None:
    toolkit_dir = tmp_path / "ToolkitNode"
    toolkit_dir.mkdir(parents=True)
    _write_manifest(toolkit_dir / "pyproject.toml")

    nested_dir = toolkit_dir / "packages" / "nested_fixture"
    nested_dir.mkdir(parents=True)
    _write_manifest(nested_dir / "pyproject.toml", standalone=False)

    manifest_loader = _load_manifest_loader(tmp_path)
    manifests = manifest_loader.find_manifest_directories()

    assert manifests == [(toolkit_dir, toolkit_dir / "pyproject.toml")]


def test_finds_nested_standalone_manifest(tmp_path: Path) -> None:
    toolkit_dir = tmp_path / "ToolkitNode"
    toolkit_dir.mkdir(parents=True)
    _write_manifest(toolkit_dir / "pyproject.toml")

    nested_dir = toolkit_dir / "packages" / "uv_sealed_worker"
    nested_dir.mkdir(parents=True)
    _write_manifest(nested_dir / "pyproject.toml", standalone=True)

    manifest_loader = _load_manifest_loader(tmp_path)
    manifests = manifest_loader.find_manifest_directories()

    assert manifests == [
        (toolkit_dir, toolkit_dir / "pyproject.toml"),
        (nested_dir, nested_dir / "pyproject.toml"),
    ]
