"""Tests for execution_model parsing and sealed-worker loader selection."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_manifest(
    *,
    package_manager: str = "uv",
    execution_model: str | None = None,
    can_isolate: bool = True,
    dependencies: list[str] | None = None,
    sealed_host_ro_paths: list[str] | None = None,
) -> dict:
    isolation: dict = {"can_isolate": can_isolate}
    if package_manager != "uv":
        isolation["package_manager"] = package_manager
    if execution_model is not None:
        isolation["execution_model"] = execution_model
    if sealed_host_ro_paths is not None:
        isolation["sealed_host_ro_paths"] = sealed_host_ro_paths

    return {
        "project": {
            "name": "test-extension",
            "dependencies": dependencies or ["numpy"],
        },
        "tool": {"comfy": {"isolation": isolation}},
    }


@pytest.fixture
def manifest_file(tmp_path):
    path = tmp_path / "pyproject.toml"
    path.write_bytes(b"")
    return path


@pytest.fixture
def loader_module(monkeypatch):
    mock_wrapper = MagicMock()
    mock_wrapper.ComfyNodeExtension = type("ComfyNodeExtension", (), {})

    iso_mod = types.ModuleType("comfy.isolation")
    iso_mod.__path__ = [  # type: ignore[attr-defined]
        str(Path(__file__).resolve().parent.parent.parent / "comfy" / "isolation")
    ]
    iso_mod.__package__ = "comfy.isolation"

    manifest_loader = types.SimpleNamespace(
        is_cache_valid=lambda *args, **kwargs: False,
        load_from_cache=lambda *args, **kwargs: None,
        save_to_cache=lambda *args, **kwargs: None,
    )
    host_policy = types.SimpleNamespace(
        load_host_policy=lambda base_path: {
            "sandbox_mode": "required",
            "allow_network": False,
            "writable_paths": [],
            "readonly_paths": [],
            "sealed_worker_ro_import_paths": [],
        }
    )
    folder_paths = types.SimpleNamespace(base_path="/fake/comfyui")

    monkeypatch.setitem(sys.modules, "comfy.isolation", iso_mod)
    monkeypatch.setitem(sys.modules, "comfy.isolation.extension_wrapper", mock_wrapper)
    monkeypatch.setitem(sys.modules, "comfy.isolation.runtime_helpers", MagicMock())
    monkeypatch.setitem(sys.modules, "comfy.isolation.manifest_loader", manifest_loader)
    monkeypatch.setitem(sys.modules, "comfy.isolation.host_policy", host_policy)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    sys.modules.pop("comfy.isolation.extension_loader", None)

    module = importlib.import_module("comfy.isolation.extension_loader")
    try:
        yield module
    finally:
        sys.modules.pop("comfy.isolation.extension_loader", None)
        comfy_pkg = sys.modules.get("comfy")
        if comfy_pkg is not None and hasattr(comfy_pkg, "isolation"):
            delattr(comfy_pkg, "isolation")


@pytest.fixture
def mock_pyisolate(loader_module):
    mock_ext = AsyncMock()
    mock_ext.list_nodes = AsyncMock(return_value={})

    mock_manager = MagicMock()
    mock_manager.load_extension = MagicMock(return_value=mock_ext)
    sealed_type = type("SealedNodeExtension", (), {})

    with patch.object(loader_module, "pyisolate") as mock_pi:
        mock_pi.ExtensionManager = MagicMock(return_value=mock_manager)
        mock_pi.SealedNodeExtension = sealed_type
        yield loader_module, mock_pi, mock_manager, mock_ext, sealed_type


def load_isolated_node(*args, **kwargs):
    return sys.modules["comfy.isolation.extension_loader"].load_isolated_node(*args, **kwargs)


@pytest.mark.asyncio
async def test_uv_sealed_worker_selects_sealed_extension_type(
    mock_pyisolate, manifest_file, tmp_path
):
    manifest = _make_manifest(execution_model="sealed_worker")

    _, mock_pi, mock_manager, _, sealed_type = mock_pyisolate

    with patch("comfy.isolation.extension_loader.tomllib") as mock_tomllib:
        mock_tomllib.load.return_value = manifest
        await load_isolated_node(
            node_dir=tmp_path,
            manifest_path=manifest_file,
            logger=MagicMock(),
            build_stub_class=MagicMock(),
            venv_root=tmp_path / "venvs",
            extension_managers=[],
        )

    extension_type = mock_pi.ExtensionManager.call_args[0][0]
    config = mock_manager.load_extension.call_args[0][0]
    assert extension_type is sealed_type
    assert config["execution_model"] == "sealed_worker"
    assert "apis" not in config


@pytest.mark.asyncio
async def test_default_uv_keeps_host_coupled_extension_type(
    mock_pyisolate, manifest_file, tmp_path
):
    manifest = _make_manifest()

    _, mock_pi, mock_manager, _, sealed_type = mock_pyisolate

    with patch("comfy.isolation.extension_loader.tomllib") as mock_tomllib:
        mock_tomllib.load.return_value = manifest
        await load_isolated_node(
            node_dir=tmp_path,
            manifest_path=manifest_file,
            logger=MagicMock(),
            build_stub_class=MagicMock(),
            venv_root=tmp_path / "venvs",
            extension_managers=[],
        )

    extension_type = mock_pi.ExtensionManager.call_args[0][0]
    config = mock_manager.load_extension.call_args[0][0]
    assert extension_type is not sealed_type
    assert "execution_model" not in config


@pytest.mark.asyncio
async def test_conda_without_execution_model_remains_sealed_worker(
    mock_pyisolate, manifest_file, tmp_path
):
    manifest = _make_manifest(package_manager="conda")
    manifest["tool"]["comfy"]["isolation"]["conda_channels"] = ["conda-forge"]
    manifest["tool"]["comfy"]["isolation"]["conda_dependencies"] = ["eccodes"]

    _, mock_pi, mock_manager, _, sealed_type = mock_pyisolate

    with patch("comfy.isolation.extension_loader.tomllib") as mock_tomllib:
        mock_tomllib.load.return_value = manifest
        await load_isolated_node(
            node_dir=tmp_path,
            manifest_path=manifest_file,
            logger=MagicMock(),
            build_stub_class=MagicMock(),
            venv_root=tmp_path / "venvs",
            extension_managers=[],
        )

    extension_type = mock_pi.ExtensionManager.call_args[0][0]
    config = mock_manager.load_extension.call_args[0][0]
    assert extension_type is sealed_type
    assert config["execution_model"] == "sealed_worker"


@pytest.mark.asyncio
async def test_sealed_worker_uses_host_policy_ro_import_paths(
    mock_pyisolate, manifest_file, tmp_path
):
    manifest = _make_manifest(execution_model="sealed_worker")

    module, _, mock_manager, _, _ = mock_pyisolate

    with (
        patch("comfy.isolation.extension_loader.tomllib") as mock_tomllib,
        patch.object(
            module,
            "load_host_policy",
            return_value={
                "sandbox_mode": "required",
                "allow_network": False,
                "writable_paths": [],
                "readonly_paths": [],
                "sealed_worker_ro_import_paths": ["/home/johnj/ComfyUI"],
            },
        ),
    ):
        mock_tomllib.load.return_value = manifest
        await load_isolated_node(
            node_dir=tmp_path,
            manifest_path=manifest_file,
            logger=MagicMock(),
            build_stub_class=MagicMock(),
            venv_root=tmp_path / "venvs",
            extension_managers=[],
        )

    config = mock_manager.load_extension.call_args[0][0]
    assert config["sealed_host_ro_paths"] == ["/home/johnj/ComfyUI"]


@pytest.mark.asyncio
async def test_host_coupled_does_not_emit_sealed_host_ro_paths(
    mock_pyisolate, manifest_file, tmp_path
):
    manifest = _make_manifest(execution_model="host-coupled")

    module, _, mock_manager, _, _ = mock_pyisolate

    with (
        patch("comfy.isolation.extension_loader.tomllib") as mock_tomllib,
        patch.object(
            module,
            "load_host_policy",
            return_value={
                "sandbox_mode": "required",
                "allow_network": False,
                "writable_paths": [],
                "readonly_paths": [],
                "sealed_worker_ro_import_paths": ["/home/johnj/ComfyUI"],
            },
        ),
    ):
        mock_tomllib.load.return_value = manifest
        await load_isolated_node(
            node_dir=tmp_path,
            manifest_path=manifest_file,
            logger=MagicMock(),
            build_stub_class=MagicMock(),
            venv_root=tmp_path / "venvs",
            extension_managers=[],
        )

    config = mock_manager.load_extension.call_args[0][0]
    assert "sealed_host_ro_paths" not in config


@pytest.mark.asyncio
async def test_sealed_worker_manifest_ro_import_paths_blocked(
    mock_pyisolate, manifest_file, tmp_path
):
    manifest = _make_manifest(
        execution_model="sealed_worker",
        sealed_host_ro_paths=["/home/johnj/ComfyUI"],
    )

    _, _, _mock_manager, _, _ = mock_pyisolate

    with patch("comfy.isolation.extension_loader.tomllib") as mock_tomllib:
        mock_tomllib.load.return_value = manifest
        with pytest.raises(ValueError, match="Manifest field 'sealed_host_ro_paths' is not allowed"):
            await load_isolated_node(
                node_dir=tmp_path,
                manifest_path=manifest_file,
                logger=MagicMock(),
                build_stub_class=MagicMock(),
                venv_root=tmp_path / "venvs",
                extension_managers=[],
            )
