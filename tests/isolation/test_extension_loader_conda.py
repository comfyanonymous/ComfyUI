"""Tests for conda config parsing in extension_loader.py (Slice 5).

These tests verify that extension_loader.py correctly parses conda-related
fields from pyproject.toml manifests and passes them into the extension config
dict given to pyisolate. The torch import chain is broken by pre-mocking
extension_wrapper before importing extension_loader.
"""

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
    conda_channels: list[str] | None = None,
    conda_dependencies: list[str] | None = None,
    conda_platforms: list[str] | None = None,
    share_torch: bool = False,
    can_isolate: bool = True,
    dependencies: list[str] | None = None,
    cuda_wheels: list[str] | None = None,
) -> dict:
    """Build a manifest dict matching tomllib.load() output."""
    isolation: dict = {"can_isolate": can_isolate}
    if package_manager != "uv":
        isolation["package_manager"] = package_manager
    if conda_channels is not None:
        isolation["conda_channels"] = conda_channels
    if conda_dependencies is not None:
        isolation["conda_dependencies"] = conda_dependencies
    if conda_platforms is not None:
        isolation["conda_platforms"] = conda_platforms
    if share_torch:
        isolation["share_torch"] = True
    if cuda_wheels is not None:
        isolation["cuda_wheels"] = cuda_wheels

    return {
        "project": {
            "name": "test-extension",
            "dependencies": dependencies or ["numpy"],
        },
        "tool": {"comfy": {"isolation": isolation}},
    }


@pytest.fixture
def manifest_file(tmp_path):
    """Create a dummy pyproject.toml so manifest_path.open('rb') succeeds."""
    path = tmp_path / "pyproject.toml"
    path.write_bytes(b"")  # content is overridden by tomllib mock
    return path


@pytest.fixture
def loader_module(monkeypatch):
    """Import extension_loader under a mocked isolation package for this test only."""
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
        yield module, mock_wrapper
    finally:
        sys.modules.pop("comfy.isolation.extension_loader", None)
        comfy_pkg = sys.modules.get("comfy")
        if comfy_pkg is not None and hasattr(comfy_pkg, "isolation"):
            delattr(comfy_pkg, "isolation")


@pytest.fixture
def mock_pyisolate(loader_module):
    """Mock pyisolate to avoid real venv creation."""
    module, mock_wrapper = loader_module
    mock_ext = AsyncMock()
    mock_ext.list_nodes = AsyncMock(return_value={})

    mock_manager = MagicMock()
    mock_manager.load_extension = MagicMock(return_value=mock_ext)
    sealed_type = type("SealedNodeExtension", (), {})

    with patch.object(module, "pyisolate") as mock_pi:
        mock_pi.ExtensionManager = MagicMock(return_value=mock_manager)
        mock_pi.SealedNodeExtension = sealed_type
        yield module, mock_pi, mock_manager, mock_ext, mock_wrapper


def load_isolated_node(*args, **kwargs):
    return sys.modules["comfy.isolation.extension_loader"].load_isolated_node(
        *args, **kwargs
    )


class TestCondaPackageManagerParsing:
    """Verify extension_loader.py parses conda config from pyproject.toml."""

    @pytest.mark.asyncio
    async def test_conda_package_manager_in_config(
        self, mock_pyisolate, manifest_file, tmp_path
    ):
        """package_manager='conda' must appear in extension_config."""

        manifest = _make_manifest(
            package_manager="conda",
            conda_channels=["conda-forge"],
            conda_dependencies=["eccodes"],
        )

        _, _, mock_manager, _, _ = mock_pyisolate

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

        config = mock_manager.load_extension.call_args[0][0]
        assert config["package_manager"] == "conda"

    @pytest.mark.asyncio
    async def test_conda_channels_in_config(
        self, mock_pyisolate, manifest_file, tmp_path
    ):
        """conda_channels must be passed through to extension_config."""

        manifest = _make_manifest(
            package_manager="conda",
            conda_channels=["conda-forge", "nvidia"],
            conda_dependencies=["eccodes"],
        )

        _, _, mock_manager, _, _ = mock_pyisolate

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

        config = mock_manager.load_extension.call_args[0][0]
        assert config["conda_channels"] == ["conda-forge", "nvidia"]

    @pytest.mark.asyncio
    async def test_conda_dependencies_in_config(
        self, mock_pyisolate, manifest_file, tmp_path
    ):
        """conda_dependencies must be passed through to extension_config."""

        manifest = _make_manifest(
            package_manager="conda",
            conda_channels=["conda-forge"],
            conda_dependencies=["eccodes", "cfgrib"],
        )

        _, _, mock_manager, _, _ = mock_pyisolate

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

        config = mock_manager.load_extension.call_args[0][0]
        assert config["conda_dependencies"] == ["eccodes", "cfgrib"]

    @pytest.mark.asyncio
    async def test_conda_platforms_in_config(
        self, mock_pyisolate, manifest_file, tmp_path
    ):
        """conda_platforms must be passed through to extension_config."""

        manifest = _make_manifest(
            package_manager="conda",
            conda_channels=["conda-forge"],
            conda_dependencies=["eccodes"],
            conda_platforms=["linux-64"],
        )

        _, _, mock_manager, _, _ = mock_pyisolate

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

        config = mock_manager.load_extension.call_args[0][0]
        assert config["conda_platforms"] == ["linux-64"]


class TestCondaForcedOverrides:
    """Verify conda forces share_torch=False, share_cuda_ipc=False."""

    @pytest.mark.asyncio
    async def test_conda_forces_share_torch_false(
        self, mock_pyisolate, manifest_file, tmp_path
    ):
        """share_torch must be forced False for conda, even if manifest says True."""

        manifest = _make_manifest(
            package_manager="conda",
            conda_channels=["conda-forge"],
            conda_dependencies=["eccodes"],
            share_torch=True,  # manifest requests True — must be overridden
        )

        _, _, mock_manager, _, _ = mock_pyisolate

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

        config = mock_manager.load_extension.call_args[0][0]
        assert config["share_torch"] is False

    @pytest.mark.asyncio
    async def test_conda_forces_share_cuda_ipc_false(
        self, mock_pyisolate, manifest_file, tmp_path
    ):
        """share_cuda_ipc must be forced False for conda."""

        manifest = _make_manifest(
            package_manager="conda",
            conda_channels=["conda-forge"],
            conda_dependencies=["eccodes"],
            share_torch=True,
        )

        _, _, mock_manager, _, _ = mock_pyisolate

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

        config = mock_manager.load_extension.call_args[0][0]
        assert config["share_cuda_ipc"] is False

    @pytest.mark.asyncio
    async def test_conda_sealed_worker_uses_host_policy_sandbox_config(
        self, mock_pyisolate, manifest_file, tmp_path
    ):
        """Conda sealed_worker must carry the host-policy sandbox config on Linux."""

        manifest = _make_manifest(
            package_manager="conda",
            conda_channels=["conda-forge"],
            conda_dependencies=["eccodes"],
        )

        _, _, mock_manager, _, _ = mock_pyisolate

        with (
            patch("comfy.isolation.extension_loader.tomllib") as mock_tomllib,
            patch(
                "comfy.isolation.extension_loader.platform.system",
                return_value="Linux",
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
        assert config["sandbox"] == {
            "network": False,
            "writable_paths": [],
            "readonly_paths": [],
        }

    @pytest.mark.asyncio
    async def test_conda_uses_sealed_extension_type(
        self, mock_pyisolate, manifest_file, tmp_path
    ):
        """Conda must not launch through ComfyNodeExtension."""

        _, mock_pi, _, _, mock_wrapper = mock_pyisolate
        manifest = _make_manifest(
            package_manager="conda",
            conda_channels=["conda-forge"],
            conda_dependencies=["eccodes"],
        )

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
        assert extension_type.__name__ == "SealedNodeExtension"
        assert extension_type is not mock_wrapper.ComfyNodeExtension


class TestUvUnchanged:
    """Verify uv extensions are NOT affected by conda changes."""

    @pytest.mark.asyncio
    async def test_uv_default_no_conda_keys(
        self, mock_pyisolate, manifest_file, tmp_path
    ):
        """Default uv extension must NOT have package_manager or conda keys."""

        manifest = _make_manifest()  # defaults: uv, no conda fields

        _, _, mock_manager, _, _ = mock_pyisolate

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

        config = mock_manager.load_extension.call_args[0][0]
        # uv extensions should not have conda-specific keys
        assert config.get("package_manager", "uv") == "uv"
        assert "conda_channels" not in config
        assert "conda_dependencies" not in config

    @pytest.mark.asyncio
    async def test_uv_keeps_comfy_extension_type(
        self, mock_pyisolate, manifest_file, tmp_path
    ):
        """uv keeps the existing ComfyNodeExtension path."""

        _, mock_pi, _, _, _ = mock_pyisolate
        manifest = _make_manifest()

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
        assert extension_type.__name__ == "ComfyNodeExtension"
        assert extension_type is not mock_pi.SealedNodeExtension
