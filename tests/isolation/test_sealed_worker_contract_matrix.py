"""Generic sealed-worker loader contract matrix tests."""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

COMFYUI_ROOT = Path(__file__).resolve().parents[2]
TEST_WORKFLOW_ROOT = COMFYUI_ROOT / "tests" / "isolation" / "workflows"
SEALED_WORKFLOW_CLASS_TYPES: dict[str, set[str]] = {
    "quick_6_uv_sealed_worker.json": {
        "EmptyLatentImage",
        "ProxyTestSealedWorker",
        "UVSealedBoltonsSlugify",
        "UVSealedLatentEcho",
        "UVSealedRuntimeProbe",
    },
    "isolation_7_uv_sealed_worker.json": {
        "EmptyLatentImage",
        "ProxyTestSealedWorker",
        "UVSealedBoltonsSlugify",
        "UVSealedLatentEcho",
        "UVSealedRuntimeProbe",
    },
    "quick_8_conda_sealed_worker.json": {
        "CondaSealedLatentEcho",
        "CondaSealedOpenWeatherDataset",
        "CondaSealedRuntimeProbe",
        "EmptyLatentImage",
        "ProxyTestCondaSealedWorker",
    },
    "isolation_9_conda_sealed_worker.json": {
        "CondaSealedLatentEcho",
        "CondaSealedOpenWeatherDataset",
        "CondaSealedRuntimeProbe",
        "EmptyLatentImage",
        "ProxyTestCondaSealedWorker",
    },
}


def _workflow_class_types(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        node["class_type"]
        for node in payload.values()
        if isinstance(node, dict) and "class_type" in node
    }


def _make_manifest(
    *,
    package_manager: str = "uv",
    execution_model: str | None = None,
    can_isolate: bool = True,
    dependencies: list[str] | None = None,
    share_torch: bool = False,
    sealed_host_ro_paths: list[str] | None = None,
) -> dict:
    isolation: dict[str, object] = {
        "can_isolate": can_isolate,
    }
    if package_manager != "uv":
        isolation["package_manager"] = package_manager
    if execution_model is not None:
        isolation["execution_model"] = execution_model
    if share_torch:
        isolation["share_torch"] = True
    if sealed_host_ro_paths is not None:
        isolation["sealed_host_ro_paths"] = sealed_host_ro_paths

    if package_manager == "conda":
        isolation["conda_channels"] = ["conda-forge"]
        isolation["conda_dependencies"] = ["numpy"]

    return {
        "project": {
            "name": "contract-extension",
            "dependencies": dependencies or ["numpy"],
        },
        "tool": {"comfy": {"isolation": isolation}},
    }


@pytest.fixture
def manifest_file(tmp_path: Path) -> Path:
    path = tmp_path / "pyproject.toml"
    path.write_bytes(b"")
    return path


def _loader_module(
    monkeypatch: pytest.MonkeyPatch, *, preload_extension_wrapper: bool
):
    mock_wrapper = MagicMock()
    mock_wrapper.ComfyNodeExtension = type("ComfyNodeExtension", (), {})

    iso_mod = types.ModuleType("comfy.isolation")
    iso_mod.__path__ = [
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
    monkeypatch.setitem(sys.modules, "comfy.isolation.runtime_helpers", MagicMock())
    monkeypatch.setitem(sys.modules, "comfy.isolation.manifest_loader", manifest_loader)
    monkeypatch.setitem(sys.modules, "comfy.isolation.host_policy", host_policy)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    if preload_extension_wrapper:
        monkeypatch.setitem(sys.modules, "comfy.isolation.extension_wrapper", mock_wrapper)
    else:
        sys.modules.pop("comfy.isolation.extension_wrapper", None)
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
def loader_module(monkeypatch: pytest.MonkeyPatch):
    yield from _loader_module(monkeypatch, preload_extension_wrapper=True)


@pytest.fixture
def sealed_loader_module(monkeypatch: pytest.MonkeyPatch):
    yield from _loader_module(monkeypatch, preload_extension_wrapper=False)


@pytest.fixture
def mocked_loader(loader_module):
    module, mock_wrapper = loader_module
    mock_ext = AsyncMock()
    mock_ext.list_nodes = AsyncMock(return_value={})

    mock_manager = MagicMock()
    mock_manager.load_extension = MagicMock(return_value=mock_ext)
    sealed_type = type("SealedNodeExtension", (), {})

    with patch.object(module, "pyisolate") as mock_pi:
        mock_pi.ExtensionManager = MagicMock(return_value=mock_manager)
        mock_pi.SealedNodeExtension = sealed_type
        yield module, mock_pi, mock_manager, sealed_type, mock_wrapper


@pytest.fixture
def sealed_mocked_loader(sealed_loader_module):
    module, mock_wrapper = sealed_loader_module
    mock_ext = AsyncMock()
    mock_ext.list_nodes = AsyncMock(return_value={})

    mock_manager = MagicMock()
    mock_manager.load_extension = MagicMock(return_value=mock_ext)
    sealed_type = type("SealedNodeExtension", (), {})

    with patch.object(module, "pyisolate") as mock_pi:
        mock_pi.ExtensionManager = MagicMock(return_value=mock_manager)
        mock_pi.SealedNodeExtension = sealed_type
        yield module, mock_pi, mock_manager, sealed_type, mock_wrapper


async def _load_node(module, manifest: dict, manifest_path: Path, tmp_path: Path) -> dict:
    with patch("comfy.isolation.extension_loader.tomllib") as mock_tomllib:
        mock_tomllib.load.return_value = manifest
        await module.load_isolated_node(
            node_dir=tmp_path,
            manifest_path=manifest_path,
            logger=MagicMock(),
            build_stub_class=MagicMock(),
            venv_root=tmp_path / "venvs",
            extension_managers=[],
        )
    manager = module.pyisolate.ExtensionManager.return_value
    return manager.load_extension.call_args[0][0]


@pytest.mark.asyncio
async def test_uv_host_coupled_default(mocked_loader, manifest_file: Path, tmp_path: Path):
    module, mock_pi, _mock_manager, sealed_type, _ = mocked_loader
    manifest = _make_manifest(package_manager="uv")

    config = await _load_node(module, manifest, manifest_file, tmp_path)

    extension_type = mock_pi.ExtensionManager.call_args[0][0]
    assert extension_type is not sealed_type
    assert "execution_model" not in config


@pytest.mark.asyncio
async def test_uv_sealed_worker_opt_in(
    sealed_mocked_loader, manifest_file: Path, tmp_path: Path
):
    module, mock_pi, _mock_manager, sealed_type, _ = sealed_mocked_loader
    manifest = _make_manifest(package_manager="uv", execution_model="sealed_worker")

    config = await _load_node(module, manifest, manifest_file, tmp_path)

    extension_type = mock_pi.ExtensionManager.call_args[0][0]
    assert extension_type is sealed_type
    assert config["execution_model"] == "sealed_worker"
    assert "apis" not in config
    assert "comfy.isolation.extension_wrapper" not in sys.modules


@pytest.mark.asyncio
async def test_conda_defaults_to_sealed_worker(
    sealed_mocked_loader, manifest_file: Path, tmp_path: Path
):
    module, mock_pi, _mock_manager, sealed_type, _ = sealed_mocked_loader
    manifest = _make_manifest(package_manager="conda")

    config = await _load_node(module, manifest, manifest_file, tmp_path)

    extension_type = mock_pi.ExtensionManager.call_args[0][0]
    assert extension_type is sealed_type
    assert config["execution_model"] == "sealed_worker"
    assert config["package_manager"] == "conda"
    assert "comfy.isolation.extension_wrapper" not in sys.modules


@pytest.mark.asyncio
async def test_conda_never_uses_comfy_extension_type(
    mocked_loader, manifest_file: Path, tmp_path: Path
):
    module, mock_pi, _mock_manager, sealed_type, mock_wrapper = mocked_loader
    manifest = _make_manifest(package_manager="conda")

    await _load_node(module, manifest, manifest_file, tmp_path)

    extension_type = mock_pi.ExtensionManager.call_args[0][0]
    assert extension_type is sealed_type
    assert extension_type is not mock_wrapper.ComfyNodeExtension


@pytest.mark.asyncio
async def test_conda_forces_share_torch_false(mocked_loader, manifest_file: Path, tmp_path: Path):
    module, _mock_pi, _mock_manager, _sealed_type, _ = mocked_loader
    manifest = _make_manifest(package_manager="conda", share_torch=True)

    config = await _load_node(module, manifest, manifest_file, tmp_path)

    assert config["share_torch"] is False


@pytest.mark.asyncio
async def test_conda_forces_share_cuda_ipc_false(
    mocked_loader, manifest_file: Path, tmp_path: Path
):
    module, _mock_pi, _mock_manager, _sealed_type, _ = mocked_loader
    manifest = _make_manifest(package_manager="conda", share_torch=True)

    config = await _load_node(module, manifest, manifest_file, tmp_path)

    assert config["share_cuda_ipc"] is False


@pytest.mark.asyncio
async def test_conda_sandbox_policy_applied(mocked_loader, manifest_file: Path, tmp_path: Path):
    module, _mock_pi, _mock_manager, _sealed_type, _ = mocked_loader
    manifest = _make_manifest(package_manager="conda")

    custom_policy = {
        "sandbox_mode": "required",
        "allow_network": True,
        "writable_paths": ["/data/write"],
        "readonly_paths": ["/data/read"],
    }

    with patch("platform.system", return_value="Linux"):
        with patch.object(module, "load_host_policy", return_value=custom_policy):
            config = await _load_node(module, manifest, manifest_file, tmp_path)

    assert config["sandbox_mode"] == "required"
    assert config["sandbox"] == {
        "network": True,
        "writable_paths": ["/data/write"],
        "readonly_paths": ["/data/read"],
    }


def test_sealed_worker_workflow_templates_present() -> None:
    missing = [
        filename
        for filename in SEALED_WORKFLOW_CLASS_TYPES
        if not (TEST_WORKFLOW_ROOT / filename).is_file()
    ]
    assert not missing, f"missing sealed-worker workflow templates: {missing}"


@pytest.mark.parametrize(
    "workflow_name,expected_class_types",
    SEALED_WORKFLOW_CLASS_TYPES.items(),
)
def test_sealed_worker_workflow_class_type_contract(
    workflow_name: str, expected_class_types: set[str]
) -> None:
    workflow_path = TEST_WORKFLOW_ROOT / workflow_name
    assert workflow_path.is_file(), f"workflow missing: {workflow_path}"

    assert _workflow_class_types(workflow_path) == expected_class_types


@pytest.mark.asyncio
async def test_sealed_worker_host_policy_ro_import_matrix(
    mocked_loader, manifest_file: Path, tmp_path: Path
):
    module, _mock_pi, _mock_manager, _sealed_type, _ = mocked_loader
    manifest = _make_manifest(package_manager="uv", execution_model="sealed_worker")

    with patch.object(
        module,
        "load_host_policy",
        return_value={
            "sandbox_mode": "required",
            "allow_network": False,
            "writable_paths": [],
            "readonly_paths": [],
            "sealed_worker_ro_import_paths": [],
        },
    ):
        default_config = await _load_node(module, manifest, manifest_file, tmp_path)

    with patch.object(
        module,
        "load_host_policy",
        return_value={
            "sandbox_mode": "required",
            "allow_network": False,
            "writable_paths": [],
            "readonly_paths": [],
            "sealed_worker_ro_import_paths": ["/home/johnj/ComfyUI"],
        },
    ):
        opt_in_config = await _load_node(module, manifest, manifest_file, tmp_path)

    assert default_config["execution_model"] == "sealed_worker"
    assert "sealed_host_ro_paths" not in default_config

    assert opt_in_config["execution_model"] == "sealed_worker"
    assert opt_in_config["sealed_host_ro_paths"] == ["/home/johnj/ComfyUI"]
    assert "apis" not in opt_in_config
