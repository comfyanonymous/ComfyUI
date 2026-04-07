"""Synthetic integration coverage for manifest plumbing and env flags.

These tests do not perform a real wheel install or a real ComfyUI E2E run.
"""

import asyncio
import logging
import os
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest

import comfy.isolation as isolation_pkg
from comfy.isolation import runtime_helpers
from comfy.isolation import extension_loader as extension_loader_module
from comfy.isolation import extension_wrapper as extension_wrapper_module
from comfy.isolation import model_patcher_proxy_utils
from comfy.isolation.extension_loader import ExtensionLoadError, load_isolated_node
from comfy.isolation.extension_wrapper import ComfyNodeExtension
from comfy.isolation.model_patcher_proxy_utils import maybe_wrap_model_for_isolation
from pyisolate._internal.environment_conda import _generate_pixi_toml


class _DummyExtension:
    def __init__(self) -> None:
        self.name = "demo-extension"

    async def stop(self) -> None:
        return None


def _write_manifest(node_dir, manifest_text: str) -> None:
    (node_dir / "pyproject.toml").write_text(manifest_text, encoding="utf-8")


def test_load_isolated_node_passes_normalized_cuda_wheels_config(tmp_path, monkeypatch):
    node_dir = tmp_path / "node"
    node_dir.mkdir()
    manifest_path = node_dir / "pyproject.toml"
    _write_manifest(
        node_dir,
        """
[project]
name = "demo-node"
dependencies = ["flash-attn>=1.0", "sageattention==0.1"]

[tool.comfy.isolation]
can_isolate = true
share_torch = true

[tool.comfy.isolation.cuda_wheels]
index_url = "https://example.invalid/cuda-wheels"
packages = ["flash_attn", "sageattention"]

[tool.comfy.isolation.cuda_wheels.package_map]
flash_attn = "flash-attn-special"
""".strip(),
    )

    captured: dict[str, object] = {}

    class DummyManager:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def load_extension(self, config):
            captured.update(config)
            return _DummyExtension()

    monkeypatch.setattr(extension_loader_module.pyisolate, "ExtensionManager", DummyManager)
    monkeypatch.setattr(
        extension_loader_module,
        "load_host_policy",
        lambda base_path: {
            "sandbox_mode": "required",
            "allow_network": False,
            "writable_paths": [],
            "readonly_paths": [],
        },
    )
    monkeypatch.setattr(extension_loader_module, "is_cache_valid", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        extension_loader_module,
        "load_from_cache",
        lambda *args, **kwargs: {"Node": {"display_name": "Node", "schema_v1": {}}},
    )
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(base_path=str(tmp_path)))

    specs = asyncio.run(
        load_isolated_node(
            node_dir,
            manifest_path,
            logging.getLogger("test"),
            lambda *args, **kwargs: object,
            tmp_path / "venvs",
            [],
        )
    )

    assert len(specs) == 1
    assert captured["sandbox_mode"] == "required"
    assert captured["cuda_wheels"] == {
        "index_url": "https://example.invalid/cuda-wheels/",
        "packages": ["flash-attn", "sageattention"],
        "package_map": {"flash-attn": "flash-attn-special"},
    }


def test_load_isolated_node_rejects_undeclared_cuda_wheel_dependency(
    tmp_path, monkeypatch
):
    node_dir = tmp_path / "node"
    node_dir.mkdir()
    manifest_path = node_dir / "pyproject.toml"
    _write_manifest(
        node_dir,
        """
[project]
name = "demo-node"
dependencies = ["numpy>=1.0"]

[tool.comfy.isolation]
can_isolate = true

[tool.comfy.isolation.cuda_wheels]
index_url = "https://example.invalid/cuda-wheels"
packages = ["flash-attn"]
""".strip(),
    )

    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(base_path=str(tmp_path)))

    with pytest.raises(ExtensionLoadError, match="undeclared dependencies"):
        asyncio.run(
            load_isolated_node(
                node_dir,
                manifest_path,
                logging.getLogger("test"),
                lambda *args, **kwargs: object,
                tmp_path / "venvs",
                [],
            )
        )


def test_conda_cuda_wheels_declared_packages_do_not_force_pixi_solve(tmp_path, monkeypatch):
    node_dir = tmp_path / "node"
    node_dir.mkdir()
    manifest_path = node_dir / "pyproject.toml"
    _write_manifest(
        node_dir,
        """
[project]
name = "demo-node"
dependencies = ["numpy>=1.0", "spconv", "cumm", "flash-attn"]

[tool.comfy.isolation]
can_isolate = true
package_manager = "conda"
conda_channels = ["conda-forge"]

[tool.comfy.isolation.cuda_wheels]
index_url = "https://example.invalid/cuda-wheels"
packages = ["spconv", "cumm", "flash-attn"]
""".strip(),
    )

    captured: dict[str, object] = {}

    class DummyManager:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def load_extension(self, config):
            captured.update(config)
            return _DummyExtension()

    monkeypatch.setattr(extension_loader_module.pyisolate, "ExtensionManager", DummyManager)
    monkeypatch.setattr(
        extension_loader_module,
        "load_host_policy",
        lambda base_path: {
            "sandbox_mode": "disabled",
            "allow_network": False,
            "writable_paths": [],
            "readonly_paths": [],
        },
    )
    monkeypatch.setattr(extension_loader_module, "is_cache_valid", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        extension_loader_module,
        "load_from_cache",
        lambda *args, **kwargs: {"Node": {"display_name": "Node", "schema_v1": {}}},
    )
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(base_path=str(tmp_path)))

    asyncio.run(
        load_isolated_node(
            node_dir,
            manifest_path,
            logging.getLogger("test"),
            lambda *args, **kwargs: object,
            tmp_path / "venvs",
            [],
        )
    )

    generated = _generate_pixi_toml(captured)
    assert 'numpy = ">=1.0"' in generated
    assert "spconv =" not in generated
    assert "cumm =" not in generated
    assert "flash-attn =" not in generated


def test_conda_cuda_wheels_loader_accepts_sam3d_contract(tmp_path, monkeypatch):
    node_dir = tmp_path / "node"
    node_dir.mkdir()
    manifest_path = node_dir / "pyproject.toml"
    _write_manifest(
        node_dir,
        """
[project]
name = "demo-node"
dependencies = [
  "torch",
  "torchvision",
  "pytorch3d",
  "gsplat",
  "nvdiffrast",
  "flash-attn",
  "sageattention",
  "spconv",
  "cumm",
]

[tool.comfy.isolation]
can_isolate = true
package_manager = "conda"
conda_channels = ["conda-forge"]

[tool.comfy.isolation.cuda_wheels]
index_url = "https://example.invalid/cuda-wheels"
packages = ["pytorch3d", "gsplat", "nvdiffrast", "flash-attn", "sageattention", "spconv", "cumm"]
""".strip(),
    )

    captured: dict[str, object] = {}

    class DummyManager:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def load_extension(self, config):
            captured.update(config)
            return _DummyExtension()

    monkeypatch.setattr(extension_loader_module.pyisolate, "ExtensionManager", DummyManager)
    monkeypatch.setattr(
        extension_loader_module,
        "load_host_policy",
        lambda base_path: {
            "sandbox_mode": "disabled",
            "allow_network": False,
            "writable_paths": [],
            "readonly_paths": [],
        },
    )
    monkeypatch.setattr(extension_loader_module, "is_cache_valid", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        extension_loader_module,
        "load_from_cache",
        lambda *args, **kwargs: {"Node": {"display_name": "Node", "schema_v1": {}}},
    )
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(base_path=str(tmp_path)))

    asyncio.run(
        load_isolated_node(
            node_dir,
            manifest_path,
            logging.getLogger("test"),
            lambda *args, **kwargs: object,
            tmp_path / "venvs",
            [],
        )
    )

    assert captured["package_manager"] == "conda"
    assert captured["cuda_wheels"] == {
        "index_url": "https://example.invalid/cuda-wheels/",
        "packages": [
            "pytorch3d",
            "gsplat",
            "nvdiffrast",
            "flash-attn",
            "sageattention",
            "spconv",
            "cumm",
        ],
        "package_map": {},
    }


def test_load_isolated_node_omits_cuda_wheels_when_not_configured(tmp_path, monkeypatch):
    node_dir = tmp_path / "node"
    node_dir.mkdir()
    manifest_path = node_dir / "pyproject.toml"
    _write_manifest(
        node_dir,
        """
[project]
name = "demo-node"
dependencies = ["numpy>=1.0"]

[tool.comfy.isolation]
can_isolate = true
""".strip(),
    )

    captured: dict[str, object] = {}

    class DummyManager:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def load_extension(self, config):
            captured.update(config)
            return _DummyExtension()

    monkeypatch.setattr(extension_loader_module.pyisolate, "ExtensionManager", DummyManager)
    monkeypatch.setattr(
        extension_loader_module,
        "load_host_policy",
        lambda base_path: {
            "sandbox_mode": "disabled",
            "allow_network": False,
            "writable_paths": [],
            "readonly_paths": [],
        },
    )
    monkeypatch.setattr(extension_loader_module, "is_cache_valid", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        extension_loader_module,
        "load_from_cache",
        lambda *args, **kwargs: {"Node": {"display_name": "Node", "schema_v1": {}}},
    )
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(base_path=str(tmp_path)))

    asyncio.run(
        load_isolated_node(
            node_dir,
            manifest_path,
            logging.getLogger("test"),
            lambda *args, **kwargs: object,
            tmp_path / "venvs",
            [],
        )
    )

    assert captured["sandbox_mode"] == "disabled"
    assert "cuda_wheels" not in captured


def test_maybe_wrap_model_for_isolation_uses_runtime_flag(monkeypatch):
    class DummyRegistry:
        def register(self, model):
            return "model-123"

    class DummyProxy:
        def __init__(self, model_id, registry, manage_lifecycle):
            self.model_id = model_id
            self.registry = registry
            self.manage_lifecycle = manage_lifecycle

    monkeypatch.setattr(model_patcher_proxy_utils.args, "use_process_isolation", True)
    monkeypatch.delenv("PYISOLATE_ISOLATION_ACTIVE", raising=False)
    monkeypatch.delenv("PYISOLATE_CHILD", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "comfy.isolation.model_patcher_proxy_registry",
        SimpleNamespace(ModelPatcherRegistry=DummyRegistry),
    )
    monkeypatch.setitem(
        sys.modules,
        "comfy.isolation.model_patcher_proxy",
        SimpleNamespace(ModelPatcherProxy=DummyProxy),
    )

    wrapped = cast(Any, maybe_wrap_model_for_isolation(object()))

    assert isinstance(wrapped, DummyProxy)
    assert getattr(wrapped, "model_id") == "model-123"
    assert getattr(wrapped, "manage_lifecycle") is True


def test_flush_transport_state_uses_child_env_without_legacy_flag(monkeypatch):
    monkeypatch.setenv("PYISOLATE_CHILD", "1")
    monkeypatch.delenv("PYISOLATE_ISOLATION_ACTIVE", raising=False)
    monkeypatch.setattr(extension_wrapper_module, "_flush_tensor_transport_state", lambda marker: 3)
    monkeypatch.setitem(
        sys.modules,
        "comfy.isolation.model_patcher_proxy_registry",
        SimpleNamespace(
            ModelPatcherRegistry=lambda: SimpleNamespace(
                sweep_pending_cleanup=lambda: 0
            )
        ),
    )

    flushed = asyncio.run(
        ComfyNodeExtension.flush_transport_state(SimpleNamespace(name="demo"))
    )

    assert flushed == 3


def test_build_stub_class_relieves_host_vram_without_legacy_flag(monkeypatch):
    relieve_calls: list[str] = []

    async def deserialize_from_isolation(result, extension):
        return result

    monkeypatch.delenv("PYISOLATE_CHILD", raising=False)
    monkeypatch.delenv("PYISOLATE_ISOLATION_ACTIVE", raising=False)
    monkeypatch.setattr(
        runtime_helpers, "_relieve_host_vram_pressure", lambda marker, logger: relieve_calls.append(marker)
    )
    monkeypatch.setattr(runtime_helpers, "scan_shm_forensics", lambda *args, **kwargs: None)
    monkeypatch.setattr(isolation_pkg, "_RUNNING_EXTENSIONS", {}, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "pyisolate._internal.model_serialization",
        SimpleNamespace(
            serialize_for_isolation=lambda payload: payload,
            deserialize_from_isolation=deserialize_from_isolation,
        ),
    )

    class DummyExtension:
        name = "demo-extension"
        module_path = os.getcwd()

        async def execute_node(self, node_name, **inputs):
            return inputs

    stub_cls = runtime_helpers.build_stub_class(
        "DemoNode",
        {"input_types": {}},
        DummyExtension(),
        {},
        logging.getLogger("test"),
    )

    result = asyncio.run(
        getattr(stub_cls, "_pyisolate_execute")(SimpleNamespace(), value=1)
    )

    assert relieve_calls == ["RUNTIME:pre_execute"]
    assert result == {"value": 1}
