from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any


COMFYUI_ROOT = Path(__file__).resolve().parents[2]
UV_SEALED_WORKER_MODULE = COMFYUI_ROOT / "tests" / "isolation" / "uv_sealed_worker" / "__init__.py"
FORBIDDEN_MINIMAL_SEALED_MODULES = (
    "torch",
    "folder_paths",
    "comfy.utils",
    "comfy.model_management",
    "main",
    "comfy.isolation.extension_wrapper",
)
FORBIDDEN_SEALED_SINGLETON_MODULES = (
    "torch",
    "folder_paths",
    "comfy.utils",
    "comfy_execution.progress",
)
FORBIDDEN_EXACT_SMALL_PROXY_MODULES = FORBIDDEN_SEALED_SINGLETON_MODULES
FORBIDDEN_MODEL_MANAGEMENT_MODULES = (
    "comfy.model_management",
)


def _load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to build import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def matching_modules(prefixes: tuple[str, ...], modules: set[str]) -> list[str]:
    return sorted(
        module_name
        for module_name in modules
        if any(
            module_name == prefix or module_name.startswith(f"{prefix}.")
            for prefix in prefixes
        )
    )


def _load_helper_proxy_service() -> Any | None:
    try:
        from comfy.isolation.proxies.helper_proxies import HelperProxiesService
    except (ImportError, AttributeError):
        return None
    return HelperProxiesService


def _load_model_management_proxy() -> Any | None:
    try:
        from comfy.isolation.proxies.model_management_proxy import ModelManagementProxy
    except (ImportError, AttributeError):
        return None
    return ModelManagementProxy


async def _capture_minimal_sealed_worker_imports() -> dict[str, object]:
    from pyisolate.sealed import SealedNodeExtension

    module_name = "tests.isolation.uv_sealed_worker_boundary_probe"
    before = set(sys.modules)
    extension = SealedNodeExtension()
    module = _load_module_from_path(module_name, UV_SEALED_WORKER_MODULE)
    try:
        await extension.on_module_loaded(module)
        node_list = await extension.list_nodes()
        node_details = await extension.get_node_details("UVSealedRuntimeProbe")
        imported = set(sys.modules) - before
        return {
            "mode": "minimal_sealed_worker",
            "node_names": sorted(node_list),
            "runtime_probe_function": node_details["function"],
            "modules": sorted(imported),
            "forbidden_matches": matching_modules(FORBIDDEN_MINIMAL_SEALED_MODULES, imported),
        }
    finally:
        sys.modules.pop(module_name, None)


def capture_minimal_sealed_worker_imports() -> dict[str, object]:
    return asyncio.run(_capture_minimal_sealed_worker_imports())


class FakeSingletonCaller:
    def __init__(self, methods: dict[str, Any], calls: list[dict[str, Any]], object_id: str):
        self._methods = methods
        self._calls = calls
        self._object_id = object_id

    def __getattr__(self, name: str):
        if name not in self._methods:
            raise AttributeError(name)

        async def method(*args: Any, **kwargs: Any) -> Any:
            self._calls.append(
                {
                    "object_id": self._object_id,
                    "method": name,
                    "args": list(args),
                    "kwargs": dict(kwargs),
                }
            )
            result = self._methods[name]
            return result(*args, **kwargs) if callable(result) else result

        return method


class FakeSingletonRPC:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._device = {"__pyisolate_torch_device__": "cpu"}
        self._services: dict[str, dict[str, Any]] = {
            "FolderPathsProxy": {
                "rpc_get_models_dir": lambda: "/sandbox/models",
                "rpc_get_folder_names_and_paths": lambda: {
                    "checkpoints": {
                        "paths": ["/sandbox/models/checkpoints"],
                        "extensions": [".ckpt", ".safetensors"],
                    }
                },
                "rpc_get_extension_mimetypes_cache": lambda: {"webp": "image"},
                "rpc_get_filename_list_cache": lambda: {},
                "rpc_get_temp_directory": lambda: "/sandbox/temp",
                "rpc_get_input_directory": lambda: "/sandbox/input",
                "rpc_get_output_directory": lambda: "/sandbox/output",
                "rpc_get_user_directory": lambda: "/sandbox/user",
                "rpc_get_annotated_filepath": self._get_annotated_filepath,
                "rpc_exists_annotated_filepath": lambda _name: False,
                "rpc_add_model_folder_path": lambda *_args, **_kwargs: None,
                "rpc_get_folder_paths": lambda folder_name: [f"/sandbox/models/{folder_name}"],
                "rpc_get_filename_list": lambda folder_name: [f"{folder_name}_fixture.safetensors"],
                "rpc_get_full_path": lambda folder_name, filename: f"/sandbox/models/{folder_name}/{filename}",
            },
            "UtilsProxy": {
                "progress_bar_hook": lambda value, total, preview=None, node_id=None: {
                    "value": value,
                    "total": total,
                    "preview": preview,
                    "node_id": node_id,
                }
            },
            "ProgressProxy": {
                "rpc_set_progress": lambda value, max_value, node_id=None, image=None: {
                    "value": value,
                    "max_value": max_value,
                    "node_id": node_id,
                    "image": image,
                }
            },
            "HelperProxiesService": {
                "rpc_restore_input_types": lambda raw: raw,
            },
            "ModelManagementProxy": {
                "rpc_call": self._model_management_rpc_call,
            },
        }

    def _model_management_rpc_call(self, method_name: str, args: Any = None, kwargs: Any = None) -> Any:
        if method_name == "get_torch_device":
            return self._device
        elif method_name == "get_torch_device_name":
            return "cpu"
        elif method_name == "get_free_memory":
            return 34359738368
        raise AssertionError(f"unexpected model_management method {method_name}")

    @staticmethod
    def _get_annotated_filepath(name: str, default_dir: str | None = None) -> str:
        if name.endswith("[output]"):
            return f"/sandbox/output/{name[:-8]}"
        if name.endswith("[input]"):
            return f"/sandbox/input/{name[:-7]}"
        if name.endswith("[temp]"):
            return f"/sandbox/temp/{name[:-6]}"
        base_dir = default_dir or "/sandbox/input"
        return f"{base_dir}/{name}"

    def create_caller(self, cls: Any, object_id: str):
        methods = self._services.get(object_id) or self._services.get(getattr(cls, "__name__", object_id))
        if methods is None:
            raise KeyError(object_id)
        return FakeSingletonCaller(methods, self.calls, object_id)


def _clear_proxy_rpcs() -> None:
    from comfy.isolation.proxies.folder_paths_proxy import FolderPathsProxy
    from comfy.isolation.proxies.progress_proxy import ProgressProxy
    from comfy.isolation.proxies.utils_proxy import UtilsProxy

    FolderPathsProxy.clear_rpc()
    ProgressProxy.clear_rpc()
    UtilsProxy.clear_rpc()
    helper_proxy_service = _load_helper_proxy_service()
    if helper_proxy_service is not None:
        helper_proxy_service.clear_rpc()
    model_management_proxy = _load_model_management_proxy()
    if model_management_proxy is not None and hasattr(model_management_proxy, "clear_rpc"):
        model_management_proxy.clear_rpc()


def prepare_sealed_singleton_proxies(fake_rpc: FakeSingletonRPC) -> None:
    os.environ["PYISOLATE_CHILD"] = "1"
    os.environ["PYISOLATE_IMPORT_TORCH"] = "0"
    _clear_proxy_rpcs()

    from comfy.isolation.proxies.folder_paths_proxy import FolderPathsProxy
    from comfy.isolation.proxies.progress_proxy import ProgressProxy
    from comfy.isolation.proxies.utils_proxy import UtilsProxy

    FolderPathsProxy.set_rpc(fake_rpc)
    ProgressProxy.set_rpc(fake_rpc)
    UtilsProxy.set_rpc(fake_rpc)
    helper_proxy_service = _load_helper_proxy_service()
    if helper_proxy_service is not None:
        helper_proxy_service.set_rpc(fake_rpc)
    model_management_proxy = _load_model_management_proxy()
    if model_management_proxy is not None and hasattr(model_management_proxy, "set_rpc"):
        model_management_proxy.set_rpc(fake_rpc)


def reset_forbidden_singleton_modules() -> None:
    for module_name in (
        "folder_paths",
        "comfy.utils",
        "comfy_execution.progress",
    ):
        sys.modules.pop(module_name, None)


class FakeExactRelayCaller:
    def __init__(self, methods: dict[str, Any], transcripts: list[dict[str, Any]], object_id: str):
        self._methods = methods
        self._transcripts = transcripts
        self._object_id = object_id

    def __getattr__(self, name: str):
        if name not in self._methods:
            raise AttributeError(name)

        async def method(*args: Any, **kwargs: Any) -> Any:
            self._transcripts.append(
                {
                    "phase": "child_call",
                    "object_id": self._object_id,
                    "method": name,
                    "args": list(args),
                    "kwargs": dict(kwargs),
                }
            )
            impl = self._methods[name]
            self._transcripts.append(
                {
                    "phase": "host_invocation",
                    "object_id": self._object_id,
                    "method": name,
                    "target": impl["target"],
                    "args": list(args),
                    "kwargs": dict(kwargs),
                }
            )
            result = impl["result"](*args, **kwargs) if callable(impl["result"]) else impl["result"]
            self._transcripts.append(
                {
                    "phase": "result",
                    "object_id": self._object_id,
                    "method": name,
                    "result": result,
                }
            )
            return result

        return method


class FakeExactRelayRPC:
    def __init__(self) -> None:
        self.transcripts: list[dict[str, Any]] = []
        self._device = {"__pyisolate_torch_device__": "cpu"}
        self._services: dict[str, dict[str, Any]] = {
            "FolderPathsProxy": {
                "rpc_get_models_dir": {
                    "target": "folder_paths.models_dir",
                    "result": "/sandbox/models",
                },
                "rpc_get_temp_directory": {
                    "target": "folder_paths.get_temp_directory",
                    "result": "/sandbox/temp",
                },
                "rpc_get_input_directory": {
                    "target": "folder_paths.get_input_directory",
                    "result": "/sandbox/input",
                },
                "rpc_get_output_directory": {
                    "target": "folder_paths.get_output_directory",
                    "result": "/sandbox/output",
                },
                "rpc_get_user_directory": {
                    "target": "folder_paths.get_user_directory",
                    "result": "/sandbox/user",
                },
                "rpc_get_folder_names_and_paths": {
                    "target": "folder_paths.folder_names_and_paths",
                    "result": {
                        "checkpoints": {
                            "paths": ["/sandbox/models/checkpoints"],
                            "extensions": [".ckpt", ".safetensors"],
                        }
                    },
                },
                "rpc_get_extension_mimetypes_cache": {
                    "target": "folder_paths.extension_mimetypes_cache",
                    "result": {"webp": "image"},
                },
                "rpc_get_filename_list_cache": {
                    "target": "folder_paths.filename_list_cache",
                    "result": {},
                },
                "rpc_get_annotated_filepath": {
                    "target": "folder_paths.get_annotated_filepath",
                    "result": lambda name, default_dir=None: FakeSingletonRPC._get_annotated_filepath(name, default_dir),
                },
                "rpc_exists_annotated_filepath": {
                    "target": "folder_paths.exists_annotated_filepath",
                    "result": False,
                },
                "rpc_add_model_folder_path": {
                    "target": "folder_paths.add_model_folder_path",
                    "result": None,
                },
                "rpc_get_folder_paths": {
                    "target": "folder_paths.get_folder_paths",
                    "result": lambda folder_name: [f"/sandbox/models/{folder_name}"],
                },
                "rpc_get_filename_list": {
                    "target": "folder_paths.get_filename_list",
                    "result": lambda folder_name: [f"{folder_name}_fixture.safetensors"],
                },
                "rpc_get_full_path": {
                    "target": "folder_paths.get_full_path",
                    "result": lambda folder_name, filename: f"/sandbox/models/{folder_name}/{filename}",
                },
            },
            "UtilsProxy": {
                "progress_bar_hook": {
                    "target": "comfy.utils.PROGRESS_BAR_HOOK",
                    "result": lambda value, total, preview=None, node_id=None: {
                        "value": value,
                        "total": total,
                        "preview": preview,
                        "node_id": node_id,
                    },
                },
            },
            "ProgressProxy": {
                "rpc_set_progress": {
                    "target": "comfy_execution.progress.get_progress_state().update_progress",
                    "result": None,
                },
            },
            "HelperProxiesService": {
                "rpc_restore_input_types": {
                    "target": "comfy.isolation.proxies.helper_proxies.restore_input_types",
                    "result": lambda raw: raw,
                }
            },
            "ModelManagementProxy": {
                "rpc_call": {
                    "target": "comfy.model_management.*",
                    "result": self._model_management_rpc_call,
                },
            },
        }

    def _model_management_rpc_call(self, method_name: str, args: Any = None, kwargs: Any = None) -> Any:
        device = {"__pyisolate_torch_device__": "cpu"}
        if method_name == "get_torch_device":
            return device
        elif method_name == "get_torch_device_name":
            return "cpu"
        elif method_name == "get_free_memory":
            return 34359738368
        raise AssertionError(f"unexpected exact-relay method {method_name}")

    def create_caller(self, cls: Any, object_id: str):
        methods = self._services.get(object_id) or self._services.get(getattr(cls, "__name__", object_id))
        if methods is None:
            raise KeyError(object_id)
        return FakeExactRelayCaller(methods, self.transcripts, object_id)


def capture_exact_small_proxy_relay() -> dict[str, object]:
    reset_forbidden_singleton_modules()
    fake_rpc = FakeExactRelayRPC()
    previous_child = os.environ.get("PYISOLATE_CHILD")
    previous_import_torch = os.environ.get("PYISOLATE_IMPORT_TORCH")
    try:
        prepare_sealed_singleton_proxies(fake_rpc)

        from comfy.isolation.proxies.folder_paths_proxy import FolderPathsProxy
        from comfy.isolation.proxies.helper_proxies import restore_input_types
        from comfy.isolation.proxies.progress_proxy import ProgressProxy
        from comfy.isolation.proxies.utils_proxy import UtilsProxy

        folder_proxy = FolderPathsProxy()
        utils_proxy = UtilsProxy()
        progress_proxy = ProgressProxy()
        before = set(sys.modules)

        restored = restore_input_types(
            {
                "required": {
                    "image": {"__pyisolate_any_type__": True, "value": "*"},
                }
            }
        )
        folder_path = folder_proxy.get_annotated_filepath("demo.png[input]")
        models_dir = folder_proxy.models_dir
        folder_names_and_paths = folder_proxy.folder_names_and_paths
        asyncio.run(utils_proxy.progress_bar_hook(2, 5, node_id="node-17"))
        progress_proxy.set_progress(1.5, 5.0, node_id="node-17")

        imported = set(sys.modules) - before
        return {
            "mode": "exact_small_proxy_relay",
            "folder_path": folder_path,
            "models_dir": models_dir,
            "folder_names_and_paths": folder_names_and_paths,
            "restored_any_type": str(restored["required"]["image"]),
            "transcripts": fake_rpc.transcripts,
            "modules": sorted(imported),
            "forbidden_matches": matching_modules(FORBIDDEN_EXACT_SMALL_PROXY_MODULES, imported),
        }
    finally:
        _clear_proxy_rpcs()
        if previous_child is None:
            os.environ.pop("PYISOLATE_CHILD", None)
        else:
            os.environ["PYISOLATE_CHILD"] = previous_child
        if previous_import_torch is None:
            os.environ.pop("PYISOLATE_IMPORT_TORCH", None)
        else:
            os.environ["PYISOLATE_IMPORT_TORCH"] = previous_import_torch


class FakeModelManagementExactRelayRPC:
    def __init__(self) -> None:
        self.transcripts: list[dict[str, object]] = []
        self._device = {"__pyisolate_torch_device__": "cpu"}
        self._services: dict[str, dict[str, Any]] = {
            "ModelManagementProxy": {
                "rpc_call": self._rpc_call,
            }
        }

    def create_caller(self, cls: Any, object_id: str):
        methods = self._services.get(object_id) or self._services.get(getattr(cls, "__name__", object_id))
        if methods is None:
            raise KeyError(object_id)
        return _ModelManagementExactRelayCaller(methods)

    def _rpc_call(self, method_name: str, args: Any, kwargs: Any) -> Any:
        self.transcripts.append(
            {
                "phase": "child_call",
                "object_id": "ModelManagementProxy",
                "method": method_name,
                "args": _json_safe(args),
                "kwargs": _json_safe(kwargs),
            }
        )
        target = f"comfy.model_management.{method_name}"
        self.transcripts.append(
            {
                "phase": "host_invocation",
                "object_id": "ModelManagementProxy",
                "method": method_name,
                "target": target,
                "args": _json_safe(args),
                "kwargs": _json_safe(kwargs),
            }
        )
        if method_name == "get_torch_device":
            result = self._device
        elif method_name == "get_torch_device_name":
            result = "cpu"
        elif method_name == "get_free_memory":
            result = 34359738368
        else:
            raise AssertionError(f"unexpected exact-relay method {method_name}")
        self.transcripts.append(
            {
                "phase": "result",
                "object_id": "ModelManagementProxy",
                "method": method_name,
                "result": _json_safe(result),
            }
        )
        return result


class _ModelManagementExactRelayCaller:
    def __init__(self, methods: dict[str, Any]):
        self._methods = methods

    def __getattr__(self, name: str):
        if name not in self._methods:
            raise AttributeError(name)

        async def method(*args: Any, **kwargs: Any) -> Any:
            impl = self._methods[name]
            return impl(*args, **kwargs) if callable(impl) else impl

        return method


def _json_safe(value: Any) -> Any:
    if callable(value):
        return f"<callable {getattr(value, '__name__', 'anonymous')}>"
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(inner) for key, inner in value.items()}
    return value


def capture_model_management_exact_relay() -> dict[str, object]:
    for module_name in FORBIDDEN_MODEL_MANAGEMENT_MODULES:
        sys.modules.pop(module_name, None)

    fake_rpc = FakeModelManagementExactRelayRPC()
    previous_child = os.environ.get("PYISOLATE_CHILD")
    previous_import_torch = os.environ.get("PYISOLATE_IMPORT_TORCH")
    try:
        os.environ["PYISOLATE_CHILD"] = "1"
        os.environ["PYISOLATE_IMPORT_TORCH"] = "0"

        from comfy.isolation.proxies.model_management_proxy import ModelManagementProxy

        if hasattr(ModelManagementProxy, "clear_rpc"):
            ModelManagementProxy.clear_rpc()
        if hasattr(ModelManagementProxy, "set_rpc"):
            ModelManagementProxy.set_rpc(fake_rpc)

        proxy = ModelManagementProxy()
        before = set(sys.modules)
        device = proxy.get_torch_device()
        device_name = proxy.get_torch_device_name(device)
        free_memory = proxy.get_free_memory(device)
        imported = set(sys.modules) - before
        return {
            "mode": "model_management_exact_relay",
            "device": str(device),
            "device_type": getattr(device, "type", None),
            "device_name": device_name,
            "free_memory": free_memory,
            "transcripts": fake_rpc.transcripts,
            "modules": sorted(imported),
            "forbidden_matches": matching_modules(FORBIDDEN_MODEL_MANAGEMENT_MODULES, imported),
        }
    finally:
        model_management_proxy = _load_model_management_proxy()
        if model_management_proxy is not None and hasattr(model_management_proxy, "clear_rpc"):
            model_management_proxy.clear_rpc()
        if previous_child is None:
            os.environ.pop("PYISOLATE_CHILD", None)
        else:
            os.environ["PYISOLATE_CHILD"] = previous_child
        if previous_import_torch is None:
            os.environ.pop("PYISOLATE_IMPORT_TORCH", None)
        else:
            os.environ["PYISOLATE_IMPORT_TORCH"] = previous_import_torch


FORBIDDEN_PROMPT_WEB_MODULES = (
    "server",
    "aiohttp",
    "comfy.isolation.extension_wrapper",
)
FORBIDDEN_EXACT_BOOTSTRAP_MODULES = (
    "comfy.isolation.adapter",
    "folder_paths",
    "comfy.utils",
    "comfy.model_management",
    "server",
    "main",
    "comfy.isolation.extension_wrapper",
)


class _PromptServiceExactRelayCaller:
    def __init__(self, methods: dict[str, Any], transcripts: list[dict[str, Any]], object_id: str):
        self._methods = methods
        self._transcripts = transcripts
        self._object_id = object_id

    def __getattr__(self, name: str):
        if name not in self._methods:
            raise AttributeError(name)

        async def method(*args: Any, **kwargs: Any) -> Any:
            self._transcripts.append(
                {
                    "phase": "child_call",
                    "object_id": self._object_id,
                    "method": name,
                    "args": _json_safe(args),
                    "kwargs": _json_safe(kwargs),
                }
            )
            impl = self._methods[name]
            self._transcripts.append(
                {
                    "phase": "host_invocation",
                    "object_id": self._object_id,
                    "method": name,
                    "target": impl["target"],
                    "args": _json_safe(args),
                    "kwargs": _json_safe(kwargs),
                }
            )
            result = impl["result"](*args, **kwargs) if callable(impl["result"]) else impl["result"]
            self._transcripts.append(
                {
                    "phase": "result",
                    "object_id": self._object_id,
                    "method": name,
                    "result": _json_safe(result),
                }
            )
            return result

        return method


class FakePromptWebRPC:
    def __init__(self) -> None:
        self.transcripts: list[dict[str, Any]] = []
        self._services = {
            "PromptServerService": {
                "ui_send_progress_text": {
                    "target": "server.PromptServer.instance.send_progress_text",
                    "result": None,
                },
                "register_route_rpc": {
                    "target": "server.PromptServer.instance.routes.add_route",
                    "result": None,
                },
            }
        }

    def create_caller(self, cls: Any, object_id: str):
        methods = self._services.get(object_id) or self._services.get(getattr(cls, "__name__", object_id))
        if methods is None:
            raise KeyError(object_id)
        return _PromptServiceExactRelayCaller(methods, self.transcripts, object_id)


class FakeWebDirectoryProxy:
    def __init__(self, transcripts: list[dict[str, Any]]):
        self._transcripts = transcripts

    def get_web_file(self, extension_name: str, relative_path: str) -> dict[str, Any]:
        self._transcripts.append(
            {
                "phase": "child_call",
                "object_id": "WebDirectoryProxy",
                "method": "get_web_file",
                "args": [extension_name, relative_path],
                "kwargs": {},
            }
        )
        self._transcripts.append(
            {
                "phase": "host_invocation",
                "object_id": "WebDirectoryProxy",
                "method": "get_web_file",
                "target": "comfy.isolation.proxies.web_directory_proxy.WebDirectoryProxy.get_web_file",
                "args": [extension_name, relative_path],
                "kwargs": {},
            }
        )
        result = {
            "content": "Y29uc29sZS5sb2coJ2RlbycpOw==",
            "content_type": "application/javascript",
        }
        self._transcripts.append(
            {
                "phase": "result",
                "object_id": "WebDirectoryProxy",
                "method": "get_web_file",
                "result": result,
            }
        )
        return result


def capture_prompt_web_exact_relay() -> dict[str, object]:
    for module_name in FORBIDDEN_PROMPT_WEB_MODULES:
        sys.modules.pop(module_name, None)

    fake_rpc = FakePromptWebRPC()

    from comfy.isolation.proxies.prompt_server_impl import PromptServerStub
    from comfy.isolation.proxies.web_directory_proxy import WebDirectoryCache

    PromptServerStub.set_rpc(fake_rpc)
    stub = PromptServerStub()
    cache = WebDirectoryCache()
    cache.register_proxy("demo_ext", FakeWebDirectoryProxy(fake_rpc.transcripts))

    before = set(sys.modules)

    def demo_handler(_request):
        return {"ok": True}

    stub.send_progress_text("hello", "node-17")
    stub.routes.get("/demo")(demo_handler)
    web_file = cache.get_file("demo_ext", "js/app.js")
    imported = set(sys.modules) - before
    return {
        "mode": "prompt_web_exact_relay",
        "web_file": {
            "content_type": web_file["content_type"] if web_file else None,
            "content": web_file["content"].decode("utf-8") if web_file else None,
        },
        "transcripts": fake_rpc.transcripts,
        "modules": sorted(imported),
        "forbidden_matches": matching_modules(FORBIDDEN_PROMPT_WEB_MODULES, imported),
    }


class FakeExactBootstrapRPC:
    def __init__(self) -> None:
        self.transcripts: list[dict[str, Any]] = []
        self._device = {"__pyisolate_torch_device__": "cpu"}
        self._services: dict[str, dict[str, Any]] = {
            "FolderPathsProxy": FakeExactRelayRPC()._services["FolderPathsProxy"],
            "HelperProxiesService": FakeExactRelayRPC()._services["HelperProxiesService"],
            "ProgressProxy": FakeExactRelayRPC()._services["ProgressProxy"],
            "UtilsProxy": FakeExactRelayRPC()._services["UtilsProxy"],
            "PromptServerService": {
                "ui_send_sync": {
                    "target": "server.PromptServer.instance.send_sync",
                    "result": None,
                },
                "ui_send": {
                    "target": "server.PromptServer.instance.send",
                    "result": None,
                },
                "ui_send_progress_text": {
                    "target": "server.PromptServer.instance.send_progress_text",
                    "result": None,
                },
                "register_route_rpc": {
                    "target": "server.PromptServer.instance.routes.add_route",
                    "result": None,
                },
            },
            "ModelManagementProxy": {
                "rpc_call": self._rpc_call,
            },
        }

    def create_caller(self, cls: Any, object_id: str):
        methods = self._services.get(object_id) or self._services.get(getattr(cls, "__name__", object_id))
        if methods is None:
            raise KeyError(object_id)
        if object_id == "ModelManagementProxy":
            return _ModelManagementExactRelayCaller(methods)
        return _PromptServiceExactRelayCaller(methods, self.transcripts, object_id)

    def _rpc_call(self, method_name: str, args: Any, kwargs: Any) -> Any:
        self.transcripts.append(
            {
                "phase": "child_call",
                "object_id": "ModelManagementProxy",
                "method": method_name,
                "args": _json_safe(args),
                "kwargs": _json_safe(kwargs),
            }
        )
        self.transcripts.append(
            {
                "phase": "host_invocation",
                "object_id": "ModelManagementProxy",
                "method": method_name,
                "target": f"comfy.model_management.{method_name}",
                "args": _json_safe(args),
                "kwargs": _json_safe(kwargs),
            }
        )
        result = self._device if method_name == "get_torch_device" else None
        self.transcripts.append(
            {
                "phase": "result",
                "object_id": "ModelManagementProxy",
                "method": method_name,
                "result": _json_safe(result),
            }
        )
        return result


def capture_exact_proxy_bootstrap_contract() -> dict[str, object]:
    from pyisolate._internal.rpc_protocol import get_child_rpc_instance, set_child_rpc_instance

    from comfy.isolation.adapter import ComfyUIAdapter
    from comfy.isolation.child_hooks import initialize_child_process
    from comfy.isolation.proxies.folder_paths_proxy import FolderPathsProxy
    from comfy.isolation.proxies.helper_proxies import HelperProxiesService
    from comfy.isolation.proxies.model_management_proxy import ModelManagementProxy
    from comfy.isolation.proxies.progress_proxy import ProgressProxy
    from comfy.isolation.proxies.prompt_server_impl import PromptServerStub
    from comfy.isolation.proxies.utils_proxy import UtilsProxy

    host_services = sorted(cls.__name__ for cls in ComfyUIAdapter().provide_rpc_services())

    for module_name in FORBIDDEN_EXACT_BOOTSTRAP_MODULES:
        sys.modules.pop(module_name, None)

    previous_child = os.environ.get("PYISOLATE_CHILD")
    previous_import_torch = os.environ.get("PYISOLATE_IMPORT_TORCH")
    os.environ["PYISOLATE_CHILD"] = "1"
    os.environ["PYISOLATE_IMPORT_TORCH"] = "0"

    _clear_proxy_rpcs()
    if hasattr(PromptServerStub, "clear_rpc"):
        PromptServerStub.clear_rpc()
    else:
        PromptServerStub._rpc = None  # type: ignore[attr-defined]
    fake_rpc = FakeExactBootstrapRPC()
    set_child_rpc_instance(fake_rpc)

    before = set(sys.modules)
    try:
        initialize_child_process()
        imported = set(sys.modules) - before
        matrix = {
            "base.py": {
                "bound": get_child_rpc_instance() is fake_rpc,
                "details": {"child_rpc_instance": get_child_rpc_instance() is fake_rpc},
            },
            "folder_paths_proxy.py": {
                "bound": "FolderPathsProxy" in host_services and FolderPathsProxy._rpc is not None,
                "details": {"host_service": "FolderPathsProxy" in host_services, "child_rpc": FolderPathsProxy._rpc is not None},
            },
            "helper_proxies.py": {
                "bound": "HelperProxiesService" in host_services and HelperProxiesService._rpc is not None,
                "details": {"host_service": "HelperProxiesService" in host_services, "child_rpc": HelperProxiesService._rpc is not None},
            },
            "model_management_proxy.py": {
                "bound": "ModelManagementProxy" in host_services and ModelManagementProxy._rpc is not None,
                "details": {"host_service": "ModelManagementProxy" in host_services, "child_rpc": ModelManagementProxy._rpc is not None},
            },
            "progress_proxy.py": {
                "bound": "ProgressProxy" in host_services and ProgressProxy._rpc is not None,
                "details": {"host_service": "ProgressProxy" in host_services, "child_rpc": ProgressProxy._rpc is not None},
            },
            "prompt_server_impl.py": {
                "bound": "PromptServerService" in host_services and PromptServerStub._rpc is not None,
                "details": {"host_service": "PromptServerService" in host_services, "child_rpc": PromptServerStub._rpc is not None},
            },
            "utils_proxy.py": {
                "bound": "UtilsProxy" in host_services and UtilsProxy._rpc is not None,
                "details": {"host_service": "UtilsProxy" in host_services, "child_rpc": UtilsProxy._rpc is not None},
            },
            "web_directory_proxy.py": {
                "bound": "WebDirectoryProxy" in host_services,
                "details": {"host_service": "WebDirectoryProxy" in host_services},
            },
        }
    finally:
        set_child_rpc_instance(None)
        if previous_child is None:
            os.environ.pop("PYISOLATE_CHILD", None)
        else:
            os.environ["PYISOLATE_CHILD"] = previous_child
        if previous_import_torch is None:
            os.environ.pop("PYISOLATE_IMPORT_TORCH", None)
        else:
            os.environ["PYISOLATE_IMPORT_TORCH"] = previous_import_torch

    omitted = sorted(name for name, status in matrix.items() if not status["bound"])
    return {
        "mode": "exact_proxy_bootstrap_contract",
        "host_services": host_services,
        "matrix": matrix,
        "omitted_proxies": omitted,
        "modules": sorted(imported),
        "forbidden_matches": matching_modules(FORBIDDEN_EXACT_BOOTSTRAP_MODULES, imported),
    }

def capture_sealed_singleton_imports() -> dict[str, object]:
    reset_forbidden_singleton_modules()
    fake_rpc = FakeSingletonRPC()
    previous_child = os.environ.get("PYISOLATE_CHILD")
    previous_import_torch = os.environ.get("PYISOLATE_IMPORT_TORCH")
    before = set(sys.modules)
    try:
        prepare_sealed_singleton_proxies(fake_rpc)

        from comfy.isolation.proxies.folder_paths_proxy import FolderPathsProxy
        from comfy.isolation.proxies.progress_proxy import ProgressProxy
        from comfy.isolation.proxies.utils_proxy import UtilsProxy

        folder_proxy = FolderPathsProxy()
        progress_proxy = ProgressProxy()
        utils_proxy = UtilsProxy()

        folder_path = folder_proxy.get_annotated_filepath("demo.png[input]")
        temp_dir = folder_proxy.get_temp_directory()
        models_dir = folder_proxy.models_dir
        asyncio.run(utils_proxy.progress_bar_hook(2, 5, node_id="node-17"))
        progress_proxy.set_progress(1.5, 5.0, node_id="node-17")

        imported = set(sys.modules) - before
        return {
            "mode": "sealed_singletons",
            "folder_path": folder_path,
            "temp_dir": temp_dir,
            "models_dir": models_dir,
            "rpc_calls": fake_rpc.calls,
            "modules": sorted(imported),
            "forbidden_matches": matching_modules(FORBIDDEN_SEALED_SINGLETON_MODULES, imported),
        }
    finally:
        _clear_proxy_rpcs()
        if previous_child is None:
            os.environ.pop("PYISOLATE_CHILD", None)
        else:
            os.environ["PYISOLATE_CHILD"] = previous_child
        if previous_import_torch is None:
            os.environ.pop("PYISOLATE_IMPORT_TORCH", None)
        else:
            os.environ["PYISOLATE_IMPORT_TORCH"] = previous_import_torch
