# pylint: disable=consider-using-from-import,import-outside-toplevel,no-member
from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Set, TYPE_CHECKING

from .proxies.helper_proxies import restore_input_types
from .shm_forensics import scan_shm_forensics

_IMPORT_TORCH = os.environ.get("PYISOLATE_IMPORT_TORCH", "1") == "1"

_ComfyNodeInternal = object
latest_io = None

if _IMPORT_TORCH:
    from comfy_api.internal import _ComfyNodeInternal
    from comfy_api.latest import _io as latest_io

if TYPE_CHECKING:
    from .extension_wrapper import ComfyNodeExtension

LOG_PREFIX = "]["
_PRE_EXEC_MIN_FREE_VRAM_BYTES = 2 * 1024 * 1024 * 1024


class _RemoteObjectRegistryCaller:
    def __init__(self, extension: Any) -> None:
        self._extension = extension

    def __getattr__(self, method_name: str) -> Any:
        async def _call(instance_id: str, *args: Any, **kwargs: Any) -> Any:
            return await self._extension.call_remote_object_method(
                instance_id,
                method_name,
                *args,
                **kwargs,
            )

        return _call


def _wrap_remote_handles_as_host_proxies(value: Any, extension: Any) -> Any:
    from pyisolate._internal.remote_handle import RemoteObjectHandle

    if isinstance(value, RemoteObjectHandle):
        if value.type_name == "ModelPatcher":
            from comfy.isolation.model_patcher_proxy import ModelPatcherProxy

            proxy = ModelPatcherProxy(value.object_id, manage_lifecycle=False)
            proxy._rpc_caller = _RemoteObjectRegistryCaller(extension)  # type: ignore[attr-defined]
            proxy._pyisolate_remote_handle = value  # type: ignore[attr-defined]
            return proxy
        if value.type_name == "VAE":
            from comfy.isolation.vae_proxy import VAEProxy

            proxy = VAEProxy(value.object_id, manage_lifecycle=False)
            proxy._rpc_caller = _RemoteObjectRegistryCaller(extension)  # type: ignore[attr-defined]
            proxy._pyisolate_remote_handle = value  # type: ignore[attr-defined]
            return proxy
        if value.type_name == "CLIP":
            from comfy.isolation.clip_proxy import CLIPProxy

            proxy = CLIPProxy(value.object_id, manage_lifecycle=False)
            proxy._rpc_caller = _RemoteObjectRegistryCaller(extension)  # type: ignore[attr-defined]
            proxy._pyisolate_remote_handle = value  # type: ignore[attr-defined]
            return proxy
        if value.type_name == "ModelSampling":
            from comfy.isolation.model_sampling_proxy import ModelSamplingProxy

            proxy = ModelSamplingProxy(value.object_id, manage_lifecycle=False)
            proxy._rpc_caller = _RemoteObjectRegistryCaller(extension)  # type: ignore[attr-defined]
            proxy._pyisolate_remote_handle = value  # type: ignore[attr-defined]
            return proxy
        return value

    if isinstance(value, dict):
        return {
            k: _wrap_remote_handles_as_host_proxies(v, extension) for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        wrapped = [_wrap_remote_handles_as_host_proxies(item, extension) for item in value]
        return type(value)(wrapped)

    return value


def _resource_snapshot() -> Dict[str, int]:
    fd_count = -1
    shm_sender_files = 0
    try:
        fd_count = len(os.listdir("/proc/self/fd"))
    except Exception:
        pass
    try:
        shm_root = Path("/dev/shm")
        if shm_root.exists():
            prefix = f"torch_{os.getpid()}_"
            shm_sender_files = sum(1 for _ in shm_root.glob(f"{prefix}*"))
    except Exception:
        pass
    return {"fd_count": fd_count, "shm_sender_files": shm_sender_files}


def _tensor_transport_summary(value: Any) -> Dict[str, int]:
    summary: Dict[str, int] = {
        "tensor_count": 0,
        "cpu_tensors": 0,
        "cuda_tensors": 0,
        "shared_cpu_tensors": 0,
        "tensor_bytes": 0,
    }
    try:
        import torch
    except Exception:
        return summary

    def visit(node: Any) -> None:
        if isinstance(node, torch.Tensor):
            summary["tensor_count"] += 1
            summary["tensor_bytes"] += int(node.numel() * node.element_size())
            if node.device.type == "cpu":
                summary["cpu_tensors"] += 1
                if node.is_shared():
                    summary["shared_cpu_tensors"] += 1
            elif node.device.type == "cuda":
                summary["cuda_tensors"] += 1
            return
        if isinstance(node, dict):
            for v in node.values():
                visit(v)
            return
        if isinstance(node, (list, tuple)):
            for v in node:
                visit(v)

    visit(value)
    return summary


def _extract_hidden_unique_id(inputs: Dict[str, Any]) -> str | None:
    for key, value in inputs.items():
        key_text = str(key)
        if "unique_id" in key_text:
            return str(value)
    return None


def _flush_tensor_transport_state(marker: str, logger: logging.Logger) -> None:
    try:
        from pyisolate import flush_tensor_keeper  # type: ignore[attr-defined]
    except Exception:
        return
    if not callable(flush_tensor_keeper):
        return
    flushed = flush_tensor_keeper()
    if flushed > 0:
        logger.debug(
            "%s %s flush_tensor_keeper released=%d", LOG_PREFIX, marker, flushed
        )


def _relieve_host_vram_pressure(marker: str, logger: logging.Logger) -> None:
    import comfy.model_management as model_management

    model_management.cleanup_models_gc()
    model_management.cleanup_models()

    device = model_management.get_torch_device()
    if not hasattr(device, "type") or device.type == "cpu":
        return

    required = max(
        model_management.minimum_inference_memory(),
        _PRE_EXEC_MIN_FREE_VRAM_BYTES,
    )
    if model_management.get_free_memory(device) < required:
        model_management.free_memory(required, device, for_dynamic=True)
        if model_management.get_free_memory(device) < required:
            model_management.free_memory(required, device, for_dynamic=False)
        model_management.cleanup_models()
        model_management.soft_empty_cache()
        logger.debug("%s %s free_memory target=%d", LOG_PREFIX, marker, required)


def _detach_shared_cpu_tensors(value: Any) -> Any:
    try:
        import torch
    except Exception:
        return value

    if isinstance(value, torch.Tensor):
        if value.device.type == "cpu" and value.is_shared():
            clone = value.clone()
            if value.requires_grad:
                clone.requires_grad_(True)
            return clone
        return value
    if isinstance(value, list):
        return [_detach_shared_cpu_tensors(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_detach_shared_cpu_tensors(v) for v in value)
    if isinstance(value, dict):
        return {k: _detach_shared_cpu_tensors(v) for k, v in value.items()}
    return value


def build_stub_class(
    node_name: str,
    info: Dict[str, object],
    extension: "ComfyNodeExtension",
    running_extensions: Dict[str, "ComfyNodeExtension"],
    logger: logging.Logger,
) -> type:
    if latest_io is None:
        raise RuntimeError("comfy_api.latest._io is required to build isolation stubs")
    is_v3 = bool(info.get("is_v3", False))
    function_name = "_pyisolate_execute"
    restored_input_types = restore_input_types(info.get("input_types", {}))

    async def _execute(self, **inputs):
        from comfy.isolation import _RUNNING_EXTENSIONS

        # Update BOTH the local dict AND the module-level dict
        running_extensions[extension.name] = extension
        _RUNNING_EXTENSIONS[extension.name] = extension
        prev_child = None
        node_unique_id = _extract_hidden_unique_id(inputs)
        summary = _tensor_transport_summary(inputs)
        resources = _resource_snapshot()
        logger.debug(
            "%s ISO:execute_start ext=%s node=%s uid=%s",
            LOG_PREFIX,
            extension.name,
            node_name,
            node_unique_id or "-",
        )
        logger.debug(
            "%s ISO:execute_start ext=%s node=%s uid=%s tensors=%d cpu=%d cuda=%d shared_cpu=%d bytes=%d fds=%d sender_shm=%d",
            LOG_PREFIX,
            extension.name,
            node_name,
            node_unique_id or "-",
            summary["tensor_count"],
            summary["cpu_tensors"],
            summary["cuda_tensors"],
            summary["shared_cpu_tensors"],
            summary["tensor_bytes"],
            resources["fd_count"],
            resources["shm_sender_files"],
        )
        scan_shm_forensics("RUNTIME:execute_start", refresh_model_context=True)
        try:
            if os.environ.get("PYISOLATE_CHILD") != "1":
                _relieve_host_vram_pressure("RUNTIME:pre_execute", logger)
                scan_shm_forensics("RUNTIME:pre_execute", refresh_model_context=True)
            from pyisolate._internal.model_serialization import (
                serialize_for_isolation,
                deserialize_from_isolation,
            )

            prev_child = os.environ.pop("PYISOLATE_CHILD", None)
            logger.debug(
                "%s ISO:serialize_start ext=%s node=%s uid=%s",
                LOG_PREFIX,
                extension.name,
                node_name,
                node_unique_id or "-",
            )
            # Unwrap NodeOutput-like dicts before serialization.
            # OUTPUT_NODE nodes return {"ui": {...}, "result": (outputs...)}
            # and the executor may pass this dict as input to downstream nodes.
            unwrapped_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, dict) and "result" in v and ("ui" in v or "__node_output__" in v):
                    result = v.get("result")
                    if isinstance(result, (tuple, list)) and len(result) > 0:
                        unwrapped_inputs[k] = result[0]
                    else:
                        unwrapped_inputs[k] = result
                else:
                    unwrapped_inputs[k] = v
            serialized = serialize_for_isolation(unwrapped_inputs)
            logger.debug(
                "%s ISO:serialize_done ext=%s node=%s uid=%s",
                LOG_PREFIX,
                extension.name,
                node_name,
                node_unique_id or "-",
            )
            logger.debug(
                "%s ISO:dispatch_start ext=%s node=%s uid=%s",
                LOG_PREFIX,
                extension.name,
                node_name,
                node_unique_id or "-",
            )
            result = await extension.execute_node(node_name, **serialized)
            logger.debug(
                "%s ISO:dispatch_done ext=%s node=%s uid=%s",
                LOG_PREFIX,
                extension.name,
                node_name,
                node_unique_id or "-",
            )
            # Reconstruct NodeOutput if the child serialized one
            if isinstance(result, dict) and result.get("__node_output__"):
                from comfy_api.latest import io as latest_io
                args_raw = result.get("args", ())
                deserialized_args = await deserialize_from_isolation(args_raw, extension)
                deserialized_args = _wrap_remote_handles_as_host_proxies(
                    deserialized_args, extension
                )
                deserialized_args = _detach_shared_cpu_tensors(deserialized_args)
                ui_raw = result.get("ui")
                deserialized_ui = None
                if ui_raw is not None:
                    deserialized_ui = await deserialize_from_isolation(ui_raw, extension)
                    deserialized_ui = _wrap_remote_handles_as_host_proxies(
                        deserialized_ui, extension
                    )
                    deserialized_ui = _detach_shared_cpu_tensors(deserialized_ui)
                scan_shm_forensics("RUNTIME:post_execute", refresh_model_context=True)
                return latest_io.NodeOutput(
                    *deserialized_args,
                    ui=deserialized_ui,
                    expand=result.get("expand"),
                    block_execution=result.get("block_execution"),
                )
            # OUTPUT_NODE: if sealed worker returned a tuple/list whose first
            # element is a {"ui": ...} dict, unwrap it for the executor.
            if (isinstance(result, (tuple, list)) and len(result) == 1
                    and isinstance(result[0], dict) and "ui" in result[0]):
                return result[0]
            deserialized = await deserialize_from_isolation(result, extension)
            deserialized = _wrap_remote_handles_as_host_proxies(deserialized, extension)
            scan_shm_forensics("RUNTIME:post_execute", refresh_model_context=True)
            return _detach_shared_cpu_tensors(deserialized)
        except ImportError:
            return await extension.execute_node(node_name, **inputs)
        except Exception:
            logger.exception(
                "%s ISO:execute_error ext=%s node=%s uid=%s",
                LOG_PREFIX,
                extension.name,
                node_name,
                node_unique_id or "-",
            )
            raise
        finally:
            if prev_child is not None:
                os.environ["PYISOLATE_CHILD"] = prev_child
            logger.debug(
                "%s ISO:execute_end ext=%s node=%s uid=%s",
                LOG_PREFIX,
                extension.name,
                node_name,
                node_unique_id or "-",
            )
            scan_shm_forensics("RUNTIME:execute_end", refresh_model_context=True)

    def _input_types(
        cls,
        include_hidden: bool = True,
        return_schema: bool = False,
        live_inputs: Any = None,
    ):
        if not is_v3:
            return restored_input_types

        inputs_copy = copy.deepcopy(restored_input_types)
        if not include_hidden:
            inputs_copy.pop("hidden", None)

        v3_data: Dict[str, Any] = {"hidden_inputs": {}}
        dynamic = inputs_copy.pop("dynamic_paths", None)
        if dynamic is not None:
            v3_data["dynamic_paths"] = dynamic

        if return_schema:
            hidden_vals = info.get("hidden", []) or []
            hidden_enums = []
            for h in hidden_vals:
                try:
                    hidden_enums.append(latest_io.Hidden(h))
                except Exception:
                    hidden_enums.append(h)

            class SchemaProxy:
                hidden = hidden_enums

            return inputs_copy, SchemaProxy, v3_data
        return inputs_copy

    def _validate_class(cls):
        return True

    def _get_node_info_v1(cls):
        node_info = copy.deepcopy(info.get("schema_v1", {}))
        relative_python_module = node_info.get("python_module")
        if not isinstance(relative_python_module, str) or not relative_python_module:
            relative_python_module = f"custom_nodes.{extension.name}"
        node_info["python_module"] = relative_python_module
        return node_info

    def _get_base_class(cls):
        return latest_io.ComfyNode

    attributes: Dict[str, object] = {
        "FUNCTION": function_name,
        "CATEGORY": info.get("category", ""),
        "OUTPUT_NODE": info.get("output_node", False),
        "RETURN_TYPES": tuple(info.get("return_types", ()) or ()),
        "RETURN_NAMES": info.get("return_names"),
        function_name: _execute,
        "_pyisolate_extension": extension,
        "_pyisolate_node_name": node_name,
        "INPUT_TYPES": classmethod(_input_types),
    }

    output_is_list = info.get("output_is_list")
    if output_is_list is not None:
        attributes["OUTPUT_IS_LIST"] = tuple(output_is_list)

    if is_v3:
        attributes["VALIDATE_CLASS"] = classmethod(_validate_class)
        attributes["GET_NODE_INFO_V1"] = classmethod(_get_node_info_v1)
        attributes["GET_BASE_CLASS"] = classmethod(_get_base_class)
        attributes["DESCRIPTION"] = info.get("description", "")
        attributes["EXPERIMENTAL"] = info.get("experimental", False)
        attributes["DEPRECATED"] = info.get("deprecated", False)
        attributes["API_NODE"] = info.get("api_node", False)
        attributes["NOT_IDEMPOTENT"] = info.get("not_idempotent", False)
        attributes["ACCEPT_ALL_INPUTS"] = info.get("accept_all_inputs", False)
        attributes["_ACCEPT_ALL_INPUTS"] = info.get("accept_all_inputs", False)
        attributes["INPUT_IS_LIST"] = info.get("input_is_list", False)

    class_name = f"PyIsolate_{node_name}".replace(" ", "_")
    bases = (_ComfyNodeInternal,) if is_v3 else ()
    stub_cls = type(class_name, bases, attributes)

    if is_v3:
        try:
            stub_cls.VALIDATE_CLASS()
        except Exception as e:
            logger.error("%s VALIDATE_CLASS failed: %s - %s", LOG_PREFIX, node_name, e)

    return stub_cls


def get_class_types_for_extension(
    extension_name: str,
    running_extensions: Dict[str, "ComfyNodeExtension"],
    specs: List[Any],
) -> Set[str]:
    extension = running_extensions.get(extension_name)
    if not extension:
        return set()

    ext_path = Path(extension.module_path)
    class_types = set()
    for spec in specs:
        if spec.module_path.resolve() == ext_path.resolve():
            class_types.add(spec.node_name)
    return class_types


__all__ = ["build_stub_class", "get_class_types_for_extension"]
