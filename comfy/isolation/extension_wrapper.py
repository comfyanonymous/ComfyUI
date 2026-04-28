# pylint: disable=consider-using-from-import,cyclic-import,import-outside-toplevel,logging-fstring-interpolation,protected-access,wrong-import-position
from __future__ import annotations

import asyncio
import torch


class AttrDict(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def copy(self):
        return AttrDict(super().copy())


import importlib
import inspect
import json
import logging
import os
import sys
import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

from pyisolate import ExtensionBase

from comfy_api.internal import _ComfyNodeInternal

LOG_PREFIX = "]["
V3_DISCOVERY_TIMEOUT = 30
_PRE_EXEC_MIN_FREE_VRAM_BYTES = 2 * 1024 * 1024 * 1024

logger = logging.getLogger(__name__)


def _run_prestartup_web_copy(module: Any, module_dir: str, web_dir_path: str) -> None:
    """Run the web asset copy step that prestartup_script.py used to do.

    If the module's web/ directory is empty and the module had a
    prestartup_script.py that copied assets from pip packages, this
    function replicates that work inside the child process.

    Generic pattern: reads _PRESTARTUP_WEB_COPY from the module if
    defined, otherwise falls back to detecting common asset packages.
    """
    import shutil

    # Already populated — nothing to do
    if os.path.isdir(web_dir_path) and any(os.scandir(web_dir_path)):
        return

    os.makedirs(web_dir_path, exist_ok=True)

    # Try module-defined copy spec first (generic hook for any node pack)
    copy_spec = getattr(module, "_PRESTARTUP_WEB_COPY", None)
    if copy_spec is not None and callable(copy_spec):
        try:
            copy_spec(web_dir_path)
            logger.info(
                "%s Ran _PRESTARTUP_WEB_COPY for %s", LOG_PREFIX, module_dir
            )
            return
        except Exception as e:
            logger.warning(
                "%s _PRESTARTUP_WEB_COPY failed for %s: %s",
                LOG_PREFIX, module_dir, e,
            )

    # Fallback: detect comfy_3d_viewers and run copy_viewer()
    try:
        from comfy_3d_viewers import copy_viewer, VIEWER_FILES
        viewers = list(VIEWER_FILES.keys())
        for viewer in viewers:
            try:
                copy_viewer(viewer, web_dir_path)
            except Exception:
                pass
        if any(os.scandir(web_dir_path)):
            logger.info(
                "%s Copied %d viewer types from comfy_3d_viewers to %s",
                LOG_PREFIX, len(viewers), web_dir_path,
            )
    except ImportError:
        pass

    # Fallback: detect comfy_dynamic_widgets
    try:
        from comfy_dynamic_widgets import get_js_path
        src = os.path.realpath(get_js_path())
        if os.path.exists(src):
            dst_dir = os.path.join(web_dir_path, "js")
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, "dynamic_widgets.js")
            shutil.copy2(src, dst)
    except ImportError:
        pass


def _read_extension_name(module_dir: str) -> str:
    """Read extension name from pyproject.toml, falling back to directory name."""
    pyproject = os.path.join(module_dir, "pyproject.toml")
    if os.path.exists(pyproject):
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        try:
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
            name = data.get("project", {}).get("name")
            if name:
                return name
        except Exception:
            pass
    return os.path.basename(module_dir)


def _flush_tensor_transport_state(marker: str) -> int:
    try:
        from pyisolate import flush_tensor_keeper  # type: ignore[attr-defined]
    except Exception:
        return 0
    if not callable(flush_tensor_keeper):
        return 0
    flushed = flush_tensor_keeper()
    if flushed > 0:
        logger.debug(
            "%s %s flush_tensor_keeper released=%d", LOG_PREFIX, marker, flushed
        )
    return flushed


def _relieve_child_vram_pressure(marker: str) -> None:
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


def _sanitize_for_transport(value):
    primitives = (str, int, float, bool, type(None))
    if isinstance(value, primitives):
        return value

    cls_name = value.__class__.__name__
    if cls_name == "FlexibleOptionalInputType":
        return {
            "__pyisolate_flexible_optional__": True,
            "type": _sanitize_for_transport(getattr(value, "type", "*")),
        }
    if cls_name == "AnyType":
        return {"__pyisolate_any_type__": True, "value": str(value)}
    if cls_name == "ByPassTypeTuple":
        return {
            "__pyisolate_bypass_tuple__": [
                _sanitize_for_transport(v) for v in tuple(value)
            ]
        }

    if isinstance(value, dict):
        return {k: _sanitize_for_transport(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return {"__pyisolate_tuple__": [_sanitize_for_transport(v) for v in value]}
    if isinstance(value, list):
        return [_sanitize_for_transport(v) for v in value]

    return str(value)


# Re-export RemoteObjectHandle from pyisolate for backward compatibility
# The canonical definition is now in pyisolate._internal.remote_handle
from pyisolate._internal.remote_handle import RemoteObjectHandle  # noqa: E402,F401


class ComfyNodeExtension(ExtensionBase):
    def __init__(self) -> None:
        super().__init__()
        self.node_classes: Dict[str, type] = {}
        self.display_names: Dict[str, str] = {}
        self.node_instances: Dict[str, Any] = {}
        self.remote_objects: Dict[str, Any] = {}
        self._route_handlers: Dict[str, Any] = {}
        self._module: Any = None
        self._metadata_ready = asyncio.Event()

    async def on_module_loaded(self, module: Any) -> None:
        try:
            self._module = module

            # Registries are initialized in host_hooks.py initialize_host_process()
            # They auto-register via ProxiedSingleton when instantiated
            # NO additional setup required here - if a registry is missing from host_hooks, it WILL fail

            self.node_classes = getattr(module, "NODE_CLASS_MAPPINGS", {}) or {}
            self.display_names = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {}) or {}
            self._register_module_routes(module)

            # Register web directory with WebDirectoryProxy (child-side)
            web_dir_attr = getattr(module, "WEB_DIRECTORY", None)
            if web_dir_attr is not None:
                module_dir = os.path.dirname(os.path.abspath(module.__file__))
                web_dir_path = os.path.abspath(os.path.join(module_dir, web_dir_attr))
                ext_name = _read_extension_name(module_dir)

                # If web dir is empty, run the copy step that prestartup_script.py did
                _run_prestartup_web_copy(module, module_dir, web_dir_path)

                if os.path.isdir(web_dir_path) and any(os.scandir(web_dir_path)):
                    from comfy.isolation.proxies.web_directory_proxy import WebDirectoryProxy
                    WebDirectoryProxy.register_web_dir(ext_name, web_dir_path)

            try:
                from comfy_api.latest import ComfyExtension

                for name, obj in inspect.getmembers(module):
                    if not (
                        inspect.isclass(obj)
                        and issubclass(obj, ComfyExtension)
                        and obj is not ComfyExtension
                    ):
                        continue
                    if not obj.__module__.startswith(module.__name__):
                        continue
                    try:
                        ext_instance = obj()
                        try:
                            await asyncio.wait_for(
                                ext_instance.on_load(), timeout=V3_DISCOVERY_TIMEOUT
                            )
                        except asyncio.TimeoutError:
                            logger.error(
                                "%s V3 Extension %s timed out in on_load()",
                                LOG_PREFIX,
                                name,
                            )
                            continue
                        try:
                            v3_nodes = await asyncio.wait_for(
                                ext_instance.get_node_list(), timeout=V3_DISCOVERY_TIMEOUT
                            )
                        except asyncio.TimeoutError:
                            logger.error(
                                "%s V3 Extension %s timed out in get_node_list()",
                                LOG_PREFIX,
                                name,
                            )
                            continue
                        for node_cls in v3_nodes:
                            if hasattr(node_cls, "GET_SCHEMA"):
                                schema = node_cls.GET_SCHEMA()
                                self.node_classes[schema.node_id] = node_cls
                                if schema.display_name:
                                    self.display_names[schema.node_id] = schema.display_name
                    except Exception as e:
                        logger.error("%s V3 Extension %s failed: %s", LOG_PREFIX, name, e)
            except ImportError:
                pass

            module_name = getattr(module, "__name__", "isolated_nodes")
            for node_cls in self.node_classes.values():
                if hasattr(node_cls, "__module__") and "/" in str(node_cls.__module__):
                    node_cls.__module__ = module_name

            self.node_instances = {}
        finally:
            self._metadata_ready.set()

    def _register_module_routes(self, module: Any) -> None:
        """Bridge legacy module-level ROUTES declarations into isolated routing."""
        routes = getattr(module, "ROUTES", None) or []
        if not routes:
            return

        from comfy.isolation.proxies.prompt_server_impl import PromptServerStub

        prompt_server = PromptServerStub()
        route_table = getattr(prompt_server, "routes", None)
        if route_table is None:
            logger.warning("%s Route registration unavailable for %s", LOG_PREFIX, module)
            return

        for route_spec in routes:
            if not isinstance(route_spec, dict):
                logger.warning("%s Ignoring non-dict ROUTES entry: %r", LOG_PREFIX, route_spec)
                continue

            method = str(route_spec.get("method", "")).strip().upper()
            path = str(route_spec.get("path", "")).strip()
            handler_ref = route_spec.get("handler")
            if not method or not path:
                logger.warning("%s Ignoring incomplete route spec: %r", LOG_PREFIX, route_spec)
                continue

            if isinstance(handler_ref, str):
                handler = getattr(module, handler_ref, None)
            else:
                handler = handler_ref
            if not callable(handler):
                logger.warning(
                    "%s Ignoring route with missing handler %r for %s %s",
                    LOG_PREFIX,
                    handler_ref,
                    method,
                    path,
                )
                continue

            decorator = getattr(route_table, method.lower(), None)
            if not callable(decorator):
                logger.warning("%s Unsupported route method %s for %s", LOG_PREFIX, method, path)
                continue

            decorator(path)(handler)
            self._route_handlers[f"{method} {path}"] = handler
            logger.info("%s buffered legacy route %s %s", LOG_PREFIX, method, path)

    async def list_nodes(self) -> Dict[str, str]:
        await asyncio.wait_for(
            self._metadata_ready.wait(), timeout=V3_DISCOVERY_TIMEOUT
        )
        return {name: self.display_names.get(name, name) for name in self.node_classes}

    async def get_node_info(self, node_name: str) -> Dict[str, Any]:
        return await self.get_node_details(node_name)

    async def get_node_details(self, node_name: str) -> Dict[str, Any]:
        await asyncio.wait_for(
            self._metadata_ready.wait(), timeout=V3_DISCOVERY_TIMEOUT
        )
        node_cls = self._get_node_class(node_name)
        is_v3 = issubclass(node_cls, _ComfyNodeInternal)

        input_types_raw = (
            node_cls.INPUT_TYPES() if hasattr(node_cls, "INPUT_TYPES") else {}
        )
        output_is_list = getattr(node_cls, "OUTPUT_IS_LIST", None)
        if output_is_list is not None:
            output_is_list = tuple(bool(x) for x in output_is_list)

        details: Dict[str, Any] = {
            "input_types": _sanitize_for_transport(input_types_raw),
            "return_types": tuple(
                str(t) for t in getattr(node_cls, "RETURN_TYPES", ())
            ),
            "return_names": getattr(node_cls, "RETURN_NAMES", None),
            "function": str(getattr(node_cls, "FUNCTION", "execute")),
            "category": str(getattr(node_cls, "CATEGORY", "")),
            "output_node": bool(getattr(node_cls, "OUTPUT_NODE", False)),
            "output_is_list": output_is_list,
            "is_v3": is_v3,
        }

        if is_v3:
            try:
                schema = node_cls.GET_SCHEMA()
                schema_v1 = asdict(schema.get_v1_info(node_cls))
                try:
                    schema_v3 = asdict(schema.get_v3_info(node_cls))
                except (AttributeError, TypeError):
                    schema_v3 = self._build_schema_v3_fallback(schema)
                details.update(
                    {
                        "schema_v1": schema_v1,
                        "schema_v3": schema_v3,
                        "hidden": [h.value for h in (schema.hidden or [])],
                        "description": getattr(schema, "description", ""),
                        "deprecated": bool(getattr(node_cls, "DEPRECATED", False)),
                        "experimental": bool(getattr(node_cls, "EXPERIMENTAL", False)),
                        "api_node": bool(getattr(node_cls, "API_NODE", False)),
                        "input_is_list": bool(
                            getattr(node_cls, "INPUT_IS_LIST", False)
                        ),
                        "not_idempotent": bool(
                            getattr(node_cls, "NOT_IDEMPOTENT", False)
                        ),
                        "accept_all_inputs": bool(
                            getattr(node_cls, "ACCEPT_ALL_INPUTS", False)
                        ),
                    }
                )
            except Exception as exc:
                logger.warning(
                    "%s V3 schema serialization failed for %s: %s",
                    LOG_PREFIX,
                    node_name,
                    exc,
                )
        return details

    def _build_schema_v3_fallback(self, schema) -> Dict[str, Any]:
        input_dict: Dict[str, Any] = {}
        output_dict: Dict[str, Any] = {}
        hidden_list: List[str] = []

        if getattr(schema, "inputs", None):
            for inp in schema.inputs:
                self._add_schema_io_v3(inp, input_dict)
        if getattr(schema, "outputs", None):
            for out in schema.outputs:
                self._add_schema_io_v3(out, output_dict)
        if getattr(schema, "hidden", None):
            for h in schema.hidden:
                hidden_list.append(getattr(h, "value", str(h)))

        return {
            "input": input_dict,
            "output": output_dict,
            "hidden": hidden_list,
            "name": getattr(schema, "node_id", None),
            "display_name": getattr(schema, "display_name", None),
            "description": getattr(schema, "description", None),
            "category": getattr(schema, "category", None),
            "output_node": getattr(schema, "is_output_node", False),
            "deprecated": getattr(schema, "is_deprecated", False),
            "experimental": getattr(schema, "is_experimental", False),
            "api_node": getattr(schema, "is_api_node", False),
        }

    def _add_schema_io_v3(self, io_obj: Any, target: Dict[str, Any]) -> None:
        io_id = getattr(io_obj, "id", None)
        if io_id is None:
            return

        io_type_fn = getattr(io_obj, "get_io_type", None)
        io_type = (
            io_type_fn() if callable(io_type_fn) else getattr(io_obj, "io_type", None)
        )

        as_dict_fn = getattr(io_obj, "as_dict", None)
        payload = as_dict_fn() if callable(as_dict_fn) else {}

        target[str(io_id)] = (io_type, payload)

    async def get_input_types(self, node_name: str) -> Dict[str, Any]:
        node_cls = self._get_node_class(node_name)
        if hasattr(node_cls, "INPUT_TYPES"):
            return node_cls.INPUT_TYPES()
        return {}

    async def execute_node(self, node_name: str, **inputs: Any) -> Tuple[Any, ...]:
        logger.debug(
            "%s ISO:child_execute_start ext=%s node=%s input_keys=%d",
            LOG_PREFIX,
            getattr(self, "name", "?"),
            node_name,
            len(inputs),
        )
        if os.environ.get("PYISOLATE_CHILD") == "1":
            _relieve_child_vram_pressure("EXT:pre_execute")

        resolved_inputs = self._resolve_remote_objects(inputs)

        instance = self._get_node_instance(node_name)
        node_cls = self._get_node_class(node_name)

        # V3 API nodes expect hidden parameters in cls.hidden, not as kwargs
        # Hidden params come through RPC as string keys like "Hidden.prompt"
        from comfy_api.latest._io import Hidden, HiddenHolder

        # Map string representations back to Hidden enum keys
        hidden_string_map = {
            "Hidden.unique_id": Hidden.unique_id,
            "Hidden.prompt": Hidden.prompt,
            "Hidden.extra_pnginfo": Hidden.extra_pnginfo,
            "Hidden.dynprompt": Hidden.dynprompt,
            "Hidden.auth_token_comfy_org": Hidden.auth_token_comfy_org,
            "Hidden.api_key_comfy_org": Hidden.api_key_comfy_org,
            # Uppercase enum VALUE forms — V3 execution engine passes these
            "UNIQUE_ID": Hidden.unique_id,
            "PROMPT": Hidden.prompt,
            "EXTRA_PNGINFO": Hidden.extra_pnginfo,
            "DYNPROMPT": Hidden.dynprompt,
            "AUTH_TOKEN_COMFY_ORG": Hidden.auth_token_comfy_org,
            "API_KEY_COMFY_ORG": Hidden.api_key_comfy_org,
        }

        # Find and extract hidden parameters (both enum and string form)
        hidden_found = {}
        keys_to_remove = []

        for key in list(resolved_inputs.keys()):
            # Check string form first (from RPC serialization)
            if key in hidden_string_map:
                hidden_found[hidden_string_map[key]] = resolved_inputs[key]
                keys_to_remove.append(key)
            # Also check enum form (direct calls)
            elif isinstance(key, Hidden):
                hidden_found[key] = resolved_inputs[key]
                keys_to_remove.append(key)

        # Remove hidden params from kwargs
        for key in keys_to_remove:
            resolved_inputs.pop(key)

        # Set hidden on node class if any hidden params found
        if hidden_found:
            if not hasattr(node_cls, "hidden") or node_cls.hidden is None:
                node_cls.hidden = HiddenHolder.from_dict(hidden_found)
            else:
                # Update existing hidden holder
                for key, value in hidden_found.items():
                    setattr(node_cls.hidden, key.value.lower(), value)

        # INPUT_IS_LIST: ComfyUI's executor passes all inputs as lists when this
        # flag is set.  The isolation RPC delivers unwrapped values, so we must
        # wrap each input in a single-element list to match the contract.
        if getattr(node_cls, "INPUT_IS_LIST", False):
            resolved_inputs = {k: [v] for k, v in resolved_inputs.items()}

        function_name = getattr(node_cls, "FUNCTION", "execute")
        if not hasattr(instance, function_name):
            raise AttributeError(f"Node {node_name} missing callable '{function_name}'")

        handler = getattr(instance, function_name)

        try:
            import torch
            if asyncio.iscoroutinefunction(handler):
                with torch.inference_mode():
                    result = await handler(**resolved_inputs)
            else:
                import functools

                def _run_with_inference_mode(**kwargs):
                    with torch.inference_mode():
                        return handler(**kwargs)

                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None, functools.partial(_run_with_inference_mode, **resolved_inputs)
                )
        except Exception:
            logger.exception(
                "%s ISO:child_execute_error ext=%s node=%s",
                LOG_PREFIX,
                getattr(self, "name", "?"),
                node_name,
            )
            raise

        if type(result).__name__ == "NodeOutput":
            node_output_dict = {
                "__node_output__": True,
                "args": self._wrap_unpicklable_objects(result.args),
            }
            if result.ui is not None:
                node_output_dict["ui"] = self._wrap_unpicklable_objects(result.ui)
            if getattr(result, "expand", None) is not None:
                node_output_dict["expand"] = result.expand
            if getattr(result, "block_execution", None) is not None:
                node_output_dict["block_execution"] = result.block_execution
            return node_output_dict
        if self._is_comfy_protocol_return(result):
            wrapped = self._wrap_unpicklable_objects(result)
            return wrapped

        if not isinstance(result, tuple):
            result = (result,)
        wrapped = self._wrap_unpicklable_objects(result)
        return wrapped

    async def flush_pending_routes(self) -> int:
        """Flush buffered route registrations to host via RPC. Called by host after node discovery."""
        from comfy.isolation.proxies.prompt_server_impl import PromptServerStub
        return await PromptServerStub.flush_child_routes()

    async def flush_transport_state(self) -> int:
        if os.environ.get("PYISOLATE_CHILD") != "1":
            return 0
        logger.debug(
            "%s ISO:child_flush_start ext=%s", LOG_PREFIX, getattr(self, "name", "?")
        )
        flushed = _flush_tensor_transport_state("EXT:workflow_end")
        try:
            from comfy.isolation.model_patcher_proxy_registry import (
                ModelPatcherRegistry,
            )

            registry = ModelPatcherRegistry()
            removed = registry.sweep_pending_cleanup()
            if removed > 0:
                logger.debug(
                    "%s EXT:workflow_end registry sweep removed=%d", LOG_PREFIX, removed
                )
        except Exception:
            logger.debug(
                "%s EXT:workflow_end registry sweep failed", LOG_PREFIX, exc_info=True
            )
        logger.debug(
            "%s ISO:child_flush_done ext=%s flushed=%d",
            LOG_PREFIX,
            getattr(self, "name", "?"),
            flushed,
        )
        return flushed

    async def get_remote_object(self, object_id: str) -> Any:
        """Retrieve a remote object by ID for host-side deserialization."""
        if object_id not in self.remote_objects:
            raise KeyError(f"Remote object {object_id} not found")

        return self.remote_objects[object_id]

    def _store_remote_object_handle(self, obj: Any) -> RemoteObjectHandle:
        object_id = str(uuid.uuid4())
        self.remote_objects[object_id] = obj
        return RemoteObjectHandle(object_id, type(obj).__name__)

    async def call_remote_object_method(
        self,
        object_id: str,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Invoke a method or attribute-backed accessor on a child-owned object."""
        obj = await self.get_remote_object(object_id)

        if method_name == "get_patcher_attr":
            return getattr(obj, args[0])
        if method_name == "get_model_options":
            return getattr(obj, "model_options")
        if method_name == "set_model_options":
            setattr(obj, "model_options", args[0])
            return None
        if method_name == "get_object_patches":
            return getattr(obj, "object_patches")
        if method_name == "get_patches":
            return getattr(obj, "patches")
        if method_name == "get_wrappers":
            return getattr(obj, "wrappers")
        if method_name == "get_callbacks":
            return getattr(obj, "callbacks")
        if method_name == "get_load_device":
            return getattr(obj, "load_device")
        if method_name == "get_offload_device":
            return getattr(obj, "offload_device")
        if method_name == "get_hook_mode":
            return getattr(obj, "hook_mode")
        if method_name == "get_parent":
            parent = getattr(obj, "parent", None)
            if parent is None:
                return None
            return self._store_remote_object_handle(parent)
        if method_name == "get_inner_model_attr":
            attr_name = args[0]
            if hasattr(obj.model, attr_name):
                return getattr(obj.model, attr_name)
            if hasattr(obj, attr_name):
                return getattr(obj, attr_name)
            return None
        if method_name == "inner_model_apply_model":
            return obj.model.apply_model(*args[0], **args[1])
        if method_name == "inner_model_extra_conds_shapes":
            return obj.model.extra_conds_shapes(*args[0], **args[1])
        if method_name == "inner_model_extra_conds":
            return obj.model.extra_conds(*args[0], **args[1])
        if method_name == "inner_model_memory_required":
            return obj.model.memory_required(*args[0], **args[1])
        if method_name == "process_latent_in":
            return obj.model.process_latent_in(*args[0], **args[1])
        if method_name == "process_latent_out":
            return obj.model.process_latent_out(*args[0], **args[1])
        if method_name == "scale_latent_inpaint":
            return obj.model.scale_latent_inpaint(*args[0], **args[1])
        if method_name.startswith("get_"):
            attr_name = method_name[4:]
            if hasattr(obj, attr_name):
                return getattr(obj, attr_name)

        target = getattr(obj, method_name)
        if callable(target):
            result = target(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            if type(result).__name__ == "ModelPatcher":
                return self._store_remote_object_handle(result)
            return result
        if args or kwargs:
            raise TypeError(f"{method_name} is not callable on remote object {object_id}")
        return target

    def _wrap_unpicklable_objects(self, data: Any) -> Any:
        if isinstance(data, (str, int, float, bool, type(None))):
            return data
        if isinstance(data, torch.Tensor):
            tensor = data.detach() if data.requires_grad else data
            if os.environ.get("PYISOLATE_CHILD") == "1" and tensor.device.type != "cpu":
                return tensor.cpu()
            return tensor

        # Special-case clip vision outputs: preserve attribute access by packing fields
        if hasattr(data, "penultimate_hidden_states") or hasattr(
            data, "last_hidden_state"
        ):
            fields = {}
            for attr in (
                "penultimate_hidden_states",
                "last_hidden_state",
                "image_embeds",
                "text_embeds",
            ):
                if hasattr(data, attr):
                    try:
                        fields[attr] = self._wrap_unpicklable_objects(
                            getattr(data, attr)
                        )
                    except Exception:
                        pass
            if fields:
                return {"__pyisolate_attribute_container__": True, "data": fields}

        # Avoid converting arbitrary objects with stateful methods (models, etc.)
        # They will be handled via RemoteObjectHandle below.

        type_name = type(data).__name__
        if type_name == "ModelPatcherProxy":
            return {"__type__": "ModelPatcherRef", "model_id": data._instance_id}
        if type_name == "CLIPProxy":
            return {"__type__": "CLIPRef", "clip_id": data._instance_id}
        if type_name == "VAEProxy":
            return {"__type__": "VAERef", "vae_id": data._instance_id}
        if type_name == "ModelSamplingProxy":
            return {"__type__": "ModelSamplingRef", "ms_id": data._instance_id}

        if isinstance(data, (list, tuple)):
            wrapped = [self._wrap_unpicklable_objects(item) for item in data]
            return tuple(wrapped) if isinstance(data, tuple) else wrapped
        if isinstance(data, dict):
            converted_dict = {
                k: self._wrap_unpicklable_objects(v) for k, v in data.items()
            }
            return {"__pyisolate_attrdict__": True, "data": converted_dict}

        from pyisolate._internal.serialization_registry import SerializerRegistry

        registry = SerializerRegistry.get_instance()
        if registry.is_data_type(type_name):
            serializer = registry.get_serializer(type_name)
            if serializer:
                return serializer(data)

        return self._store_remote_object_handle(data)

    def _resolve_remote_objects(self, data: Any) -> Any:
        if isinstance(data, RemoteObjectHandle):
            if data.object_id not in self.remote_objects:
                raise KeyError(f"Remote object {data.object_id} not found")
            return self.remote_objects[data.object_id]

        if isinstance(data, dict):
            ref_type = data.get("__type__")
            if ref_type in ("CLIPRef", "ModelPatcherRef", "VAERef"):
                from pyisolate._internal.model_serialization import (
                    deserialize_proxy_result,
                )

                return deserialize_proxy_result(data)
            if ref_type == "ModelSamplingRef":
                from pyisolate._internal.model_serialization import (
                    deserialize_proxy_result,
                )

                return deserialize_proxy_result(data)
            return {k: self._resolve_remote_objects(v) for k, v in data.items()}

        if isinstance(data, (list, tuple)):
            resolved = [self._resolve_remote_objects(item) for item in data]
            return tuple(resolved) if isinstance(data, tuple) else resolved
        return data

    def _get_node_class(self, node_name: str) -> type:
        if node_name not in self.node_classes:
            raise KeyError(f"Unknown node: {node_name}")
        return self.node_classes[node_name]

    def _get_node_instance(self, node_name: str) -> Any:
        if node_name not in self.node_instances:
            if node_name not in self.node_classes:
                raise KeyError(f"Unknown node: {node_name}")
            self.node_instances[node_name] = self.node_classes[node_name]()
        return self.node_instances[node_name]

    async def before_module_loaded(self) -> None:
        try:
            from comfy.isolation import initialize_proxies

            initialize_proxies()
        except Exception as e:
            logger.error(
                "%s before_module_loaded initialize_proxies FAILED: %s", LOG_PREFIX, e
            )

        await super().before_module_loaded()
        try:
            from comfy_api.latest import ComfyAPI_latest
            from .proxies.progress_proxy import ProgressProxy

            ComfyAPI_latest.Execution = ProgressProxy
            # ComfyAPI_latest.execution = ProgressProxy()  # Eliminated to avoid Singleton collision
            # fp_proxy = FolderPathsProxy()                 # Eliminated to avoid Singleton collision
            # latest_ui.folder_paths = fp_proxy
            # latest_resources.folder_paths = fp_proxy
        except Exception:
            pass

    async def call_route_handler(
        self,
        handler_module: str,
        handler_func: str,
        request_data: Dict[str, Any],
    ) -> Any:
        cache_key = f"{handler_module}.{handler_func}"
        if cache_key not in self._route_handlers:
            if self._module is not None and hasattr(self._module, "__file__"):
                node_dir = os.path.dirname(self._module.__file__)
                if node_dir not in sys.path:
                    sys.path.insert(0, node_dir)
            try:
                module = importlib.import_module(handler_module)
                self._route_handlers[cache_key] = getattr(module, handler_func)
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Route handler not found: {cache_key}") from e

        handler = self._route_handlers[cache_key]
        mock_request = MockRequest(request_data)

        if asyncio.iscoroutinefunction(handler):
            result = await handler(mock_request)
        else:
            result = handler(mock_request)
        return self._serialize_response(result)

    def _is_comfy_protocol_return(self, result: Any) -> bool:
        """
        Check if the result matches the ComfyUI 'Protocol Return' schema.

        A Protocol Return is a dictionary containing specific reserved keys that
        ComfyUI's execution engine interprets as instructions (UI updates,
        Workflow expansion, etc.) rather than purely data outputs.

        Schema:
           - Must be a dict
           - Must contain at least one of: 'ui', 'result', 'expand'
        """
        if not isinstance(result, dict):
            return False
        return any(key in result for key in ("ui", "result", "expand"))

    def _serialize_response(self, response: Any) -> Dict[str, Any]:
        if response is None:
            return {"type": "text", "body": "", "status": 204}
        if isinstance(response, dict):
            return {"type": "json", "body": response, "status": 200}
        if isinstance(response, str):
            return {"type": "text", "body": response, "status": 200}
        if hasattr(response, "text") and hasattr(response, "status"):
            return {
                "type": "text",
                "body": response.text
                if hasattr(response, "text")
                else str(response.body),
                "status": response.status,
                "headers": dict(response.headers)
                if hasattr(response, "headers")
                else {},
            }
        if hasattr(response, "body") and hasattr(response, "status"):
            body = response.body
            if isinstance(body, bytes):
                try:
                    return {
                        "type": "text",
                        "body": body.decode("utf-8"),
                        "status": response.status,
                    }
                except UnicodeDecodeError:
                    return {
                        "type": "binary",
                        "body": body.hex(),
                        "status": response.status,
                    }
            return {"type": "json", "body": body, "status": response.status}
        return {"type": "text", "body": str(response), "status": 200}


class MockRequest:
    def __init__(self, data: Dict[str, Any]):
        self.method = data.get("method", "GET")
        self.path = data.get("path", "/")
        self.query = data.get("query", {})
        self._body = data.get("body", {})
        self._text = data.get("text", "")
        self.headers = data.get("headers", {})
        self.content_type = data.get(
            "content_type", self.headers.get("Content-Type", "application/json")
        )
        self.match_info = data.get("match_info", {})

    async def json(self) -> Any:
        if isinstance(self._body, dict):
            return self._body
        if isinstance(self._body, str):
            return json.loads(self._body)
        return {}

    async def post(self) -> Dict[str, Any]:
        if isinstance(self._body, dict):
            return self._body
        return {}

    async def text(self) -> str:
        if self._text:
            return self._text
        if isinstance(self._body, str):
            return self._body
        if isinstance(self._body, dict):
            return json.dumps(self._body)
        return ""

    async def read(self) -> bytes:
        return (await self.text()).encode("utf-8")
