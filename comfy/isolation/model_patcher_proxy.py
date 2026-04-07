# pylint: disable=bare-except,consider-using-from-import,import-outside-toplevel,protected-access
# RPC proxy for ModelPatcher (parent process)
from __future__ import annotations

import logging
from typing import Any, Optional, List, Set, Dict, Callable

from comfy.isolation.proxies.base import (
    IS_CHILD_PROCESS,
    BaseProxy,
)
from comfy.isolation.model_patcher_proxy_registry import (
    ModelPatcherRegistry,
    AutoPatcherEjector,
)

logger = logging.getLogger(__name__)


class ModelPatcherProxy(BaseProxy[ModelPatcherRegistry]):
    _registry_class = ModelPatcherRegistry
    __module__ = "comfy.model_patcher"
    _APPLY_MODEL_GUARD_PADDING_BYTES = 32 * 1024 * 1024

    def _spawn_related_proxy(self, instance_id: str) -> "ModelPatcherProxy":
        proxy = ModelPatcherProxy(
            instance_id,
            self._registry,
            manage_lifecycle=not IS_CHILD_PROCESS,
        )
        if getattr(self, "_rpc_caller", None) is not None:
            proxy._rpc_caller = self._rpc_caller
        return proxy

    def _get_rpc(self) -> Any:
        if self._rpc_caller is None:
            from pyisolate._internal.rpc_protocol import get_child_rpc_instance

            rpc = get_child_rpc_instance()
            if rpc is not None:
                self._rpc_caller = rpc.create_caller(
                    self._registry_class, self._registry_class.get_remote_id()
                )
            else:
                self._rpc_caller = self._registry
        return self._rpc_caller

    def get_all_callbacks(self, call_type: str = None) -> Any:
        return self._call_rpc("get_all_callbacks", call_type)

    def get_all_wrappers(self, wrapper_type: str = None) -> Any:
        return self._call_rpc("get_all_wrappers", wrapper_type)

    def _load_list(self, *args, **kwargs) -> Any:
        return self._call_rpc("load_list_internal", *args, **kwargs)

    def prepare_hook_patches_current_keyframe(
        self, t: Any, hook_group: Any, model_options: Any
    ) -> None:
        self._call_rpc(
            "prepare_hook_patches_current_keyframe", t, hook_group, model_options
        )

    def add_hook_patches(
        self,
        hook: Any,
        patches: Any,
        strength_patch: float = 1.0,
        strength_model: float = 1.0,
    ) -> None:
        self._call_rpc(
            "add_hook_patches", hook, patches, strength_patch, strength_model
        )

    def clear_cached_hook_weights(self) -> None:
        self._call_rpc("clear_cached_hook_weights")

    def get_combined_hook_patches(self, hooks: Any) -> Any:
        return self._call_rpc("get_combined_hook_patches", hooks)

    def get_additional_models_with_key(self, key: str) -> Any:
        return self._call_rpc("get_additional_models_with_key", key)

    @property
    def object_patches(self) -> Any:
        return self._call_rpc("get_object_patches")

    @property
    def patches(self) -> Any:
        res = self._call_rpc("get_patches")
        if isinstance(res, dict):
            new_res = {}
            for k, v in res.items():
                new_list = []
                for item in v:
                    if isinstance(item, list):
                        new_list.append(tuple(item))
                    else:
                        new_list.append(item)
                new_res[k] = new_list
            return new_res
        return res

    @property
    def pinned(self) -> Set:
        val = self._call_rpc("get_patcher_attr", "pinned")
        return set(val) if val is not None else set()

    @property
    def hook_patches(self) -> Dict:
        val = self._call_rpc("get_patcher_attr", "hook_patches")
        if val is None:
            return {}
        try:
            from comfy.hooks import _HookRef
            import json

            new_val = {}
            for k, v in val.items():
                if isinstance(k, str):
                    if k.startswith("PYISOLATE_HOOKREF:"):
                        ref_id = k.split(":", 1)[1]
                        h = _HookRef()
                        h._pyisolate_id = ref_id
                        new_val[h] = v
                    elif k.startswith("__pyisolate_key__"):
                        try:
                            json_str = k[len("__pyisolate_key__") :]
                            data = json.loads(json_str)
                            ref_id = None
                            if isinstance(data, list):
                                for item in data:
                                    if (
                                        isinstance(item, list)
                                        and len(item) == 2
                                        and item[0] == "id"
                                    ):
                                        ref_id = item[1]
                                        break
                            if ref_id:
                                h = _HookRef()
                                h._pyisolate_id = ref_id
                                new_val[h] = v
                            else:
                                new_val[k] = v
                        except Exception:
                            new_val[k] = v
                    else:
                        new_val[k] = v
                else:
                    new_val[k] = v
            return new_val
        except ImportError:
            return val

    def set_hook_mode(self, hook_mode: Any) -> None:
        self._call_rpc("set_hook_mode", hook_mode)

    def register_all_hook_patches(
        self,
        hooks: Any,
        target_dict: Any,
        model_options: Any = None,
        registered: Any = None,
    ) -> None:
        self._call_rpc(
            "register_all_hook_patches", hooks, target_dict, model_options, registered
        )

    def is_clone(self, other: Any) -> bool:
        if isinstance(other, ModelPatcherProxy):
            return self._call_rpc("is_clone_by_id", other._instance_id)
        return False

    def clone(self) -> ModelPatcherProxy:
        new_id = self._call_rpc("clone")
        return self._spawn_related_proxy(new_id)

    def clone_has_same_weights(self, clone: Any) -> bool:
        if isinstance(clone, ModelPatcherProxy):
            return self._call_rpc("clone_has_same_weights_by_id", clone._instance_id)
        if not IS_CHILD_PROCESS:
            return self._call_rpc("is_clone", clone)
        return False

    def get_model_object(self, name: str) -> Any:
        return self._call_rpc("get_model_object", name)

    @property
    def model_options(self) -> dict:
        data = self._call_rpc("get_model_options")
        import json

        def _decode_keys(obj):
            if isinstance(obj, dict):
                new_d = {}
                for k, v in obj.items():
                    if isinstance(k, str) and k.startswith("__pyisolate_key__"):
                        try:
                            json_str = k[17:]
                            val = json.loads(json_str)
                            if isinstance(val, list):
                                val = tuple(val)
                            new_d[val] = _decode_keys(v)
                        except:
                            new_d[k] = _decode_keys(v)
                    else:
                        new_d[k] = _decode_keys(v)
                return new_d
            if isinstance(obj, list):
                return [_decode_keys(x) for x in obj]
            return obj

        return _decode_keys(data)

    @model_options.setter
    def model_options(self, value: dict) -> None:
        self._call_rpc("set_model_options", value)

    def apply_hooks(self, hooks: Any) -> Any:
        return self._call_rpc("apply_hooks", hooks)

    def prepare_state(self, timestep: Any) -> Any:
        return self._call_rpc("prepare_state", timestep)

    def restore_hook_patches(self) -> None:
        self._call_rpc("restore_hook_patches")

    def unpatch_hooks(self, whitelist_keys_set: Optional[Set[str]] = None) -> None:
        self._call_rpc("unpatch_hooks", whitelist_keys_set)

    def model_patches_to(self, device: Any) -> Any:
        return self._call_rpc("model_patches_to", device)

    def partially_load(
        self, device: Any, extra_memory: Any, force_patch_weights: bool = False
    ) -> Any:
        return self._call_rpc(
            "partially_load", device, extra_memory, force_patch_weights
        )

    def partially_unload(
        self, device_to: Any, memory_to_free: int = 0, force_patch_weights: bool = False
    ) -> int:
        return self._call_rpc(
            "partially_unload", device_to, memory_to_free, force_patch_weights
        )

    def load(
        self,
        device_to: Any = None,
        lowvram_model_memory: int = 0,
        force_patch_weights: bool = False,
        full_load: bool = False,
    ) -> None:
        self._call_rpc(
            "load", device_to, lowvram_model_memory, force_patch_weights, full_load
        )

    def patch_model(
        self,
        device_to: Any = None,
        lowvram_model_memory: int = 0,
        load_weights: bool = True,
        force_patch_weights: bool = False,
    ) -> Any:
        self._call_rpc(
            "patch_model",
            device_to,
            lowvram_model_memory,
            load_weights,
            force_patch_weights,
        )
        return self

    def unpatch_model(
        self, device_to: Any = None, unpatch_weights: bool = True
    ) -> None:
        self._call_rpc("unpatch_model", device_to, unpatch_weights)

    def detach(self, unpatch_all: bool = True) -> Any:
        self._call_rpc("detach", unpatch_all)
        return self.model

    def _cpu_tensor_bytes(self, obj: Any) -> int:
        import torch

        if isinstance(obj, torch.Tensor):
            if obj.device.type == "cpu":
                return obj.nbytes
            return 0
        if isinstance(obj, dict):
            return sum(self._cpu_tensor_bytes(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(self._cpu_tensor_bytes(v) for v in obj)
        return 0

    def _ensure_apply_model_headroom(self, required_bytes: int) -> bool:
        if required_bytes <= 0:
            return True

        import torch
        import comfy.model_management as model_management

        target_raw = self.load_device
        try:
            if isinstance(target_raw, torch.device):
                target = target_raw
            elif isinstance(target_raw, str):
                target = torch.device(target_raw)
            elif isinstance(target_raw, int):
                target = torch.device(f"cuda:{target_raw}")
            else:
                target = torch.device(target_raw)
        except Exception:
            return True

        if target.type != "cuda":
            return True

        required = required_bytes + self._APPLY_MODEL_GUARD_PADDING_BYTES
        if model_management.get_free_memory(target) >= required:
            return True

        model_management.cleanup_models_gc()
        model_management.cleanup_models()
        model_management.soft_empty_cache()

        if model_management.get_free_memory(target) < required:
            model_management.free_memory(required, target, for_dynamic=True)
            model_management.soft_empty_cache()

        if model_management.get_free_memory(target) < required:
            # Escalate to non-dynamic unloading before dispatching CUDA transfer.
            model_management.free_memory(required, target, for_dynamic=False)
            model_management.soft_empty_cache()

        if model_management.get_free_memory(target) < required:
            model_management.load_models_gpu(
                [self],
                minimum_memory_required=required,
            )

        return model_management.get_free_memory(target) >= required

    def apply_model(self, *args, **kwargs) -> Any:
        import torch

        def _preferred_device() -> Any:
            for value in args:
                if isinstance(value, torch.Tensor):
                    return value.device
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    return value.device
            return None

        def _move_result_to_device(obj: Any, device: Any) -> Any:
            if device is None:
                return obj
            if isinstance(obj, torch.Tensor):
                return obj.to(device) if obj.device != device else obj
            if isinstance(obj, dict):
                return {k: _move_result_to_device(v, device) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_move_result_to_device(v, device) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_move_result_to_device(v, device) for v in obj)
            return obj

        # DynamicVRAM models must keep load/offload decisions in host process.
        # Child-side CUDA staging here can deadlock before first inference RPC.
        if self.is_dynamic():
            out = self._call_rpc("inner_model_apply_model", args, kwargs)
            return _move_result_to_device(out, _preferred_device())

        required_bytes = self._cpu_tensor_bytes(args) + self._cpu_tensor_bytes(kwargs)
        self._ensure_apply_model_headroom(required_bytes)

        def _to_cuda(obj: Any) -> Any:
            if isinstance(obj, torch.Tensor) and obj.device.type == "cpu":
                return obj.to("cuda")
            if isinstance(obj, dict):
                return {k: _to_cuda(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_cuda(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_to_cuda(v) for v in obj)
            return obj

        try:
            args_cuda = _to_cuda(args)
            kwargs_cuda = _to_cuda(kwargs)
        except torch.OutOfMemoryError:
            self._ensure_apply_model_headroom(required_bytes)
            args_cuda = _to_cuda(args)
            kwargs_cuda = _to_cuda(kwargs)

        out = self._call_rpc("inner_model_apply_model", args_cuda, kwargs_cuda)
        return _move_result_to_device(out, _preferred_device())

    def model_state_dict(self, filter_prefix: Optional[str] = None) -> Any:
        keys = self._call_rpc("model_state_dict", filter_prefix)
        return dict.fromkeys(keys, None)

    def add_patches(self, *args: Any, **kwargs: Any) -> Any:
        res = self._call_rpc("add_patches", *args, **kwargs)
        if isinstance(res, list):
            return [tuple(x) if isinstance(x, list) else x for x in res]
        return res

    def get_key_patches(self, filter_prefix: Optional[str] = None) -> Any:
        return self._call_rpc("get_key_patches", filter_prefix)

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        self._call_rpc("patch_weight_to_device", key, device_to, inplace_update)

    def pin_weight_to_device(self, key):
        self._call_rpc("pin_weight_to_device", key)

    def unpin_weight(self, key):
        self._call_rpc("unpin_weight", key)

    def unpin_all_weights(self):
        self._call_rpc("unpin_all_weights")

    def calculate_weight(self, patches, weight, key, intermediate_dtype=None):
        return self._call_rpc(
            "calculate_weight", patches, weight, key, intermediate_dtype
        )

    def inject_model(self) -> None:
        self._call_rpc("inject_model")

    def eject_model(self) -> None:
        self._call_rpc("eject_model")

    def use_ejected(self, skip_and_inject_on_exit_only: bool = False) -> Any:
        return AutoPatcherEjector(
            self, skip_and_inject_on_exit_only=skip_and_inject_on_exit_only
        )

    @property
    def is_injected(self) -> bool:
        return self._call_rpc("get_is_injected")

    @property
    def skip_injection(self) -> bool:
        return self._call_rpc("get_skip_injection")

    @skip_injection.setter
    def skip_injection(self, value: bool) -> None:
        self._call_rpc("set_skip_injection", value)

    def clean_hooks(self) -> None:
        self._call_rpc("clean_hooks")

    def pre_run(self) -> None:
        self._call_rpc("pre_run")

    def cleanup(self) -> None:
        try:
            self._call_rpc("cleanup")
        except Exception:
            logger.debug(
                "ModelPatcherProxy cleanup RPC failed for %s",
                self._instance_id,
                exc_info=True,
            )
        finally:
            super().cleanup()

    @property
    def model(self) -> _InnerModelProxy:
        return _InnerModelProxy(self)

    def __getattr__(self, name: str) -> Any:
        _whitelisted_attrs = {
            "hook_patches_backup",
            "hook_backup",
            "cached_hook_patches",
            "current_hooks",
            "forced_hooks",
            "is_clip",
            "patches_uuid",
            "pinned",
            "attachments",
            "additional_models",
            "injections",
            "hook_patches",
            "model_lowvram",
            "model_loaded_weight_memory",
            "backup",
            "object_patches_backup",
            "weight_wrapper_patches",
            "weight_inplace_update",
            "force_cast_weights",
        }
        if name in _whitelisted_attrs:
            return self._call_rpc("get_patcher_attr", name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def load_lora(
        self,
        lora_path: str,
        strength_model: float,
        clip: Optional[Any] = None,
        strength_clip: float = 1.0,
    ) -> tuple:
        clip_id = None
        if clip is not None:
            clip_id = getattr(clip, "_instance_id", getattr(clip, "_clip_id", None))
        result = self._call_rpc(
            "load_lora", lora_path, strength_model, clip_id, strength_clip
        )
        new_model = None
        if result.get("model_id"):
            new_model = self._spawn_related_proxy(result["model_id"])
        new_clip = None
        if result.get("clip_id"):
            from comfy.isolation.clip_proxy import CLIPProxy

            new_clip = CLIPProxy(result["clip_id"])
        return (new_model, new_clip)

    @property
    def load_device(self) -> Any:
        return self._call_rpc("get_load_device")

    @property
    def offload_device(self) -> Any:
        return self._call_rpc("get_offload_device")

    @property
    def device(self) -> Any:
        return self.load_device

    def current_loaded_device(self) -> Any:
        return self._call_rpc("current_loaded_device")

    @property
    def size(self) -> int:
        return self._call_rpc("get_size")

    def model_size(self) -> Any:
        return self._call_rpc("model_size")

    def loaded_size(self) -> Any:
        return self._call_rpc("loaded_size")

    def get_ram_usage(self) -> int:
        return self._call_rpc("get_ram_usage")

    def lowvram_patch_counter(self) -> int:
        return self._call_rpc("lowvram_patch_counter")

    def memory_required(self, input_shape: Any) -> Any:
        return self._call_rpc("memory_required", input_shape)

    def get_operation_state(self) -> Dict[str, Any]:
        state = self._call_rpc("get_operation_state")
        return state if isinstance(state, dict) else {}

    def wait_for_idle(self, timeout_ms: int = 0) -> bool:
        return bool(self._call_rpc("wait_for_idle", timeout_ms))

    def is_dynamic(self) -> bool:
        return bool(self._call_rpc("is_dynamic"))

    def get_free_memory(self, device: Any) -> Any:
        return self._call_rpc("get_free_memory", device)

    def partially_unload_ram(self, ram_to_unload: int) -> Any:
        return self._call_rpc("partially_unload_ram", ram_to_unload)

    def model_dtype(self) -> Any:
        res = self._call_rpc("model_dtype")
        if isinstance(res, str) and res.startswith("torch."):
            try:
                import torch

                attr = res.split(".")[-1]
                if hasattr(torch, attr):
                    return getattr(torch, attr)
            except ImportError:
                pass
        return res

    @property
    def hook_mode(self) -> Any:
        return self._call_rpc("get_hook_mode")

    @hook_mode.setter
    def hook_mode(self, value: Any) -> None:
        self._call_rpc("set_hook_mode", value)

    def set_model_sampler_cfg_function(
        self, sampler_cfg_function: Any, disable_cfg1_optimization: bool = False
    ) -> None:
        self._call_rpc(
            "set_model_sampler_cfg_function",
            sampler_cfg_function,
            disable_cfg1_optimization,
        )

    def set_model_sampler_post_cfg_function(
        self, post_cfg_function: Any, disable_cfg1_optimization: bool = False
    ) -> None:
        self._call_rpc(
            "set_model_sampler_post_cfg_function",
            post_cfg_function,
            disable_cfg1_optimization,
        )

    def set_model_sampler_pre_cfg_function(
        self, pre_cfg_function: Any, disable_cfg1_optimization: bool = False
    ) -> None:
        self._call_rpc(
            "set_model_sampler_pre_cfg_function",
            pre_cfg_function,
            disable_cfg1_optimization,
        )

    def set_model_sampler_calc_cond_batch_function(self, fn: Any) -> None:
        self._call_rpc("set_model_sampler_calc_cond_batch_function", fn)

    def set_model_unet_function_wrapper(self, unet_wrapper_function: Any) -> None:
        self._call_rpc("set_model_unet_function_wrapper", unet_wrapper_function)

    def set_model_denoise_mask_function(self, denoise_mask_function: Any) -> None:
        self._call_rpc("set_model_denoise_mask_function", denoise_mask_function)

    def set_model_patch(self, patch: Any, name: str) -> None:
        self._call_rpc("set_model_patch", patch, name)

    def set_model_patch_replace(
        self,
        patch: Any,
        name: str,
        block_name: str,
        number: int,
        transformer_index: Optional[int] = None,
    ) -> None:
        self._call_rpc(
            "set_model_patch_replace",
            patch,
            name,
            block_name,
            number,
            transformer_index,
        )

    def set_model_attn1_patch(self, patch: Any) -> None:
        self.set_model_patch(patch, "attn1_patch")

    def set_model_attn2_patch(self, patch: Any) -> None:
        self.set_model_patch(patch, "attn2_patch")

    def set_model_attn1_replace(
        self,
        patch: Any,
        block_name: str,
        number: int,
        transformer_index: Optional[int] = None,
    ) -> None:
        self.set_model_patch_replace(
            patch, "attn1", block_name, number, transformer_index
        )

    def set_model_attn2_replace(
        self,
        patch: Any,
        block_name: str,
        number: int,
        transformer_index: Optional[int] = None,
    ) -> None:
        self.set_model_patch_replace(
            patch, "attn2", block_name, number, transformer_index
        )

    def set_model_attn1_output_patch(self, patch: Any) -> None:
        self.set_model_patch(patch, "attn1_output_patch")

    def set_model_attn2_output_patch(self, patch: Any) -> None:
        self.set_model_patch(patch, "attn2_output_patch")

    def set_model_input_block_patch(self, patch: Any) -> None:
        self.set_model_patch(patch, "input_block_patch")

    def set_model_input_block_patch_after_skip(self, patch: Any) -> None:
        self.set_model_patch(patch, "input_block_patch_after_skip")

    def set_model_output_block_patch(self, patch: Any) -> None:
        self.set_model_patch(patch, "output_block_patch")

    def set_model_emb_patch(self, patch: Any) -> None:
        self.set_model_patch(patch, "emb_patch")

    def set_model_forward_timestep_embed_patch(self, patch: Any) -> None:
        self.set_model_patch(patch, "forward_timestep_embed_patch")

    def set_model_double_block_patch(self, patch: Any) -> None:
        self.set_model_patch(patch, "double_block")

    def set_model_post_input_patch(self, patch: Any) -> None:
        self.set_model_patch(patch, "post_input")

    def set_model_rope_options(
        self,
        scale_x=1.0,
        shift_x=0.0,
        scale_y=1.0,
        shift_y=0.0,
        scale_t=1.0,
        shift_t=0.0,
        **kwargs: Any,
    ) -> None:
        options = {
            "scale_x": scale_x,
            "shift_x": shift_x,
            "scale_y": scale_y,
            "shift_y": shift_y,
            "scale_t": scale_t,
            "shift_t": shift_t,
        }
        options.update(kwargs)
        self._call_rpc("set_model_rope_options", options)

    def set_model_compute_dtype(self, dtype: Any) -> None:
        self._call_rpc("set_model_compute_dtype", dtype)

    def add_object_patch(self, name: str, obj: Any) -> None:
        self._call_rpc("add_object_patch", name, obj)

    def add_weight_wrapper(self, name: str, function: Any) -> None:
        self._call_rpc("add_weight_wrapper", name, function)

    def add_wrapper_with_key(self, wrapper_type: Any, key: str, fn: Any) -> None:
        self._call_rpc("add_wrapper_with_key", wrapper_type, key, fn)

    def add_wrapper(self, wrapper_type: str, wrapper: Callable) -> None:
        self.add_wrapper_with_key(wrapper_type, None, wrapper)

    def remove_wrappers_with_key(self, wrapper_type: str, key: str) -> None:
        self._call_rpc("remove_wrappers_with_key", wrapper_type, key)

    @property
    def wrappers(self) -> Any:
        return self._call_rpc("get_wrappers")

    def add_callback_with_key(self, call_type: str, key: str, callback: Any) -> None:
        self._call_rpc("add_callback_with_key", call_type, key, callback)

    def add_callback(self, call_type: str, callback: Any) -> None:
        self.add_callback_with_key(call_type, None, callback)

    def remove_callbacks_with_key(self, call_type: str, key: str) -> None:
        self._call_rpc("remove_callbacks_with_key", call_type, key)

    @property
    def callbacks(self) -> Any:
        return self._call_rpc("get_callbacks")

    def set_attachments(self, key: str, attachment: Any) -> None:
        self._call_rpc("set_attachments", key, attachment)

    def get_attachment(self, key: str) -> Any:
        return self._call_rpc("get_attachment", key)

    def remove_attachments(self, key: str) -> None:
        self._call_rpc("remove_attachments", key)

    def set_injections(self, key: str, injections: Any) -> None:
        self._call_rpc("set_injections", key, injections)

    def get_injections(self, key: str) -> Any:
        return self._call_rpc("get_injections", key)

    def remove_injections(self, key: str) -> None:
        self._call_rpc("remove_injections", key)

    def set_additional_models(self, key: str, models: Any) -> None:
        ids = [m._instance_id for m in models]
        self._call_rpc("set_additional_models", key, ids)

    def remove_additional_models(self, key: str) -> None:
        self._call_rpc("remove_additional_models", key)

    def get_nested_additional_models(self) -> Any:
        return self._call_rpc("get_nested_additional_models")

    def get_additional_models(self) -> List[ModelPatcherProxy]:
        ids = self._call_rpc("get_additional_models")
        return [self._spawn_related_proxy(mid) for mid in ids]

    def model_patches_models(self) -> Any:
        return self._call_rpc("model_patches_models")

    @property
    def parent(self) -> Any:
        return self._call_rpc("get_parent")

    def model_mmap_residency(self, free: bool = False) -> tuple:
        result = self._call_rpc("model_mmap_residency", free)
        if isinstance(result, list):
            return tuple(result)
        return result

    def pinned_memory_size(self) -> int:
        return self._call_rpc("pinned_memory_size")

    def get_non_dynamic_delegate(self) -> ModelPatcherProxy:
        new_id = self._call_rpc("get_non_dynamic_delegate")
        return self._spawn_related_proxy(new_id)

    def disable_model_cfg1_optimization(self) -> None:
        self._call_rpc("disable_model_cfg1_optimization")

    def set_model_noise_refiner_patch(self, patch: Any) -> None:
        self.set_model_patch(patch, "noise_refiner")


class _InnerModelProxy:
    def __init__(self, parent: ModelPatcherProxy):
        self._parent = parent
        self._model_sampling = None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name == "model_config":
            from types import SimpleNamespace

            data = self._parent._call_rpc("get_inner_model_attr", name)
            if isinstance(data, dict):
                return SimpleNamespace(**data)
            return data
        if name in (
            "latent_format",
            "model_type",
            "current_weight_patches_uuid",
        ):
            return self._parent._call_rpc("get_inner_model_attr", name)
        if name == "load_device":
            return self._parent._call_rpc("get_inner_model_attr", "load_device")
        if name == "device":
            return self._parent._call_rpc("get_inner_model_attr", "device")
        if name == "current_patcher":
            proxy = ModelPatcherProxy(
                self._parent._instance_id,
                self._parent._registry,
                manage_lifecycle=False,
            )
            if getattr(self._parent, "_rpc_caller", None) is not None:
                proxy._rpc_caller = self._parent._rpc_caller
            return proxy
        if name == "model_sampling":
            if self._model_sampling is None:
                self._model_sampling = self._parent._call_rpc(
                    "get_model_object", "model_sampling"
                )
            return self._model_sampling
        if name == "extra_conds_shapes":
            return lambda *a, **k: self._parent._call_rpc(
                "inner_model_extra_conds_shapes", a, k
            )
        if name == "extra_conds":
            return lambda *a, **k: self._parent._call_rpc(
                "inner_model_extra_conds", a, k
            )
        if name == "memory_required":
            return lambda *a, **k: self._parent._call_rpc(
                "inner_model_memory_required", a, k
            )
        if name == "apply_model":
            # Delegate to parent's method to get the CPU->CUDA optimization
            return self._parent.apply_model
        if name == "process_latent_in":
            return lambda *a, **k: self._parent._call_rpc("process_latent_in", a, k)
        if name == "process_latent_out":
            return lambda *a, **k: self._parent._call_rpc("process_latent_out", a, k)
        if name == "scale_latent_inpaint":
            return lambda *a, **k: self._parent._call_rpc("scale_latent_inpaint", a, k)
        if name == "diffusion_model":
            return self._parent._call_rpc("get_inner_model_attr", "diffusion_model")
        raise AttributeError(f"'{name}' not supported on isolated InnerModel")
