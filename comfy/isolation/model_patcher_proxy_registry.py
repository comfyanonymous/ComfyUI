# pylint: disable=import-outside-toplevel,logging-fstring-interpolation,protected-access,unused-import
# RPC server for ModelPatcher isolation (child process)
from __future__ import annotations

import asyncio
import gc
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional, List

try:
    from comfy.model_patcher import AutoPatcherEjector
except ImportError:

    class AutoPatcherEjector:
        def __init__(self, model, skip_and_inject_on_exit_only=False):
            self.model = model
            self.skip_and_inject_on_exit_only = skip_and_inject_on_exit_only
            self.prev_skip_injection = False
            self.was_injected = False

        def __enter__(self):
            self.was_injected = False
            self.prev_skip_injection = self.model.skip_injection
            if self.skip_and_inject_on_exit_only:
                self.model.skip_injection = True
            if self.model.is_injected:
                self.model.eject_model()
                self.was_injected = True

        def __exit__(self, *args):
            if self.skip_and_inject_on_exit_only:
                self.model.skip_injection = self.prev_skip_injection
                self.model.inject_model()
            if self.was_injected and not self.model.skip_injection:
                self.model.inject_model()
            self.model.skip_injection = self.prev_skip_injection


from comfy.isolation.proxies.base import (
    BaseRegistry,
    detach_if_grad,
)

logger = logging.getLogger(__name__)


@dataclass
class _OperationState:
    lease: threading.Lock = field(default_factory=threading.Lock)
    active_count: int = 0
    active_by_method: dict[str, int] = field(default_factory=dict)
    total_operations: int = 0
    last_method: Optional[str] = None
    last_started_ts: Optional[float] = None
    last_ended_ts: Optional[float] = None
    last_elapsed_ms: Optional[float] = None
    last_error: Optional[str] = None
    last_thread_id: Optional[int] = None
    last_loop_id: Optional[int] = None


class ModelPatcherRegistry(BaseRegistry[Any]):
    _type_prefix = "model"

    def __init__(self) -> None:
        super().__init__()
        self._pending_cleanup_ids: set[str] = set()
        self._operation_states: dict[str, _OperationState] = {}
        self._operation_state_cv = threading.Condition(self._lock)

    def _get_or_create_operation_state(self, instance_id: str) -> _OperationState:
        state = self._operation_states.get(instance_id)
        if state is None:
            state = _OperationState()
            self._operation_states[instance_id] = state
        return state

    def _begin_operation(self, instance_id: str, method_name: str) -> tuple[float, float]:
        start_epoch = time.time()
        start_perf = time.perf_counter()
        with self._operation_state_cv:
            state = self._get_or_create_operation_state(instance_id)
            state.active_count += 1
            state.active_by_method[method_name] = (
                state.active_by_method.get(method_name, 0) + 1
            )
            state.total_operations += 1
            state.last_method = method_name
            state.last_started_ts = start_epoch
            state.last_thread_id = threading.get_ident()
            try:
                state.last_loop_id = id(asyncio.get_running_loop())
            except RuntimeError:
                state.last_loop_id = None
        logger.debug(
            "ISO:registry_op_start instance_id=%s method=%s start_ts=%.6f thread=%s loop=%s",
            instance_id,
            method_name,
            start_epoch,
            threading.get_ident(),
            state.last_loop_id,
        )
        return start_epoch, start_perf

    def _end_operation(
        self,
        instance_id: str,
        method_name: str,
        start_perf: float,
        error: Optional[BaseException] = None,
    ) -> None:
        end_epoch = time.time()
        elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
        with self._operation_state_cv:
            state = self._get_or_create_operation_state(instance_id)
            state.active_count = max(0, state.active_count - 1)
            if method_name in state.active_by_method:
                remaining = state.active_by_method[method_name] - 1
                if remaining <= 0:
                    state.active_by_method.pop(method_name, None)
                else:
                    state.active_by_method[method_name] = remaining
            state.last_ended_ts = end_epoch
            state.last_elapsed_ms = elapsed_ms
            state.last_error = None if error is None else repr(error)
            if state.active_count == 0:
                self._operation_state_cv.notify_all()
        logger.debug(
            "ISO:registry_op_end instance_id=%s method=%s end_ts=%.6f elapsed_ms=%.3f error=%s",
            instance_id,
            method_name,
            end_epoch,
            elapsed_ms,
            None if error is None else type(error).__name__,
        )

    def _run_operation_with_lease(self, instance_id: str, method_name: str, fn):
        with self._operation_state_cv:
            state = self._get_or_create_operation_state(instance_id)
            lease = state.lease
        with lease:
            _, start_perf = self._begin_operation(instance_id, method_name)
            try:
                result = fn()
            except Exception as exc:
                self._end_operation(instance_id, method_name, start_perf, error=exc)
                raise
            self._end_operation(instance_id, method_name, start_perf)
            return result

    def _snapshot_operation_state(self, instance_id: str) -> dict[str, Any]:
        with self._operation_state_cv:
            state = self._operation_states.get(instance_id)
            if state is None:
                return {
                    "instance_id": instance_id,
                    "active_count": 0,
                    "active_methods": [],
                    "total_operations": 0,
                    "last_method": None,
                    "last_started_ts": None,
                    "last_ended_ts": None,
                    "last_elapsed_ms": None,
                    "last_error": None,
                    "last_thread_id": None,
                    "last_loop_id": None,
                }
            return {
                "instance_id": instance_id,
                "active_count": state.active_count,
                "active_methods": sorted(state.active_by_method.keys()),
                "total_operations": state.total_operations,
                "last_method": state.last_method,
                "last_started_ts": state.last_started_ts,
                "last_ended_ts": state.last_ended_ts,
                "last_elapsed_ms": state.last_elapsed_ms,
                "last_error": state.last_error,
                "last_thread_id": state.last_thread_id,
                "last_loop_id": state.last_loop_id,
            }

    def unregister_sync(self, instance_id: str) -> None:
        with self._operation_state_cv:
            instance = self._registry.pop(instance_id, None)
            if instance is not None:
                self._id_map.pop(id(instance), None)
            self._pending_cleanup_ids.discard(instance_id)
            self._operation_states.pop(instance_id, None)
            self._operation_state_cv.notify_all()

    async def get_operation_state(self, instance_id: str) -> dict[str, Any]:
        return self._snapshot_operation_state(instance_id)

    async def get_all_operation_states(self) -> dict[str, dict[str, Any]]:
        with self._operation_state_cv:
            ids = sorted(self._operation_states.keys())
        return {instance_id: self._snapshot_operation_state(instance_id) for instance_id in ids}

    async def wait_for_idle(self, instance_id: str, timeout_ms: int = 0) -> bool:
        timeout_s = None if timeout_ms <= 0 else (timeout_ms / 1000.0)
        deadline = None if timeout_s is None else (time.monotonic() + timeout_s)
        with self._operation_state_cv:
            while True:
                active = self._operation_states.get(instance_id)
                if active is None or active.active_count == 0:
                    return True
                if deadline is None:
                    self._operation_state_cv.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._operation_state_cv.wait(timeout=remaining)

    async def wait_all_idle(self, timeout_ms: int = 0) -> bool:
        timeout_s = None if timeout_ms <= 0 else (timeout_ms / 1000.0)
        deadline = None if timeout_s is None else (time.monotonic() + timeout_s)
        with self._operation_state_cv:
            while True:
                has_active = any(
                    state.active_count > 0 for state in self._operation_states.values()
                )
                if not has_active:
                    return True
                if deadline is None:
                    self._operation_state_cv.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._operation_state_cv.wait(timeout=remaining)

    async def clone(self, instance_id: str) -> str:
        instance = self._get_instance(instance_id)
        new_model = instance.clone()
        return self.register(new_model)

    async def is_clone(self, instance_id: str, other: Any) -> bool:
        instance = self._get_instance(instance_id)
        if hasattr(other, "model"):
            return instance.is_clone(other)
        return False

    async def get_model_object(self, instance_id: str, name: str) -> Any:
        instance = self._get_instance(instance_id)
        if name == "model":
            return f"<ModelObject: {type(instance.model).__name__}>"
        result = instance.get_model_object(name)
        if name == "model_sampling":
            # Return inline serialization so the child reconstructs the real
            # class with correct isinstance behavior. Returning a
            # ModelSamplingProxy breaks isinstance checks (e.g.
            # offset_first_sigma_for_snr in k_diffusion/sampling.py:173).
            return self._serialize_model_sampling_inline(result)

        return detach_if_grad(result)

    @staticmethod
    def _serialize_model_sampling_inline(obj: Any) -> dict:
        """Serialize a ModelSampling object as inline data for the child to reconstruct."""
        import torch
        import base64
        import io as _io

        bases = []
        for base in type(obj).__mro__:
            if base.__module__ == "comfy.model_sampling" and base.__name__ != "object":
                bases.append(base.__name__)

        sd = obj.state_dict()
        sd_serialized = {}
        for k, v in sd.items():
            buf = _io.BytesIO()
            torch.save(v, buf)
            sd_serialized[k] = base64.b64encode(buf.getvalue()).decode("ascii")

        plain_attrs = {}
        for k, v in obj.__dict__.items():
            if k.startswith("_"):
                continue
            if isinstance(v, (bool, int, float, str)):
                plain_attrs[k] = v

        return {
            "__type__": "ModelSamplingInline",
            "bases": bases,
            "state_dict": sd_serialized,
            "attrs": plain_attrs,
        }

    async def get_model_options(self, instance_id: str) -> dict:
        instance = self._get_instance(instance_id)
        import copy

        opts = copy.deepcopy(instance.model_options)
        return self._sanitize_rpc_result(opts)

    async def set_model_options(self, instance_id: str, options: dict) -> None:
        self._get_instance(instance_id).model_options = options

    async def get_patcher_attr(self, instance_id: str, name: str) -> Any:
        return self._sanitize_rpc_result(
            getattr(self._get_instance(instance_id), name, None)
        )

    async def model_state_dict(self, instance_id: str, filter_prefix=None) -> Any:
        instance = self._get_instance(instance_id)
        sd_keys = instance.model.state_dict().keys()
        return dict.fromkeys(sd_keys, None)

    def _sanitize_rpc_result(self, obj, seen=None):
        if seen is None:
            seen = set()
        if obj is None:
            return None
        if isinstance(obj, (bool, int, float, str)):
            if isinstance(obj, str) and len(obj) > 500000:
                return f"<Truncated String len={len(obj)}>"
            return obj
        obj_id = id(obj)
        if obj_id in seen:
            return None
        seen.add(obj_id)
        if isinstance(obj, (list, tuple)):
            return [self._sanitize_rpc_result(x, seen) for x in obj]
        if isinstance(obj, set):
            return [self._sanitize_rpc_result(x, seen) for x in obj]
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                if isinstance(k, tuple):
                    import json

                    try:
                        key_str = "__pyisolate_key__" + json.dumps(list(k))
                        new_dict[key_str] = self._sanitize_rpc_result(v, seen)
                    except Exception:
                        new_dict[str(k)] = self._sanitize_rpc_result(v, seen)
                else:
                    new_dict[str(k)] = self._sanitize_rpc_result(v, seen)
            return new_dict
        if (
            hasattr(obj, "__dict__")
            and not hasattr(obj, "__get__")
            and not hasattr(obj, "__call__")
        ):
            return self._sanitize_rpc_result(obj.__dict__, seen)
        if hasattr(obj, "items") and hasattr(obj, "get"):
            return {str(k): self._sanitize_rpc_result(v, seen) for k, v in obj.items()}
        return None

    async def get_load_device(self, instance_id: str) -> Any:
        return self._get_instance(instance_id).load_device

    async def get_offload_device(self, instance_id: str) -> Any:
        return self._get_instance(instance_id).offload_device

    async def current_loaded_device(self, instance_id: str) -> Any:
        return self._get_instance(instance_id).current_loaded_device()

    async def get_size(self, instance_id: str) -> int:
        return self._get_instance(instance_id).size

    async def model_size(self, instance_id: str) -> Any:
        return self._get_instance(instance_id).model_size()

    async def loaded_size(self, instance_id: str) -> Any:
        return self._get_instance(instance_id).loaded_size()

    async def get_ram_usage(self, instance_id: str) -> int:
        return self._get_instance(instance_id).get_ram_usage()

    async def model_mmap_residency(self, instance_id: str, free: bool = False) -> tuple:
        return self._get_instance(instance_id).model_mmap_residency(free=free)

    async def pinned_memory_size(self, instance_id: str) -> int:
        return self._get_instance(instance_id).pinned_memory_size()

    async def get_non_dynamic_delegate(self, instance_id: str) -> str:
        instance = self._get_instance(instance_id)
        delegate = instance.get_non_dynamic_delegate()
        return self.register(delegate)

    async def disable_model_cfg1_optimization(self, instance_id: str) -> None:
        self._get_instance(instance_id).disable_model_cfg1_optimization()

    async def lowvram_patch_counter(self, instance_id: str) -> int:
        return self._get_instance(instance_id).lowvram_patch_counter()

    async def memory_required(self, instance_id: str, input_shape: Any) -> Any:
        return self._run_operation_with_lease(
            instance_id,
            "memory_required",
            lambda: self._get_instance(instance_id).memory_required(input_shape),
        )

    async def is_dynamic(self, instance_id: str) -> bool:
        instance = self._get_instance(instance_id)
        if hasattr(instance, "is_dynamic"):
            return bool(instance.is_dynamic())
        return False

    async def get_free_memory(self, instance_id: str, device: Any) -> Any:
        instance = self._get_instance(instance_id)
        if hasattr(instance, "get_free_memory"):
            return instance.get_free_memory(device)
        import comfy.model_management

        return comfy.model_management.get_free_memory(device)

    async def partially_unload_ram(self, instance_id: str, ram_to_unload: int) -> Any:
        instance = self._get_instance(instance_id)
        if hasattr(instance, "partially_unload_ram"):
            return instance.partially_unload_ram(ram_to_unload)
        return None

    async def model_dtype(self, instance_id: str) -> Any:
        return self._run_operation_with_lease(
            instance_id,
            "model_dtype",
            lambda: self._get_instance(instance_id).model_dtype(),
        )

    async def model_patches_to(self, instance_id: str, device: Any) -> Any:
        return self._get_instance(instance_id).model_patches_to(device)

    async def partially_load(
        self,
        instance_id: str,
        device: Any,
        extra_memory: Any,
        force_patch_weights: bool = False,
    ) -> Any:
        return self._run_operation_with_lease(
            instance_id,
            "partially_load",
            lambda: self._get_instance(instance_id).partially_load(
                device, extra_memory, force_patch_weights=force_patch_weights
            ),
        )

    async def partially_unload(
        self,
        instance_id: str,
        device_to: Any,
        memory_to_free: int = 0,
        force_patch_weights: bool = False,
    ) -> int:
        return self._run_operation_with_lease(
            instance_id,
            "partially_unload",
            lambda: self._get_instance(instance_id).partially_unload(
                device_to, memory_to_free, force_patch_weights
            ),
        )

    async def load(
        self,
        instance_id: str,
        device_to: Any = None,
        lowvram_model_memory: int = 0,
        force_patch_weights: bool = False,
        full_load: bool = False,
    ) -> None:
        self._run_operation_with_lease(
            instance_id,
            "load",
            lambda: self._get_instance(instance_id).load(
                device_to, lowvram_model_memory, force_patch_weights, full_load
            ),
        )

    async def patch_model(
        self,
        instance_id: str,
        device_to: Any = None,
        lowvram_model_memory: int = 0,
        load_weights: bool = True,
        force_patch_weights: bool = False,
    ) -> None:
        def _invoke() -> None:
            try:
                self._get_instance(instance_id).patch_model(
                    device_to, lowvram_model_memory, load_weights, force_patch_weights
                )
            except AttributeError as e:
                logger.error(
                    f"Isolation Error: Failed to patch model attribute: {e}. Skipping."
                )
                return

        self._run_operation_with_lease(instance_id, "patch_model", _invoke)

    async def unpatch_model(
        self, instance_id: str, device_to: Any = None, unpatch_weights: bool = True
    ) -> None:
        self._run_operation_with_lease(
            instance_id,
            "unpatch_model",
            lambda: self._get_instance(instance_id).unpatch_model(
                device_to, unpatch_weights
            ),
        )

    async def detach(self, instance_id: str, unpatch_all: bool = True) -> None:
        self._get_instance(instance_id).detach(unpatch_all)

    async def prepare_state(self, instance_id: str, timestep: Any) -> Any:
        instance = self._get_instance(instance_id)
        cp = getattr(instance.model, "current_patcher", instance)
        if cp is None:
            cp = instance
        return cp.prepare_state(timestep)

    async def pre_run(self, instance_id: str) -> None:
        self._get_instance(instance_id).pre_run()

    async def cleanup(self, instance_id: str) -> None:
        def _invoke() -> None:
            try:
                instance = self._get_instance(instance_id)
            except Exception:
                logger.debug(
                    "ModelPatcher cleanup requested for missing instance %s",
                    instance_id,
                    exc_info=True,
                )
                return

            try:
                instance.cleanup()
            finally:
                with self._lock:
                    self._pending_cleanup_ids.add(instance_id)
                gc.collect()

        self._run_operation_with_lease(instance_id, "cleanup", _invoke)

    def sweep_pending_cleanup(self) -> int:
        removed = 0
        with self._operation_state_cv:
            pending_ids = list(self._pending_cleanup_ids)
            self._pending_cleanup_ids.clear()
            for instance_id in pending_ids:
                instance = self._registry.pop(instance_id, None)
                if instance is None:
                    continue
                self._id_map.pop(id(instance), None)
                self._operation_states.pop(instance_id, None)
                removed += 1
            self._operation_state_cv.notify_all()

        gc.collect()
        return removed

    def purge_all(self) -> int:
        with self._operation_state_cv:
            removed = len(self._registry)
            self._registry.clear()
            self._id_map.clear()
            self._pending_cleanup_ids.clear()
            self._operation_states.clear()
            self._operation_state_cv.notify_all()
        gc.collect()
        return removed

    async def apply_hooks(self, instance_id: str, hooks: Any) -> Any:
        instance = self._get_instance(instance_id)
        cp = getattr(instance.model, "current_patcher", instance)
        if cp is None:
            cp = instance
        return cp.apply_hooks(hooks=hooks)

    async def clean_hooks(self, instance_id: str) -> None:
        self._get_instance(instance_id).clean_hooks()

    async def restore_hook_patches(self, instance_id: str) -> None:
        self._get_instance(instance_id).restore_hook_patches()

    async def unpatch_hooks(
        self, instance_id: str, whitelist_keys_set: Optional[set] = None
    ) -> None:
        self._get_instance(instance_id).unpatch_hooks(whitelist_keys_set)

    async def register_all_hook_patches(
        self,
        instance_id: str,
        hooks: Any,
        target_dict: Any,
        model_options: Any,
        registered: Any,
    ) -> None:
        from types import SimpleNamespace
        import comfy.hooks

        instance = self._get_instance(instance_id)
        if isinstance(hooks, SimpleNamespace) or hasattr(hooks, "__dict__"):
            hook_data = hooks.__dict__ if hasattr(hooks, "__dict__") else hooks
            new_hooks = comfy.hooks.HookGroup()
            if hasattr(hook_data, "hooks"):
                new_hooks.hooks = (
                    hook_data["hooks"]
                    if isinstance(hook_data, dict)
                    else hook_data.hooks
                )
            hooks = new_hooks
        instance.register_all_hook_patches(
            hooks, target_dict, model_options, registered
        )

    async def get_hook_mode(self, instance_id: str) -> Any:
        return getattr(self._get_instance(instance_id), "hook_mode", None)

    async def set_hook_mode(self, instance_id: str, value: Any) -> None:
        setattr(self._get_instance(instance_id), "hook_mode", value)

    async def inject_model(self, instance_id: str) -> None:
        instance = self._get_instance(instance_id)
        try:
            instance.inject_model()
        except AttributeError as e:
            if "inject" in str(e):
                logger.error(
                    "Isolation Error: Injector object lost method code during serialization. Cannot inject. Skipping."
                )
                return
            raise e

    async def eject_model(self, instance_id: str) -> None:
        self._get_instance(instance_id).eject_model()

    async def get_is_injected(self, instance_id: str) -> bool:
        return self._get_instance(instance_id).is_injected

    async def set_skip_injection(self, instance_id: str, value: bool) -> None:
        self._get_instance(instance_id).skip_injection = value

    async def get_skip_injection(self, instance_id: str) -> bool:
        return self._get_instance(instance_id).skip_injection

    async def set_model_sampler_cfg_function(
        self,
        instance_id: str,
        sampler_cfg_function: Any,
        disable_cfg1_optimization: bool = False,
    ) -> None:
        if not callable(sampler_cfg_function):
            logger.error(
                f"set_model_sampler_cfg_function: Expected callable, got {type(sampler_cfg_function)}. Skipping."
            )
            return
        self._get_instance(instance_id).set_model_sampler_cfg_function(
            sampler_cfg_function, disable_cfg1_optimization
        )

    async def set_model_sampler_post_cfg_function(
        self,
        instance_id: str,
        post_cfg_function: Any,
        disable_cfg1_optimization: bool = False,
    ) -> None:
        self._get_instance(instance_id).set_model_sampler_post_cfg_function(
            post_cfg_function, disable_cfg1_optimization
        )

    async def set_model_sampler_pre_cfg_function(
        self,
        instance_id: str,
        pre_cfg_function: Any,
        disable_cfg1_optimization: bool = False,
    ) -> None:
        self._get_instance(instance_id).set_model_sampler_pre_cfg_function(
            pre_cfg_function, disable_cfg1_optimization
        )

    async def set_model_sampler_calc_cond_batch_function(
        self, instance_id: str, fn: Any
    ) -> None:
        self._get_instance(instance_id).set_model_sampler_calc_cond_batch_function(fn)

    async def set_model_unet_function_wrapper(
        self, instance_id: str, unet_wrapper_function: Any
    ) -> None:
        self._get_instance(instance_id).set_model_unet_function_wrapper(
            unet_wrapper_function
        )

    async def set_model_denoise_mask_function(
        self, instance_id: str, denoise_mask_function: Any
    ) -> None:
        self._get_instance(instance_id).set_model_denoise_mask_function(
            denoise_mask_function
        )

    async def set_model_patch(self, instance_id: str, patch: Any, name: str) -> None:
        self._get_instance(instance_id).set_model_patch(patch, name)

    async def set_model_patch_replace(
        self,
        instance_id: str,
        patch: Any,
        name: str,
        block_name: str,
        number: int,
        transformer_index: Optional[int] = None,
    ) -> None:
        self._get_instance(instance_id).set_model_patch_replace(
            patch, name, block_name, number, transformer_index
        )

    async def set_model_input_block_patch(self, instance_id: str, patch: Any) -> None:
        self._get_instance(instance_id).set_model_input_block_patch(patch)

    async def set_model_input_block_patch_after_skip(
        self, instance_id: str, patch: Any
    ) -> None:
        self._get_instance(instance_id).set_model_input_block_patch_after_skip(patch)

    async def set_model_output_block_patch(self, instance_id: str, patch: Any) -> None:
        self._get_instance(instance_id).set_model_output_block_patch(patch)

    async def set_model_emb_patch(self, instance_id: str, patch: Any) -> None:
        self._get_instance(instance_id).set_model_emb_patch(patch)

    async def set_model_forward_timestep_embed_patch(
        self, instance_id: str, patch: Any
    ) -> None:
        self._get_instance(instance_id).set_model_forward_timestep_embed_patch(patch)

    async def set_model_double_block_patch(self, instance_id: str, patch: Any) -> None:
        self._get_instance(instance_id).set_model_double_block_patch(patch)

    async def set_model_post_input_patch(self, instance_id: str, patch: Any) -> None:
        self._get_instance(instance_id).set_model_post_input_patch(patch)

    async def set_model_rope_options(self, instance_id: str, options: dict) -> None:
        self._get_instance(instance_id).set_model_rope_options(**options)

    async def set_model_compute_dtype(self, instance_id: str, dtype: Any) -> None:
        self._get_instance(instance_id).set_model_compute_dtype(dtype)

    async def clone_has_same_weights_by_id(
        self, instance_id: str, other_id: str
    ) -> bool:
        instance = self._get_instance(instance_id)
        other = self._get_instance(other_id)
        if not other:
            return False
        return instance.clone_has_same_weights(other)

    async def load_list_internal(self, instance_id: str, *args, **kwargs) -> Any:
        return self._get_instance(instance_id)._load_list(*args, **kwargs)

    async def is_clone_by_id(self, instance_id: str, other_id: str) -> bool:
        instance = self._get_instance(instance_id)
        other = self._get_instance(other_id)
        if hasattr(instance, "is_clone"):
            return instance.is_clone(other)
        return False

    async def add_object_patch(self, instance_id: str, name: str, obj: Any) -> None:
        self._get_instance(instance_id).add_object_patch(name, obj)

    async def add_weight_wrapper(
        self, instance_id: str, name: str, function: Any
    ) -> None:
        self._get_instance(instance_id).add_weight_wrapper(name, function)

    async def add_wrapper_with_key(
        self, instance_id: str, wrapper_type: Any, key: str, fn: Any
    ) -> None:
        self._get_instance(instance_id).add_wrapper_with_key(wrapper_type, key, fn)

    async def remove_wrappers_with_key(
        self, instance_id: str, wrapper_type: str, key: str
    ) -> None:
        self._get_instance(instance_id).remove_wrappers_with_key(wrapper_type, key)

    async def get_wrappers(
        self, instance_id: str, wrapper_type: str = None, key: str = None
    ) -> Any:
        if wrapper_type is None and key is None:
            return self._sanitize_rpc_result(
                getattr(self._get_instance(instance_id), "wrappers", {})
            )
        return self._sanitize_rpc_result(
            self._get_instance(instance_id).get_wrappers(wrapper_type, key)
        )

    async def get_all_wrappers(self, instance_id: str, wrapper_type: str = None) -> Any:
        return self._sanitize_rpc_result(
            getattr(self._get_instance(instance_id), "get_all_wrappers", lambda x: [])(
                wrapper_type
            )
        )

    async def add_callback_with_key(
        self, instance_id: str, call_type: str, key: str, callback: Any
    ) -> None:
        self._get_instance(instance_id).add_callback_with_key(call_type, key, callback)

    async def remove_callbacks_with_key(
        self, instance_id: str, call_type: str, key: str
    ) -> None:
        self._get_instance(instance_id).remove_callbacks_with_key(call_type, key)

    async def get_callbacks(
        self, instance_id: str, call_type: str = None, key: str = None
    ) -> Any:
        if call_type is None and key is None:
            return self._sanitize_rpc_result(
                getattr(self._get_instance(instance_id), "callbacks", {})
            )
        return self._sanitize_rpc_result(
            self._get_instance(instance_id).get_callbacks(call_type, key)
        )

    async def get_all_callbacks(self, instance_id: str, call_type: str = None) -> Any:
        return self._sanitize_rpc_result(
            getattr(self._get_instance(instance_id), "get_all_callbacks", lambda x: [])(
                call_type
            )
        )

    async def set_attachments(
        self, instance_id: str, key: str, attachment: Any
    ) -> None:
        self._get_instance(instance_id).set_attachments(key, attachment)

    async def get_attachment(self, instance_id: str, key: str) -> Any:
        return self._sanitize_rpc_result(
            self._get_instance(instance_id).get_attachment(key)
        )

    async def remove_attachments(self, instance_id: str, key: str) -> None:
        self._get_instance(instance_id).remove_attachments(key)

    async def set_injections(self, instance_id: str, key: str, injections: Any) -> None:
        self._get_instance(instance_id).set_injections(key, injections)

    async def get_injections(self, instance_id: str, key: str) -> Any:
        return self._sanitize_rpc_result(
            self._get_instance(instance_id).get_injections(key)
        )

    async def remove_injections(self, instance_id: str, key: str) -> None:
        self._get_instance(instance_id).remove_injections(key)

    async def set_additional_models(
        self, instance_id: str, key: str, models: Any
    ) -> None:
        self._get_instance(instance_id).set_additional_models(key, models)

    async def remove_additional_models(self, instance_id: str, key: str) -> None:
        self._get_instance(instance_id).remove_additional_models(key)

    async def get_nested_additional_models(self, instance_id: str) -> Any:
        return self._sanitize_rpc_result(
            self._get_instance(instance_id).get_nested_additional_models()
        )

    async def get_additional_models(self, instance_id: str) -> List[str]:
        models = self._get_instance(instance_id).get_additional_models()
        return [self.register(m) for m in models]

    async def get_additional_models_with_key(self, instance_id: str, key: str) -> Any:
        return self._sanitize_rpc_result(
            self._get_instance(instance_id).get_additional_models_with_key(key)
        )

    async def model_patches_models(self, instance_id: str) -> Any:
        return self._sanitize_rpc_result(
            self._get_instance(instance_id).model_patches_models()
        )

    async def get_patches(self, instance_id: str) -> Any:
        return self._sanitize_rpc_result(self._get_instance(instance_id).patches.copy())

    async def get_object_patches(self, instance_id: str) -> Any:
        return self._sanitize_rpc_result(
            self._get_instance(instance_id).object_patches.copy()
        )

    async def add_patches(
        self,
        instance_id: str,
        patches: Any,
        strength_patch: float = 1.0,
        strength_model: float = 1.0,
    ) -> Any:
        return self._get_instance(instance_id).add_patches(
            patches, strength_patch, strength_model
        )

    async def get_key_patches(
        self, instance_id: str, filter_prefix: Optional[str] = None
    ) -> Any:
        res = self._get_instance(instance_id).get_key_patches()
        if filter_prefix:
            res = {k: v for k, v in res.items() if k.startswith(filter_prefix)}
        safe_res = {}
        for k, v in res.items():
            safe_res[k] = [
                f"<Tensor shape={t.shape} dtype={t.dtype}>"
                if hasattr(t, "shape")
                else str(t)
                for t in v
            ]
        return safe_res

    async def add_hook_patches(
        self,
        instance_id: str,
        hook: Any,
        patches: Any,
        strength_patch: float = 1.0,
        strength_model: float = 1.0,
    ) -> None:
        if hasattr(hook, "hook_ref") and isinstance(hook.hook_ref, dict):
            try:
                hook.hook_ref = tuple(sorted(hook.hook_ref.items()))
            except Exception:
                hook.hook_ref = None
        self._get_instance(instance_id).add_hook_patches(
            hook, patches, strength_patch, strength_model
        )

    async def get_combined_hook_patches(self, instance_id: str, hooks: Any) -> Any:
        if hooks is not None and hasattr(hooks, "hooks"):
            for hook in getattr(hooks, "hooks", []):
                hook_ref = getattr(hook, "hook_ref", None)
                if isinstance(hook_ref, dict):
                    try:
                        hook.hook_ref = tuple(sorted(hook_ref.items()))
                    except Exception:
                        hook.hook_ref = None
        res = self._get_instance(instance_id).get_combined_hook_patches(hooks)
        return self._sanitize_rpc_result(res)

    async def clear_cached_hook_weights(self, instance_id: str) -> None:
        self._get_instance(instance_id).clear_cached_hook_weights()

    async def prepare_hook_patches_current_keyframe(
        self, instance_id: str, t: Any, hook_group: Any, model_options: Any
    ) -> None:
        self._get_instance(instance_id).prepare_hook_patches_current_keyframe(
            t, hook_group, model_options
        )

    async def get_parent(self, instance_id: str) -> Any:
        return getattr(self._get_instance(instance_id), "parent", None)

    async def patch_weight_to_device(
        self,
        instance_id: str,
        key: str,
        device_to: Any = None,
        inplace_update: bool = False,
    ) -> None:
        self._get_instance(instance_id).patch_weight_to_device(
            key, device_to, inplace_update
        )

    async def pin_weight_to_device(self, instance_id: str, key: str) -> None:
        instance = self._get_instance(instance_id)
        if hasattr(instance, "pinned") and isinstance(instance.pinned, list):
            instance.pinned = set(instance.pinned)
        instance.pin_weight_to_device(key)

    async def unpin_weight(self, instance_id: str, key: str) -> None:
        instance = self._get_instance(instance_id)
        if hasattr(instance, "pinned") and isinstance(instance.pinned, list):
            instance.pinned = set(instance.pinned)
        instance.unpin_weight(key)

    async def unpin_all_weights(self, instance_id: str) -> None:
        instance = self._get_instance(instance_id)
        if hasattr(instance, "pinned") and isinstance(instance.pinned, list):
            instance.pinned = set(instance.pinned)
        instance.unpin_all_weights()

    async def calculate_weight(
        self,
        instance_id: str,
        patches: Any,
        weight: Any,
        key: str,
        intermediate_dtype: Any = float,
    ) -> Any:
        return detach_if_grad(
            self._get_instance(instance_id).calculate_weight(
                patches, weight, key, intermediate_dtype
            )
        )

    async def get_inner_model_attr(self, instance_id: str, name: str) -> Any:
        try:
            value = getattr(self._get_instance(instance_id).model, name)
            if name == "model_config":
                value = self._extract_model_config(value)
            return self._sanitize_rpc_result(value)
        except AttributeError:
            return None

    @staticmethod
    def _extract_model_config(config: Any) -> dict:
        """Extract JSON-safe attributes from a model config object.

        ComfyUI model config classes (supported_models_base.BASE subclasses)
        have a permissive __getattr__ that returns None for any unknown
        attribute instead of raising AttributeError. This defeats hasattr-based
        duck-typing in _sanitize_rpc_result, causing TypeError when it tries
        to call obj.items() (which resolves to None). We extract the real
        class-level and instance-level attributes into a plain dict.
        """
        # Attributes consumed by ModelSampling*.__init__ and other callers
        _CONFIG_KEYS = (
            "sampling_settings",
            "unet_config",
            "unet_extra_config",
            "latent_format",
            "manual_cast_dtype",
            "custom_operations",
            "optimizations",
            "memory_usage_factor",
            "supported_inference_dtypes",
        )
        result: dict = {}
        for key in _CONFIG_KEYS:
            # Use type(config).__dict__ first (class attrs), then instance __dict__
            # to avoid triggering the permissive __getattr__
            if key in type(config).__dict__:
                val = type(config).__dict__[key]
                # Skip classmethods/staticmethods/descriptors
                if not callable(val) or isinstance(val, (dict, list, tuple)):
                    result[key] = val
            elif hasattr(config, "__dict__") and key in config.__dict__:
                result[key] = config.__dict__[key]
        # Also include instance overrides (e.g. set_inference_dtype sets unet_config['dtype'])
        if hasattr(config, "__dict__"):
            for key, val in config.__dict__.items():
                if key in _CONFIG_KEYS:
                    result[key] = val
        return result

    async def inner_model_memory_required(
        self, instance_id: str, args: tuple, kwargs: dict
    ) -> Any:
        return self._run_operation_with_lease(
            instance_id,
            "inner_model_memory_required",
            lambda: self._get_instance(instance_id).model.memory_required(
                *args, **kwargs
            ),
        )

    async def inner_model_extra_conds_shapes(
        self, instance_id: str, args: tuple, kwargs: dict
    ) -> Any:
        return self._run_operation_with_lease(
            instance_id,
            "inner_model_extra_conds_shapes",
            lambda: self._get_instance(instance_id).model.extra_conds_shapes(
                *args, **kwargs
            ),
        )

    async def inner_model_extra_conds(
        self, instance_id: str, args: tuple, kwargs: dict
    ) -> Any:
        def _invoke() -> Any:
            result = self._get_instance(instance_id).model.extra_conds(*args, **kwargs)
            try:
                import torch
                import comfy.conds
            except Exception:
                return result

            def _to_cpu(obj: Any) -> Any:
                if torch.is_tensor(obj):
                    return obj.detach().cpu() if obj.device.type != "cpu" else obj
                if isinstance(obj, dict):
                    return {k: _to_cpu(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_to_cpu(v) for v in obj]
                if isinstance(obj, tuple):
                    return tuple(_to_cpu(v) for v in obj)
                if isinstance(obj, comfy.conds.CONDRegular):
                    return type(obj)(_to_cpu(obj.cond))
                return obj

            return _to_cpu(result)

        return self._run_operation_with_lease(instance_id, "inner_model_extra_conds", _invoke)

    async def inner_model_state_dict(
        self, instance_id: str, args: tuple, kwargs: dict
    ) -> Any:
        sd = self._get_instance(instance_id).model.state_dict(*args, **kwargs)
        return {
            k: {"numel": v.numel(), "element_size": v.element_size()}
            for k, v in sd.items()
        }

    async def inner_model_apply_model(
        self, instance_id: str, args: tuple, kwargs: dict
    ) -> Any:
        def _invoke() -> Any:
            import torch

            instance = self._get_instance(instance_id)
            target = getattr(instance, "load_device", None)
            if target is None and args and hasattr(args[0], "device"):
                target = args[0].device
            elif target is None:
                for v in kwargs.values():
                    if hasattr(v, "device"):
                        target = v.device
                        break

            def _move(obj):
                if target is None:
                    return obj
                if isinstance(obj, (tuple, list)):
                    return type(obj)(_move(o) for o in obj)
                if hasattr(obj, "to"):
                    return obj.to(target)
                return obj

            moved_args = tuple(_move(a) for a in args)
            moved_kwargs = {k: _move(v) for k, v in kwargs.items()}
            result = instance.model.apply_model(*moved_args, **moved_kwargs)
            moved_result = detach_if_grad(_move(result))

            # DynamicVRAM + isolation: returning CUDA tensors across RPC can stall
            # at the transport boundary. Marshal dynamic-path results as CPU and let
            # the proxy restore device placement in the child process.
            is_dynamic_fn = getattr(instance, "is_dynamic", None)
            if callable(is_dynamic_fn) and is_dynamic_fn():
                def _to_cpu(obj: Any) -> Any:
                    if torch.is_tensor(obj):
                        return obj.detach().cpu() if obj.device.type != "cpu" else obj
                    if isinstance(obj, dict):
                        return {k: _to_cpu(v) for k, v in obj.items()}
                    if isinstance(obj, list):
                        return [_to_cpu(v) for v in obj]
                    if isinstance(obj, tuple):
                        return tuple(_to_cpu(v) for v in obj)
                    return obj

                return _to_cpu(moved_result)
            return moved_result

        return self._run_operation_with_lease(instance_id, "inner_model_apply_model", _invoke)

    async def process_latent_in(
        self, instance_id: str, args: tuple, kwargs: dict
    ) -> Any:
        import torch

        def _invoke() -> Any:
            instance = self._get_instance(instance_id)
            result = detach_if_grad(instance.model.process_latent_in(*args, **kwargs))

            # DynamicVRAM + isolation: returning CUDA tensors across RPC can stall
            # at the transport boundary. Marshal dynamic-path results as CPU and let
            # the proxy restore placement when needed.
            is_dynamic_fn = getattr(instance, "is_dynamic", None)
            if callable(is_dynamic_fn) and is_dynamic_fn():
                def _to_cpu(obj: Any) -> Any:
                    if torch.is_tensor(obj):
                        return obj.detach().cpu() if obj.device.type != "cpu" else obj
                    if isinstance(obj, dict):
                        return {k: _to_cpu(v) for k, v in obj.items()}
                    if isinstance(obj, list):
                        return [_to_cpu(v) for v in obj]
                    if isinstance(obj, tuple):
                        return tuple(_to_cpu(v) for v in obj)
                    return obj

                return _to_cpu(result)
            return result

        return self._run_operation_with_lease(instance_id, "process_latent_in", _invoke)

    async def process_latent_out(
        self, instance_id: str, args: tuple, kwargs: dict
    ) -> Any:
        import torch

        def _invoke() -> Any:
            instance = self._get_instance(instance_id)
            result = instance.model.process_latent_out(*args, **kwargs)
            moved_result = None
            try:
                target = None
                if args and hasattr(args[0], "device"):
                    target = args[0].device
                elif kwargs:
                    for v in kwargs.values():
                        if hasattr(v, "device"):
                            target = v.device
                            break
                if target is not None and hasattr(result, "to"):
                    moved_result = detach_if_grad(result.to(target))
            except Exception:
                logger.debug(
                    "process_latent_out: failed to move result to target device",
                    exc_info=True,
                )
            if moved_result is None:
                moved_result = detach_if_grad(result)

            is_dynamic_fn = getattr(instance, "is_dynamic", None)
            if callable(is_dynamic_fn) and is_dynamic_fn():
                def _to_cpu(obj: Any) -> Any:
                    if torch.is_tensor(obj):
                        return obj.detach().cpu() if obj.device.type != "cpu" else obj
                    if isinstance(obj, dict):
                        return {k: _to_cpu(v) for k, v in obj.items()}
                    if isinstance(obj, list):
                        return [_to_cpu(v) for v in obj]
                    if isinstance(obj, tuple):
                        return tuple(_to_cpu(v) for v in obj)
                    return obj

                return _to_cpu(moved_result)
            return moved_result

        return self._run_operation_with_lease(instance_id, "process_latent_out", _invoke)

    async def scale_latent_inpaint(
        self, instance_id: str, args: tuple, kwargs: dict
    ) -> Any:
        import torch

        def _invoke() -> Any:
            instance = self._get_instance(instance_id)
            result = instance.model.scale_latent_inpaint(*args, **kwargs)
            moved_result = None
            try:
                target = None
                if args and hasattr(args[0], "device"):
                    target = args[0].device
                elif kwargs:
                    for v in kwargs.values():
                        if hasattr(v, "device"):
                            target = v.device
                            break
                if target is not None and hasattr(result, "to"):
                    moved_result = detach_if_grad(result.to(target))
            except Exception:
                logger.debug(
                    "scale_latent_inpaint: failed to move result to target device",
                    exc_info=True,
                )
            if moved_result is None:
                moved_result = detach_if_grad(result)

            is_dynamic_fn = getattr(instance, "is_dynamic", None)
            if callable(is_dynamic_fn) and is_dynamic_fn():
                def _to_cpu(obj: Any) -> Any:
                    if torch.is_tensor(obj):
                        return obj.detach().cpu() if obj.device.type != "cpu" else obj
                    if isinstance(obj, dict):
                        return {k: _to_cpu(v) for k, v in obj.items()}
                    if isinstance(obj, list):
                        return [_to_cpu(v) for v in obj]
                    if isinstance(obj, tuple):
                        return tuple(_to_cpu(v) for v in obj)
                    return obj

                return _to_cpu(moved_result)
            return moved_result

        return self._run_operation_with_lease(
            instance_id, "scale_latent_inpaint", _invoke
        )

    async def load_lora(
        self,
        instance_id: str,
        lora_path: str,
        strength_model: float,
        clip_id: Optional[str] = None,
        strength_clip: float = 1.0,
    ) -> dict:
        import comfy.utils
        import comfy.sd
        import folder_paths
        from comfy.isolation.clip_proxy import CLIPRegistry

        model = self._get_instance(instance_id)
        clip = None
        if clip_id:
            clip = CLIPRegistry()._get_instance(clip_id)
        lora_full_path = folder_paths.get_full_path("loras", lora_path)
        if lora_full_path is None:
            raise ValueError(f"LoRA file not found: {lora_path}")
        lora = comfy.utils.load_torch_file(lora_full_path)
        new_model, new_clip = comfy.sd.load_lora_for_models(
            model, clip, lora, strength_model, strength_clip
        )
        new_model_id = self.register(new_model) if new_model else None
        new_clip_id = (
            CLIPRegistry().register(new_clip) if (new_clip and clip_id) else None
        )
        return {"model_id": new_model_id, "clip_id": new_clip_id}
