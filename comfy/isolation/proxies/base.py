# pylint: disable=global-statement,import-outside-toplevel,protected-access
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import threading
import time
import weakref
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

try:
    from pyisolate import ProxiedSingleton
except ImportError:

    class ProxiedSingleton:  # type: ignore[no-redef]
        pass


logger = logging.getLogger(__name__)

IS_CHILD_PROCESS = os.environ.get("PYISOLATE_CHILD") == "1"
_thread_local = threading.local()
T = TypeVar("T")


def get_thread_loop() -> asyncio.AbstractEventLoop:
    loop = getattr(_thread_local, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _thread_local.loop = loop
    return loop


def run_coro_in_new_loop(coro: Any) -> Any:
    result_box: Dict[str, Any] = {}
    exc_box: Dict[str, BaseException] = {}

    def runner() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_box["value"] = loop.run_until_complete(coro)
        except Exception as exc:  # noqa: BLE001
            exc_box["exc"] = exc
        finally:
            loop.close()

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join()
    if "exc" in exc_box:
        raise exc_box["exc"]
    return result_box.get("value")


def detach_if_grad(obj: Any) -> Any:
    try:
        import torch
    except Exception:
        return obj

    if isinstance(obj, torch.Tensor):
        return obj.detach() if obj.requires_grad else obj
    if isinstance(obj, (list, tuple)):
        return type(obj)(detach_if_grad(x) for x in obj)
    if isinstance(obj, dict):
        return {k: detach_if_grad(v) for k, v in obj.items()}
    return obj


class BaseRegistry(ProxiedSingleton, Generic[T]):
    _type_prefix: str = "base"

    def __init__(self) -> None:
        if hasattr(ProxiedSingleton, "__init__") and ProxiedSingleton is not object:
            super().__init__()
        self._registry: Dict[str, T] = {}
        self._id_map: Dict[int, str] = {}
        self._counter = 0
        self._lock = threading.Lock()

    def register(self, instance: T) -> str:
        with self._lock:
            obj_id = id(instance)
            if obj_id in self._id_map:
                return self._id_map[obj_id]
            instance_id = f"{self._type_prefix}_{self._counter}"
            self._counter += 1
            self._registry[instance_id] = instance
            self._id_map[obj_id] = instance_id
        return instance_id

    def unregister_sync(self, instance_id: str) -> None:
        with self._lock:
            instance = self._registry.pop(instance_id, None)
            if instance:
                self._id_map.pop(id(instance), None)

    def _get_instance(self, instance_id: str) -> T:
        if IS_CHILD_PROCESS:
            raise RuntimeError(
                f"[{self.__class__.__name__}] _get_instance called in child"
            )
        with self._lock:
            instance = self._registry.get(instance_id)
        if instance is None:
            raise ValueError(f"{instance_id} not found")
        return instance


_GLOBAL_LOOP: Optional[asyncio.AbstractEventLoop] = None


def set_global_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _GLOBAL_LOOP
    _GLOBAL_LOOP = loop


def run_sync_rpc_coro(coro: Any, timeout_ms: Optional[int] = None) -> Any:
    if timeout_ms is not None:
        coro = asyncio.wait_for(coro, timeout=timeout_ms / 1000.0)

    try:
        if _GLOBAL_LOOP is not None and _GLOBAL_LOOP.is_running():
            try:
                curr_loop = asyncio.get_running_loop()
                if curr_loop is _GLOBAL_LOOP:
                    pass
            except RuntimeError:
                future = asyncio.run_coroutine_threadsafe(coro, _GLOBAL_LOOP)
                return future.result(
                    timeout=(timeout_ms / 1000.0) if timeout_ms is not None else None
                )

        try:
            asyncio.get_running_loop()
            return run_coro_in_new_loop(coro)
        except RuntimeError:
            loop = get_thread_loop()
            return loop.run_until_complete(coro)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"Isolation RPC timeout (timeout_ms={timeout_ms})") from exc
    except concurrent.futures.TimeoutError as exc:
        raise TimeoutError(f"Isolation RPC timeout (timeout_ms={timeout_ms})") from exc


def call_singleton_rpc(
    caller: Any,
    method_name: str,
    *args: Any,
    timeout_ms: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    if caller is None:
        raise RuntimeError(f"No RPC caller available for {method_name}")
    method = getattr(caller, method_name)
    return run_sync_rpc_coro(method(*args, **kwargs), timeout_ms=timeout_ms)


class BaseProxy(Generic[T]):
    _registry_class: type = BaseRegistry  # type: ignore[type-arg]
    __module__: str = "comfy.isolation.proxies.base"
    _TIMEOUT_RPC_METHODS = frozenset(
        {
            "partially_load",
            "partially_unload",
            "load",
            "patch_model",
            "unpatch_model",
            "inner_model_apply_model",
            "memory_required",
            "model_dtype",
            "inner_model_memory_required",
            "inner_model_extra_conds_shapes",
            "inner_model_extra_conds",
            "process_latent_in",
            "process_latent_out",
            "scale_latent_inpaint",
        }
    )

    def __init__(
        self,
        instance_id: str,
        registry: Optional[Any] = None,
        manage_lifecycle: bool = False,
    ) -> None:
        self._instance_id = instance_id
        self._rpc_caller: Optional[Any] = None
        self._registry = registry if registry is not None else self._registry_class()
        self._manage_lifecycle = manage_lifecycle
        self._cleaned_up = False
        if manage_lifecycle and not IS_CHILD_PROCESS:
            self._finalizer = weakref.finalize(
                self, self._registry.unregister_sync, instance_id
            )

    def _get_rpc(self) -> Any:
        if self._rpc_caller is None:
            from pyisolate._internal.rpc_protocol import get_child_rpc_instance

            rpc = get_child_rpc_instance()
            if rpc is None:
                raise RuntimeError(f"[{self.__class__.__name__}] No RPC in child")
            self._rpc_caller = rpc.create_caller(
                self._registry_class, self._registry_class.get_remote_id()
            )
        return self._rpc_caller

    def _rpc_timeout_ms_for_method(self, method_name: str) -> Optional[int]:
        if method_name not in self._TIMEOUT_RPC_METHODS:
            return None
        try:
            timeout_ms = int(
                os.environ.get("COMFY_ISOLATION_LOAD_RPC_TIMEOUT_MS", "120000")
            )
        except ValueError:
            timeout_ms = 120000
        return max(1, timeout_ms)

    def _call_rpc(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        rpc = self._get_rpc()
        method = getattr(rpc, method_name)
        timeout_ms = self._rpc_timeout_ms_for_method(method_name)
        coro = method(self._instance_id, *args, **kwargs)
        if timeout_ms is not None:
            coro = asyncio.wait_for(coro, timeout=timeout_ms / 1000.0)

        start_epoch = time.time()
        start_perf = time.perf_counter()
        thread_id = threading.get_ident()
        try:
            running_loop = asyncio.get_running_loop()
            loop_id: Optional[int] = id(running_loop)
        except RuntimeError:
            loop_id = None
        logger.debug(
            "ISO:rpc_start proxy=%s method=%s instance_id=%s start_ts=%.6f "
            "thread=%s loop=%s timeout_ms=%s",
            self.__class__.__name__,
            method_name,
            self._instance_id,
            start_epoch,
            thread_id,
            loop_id,
            timeout_ms,
        )

        try:
            return run_sync_rpc_coro(coro, timeout_ms=timeout_ms)
        except TimeoutError as exc:
            raise TimeoutError(
                f"Isolation RPC timeout in {self.__class__.__name__}.{method_name} "
                f"(instance_id={self._instance_id}, timeout_ms={timeout_ms})"
            ) from exc
        finally:
            end_epoch = time.time()
            elapsed_ms = (time.perf_counter() - start_perf) * 1000.0
            logger.debug(
                "ISO:rpc_end proxy=%s method=%s instance_id=%s end_ts=%.6f "
                "elapsed_ms=%.3f thread=%s loop=%s",
                self.__class__.__name__,
                method_name,
                self._instance_id,
                end_epoch,
                elapsed_ms,
                thread_id,
                loop_id,
            )

    def __getstate__(self) -> Dict[str, Any]:
        return {"_instance_id": self._instance_id}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self._instance_id = state["_instance_id"]
        self._rpc_caller = None
        self._registry = self._registry_class()
        self._manage_lifecycle = False
        self._cleaned_up = False

    def cleanup(self) -> None:
        if self._cleaned_up or IS_CHILD_PROCESS:
            return
        self._cleaned_up = True
        finalizer = getattr(self, "_finalizer", None)
        if finalizer is not None:
            finalizer.detach()
        self._registry.unregister_sync(self._instance_id)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self._instance_id}>"


def create_rpc_method(method_name: str) -> Callable[..., Any]:
    def method(self: BaseProxy[Any], *args: Any, **kwargs: Any) -> Any:
        return self._call_rpc(method_name, *args, **kwargs)

    method.__name__ = method_name
    return method
