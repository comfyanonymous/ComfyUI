from __future__ import annotations

import os
from typing import Any, Optional

from pyisolate import ProxiedSingleton

from .base import call_singleton_rpc


def _mm():
    import comfy.model_management

    return comfy.model_management


def _is_child_process() -> bool:
    return os.environ.get("PYISOLATE_CHILD") == "1"


class TorchDeviceProxy:
    def __init__(self, device_str: str):
        self._device_str = device_str
        if ":" in device_str:
            device_type, index = device_str.split(":", 1)
            self.type = device_type
            self.index = int(index)
        else:
            self.type = device_str
            self.index = None

    def __str__(self) -> str:
        return self._device_str

    def __repr__(self) -> str:
        return f"TorchDeviceProxy({self._device_str!r})"


def _serialize_value(value: Any) -> Any:
    value_type = type(value)
    if value_type.__module__ == "torch" and value_type.__name__ == "device":
        return {"__pyisolate_torch_device__": str(value)}
    if isinstance(value, TorchDeviceProxy):
        return {"__pyisolate_torch_device__": str(value)}
    if isinstance(value, tuple):
        return {"__pyisolate_tuple__": [_serialize_value(item) for item in value]}
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_value(inner) for key, inner in value.items()}
    return value


def _deserialize_value(value: Any) -> Any:
    if isinstance(value, dict):
        if "__pyisolate_torch_device__" in value:
            return TorchDeviceProxy(value["__pyisolate_torch_device__"])
        if "__pyisolate_tuple__" in value:
            return tuple(_deserialize_value(item) for item in value["__pyisolate_tuple__"])
        return {key: _deserialize_value(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_deserialize_value(item) for item in value]
    return value


def _normalize_argument(value: Any) -> Any:
    if isinstance(value, TorchDeviceProxy):
        import torch

        return torch.device(str(value))
    if isinstance(value, dict):
        if "__pyisolate_torch_device__" in value:
            import torch

            return torch.device(value["__pyisolate_torch_device__"])
        if "__pyisolate_tuple__" in value:
            return tuple(_normalize_argument(item) for item in value["__pyisolate_tuple__"])
        return {key: _normalize_argument(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_normalize_argument(item) for item in value]
    return value


class ModelManagementProxy(ProxiedSingleton):
    """
    Exact-relay proxy for comfy.model_management.
    Child calls never import comfy.model_management directly; they serialize
    arguments, relay to host, and deserialize the host result back.
    """

    _rpc: Optional[Any] = None

    @classmethod
    def set_rpc(cls, rpc: Any) -> None:
        cls._rpc = rpc.create_caller(cls, cls.get_remote_id())

    @classmethod
    def clear_rpc(cls) -> None:
        cls._rpc = None

    @classmethod
    def _get_caller(cls) -> Any:
        if cls._rpc is None:
            raise RuntimeError("ModelManagementProxy RPC caller is not configured")
        return cls._rpc

    def _relay_call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        payload = call_singleton_rpc(
            self._get_caller(),
            "rpc_call",
            method_name,
            _serialize_value(args),
            _serialize_value(kwargs),
        )
        return _deserialize_value(payload)

    @property
    def VRAMState(self):
        return _mm().VRAMState

    @property
    def CPUState(self):
        return _mm().CPUState

    @property
    def OOM_EXCEPTION(self):
        return _mm().OOM_EXCEPTION

    def __getattr__(self, name: str):
        if _is_child_process():
            def child_method(*args: Any, **kwargs: Any) -> Any:
                return self._relay_call(name, *args, **kwargs)

            return child_method
        return getattr(_mm(), name)

    async def rpc_call(self, method_name: str, args: Any, kwargs: Any) -> Any:
        normalized_args = _normalize_argument(_deserialize_value(args))
        normalized_kwargs = _normalize_argument(_deserialize_value(kwargs))
        method = getattr(_mm(), method_name)
        result = method(*normalized_args, **normalized_kwargs)
        return _serialize_value(result)
