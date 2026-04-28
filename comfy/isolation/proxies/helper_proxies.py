from __future__ import annotations

import os
from typing import Any, Dict, Optional

from pyisolate import ProxiedSingleton

from .base import call_singleton_rpc


class AnyTypeProxy(str):
    """Replacement for custom AnyType objects used by some nodes."""

    def __new__(cls, value: str = "*"):
        return super().__new__(cls, value)

    def __ne__(self, other):  # type: ignore[override]
        return False


class FlexibleOptionalInputProxy(dict):
    """Replacement for FlexibleOptionalInputType to allow dynamic inputs."""

    def __init__(self, flex_type, data: Optional[Dict[str, object]] = None):
        super().__init__()
        self.type = flex_type
        if data:
            self.update(data)

    def __getitem__(self, key):  # type: ignore[override]
        return (self.type,)

    def __contains__(self, key):  # type: ignore[override]
        return True


class ByPassTypeTupleProxy(tuple):
    """Replacement for ByPassTypeTuple to mirror wildcard fallback behavior."""

    def __new__(cls, values):
        return super().__new__(cls, values)

    def __getitem__(self, index):  # type: ignore[override]
        if index >= len(self):
            return AnyTypeProxy("*")
        return super().__getitem__(index)


def _restore_special_value(value: Any) -> Any:
    if isinstance(value, dict):
        if value.get("__pyisolate_any_type__"):
            return AnyTypeProxy(value.get("value", "*"))
        if value.get("__pyisolate_flexible_optional__"):
            flex_type = _restore_special_value(value.get("type"))
            data_raw = value.get("data")
            data = (
                {k: _restore_special_value(v) for k, v in data_raw.items()}
                if isinstance(data_raw, dict)
                else {}
            )
            return FlexibleOptionalInputProxy(flex_type, data)
        if value.get("__pyisolate_tuple__") is not None:
            return tuple(
                _restore_special_value(v) for v in value["__pyisolate_tuple__"]
            )
        if value.get("__pyisolate_bypass_tuple__") is not None:
            return ByPassTypeTupleProxy(
                tuple(
                    _restore_special_value(v)
                    for v in value["__pyisolate_bypass_tuple__"]
                )
            )
        return {k: _restore_special_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore_special_value(v) for v in value]
    return value


def _serialize_special_value(value: Any) -> Any:
    if isinstance(value, AnyTypeProxy):
        return {"__pyisolate_any_type__": True, "value": str(value)}
    if isinstance(value, FlexibleOptionalInputProxy):
        return {
            "__pyisolate_flexible_optional__": True,
            "type": _serialize_special_value(value.type),
            "data": {k: _serialize_special_value(v) for k, v in value.items()},
        }
    if isinstance(value, ByPassTypeTupleProxy):
        return {
            "__pyisolate_bypass_tuple__": [_serialize_special_value(v) for v in value]
        }
    if isinstance(value, tuple):
        return {"__pyisolate_tuple__": [_serialize_special_value(v) for v in value]}
    if isinstance(value, list):
        return [_serialize_special_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_special_value(v) for k, v in value.items()}
    return value


def _restore_input_types_local(raw: Dict[str, object]) -> Dict[str, object]:
    if not isinstance(raw, dict):
        return raw  # type: ignore[return-value]

    restored: Dict[str, object] = {}
    for section, entries in raw.items():
        if isinstance(entries, dict) and entries.get("__pyisolate_flexible_optional__"):
            restored[section] = _restore_special_value(entries)
        elif isinstance(entries, dict):
            restored[section] = {
                k: _restore_special_value(v) for k, v in entries.items()
            }
        else:
            restored[section] = _restore_special_value(entries)
    return restored


class HelperProxiesService(ProxiedSingleton):
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
            raise RuntimeError("HelperProxiesService RPC caller is not configured")
        return cls._rpc

    async def rpc_restore_input_types(self, raw: Dict[str, object]) -> Dict[str, object]:
        restored = _restore_input_types_local(raw)
        return _serialize_special_value(restored)


def restore_input_types(raw: Dict[str, object]) -> Dict[str, object]:
    """Restore serialized INPUT_TYPES payload back into ComfyUI-compatible objects."""
    if os.environ.get("PYISOLATE_CHILD") == "1":
        payload = call_singleton_rpc(
            HelperProxiesService._get_caller(),
            "rpc_restore_input_types",
            raw,
        )
        return _restore_input_types_local(payload)
    return _restore_input_types_local(raw)


__all__ = [
    "AnyTypeProxy",
    "FlexibleOptionalInputProxy",
    "ByPassTypeTupleProxy",
    "HelperProxiesService",
    "restore_input_types",
]
