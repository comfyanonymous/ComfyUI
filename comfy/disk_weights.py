"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Comfy

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import collections
import weakref
from dataclasses import dataclass
from typing import Dict, Optional

import torch


ALLOW_GDS = False
PIN_IF_CPU = False
DISK_WEIGHTS_ENABLED = False


@dataclass
class DiskTensorRef:
    state_dict: object
    key: str
    meta: object
    requires_grad: bool
    is_buffer: bool

    def load(self, device: torch.device, allow_gds: bool, pin_if_cpu: bool) -> torch.Tensor:
        dtype = getattr(self.meta, "dtype", None)
        if hasattr(self.state_dict, "get_tensor"):
            return self.state_dict.get_tensor(
                self.key,
                device=device,
                dtype=dtype,
                allow_gds=allow_gds,
                pin_if_cpu=pin_if_cpu,
            )
        tensor = self.state_dict[self.key]
        if device is not None and tensor.device != device:
            tensor = tensor.to(device=device)
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        return tensor


class DiskWeightRegistry:
    def __init__(self):
        self._registry = weakref.WeakKeyDictionary()

    def register(self, module: torch.nn.Module, name: str, ref: DiskTensorRef):
        module_refs = self._registry.setdefault(module, {})
        module_refs[name] = ref

    def get(self, module: torch.nn.Module) -> Optional[Dict[str, DiskTensorRef]]:
        return self._registry.get(module)

    def has(self, module: torch.nn.Module) -> bool:
        return module in self._registry


@dataclass
class CacheEntry:
    module_ref: weakref.ReferenceType
    name: str
    size_bytes: int
    is_buffer: bool


class DiskWeightCache:
    def __init__(self, max_bytes: int = 0):
        self.max_bytes = max_bytes
        self.current_bytes = 0
        self._entries: "collections.OrderedDict[tuple[int, str], CacheEntry]" = collections.OrderedDict()

    def set_limit(self, max_bytes: int):
        self.max_bytes = max_bytes
        self._evict_if_needed()

    def _entry_key(self, module: torch.nn.Module, name: str) -> tuple[int, str]:
        return (id(module), name)

    def record(self, module: torch.nn.Module, name: str, tensor: torch.Tensor, is_buffer: bool):
        if tensor.device.type != "cpu":
            return
        size_bytes = tensor.numel() * tensor.element_size()
        key = self._entry_key(module, name)
        if key in self._entries:
            entry = self._entries.pop(key)
            self.current_bytes -= entry.size_bytes
        module_ref = weakref.ref(module, self._drop_module_entries)
        self._entries[key] = CacheEntry(module_ref=module_ref, name=name, size_bytes=size_bytes, is_buffer=is_buffer)
        self.current_bytes += size_bytes
        self._evict_if_needed()

    def touch(self, module: torch.nn.Module, name: str):
        key = self._entry_key(module, name)
        if key in self._entries:
            entry = self._entries.pop(key)
            self._entries[key] = entry

    def evict_bytes(self, bytes_to_free: int):
        freed = 0
        while self._entries and freed < bytes_to_free:
            _, entry = self._entries.popitem(last=False)
            freed += entry.size_bytes
            self.current_bytes -= entry.size_bytes
            module = entry.module_ref()
            if module is not None:
                _evict_module_weight(module, entry.name, entry.is_buffer)
        return freed

    def _drop_module_entries(self, module_ref: weakref.ReferenceType):
        to_remove = []
        for key, entry in self._entries.items():
            if entry.module_ref is module_ref:
                to_remove.append(key)
        for key in to_remove:
            entry = self._entries.pop(key)
            self.current_bytes -= entry.size_bytes

    def _evict_if_needed(self):
        while self._entries and self.current_bytes > self.max_bytes:
            _, entry = self._entries.popitem(last=False)
            self.current_bytes -= entry.size_bytes
            module = entry.module_ref()
            if module is not None:
                _evict_module_weight(module, entry.name, entry.is_buffer)


REGISTRY = DiskWeightRegistry()
CACHE = DiskWeightCache(0)


def configure(cache_bytes: int, allow_gds: bool, pin_if_cpu: bool, enabled: bool = True):
    global ALLOW_GDS, PIN_IF_CPU, DISK_WEIGHTS_ENABLED
    ALLOW_GDS = allow_gds
    PIN_IF_CPU = pin_if_cpu
    DISK_WEIGHTS_ENABLED = enabled
    CACHE.set_limit(cache_bytes if enabled else 0)
    if not enabled:
        CACHE._entries.clear()
        CACHE.current_bytes = 0


def disk_weights_enabled() -> bool:
    return DISK_WEIGHTS_ENABLED


def register_module_weights(module: torch.nn.Module, state_dict, prefix: str = ""):
    if not disk_weights_enabled():
        return
    if not hasattr(state_dict, "meta") or not hasattr(state_dict, "get_tensor"):
        return
    for name, param in module.named_parameters(recurse=True):
        key = f"{prefix}{name}" if prefix else name
        if key in state_dict:
            meta = state_dict.meta(key)
            ref = DiskTensorRef(state_dict=state_dict, key=key, meta=meta, requires_grad=param.requires_grad, is_buffer=False)
            REGISTRY.register(module, name, ref)
            if param.device.type == "cpu":
                CACHE.record(module, name, param, is_buffer=False)
    for name, buf in module.named_buffers(recurse=True):
        key = f"{prefix}{name}" if prefix else name
        if key in state_dict and buf is not None:
            meta = state_dict.meta(key)
            ref = DiskTensorRef(state_dict=state_dict, key=key, meta=meta, requires_grad=False, is_buffer=True)
            REGISTRY.register(module, name, ref)
            if buf.device.type == "cpu":
                CACHE.record(module, name, buf, is_buffer=True)


def _evict_module_weight(module: torch.nn.Module, name: str, is_buffer: bool):
    ref = REGISTRY.get(module)
    if not ref or name not in ref:
        return
    disk_ref = ref[name]
    shape = getattr(disk_ref.meta, "shape", None)
    dtype = getattr(disk_ref.meta, "dtype", None)
    if shape is None or dtype is None:
        return
    meta_tensor = torch.empty(shape, dtype=dtype, device="meta")
    if is_buffer:
        module._buffers[name] = meta_tensor
    else:
        module._parameters[name] = torch.nn.Parameter(meta_tensor, requires_grad=disk_ref.requires_grad)


def _find_tensor_device(args, kwargs) -> Optional[torch.device]:
    def check(obj):
        if torch.is_tensor(obj):
            return obj.device
        if isinstance(obj, (list, tuple)):
            for item in obj:
                dev = check(item)
                if dev is not None:
                    return dev
        if isinstance(obj, dict):
            for item in obj.values():
                dev = check(item)
                if dev is not None:
                    return dev
        return None

    dev = check(args)
    if dev is not None:
        return dev
    return check(kwargs)


def ensure_module_materialized(module: torch.nn.Module, target_device: torch.device):
    refs = REGISTRY.get(module)
    if not refs:
        return
    for name, disk_ref in refs.items():
        if name in module._parameters:
            current = module._parameters[name]
            is_buffer = False
        elif name in module._buffers:
            current = module._buffers[name]
            is_buffer = True
        else:
            continue
        if current is None:
            continue
        if current.device.type != "meta":
            if current.device.type == "cpu":
                CACHE.touch(module, name)
            continue
        tensor = disk_ref.load(target_device, ALLOW_GDS, PIN_IF_CPU)
        if is_buffer:
            module._buffers[name] = tensor
        else:
            module._parameters[name] = torch.nn.Parameter(tensor, requires_grad=disk_ref.requires_grad)
        if tensor.device.type == "cpu":
            CACHE.record(module, name, tensor, is_buffer=is_buffer)


def disk_weight_pre_hook(module: torch.nn.Module, args, kwargs):
    if not REGISTRY.has(module):
        return
    if getattr(module, "comfy_cast_weights", False):
        target_device = torch.device("cpu")
    else:
        target_device = _find_tensor_device(args, kwargs) or torch.device("cpu")
    ensure_module_materialized(module, target_device)


def attach_disk_weight_hooks(model: torch.nn.Module):
    if not disk_weights_enabled():
        return
    for module in model.modules():
        if getattr(module, "_disk_weight_hook_attached", False):
            continue
        module.register_forward_pre_hook(disk_weight_pre_hook)
        module._disk_weight_hook_attached = True


def evict_ram_cache(bytes_to_free: int):
    if bytes_to_free <= 0:
        return 0
    return CACHE.evict_bytes(bytes_to_free)
