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
import ctypes
import importlib
import importlib.util
import os
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch


_FST_MODULE = None
_FST_LOCK = threading.Lock()
_FST_LOADED = False
_GDS_INITIALIZED = False
_MISSING = object()
_NOGDS_CHUNK_BYTES_DEFAULT = 64 * 1024 * 1024
_PINNED_INFLIGHT = collections.deque()


def _reap_pinned_inflight():
    if not _PINNED_INFLIGHT:
        return
    pending = collections.deque()
    while _PINNED_INFLIGHT:
        event, tensor = _PINNED_INFLIGHT.popleft()
        if event.query():
            continue
        pending.append((event, tensor))
    _PINNED_INFLIGHT.extend(pending)


def _require_fastsafetensors():
    global _FST_MODULE
    with _FST_LOCK:
        if _FST_MODULE is None:
            if importlib.util.find_spec("fastsafetensors") is None:
                raise ImportError(
                    "fastsafetensors is required for safetensors streaming. "
                    "Install it with: pip install 'fastsafetensors @ https://github.com/"
                    "foundation-model-stack/fastsafetensors/archive/refs/heads/main.zip'"
                )
            _FST_MODULE = importlib.import_module("fastsafetensors")
    return _FST_MODULE


def _init_fastsafetensors_lib():
    global _FST_LOADED
    fst = _require_fastsafetensors()
    if not _FST_LOADED:
        fst.cpp.load_library_functions()
        _FST_LOADED = True
    return fst


def _init_gds():
    global _GDS_INITIALIZED
    fst = _init_fastsafetensors_lib()
    if not _GDS_INITIALIZED:
        if fst.cpp.init_gds() != 0:
            raise RuntimeError("fastsafetensors init_gds() failed")
        _GDS_INITIALIZED = True


@dataclass(frozen=True)
class TensorMeta:
    dtype: torch.dtype
    shape: Tuple[int, ...]
    numel: int
    nbytes: int
    data_offsets: Tuple[int, int]
    filename: str
    fst_dtype: object
    strides: Tuple[int, ...]


class SafeTensorIndex:
    def __init__(self, filename: str):
        fst = _init_fastsafetensors_lib()
        framework = fst.frameworks.get_framework_op("pytorch")
        metadata = fst.common.SafeTensorsMetadata.from_file(filename, framework)
        self._filename = filename
        self._metadata = metadata
        self._framework = framework
        from fastsafetensors.frameworks import _torch as fst_torch
        self._dtype_map = fst_torch.dtype_convert
        self._tensor_meta: Dict[str, TensorMeta] = {}
        for key, frame in metadata.tensors.items():
            torch_dtype = self._dtype_map.get(frame.dtype, None)
            if torch_dtype is None:
                raise ValueError(f"Unsupported safetensors dtype {frame.dtype} in {filename}")
            numel = 1
            for s in frame.shape:
                numel *= s
            nbytes = numel * framework.get_dtype_size(frame.dtype)
            self._tensor_meta[key] = TensorMeta(
                dtype=torch_dtype,
                shape=tuple(frame.shape),
                numel=numel,
                nbytes=nbytes,
                data_offsets=(frame.data_offsets[0], frame.data_offsets[1]),
                filename=filename,
                fst_dtype=frame.dtype,
                strides=tuple(frame.strides),
            )

    def keys(self) -> Iterable[str]:
        return self._tensor_meta.keys()

    def has(self, key: str) -> bool:
        return key in self._tensor_meta

    def meta(self, key: str) -> TensorMeta:
        return self._tensor_meta[key]

    def metadata(self):
        return self._metadata.metadata

    @property
    def header_length(self) -> int:
        return self._metadata.header_length

    @property
    def size_bytes(self) -> int:
        return self._metadata.size_bytes


class _SafeTensorFile:
    def __init__(self, filename: str, index: SafeTensorIndex):
        self.filename = filename
        self.index = index
        self._fd: Optional[int] = None
        self._gds_handle = None
        self._gds_reader = None
        self._nogds_reader = None
        self._refcount = 1

    def acquire(self) -> "_SafeTensorFile":
        self._refcount += 1
        return self

    def release(self):
        self._refcount -= 1
        if self._refcount <= 0:
            self.close()

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        self._gds_handle = None

    def _ensure_fd(self) -> int:
        if self._fd is None:
            self._fd = os.open(self.filename, os.O_RDONLY, 0o644)
        return self._fd

    def _ensure_nogds_reader(self, use_cuda: bool):
        fst = _init_fastsafetensors_lib()
        if self._nogds_reader is None:
            self._nogds_reader = fst.cpp.nogds_file_reader(
                False, 16 * 1024, 16, use_cuda
            )
        return self._nogds_reader

    def _ensure_gds_reader(self, use_cuda: bool):
        fst = _init_fastsafetensors_lib()
        if self._gds_reader is None:
            self._gds_reader = fst.cpp.gds_file_reader(16, use_cuda)
        return self._gds_reader

    def _ensure_gds_handle(self, use_cuda: bool):
        if self._gds_handle is None:
            fst = _init_fastsafetensors_lib()
            framework = fst.frameworks.get_framework_op("pytorch")
            o_direct = _get_gds_o_direct(framework)
            self._gds_handle = fst.cpp.gds_file_handle(self.filename, o_direct, use_cuda)
        return self._gds_handle

    def read_tensor(
        self,
        meta: TensorMeta,
        device: torch.device,
        dtype: Optional[torch.dtype],
        allow_gds: bool,
        pin_if_cpu: bool,
    ) -> torch.Tensor:
        fst = _init_fastsafetensors_lib()
        framework = fst.frameworks.get_framework_op("pytorch")
        device_is_cuda = device.type == "cuda"
        if device_is_cuda and allow_gds:
            _ensure_gds_ready(device)
            tensor = self._read_tensor_gds(
                fst, framework, meta, device, dtype
            )
            return tensor

        target_dtype = dtype
        if device_is_cuda and pin_if_cpu:
            _reap_pinned_inflight()
            target_dtype = None
        cpu_tensor = self._read_tensor_nogds(
            fst, framework, meta, torch.device("cpu"), target_dtype, pin_memory=bool(device_is_cuda and pin_if_cpu)
        )
        if device_is_cuda:
            gpu_tensor = torch.empty_like(cpu_tensor, device=device)
            gpu_tensor.copy_(cpu_tensor, non_blocking=pin_if_cpu)
            if pin_if_cpu:
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(device))
                _PINNED_INFLIGHT.append((event, cpu_tensor))
            if dtype is not None and dtype != gpu_tensor.dtype:
                gpu_tensor = gpu_tensor.to(dtype=dtype)
            return gpu_tensor
        return cpu_tensor

    def _aligned_range(self, abs_start: int, length: int) -> Tuple[int, int, int]:
        fst = _init_fastsafetensors_lib()
        align = fst.cpp.get_alignment_size()
        aligned_offset = (abs_start // align) * align
        head = abs_start - aligned_offset
        aligned_length = length + head
        tail = aligned_length % align
        if tail:
            aligned_length += align - tail
        return aligned_offset, aligned_length, head

    def _read_tensor_nogds(
        self,
        fst,
        framework,
        meta: TensorMeta,
        device: torch.device,
        dtype: Optional[torch.dtype],
        *,
        pin_memory: bool = False,
    ) -> torch.Tensor:
        fd = self._ensure_fd()
        reader = self._ensure_nogds_reader(use_cuda=False)
        abs_start = self.index.header_length + meta.data_offsets[0]
        length = meta.data_offsets[1] - meta.data_offsets[0]
        chunk_bytes = int(os.getenv("COMFY_SAFETENSORS_NOGDS_CHUNK_BYTES", _NOGDS_CHUNK_BYTES_DEFAULT))
        chunk_bytes = max(1, chunk_bytes)
        ptr_align = framework.get_device_ptr_align()
        dest_tensor = torch.empty_strided(meta.shape, meta.strides, dtype=meta.dtype, device="cpu", pin_memory=pin_memory)
        buffer_length = 0
        buf_ptr = None
        gbuf = None
        try:
            chunk_offset = 0
            while chunk_offset < length:
                chunk_len = min(length - chunk_offset, chunk_bytes)
                aligned_offset, aligned_length, head = self._aligned_range(abs_start + chunk_offset, chunk_len)
                if aligned_offset + aligned_length > self.index.size_bytes:
                    aligned_length = max(0, self.index.size_bytes - aligned_offset)
                needed = aligned_length + ptr_align
                if buf_ptr is None or needed > buffer_length:
                    if buf_ptr is not None:
                        fst.cpp.cpu_free(buf_ptr)
                    buffer_length = needed
                    buf_ptr = fst.cpp.cpu_malloc(buffer_length)
                    gbuf = fst.cpp.gds_device_buffer(buf_ptr, buffer_length, False)
                ptr_off = (- (gbuf.get_base_address() + head)) % ptr_align
                req = reader.submit_read(fd, gbuf, aligned_offset, aligned_length, ptr_off)
                if reader.wait_read(req) < 0:
                    raise RuntimeError("nogds_file_reader read failed")
                src_ptr = gbuf.get_base_address() + ptr_off + head
                dest_ptr = dest_tensor.data_ptr() + chunk_offset
                ctypes.memmove(dest_ptr, src_ptr, chunk_len)
                chunk_offset += chunk_len
        except Exception:
            if buf_ptr is not None:
                fst.cpp.cpu_free(buf_ptr)
            raise
        if buf_ptr is not None:
            fst.cpp.cpu_free(buf_ptr)
        if dtype is not None and dtype != dest_tensor.dtype:
            dest_tensor = dest_tensor.to(dtype=dtype)
        return dest_tensor

    def _read_tensor_gds(
        self,
        fst,
        framework,
        meta: TensorMeta,
        device: torch.device,
        dtype: Optional[torch.dtype],
    ) -> torch.Tensor:
        reader = self._ensure_gds_reader(use_cuda=True)
        handle = self._ensure_gds_handle(use_cuda=True)
        abs_start = self.index.header_length + meta.data_offsets[0]
        length = meta.data_offsets[1] - meta.data_offsets[0]
        aligned_offset, aligned_length, head = self._aligned_range(abs_start, length)
        if aligned_offset + aligned_length > self.index.size_bytes:
            aligned_length = max(0, self.index.size_bytes - aligned_offset)
        ptr_align = framework.get_device_ptr_align()
        buffer_length = aligned_length + ptr_align
        fst_device = _fst_device_from_torch(fst, device)
        gbuf = framework.alloc_tensor_memory(buffer_length, fst_device)
        ptr_off = (- (gbuf.get_base_address() + head)) % ptr_align
        file_length = self.index.size_bytes
        req = reader.submit_read(
            handle, gbuf, aligned_offset, aligned_length, ptr_off, file_length
        )
        if reader.wait_read(req) < 0:
            framework.free_tensor_memory(gbuf, fst_device)
            raise RuntimeError("gds_file_reader read failed")
        owner = _BufferOwner(lambda: framework.free_tensor_memory(gbuf, fst_device))
        tensor = _dlpack_tensor_from_buffer(
            fst, framework, gbuf.get_base_address() + ptr_off + head, meta, device, owner
        )
        if dtype is not None and dtype != tensor.dtype:
            tensor = tensor.to(dtype=dtype)
        return tensor


def _fst_device_from_torch(fst, device: torch.device):
    if device.type == "cuda" and device.index is not None:
        return fst.st_types.Device.from_str(f"cuda:{device.index}")
    return fst.st_types.Device.from_str(device.type)


class _BufferOwner:
    def __init__(self, free_fn):
        self._free_fn = free_fn

    def __del__(self):
        try:
            self._free_fn()
        except Exception:
            pass


def _dlpack_tensor_from_buffer(
    fst,
    framework,
    ptr: int,
    meta: TensorMeta,
    device: torch.device,
    owner: Optional[_BufferOwner],
) -> torch.Tensor:
    disk_dtype = framework.as_workaround_dtype(meta.fst_dtype)
    dev = _fst_device_from_torch(fst, device)
    dl_tensor = fst.dlpack.from_cuda_buffer(ptr, list(meta.shape), list(meta.strides), disk_dtype, dev)
    torch_tensor = framework.from_dlpack(dl_tensor, dev, disk_dtype).real_tensor
    if disk_dtype != meta.fst_dtype:
        torch_tensor = torch_tensor.view(meta.dtype)
    if owner is not None:
        torch_tensor._comfy_disk_buffer_owner = owner
    return torch_tensor


def _validate_dtype_conversion(src: torch.dtype, dst: torch.dtype):
    return


def _get_gds_o_direct(framework) -> bool:
    cuda_ver = framework.get_cuda_ver()
    if cuda_ver and cuda_ver != "0.0":
        ver_parts = cuda_ver.split("-", 1)
        if len(ver_parts) == 2:
            cudavers = list(map(int, ver_parts[1].split(".")))
            if ver_parts[0] == "cuda":
                return not (cudavers[0] > 12 or (cudavers[0] == 12 and cudavers[1] >= 2))
            return True
    return True


def _ensure_gds_ready(device: torch.device):
    fst = _init_fastsafetensors_lib()
    if not fst.common.is_gpu_found():
        raise RuntimeError(
            "GPUDirect requested but GPU runtime library is missing. "
            "Disable GPUDirect by omitting --weights-gds to use disk->RAM->GPU."
        )
    gds_supported = fst.cpp.is_gds_supported(device.index if device.index is not None else 0)
    if gds_supported < 0:
        raise RuntimeError(
            "GPUDirect requested but is_gds_supported() failed. "
            "Disable GPUDirect by omitting --weights-gds to use disk->RAM->GPU."
        )
    if not fst.cpp.is_cufile_found():
        raise RuntimeError(
            "GPUDirect requested but libcufile is missing. "
            "Disable GPUDirect by omitting --weights-gds to use disk->RAM->GPU."
        )
    if gds_supported == 0:
        raise RuntimeError(
            "GPUDirect requested but GDS is unsupported on this platform. "
            "Disable GPUDirect by omitting --weights-gds to use disk->RAM->GPU."
        )
    _init_gds()


class StreamStateDict(collections.abc.MutableMapping):
    is_stream_state_dict = True

    def __init__(
        self,
        index: SafeTensorIndex,
        file: _SafeTensorFile,
        device: torch.device,
        allow_gds: bool = False,
    ):
        self._index = index
        self._file = file
        self._device = device
        self._allow_gds = allow_gds
        self._overrides: Dict[str, torch.Tensor] = {}
        self._deleted: set[str] = set()

    @classmethod
    def from_file(cls, filename: str, device: torch.device, allow_gds: bool = False) -> "StreamStateDict":
        index = SafeTensorIndex(filename)
        file = _SafeTensorFile(filename, index)
        return cls(index, file, device, allow_gds=allow_gds)

    def close(self):
        if self._file is not None:
            self._file.release()
            self._file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def meta(self, key: str) -> TensorMeta:
        if key in self._overrides:
            t = self._overrides[key]
            numel = t.numel()
            return TensorMeta(
                dtype=t.dtype,
                shape=tuple(t.shape),
                numel=numel,
                nbytes=numel * t.element_size(),
                data_offsets=(0, numel * t.element_size()),
                filename="<override>",
                fst_dtype=None,
                strides=tuple(t.stride()),
            )
        if key in self._deleted:
            raise KeyError(key)
        return self._index.meta(key)

    def get_tensor(
        self,
        key: str,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        allow_gds: Optional[bool] = None,
        pin_if_cpu: bool = False,
    ) -> torch.Tensor:
        if key in self._overrides:
            t = self._overrides[key]
            if device is not None and t.device != device:
                t = t.to(device=device)
            if dtype is not None and t.dtype != dtype:
                _validate_dtype_conversion(t.dtype, dtype)
                t = t.to(dtype=dtype)
            return t
        if key in self._deleted:
            raise KeyError(key)
        if device is None:
            device = self._device
        if device.type == "meta":
            meta = self._index.meta(key)
            target_dtype = dtype or meta.dtype
            if dtype is not None and dtype != meta.dtype:
                _validate_dtype_conversion(meta.dtype, dtype)
            return torch.empty(meta.shape, dtype=target_dtype, device="meta")
        if allow_gds is None:
            allow_gds = self._allow_gds
        meta = self._index.meta(key)
        return self._file.read_tensor(meta, device, dtype, allow_gds, pin_if_cpu)

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.get_tensor(key)

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        self._overrides[key] = value
        self._deleted.discard(key)

    def __delitem__(self, key: str) -> None:
        if key in self._overrides:
            del self._overrides[key]
            return
        if key in self._deleted:
            raise KeyError(key)
        if self._index.has(key):
            self._deleted.add(key)
            return
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        for k in self._index.keys():
            if k in self._deleted:
                continue
            if k in self._overrides:
                continue
            yield k
        for k in self._overrides.keys():
            yield k

    def __len__(self) -> int:
        base = len(self._index.keys())
        return base - len(self._deleted) + len(self._overrides)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if key in self._deleted:
            return False
        if key in self._overrides:
            return True
        return self._index.has(key)

    def pop(self, key: str, default: object = _MISSING) -> torch.Tensor:
        if key in self._overrides:
            return self._overrides.pop(key)
        if key in self._deleted:
            if default is _MISSING:
                raise KeyError(key)
            return default
        if self._index.has(key):
            value = self.get_tensor(key)
            self._deleted.add(key)
            self._overrides.pop(key, None)
            return value
        if default is _MISSING:
            raise KeyError(key)
        return default

    def copy(self) -> "StreamStateDict":
        new = StreamStateDict(self._index, self._file.acquire(), self._device, allow_gds=self._allow_gds)
        new._overrides = dict(self._overrides)
        new._deleted = set(self._deleted)
        return new

    def metadata(self):
        return self._index.metadata()


class _BaseViewStateDict(MutableMapping):
    is_stream_state_dict = True

    def __init__(self, base: MutableMapping, mutate_base: bool = False):
        self._base = base
        self._mutate_base = mutate_base
        self._overrides: Dict[str, torch.Tensor] = {}
        self._deleted: set[str] = set()

    def _resolve_base_key(self, key: str) -> Optional[str]:
        return key

    def _iter_base_keys(self) -> Iterable[str]:
        return self._base.keys()

    def get_tensor(
        self,
        key: str,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        allow_gds: Optional[bool] = None,
        pin_if_cpu: bool = False,
    ) -> torch.Tensor:
        if key in self._overrides:
            t = self._overrides[key]
            if device is not None and t.device != device:
                t = t.to(device=device)
            if dtype is not None and t.dtype != dtype:
                _validate_dtype_conversion(t.dtype, dtype)
                t = t.to(dtype=dtype)
            return t
        base_key = self._resolve_base_key(key)
        if base_key is None or key in self._deleted:
            raise KeyError(key)
        if hasattr(self._base, "get_tensor"):
            return self._base.get_tensor(
                base_key, device=device, dtype=dtype, allow_gds=allow_gds, pin_if_cpu=pin_if_cpu
            )
        t = self._base[base_key]
        if device is not None and t.device != device:
            t = t.to(device=device)
        if dtype is not None and t.dtype != dtype:
            _validate_dtype_conversion(t.dtype, dtype)
            t = t.to(dtype=dtype)
        return t

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.get_tensor(key)

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        base_key = self._resolve_base_key(key)
        if self._mutate_base and base_key is not None and base_key in self._base:
            self._base[base_key] = value
        else:
            self._overrides[key] = value
        self._deleted.discard(key)

    def __delitem__(self, key: str) -> None:
        if key in self._overrides:
            del self._overrides[key]
            return
        base_key = self._resolve_base_key(key)
        if base_key is None or key in self._deleted:
            raise KeyError(key)
        if self._mutate_base and base_key in self._base:
            del self._base[base_key]
        else:
            self._deleted.add(key)

    def __iter__(self) -> Iterator[str]:
        for k in self._iter_base_keys():
            if k in self._deleted:
                continue
            yield k
        for k in self._overrides.keys():
            yield k

    def __len__(self) -> int:
        base_keys = list(self._iter_base_keys())
        return len(base_keys) - len(self._deleted) + len(self._overrides)

    def pop(self, key: str, default: object = _MISSING) -> torch.Tensor:
        if key in self._overrides:
            return self._overrides.pop(key)
        base_key = self._resolve_base_key(key)
        if base_key is None or key in self._deleted:
            if default is _MISSING:
                raise KeyError(key)
            return default
        if self._mutate_base:
            try:
                return self._base.pop(base_key)
            except KeyError:
                if default is _MISSING:
                    raise
                return default
        value = self.get_tensor(key)
        self._deleted.add(key)
        self._overrides.pop(key, None)
        return value

    def meta(self, key: str):
        if key in self._overrides:
            t = self._overrides[key]
            numel = t.numel()
            return SimpleNamespace(
                dtype=t.dtype,
                shape=tuple(t.shape),
                numel=numel,
                nbytes=numel * t.element_size(),
            )
        base_key = self._resolve_base_key(key)
        if base_key is None or key in self._deleted:
            raise KeyError(key)
        if hasattr(self._base, "meta"):
            return self._base.meta(base_key)
        t = self._base[base_key]
        numel = t.numel()
        return SimpleNamespace(
            dtype=t.dtype,
            shape=tuple(t.shape),
            numel=numel,
            nbytes=numel * t.element_size(),
        )


class DeviceViewStateDict(_BaseViewStateDict):
    def __init__(
        self,
        base: MutableMapping,
        device: torch.device,
        allow_gds: Optional[bool] = None,
        pin_if_cpu: bool = False,
        mutate_base: bool = False,
    ):
        super().__init__(base, mutate_base=mutate_base)
        self._device = device
        self._allow_gds = allow_gds
        self._pin_if_cpu = pin_if_cpu

    def get_tensor(
        self,
        key: str,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        allow_gds: Optional[bool] = None,
        pin_if_cpu: bool = False,
    ) -> torch.Tensor:
        device = self._device if device is None else device
        allow_gds = self._allow_gds if allow_gds is None else allow_gds
        pin_if_cpu = self._pin_if_cpu if not pin_if_cpu else pin_if_cpu
        return super().get_tensor(
            key, device=device, dtype=dtype, allow_gds=allow_gds, pin_if_cpu=pin_if_cpu
        )

    def meta(self, key: str):
        if key in self._overrides:
            t = self._overrides[key]
            numel = t.numel()
            return SimpleNamespace(
                dtype=t.dtype,
                shape=tuple(t.shape),
                numel=numel,
                nbytes=numel * t.element_size(),
            )
        base_key = self._resolve_base_key(key)
        if base_key is None or key in self._deleted:
            raise KeyError(key)
        if hasattr(self._base, "meta"):
            return self._base.meta(base_key)
        t = self._base[base_key]
        numel = t.numel()
        return SimpleNamespace(
            dtype=t.dtype,
            shape=tuple(t.shape),
            numel=numel,
            nbytes=numel * t.element_size(),
        )

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.get_tensor(key)

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        base_key = self._resolve_base_key(key)
        if self._mutate_base and base_key is not None and base_key in self._base:
            self._base[base_key] = value
        else:
            self._overrides[key] = value
        self._deleted.discard(key)

    def __delitem__(self, key: str) -> None:
        if key in self._overrides:
            del self._overrides[key]
            return
        base_key = self._resolve_base_key(key)
        if base_key is None or key in self._deleted:
            raise KeyError(key)
        if self._mutate_base and base_key in self._base:
            del self._base[base_key]
        else:
            self._deleted.add(key)

    def __iter__(self) -> Iterator[str]:
        for k in self._iter_base_keys():
            if k in self._deleted:
                continue
            yield k
        for k in self._overrides.keys():
            yield k

    def __len__(self) -> int:
        base_keys = list(self._iter_base_keys())
        return len(base_keys) - len(self._deleted) + len(self._overrides)

    def pop(self, key: str, default: object = _MISSING) -> torch.Tensor:
        if key in self._overrides:
            return self._overrides.pop(key)
        base_key = self._resolve_base_key(key)
        if base_key is None or key in self._deleted:
            if default is _MISSING:
                raise KeyError(key)
            return default
        if self._mutate_base:
            try:
                return self._base.pop(base_key)
            except KeyError:
                if default is _MISSING:
                    raise
                return default
        value = self.get_tensor(key)
        self._deleted.add(key)
        self._overrides.pop(key, None)
        return value


class FilterViewStateDict(_BaseViewStateDict):
    def __init__(self, base: MutableMapping, predicate, mutate_base: bool = False):
        super().__init__(base, mutate_base=mutate_base)
        self._predicate = predicate

    def _resolve_base_key(self, key: str) -> Optional[str]:
        if self._predicate(key):
            return key
        return None

    def _iter_base_keys(self) -> Iterable[str]:
        for k in self._base.keys():
            if self._predicate(k):
                yield k


class PrefixViewStateDict(_BaseViewStateDict):
    def __init__(self, base: MutableMapping, source_prefix: str, target_prefix: str = "", mutate_base: bool = False):
        super().__init__(base, mutate_base=mutate_base)
        self._source_prefix = source_prefix
        self._target_prefix = target_prefix
        self._mapping: Dict[str, str] = {}
        self._reverse: Dict[str, str] = {}
        for k in base.keys():
            if not k.startswith(source_prefix):
                continue
            view_key = f"{target_prefix}{k[len(source_prefix):]}"
            self._mapping[k] = view_key
            self._reverse[view_key] = k

    def _resolve_base_key(self, key: str) -> Optional[str]:
        return self._reverse.get(key)

    def _iter_base_keys(self) -> Iterable[str]:
        return self._reverse.keys()


class RenameViewStateDict(_BaseViewStateDict):
    def __init__(
        self,
        base: MutableMapping,
        replace_prefix: Mapping[str, str],
        filter_keys: bool = False,
        mutate_base: bool = False,
    ):
        super().__init__(base, mutate_base=mutate_base)
        self._filter_keys = filter_keys
        self._replace = list(replace_prefix.items())
        self._mapping: Dict[str, str] = {}
        self._reverse: Dict[str, str] = {}
        for k in base.keys():
            view_key = self._replace_key(k)
            if view_key is None:
                continue
            self._mapping[k] = view_key
            self._reverse[view_key] = k

    def _replace_key(self, key: str) -> Optional[str]:
        for rp, dst in self._replace:
            if key.startswith(rp):
                return f"{dst}{key[len(rp):]}"
        if self._filter_keys:
            return None
        return key

    def _resolve_base_key(self, key: str) -> Optional[str]:
        return self._reverse.get(key)

    def _iter_base_keys(self) -> Iterable[str]:
        return self._reverse.keys()


class MergedStateDict(MutableMapping):
    is_stream_state_dict = True

    def __init__(self, *mappings: MutableMapping):
        self._mappings = list(mappings)
        self._overrides: Dict[str, torch.Tensor] = {}
        self._deleted: set[str] = set()

    def __getitem__(self, key: str) -> torch.Tensor:
        if key in self._overrides:
            return self._overrides[key]
        if key in self._deleted:
            raise KeyError(key)
        for mapping in reversed(self._mappings):
            if key in mapping:
                if hasattr(mapping, "get_tensor"):
                    return mapping.get_tensor(key)
                return mapping[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        self._overrides[key] = value
        self._deleted.discard(key)

    def __delitem__(self, key: str) -> None:
        if key in self._overrides:
            del self._overrides[key]
            return
        if key in self._deleted:
            raise KeyError(key)
        if any(key in mapping for mapping in self._mappings):
            self._deleted.add(key)
            return
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        seen = set()
        for mapping in self._mappings:
            for key in mapping.keys():
                if key in self._deleted or key in seen:
                    continue
                seen.add(key)
                yield key
        for key in self._overrides.keys():
            if key not in seen:
                yield key

    def __len__(self) -> int:
        return len(list(self.__iter__()))

    def meta(self, key: str):
        if key in self._overrides:
            t = self._overrides[key]
            numel = t.numel()
            return SimpleNamespace(
                dtype=t.dtype,
                shape=tuple(t.shape),
                numel=numel,
                nbytes=numel * t.element_size(),
            )
        if key in self._deleted:
            raise KeyError(key)
        for mapping in reversed(self._mappings):
            if key in mapping:
                if hasattr(mapping, "meta"):
                    return mapping.meta(key)
                t = mapping[key]
                numel = t.numel()
                return SimpleNamespace(
                    dtype=t.dtype,
                    shape=tuple(t.shape),
                    numel=numel,
                    nbytes=numel * t.element_size(),
                )
        raise KeyError(key)


class MappedStateDict(_BaseViewStateDict):
    def __init__(self, base: MutableMapping, key_map: Mapping[str, str], mutate_base: bool = False):
        super().__init__(base, mutate_base=mutate_base)
        self._base_to_view = dict(key_map)
        self._view_to_base = {v: k for k, v in key_map.items()}

    def _resolve_base_key(self, key: str) -> Optional[str]:
        return self._view_to_base.get(key)

    def _iter_base_keys(self) -> Iterable[str]:
        return self._view_to_base.keys()


def stream_load_state_dict(model, state_dict, strict: bool = False, assign: bool = False):
    if getattr(state_dict, "is_stream_state_dict", False) and hasattr(state_dict, "copy"):
        state_dict = state_dict.copy()
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, "_metadata", None)

    def load(module, local_state_dict, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        if assign:
            local_metadata["assign_to_params_buffers"] = assign
        module._load_from_state_dict(
            local_state_dict,
            prefix,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for name, child in module._modules.items():
            if child is not None:
                child_prefix = f"{prefix}{name}."
                child_state_dict = FilterViewStateDict(
                    local_state_dict, lambda k, p=child_prefix: k.startswith(p), mutate_base=False
                )
                load(child, child_state_dict, child_prefix)
        incompatible = torch.nn.modules.module._IncompatibleKeys(missing_keys, unexpected_keys)
        for hook in module._load_state_dict_post_hooks.values():
            out = hook(module, incompatible)
            if out is not None:
                raise RuntimeError("load_state_dict post hook returned a value, which is unsupported.")

    load(model, state_dict)
    if strict:
        if len(unexpected_keys) > 0:
            error_msgs.insert(0, 'Unexpected key(s) in state_dict: {}. '.format(', '.join(f'"{k}"' for k in unexpected_keys)))
        if len(missing_keys) > 0:
            error_msgs.insert(0, 'Missing key(s) in state_dict: {}. '.format(', '.join(f'"{k}"' for k in missing_keys)))
    if len(error_msgs) > 0:
        raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(model.__class__.__name__, "\n\t".join(error_msgs)))
    return torch.nn.modules.module._IncompatibleKeys(missing_keys, unexpected_keys)
