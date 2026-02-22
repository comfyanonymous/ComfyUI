import math
import torch
from typing import NamedTuple

from comfy.quant_ops import QuantizedTensor

class TensorGeometry(NamedTuple):
    shape: any
    dtype: torch.dtype

    def element_size(self):
        info = torch.finfo(self.dtype) if self.dtype.is_floating_point else torch.iinfo(self.dtype)
        return info.bits // 8

    def numel(self):
        return math.prod(self.shape)

def tensors_to_geometries(tensors, dtype=None):
    geometries = []
    for t in tensors:
        if t is None or isinstance(t, QuantizedTensor):
            geometries.append(t)
            continue
        tdtype = t.dtype
        if hasattr(t, "_model_dtype"):
            tdtype = t._model_dtype
        if dtype is not None:
            tdtype = dtype
        geometries.append(TensorGeometry(shape=t.shape, dtype=tdtype))
    return geometries

def vram_aligned_size(tensor):
    if isinstance(tensor, list):
        return sum([vram_aligned_size(t) for t in tensor])

    if isinstance(tensor, QuantizedTensor):
        inner_tensors, _ = tensor.__tensor_flatten__()
        return vram_aligned_size([ getattr(tensor, attr) for attr in inner_tensors ])

    if tensor is None:
        return 0

    size = tensor.numel() * tensor.element_size()
    aligment_req = 1024
    return (size + aligment_req - 1) // aligment_req * aligment_req

def interpret_gathered_like(tensors, gathered):
    offset = 0
    dest_views = []

    if gathered.dim() != 1 or gathered.element_size() != 1:
        raise ValueError(f"Buffer must be 1D and single-byte (got {gathered.dim()}D {gathered.dtype})")

    for tensor in tensors:

        if tensor is None:
            dest_views.append(None)
            continue

        if isinstance(tensor, QuantizedTensor):
            inner_tensors, qt_ctx = tensor.__tensor_flatten__()
            templates = { attr: getattr(tensor, attr) for attr in inner_tensors }
        else:
            templates = { "data": tensor }

        actuals = {}
        for attr, template in templates.items():
            size = template.numel() * template.element_size()
            if offset + size > gathered.numel():
                raise ValueError(f"Buffer too small: needs {offset + size} bytes, but only has {gathered.numel()}. ")
            actuals[attr] = gathered[offset:offset+size].view(dtype=template.dtype).view(template.shape)
            offset += vram_aligned_size(template)

        if isinstance(tensor, QuantizedTensor):
            dest_views.append(QuantizedTensor.__tensor_unflatten__(actuals, qt_ctx, 0, 0))
        else:
            dest_views.append(actuals["data"])

    return dest_views

aimdo_enabled = False
