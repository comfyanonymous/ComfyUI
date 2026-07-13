"""
ComfyUI TorchAO INT4 weight-only quantization backend.
...
"""
import torch

_AVAILABLE = False
try:
    import torchao
    from .linear import TINT4Linear
    from .quantize import quantize_model, reconstruct_int4_state_dict
    from .quarot import build_hadamard, rotate_weight
    _AVAILABLE = True
except ImportError:
    pass

if not _AVAILABLE:
    raise ImportError(
        "torchao is required for INT4 quantization. "
        "Install: pip install torchao>=0.17.0"
    )

__all__ = [
    "TINT4Linear",
    "quantize_model",
    "reconstruct_int4_state_dict",
    "build_hadamard",
    "rotate_weight",
]
