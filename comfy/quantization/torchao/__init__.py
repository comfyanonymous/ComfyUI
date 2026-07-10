"""
ComfyUI TorchAO INT4 weight-only quantization backend.

W4A16 scheme: weights quantized to INT4 via torchao,
activations remain FP16. Supports CUDA, Intel XPU, and CPU.

Includes:
- TINT4Linear: INT4 linear layer with LoRA forward injection + QuaRot
- quantize_model: torchao INT4 quantization entry point
- reconstruct_int4_state_dict: rebuild TINT4Linear from safetensors
- build_hadamard / rotate_weight: QuaRot Hadamard rotation
"""

from .linear import TINT4Linear
from .quantize import quantize_model, reconstruct_int4_state_dict
from .quarot import build_hadamard, rotate_weight

__all__ = [
    "TINT4Linear",
    "quantize_model",
    "reconstruct_int4_state_dict",
    "build_hadamard",
    "rotate_weight",
]
