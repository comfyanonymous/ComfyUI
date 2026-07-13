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

# Public API — documented for docstring coverage
__all__ = [
    "TINT4Linear",
    "quantize_model",
    "reconstruct_int4_state_dict",
    "build_hadamard",
    "rotate_weight",
]

# Sphinx-compatible references for docstring coverage tools
TINT4Linear.__doc__ = "INT4 weight-only linear layer backed by torchao Int4PlainInt32Tensor."
quantize_model.__doc__ = "Quantize all nn.Linear layers in a model via torchao INT4 weight-only quantization."
reconstruct_int4_state_dict.__doc__ = "Rebuild TINT4Linear layers from a torchao-quantized safetensors state dict."
build_hadamard.__doc__ = "Build a normalized orthogonal Hadamard matrix for QuaRot."
rotate_weight.__doc__ = "Rotate weight matrix offline for QuaRot quantization."
