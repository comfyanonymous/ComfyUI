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

# ★ FIX 3: 删除 __doc__ 覆写（5 行）
# 原因：这些类/函数的 .py 文件里已有 docstring，重复赋值会：
#   1. 覆盖原始文档（如果两处不一致会让人困惑）
#   2. 干扰 Sphinx 等文档工具的正确引用
#   3. 违反 DRY 原则——文档应该只在源码处维护一份
